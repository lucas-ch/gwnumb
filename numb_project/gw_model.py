from typing import Any, Callable, Mapping

from shimmer import ContrastiveLoss, DomainModule, GWDecoder, GWEncoder, GWLosses2Domains, GWModule, GlobalWorkspaceBase, LatentsDomainGroupsT, LossCoefs, LossOutput, ModelModeT, RawDomainGroupT, RawDomainGroupsT, SelectionBase, SingleDomainSelection, combine_loss
from shimmer.modules.losses import GWLosses
import torch
from torch import nn
from torch.optim import AdamW
import torch.nn.functional as F

from numb_project.constants import BASE, DEVICE
from numb_project.domain_module import LoadedDomainConfig, load_domain_modules
from numb_project.operation_rotation_module import GwRotateOperationModule, PixelRotateOperationModule
from numb_project.utils import angle_to_cos_sin, rotate_images
from numb_project.utils import get_from_dict_or_val, merge_metrics

class MyGlobalWorkspace(GlobalWorkspaceBase):
    def __init__(
        self,
        gw_mod: GWModule,
        selection_mod: SelectionBase,
        loss_mod: GWLosses,
        operation_rotate_pixel_mod: PixelRotateOperationModule,
        operation_rotate_gw_mod: GwRotateOperationModule,
        optim_lr: float = 1e-3,
    ) -> None:
        super().__init__(gw_mod, selection_mod, loss_mod, optim_lr)
        self.operation_rotate_pixel_module = operation_rotate_pixel_mod
        self.operation_rotate_gw_module = operation_rotate_gw_mod

    def configure_optimizers(self) -> dict[str, AdamW]:
        optimizer = AdamW(
            self.parameters(),
            lr=self.optim_lr,
            weight_decay=self.optim_weight_decay,
        )
        return {"optimizer": optimizer}

    def generic_step(
        self,
        batch: RawDomainGroupsT,
        mode: ModelModeT,
    ) -> torch.Tensor:
        domain_latents = self.encode_domains(batch)
        raw_group = batch[frozenset({"digit", "image"})]
                
        loss_output = self.loss_mod.step(
            batch, domain_latents, mode,
        )

        batch_size = raw_group["digit"].shape[0]
        for name, metric in loss_output.all.items():
            self.log(f"{mode}/{name}", metric, batch_size=batch_size, add_dataloader_idx=False)

        return loss_output.loss

class MyCustomGWLosses(GWLosses2Domains):
    def __init__(
        self,
        gw_mod: GWModule,
        selection_mod: SelectionBase,
        domain_mods: dict[str, DomainModule],
        loss_coefs: LossCoefs | Mapping[str, float],
        representation_loss_coefs: LossCoefs | Mapping[str, float],
        contrastive_fn: Callable[[torch.Tensor, torch.Tensor], LossOutput],
        operation_rotate_pixel_mod: PixelRotateOperationModule,
        operation_rotate_gw_mod: GwRotateOperationModule,
    ) -> None:
        super().__init__(gw_mod, selection_mod, domain_mods, loss_coefs, contrastive_fn)
        self.operation_rotate_pixel_mod = operation_rotate_pixel_mod
        self.operation_rotate_gw_mod = operation_rotate_gw_mod
        self.representation_loss_coefs = representation_loss_coefs

    def _rotated_batch(
        self, raw_data: RawDomainGroupsT
    ) -> tuple[LatentsDomainGroupsT, RawDomainGroupsT]:
        """Batch tourné (cf make_rotated_gw_batch) et ses latents de domaines."""
        raw_rot = make_rotated_gw_batch(raw_data)
        with torch.no_grad():
            latents_rot = {
                domains: {domain: self.domain_mods[domain].encode(x) for domain, x in group.items()}
                for domains, group in raw_rot.items()
            }
        return latents_rot, raw_rot

    def compute_representation_loss(
        self, raw_data: RawDomainGroupsT, domain_latents: LatentsDomainGroupsT
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        metrics = {}
        metrics.update(self.demi_cycle_loss(domain_latents, raw_data))
        metrics.update(self.cycle_loss(domain_latents, raw_data))
        metrics.update(self.translation_loss(domain_latents, raw_data))
        metrics.update(self.contrastive_loss(domain_latents))

        representation_loss = combine_loss(metrics, self.representation_loss_coefs)

        return representation_loss, metrics

    def compute_representation_rotated_loss(
        self,
        raw_rot: RawDomainGroupsT,
        latents_rot: LatentsDomainGroupsT,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.compute_representation_loss(raw_rot, latents_rot)

    def step(
        self,
        raw_data: RawDomainGroupsT,
        domain_latents: LatentsDomainGroupsT,
        mode: ModelModeT,
    ) -> LossOutput:
        loss_config = self.loss_coefs

        latents_rot, raw_rot = self._rotated_batch(raw_data)

        representation_loss, metrics = self.compute_representation_loss(raw_data, domain_latents)
        representation_loss_rotated, metrics_rotated = self.compute_representation_rotated_loss(raw_rot, latents_rot)
        metrics = merge_metrics(metrics, metrics_rotated)

        rotate_pixel_loss = self.operation_rotate_pixel_mod.loss(self.gw_mod, raw_data)
        rotate_pixel_loss_rotated = self.operation_rotate_pixel_mod.loss(self.gw_mod, raw_rot)
        rotate_gw_loss = self.operation_rotate_gw_mod.loss(self.gw_mod, raw_data)
        rotate_gw_loss_rotated = self.operation_rotate_gw_mod.loss(self.gw_mod, raw_rot)

        metrics["rotate_pixel_loss"] = rotate_pixel_loss
        metrics["rotate_pixel_loss_rotated"] = rotate_pixel_loss_rotated
        metrics["rotate_gw_loss"] = rotate_gw_loss
        metrics["rotate_gw_loss_rotated"] = rotate_gw_loss_rotated
        for name, value in self.operation_rotate_gw_mod.metrics(self.gw_mod, raw_data).items():
            metrics[f"rotate_gw_{name}"] = value
        for name, value in self.operation_rotate_gw_mod.metrics(self.gw_mod, raw_rot).items():
            metrics[f"rotate_gw_{name}_rotated"] = value

        total_loss = (
            loss_config['representation_loss'] * representation_loss
            + loss_config['representation_loss_rotated'] * representation_loss_rotated
            + loss_config['rotate_pixel_loss'] * rotate_pixel_loss
            + loss_config['rotate_pixel_loss_rotated'] * rotate_pixel_loss_rotated
            + loss_config['rotate_gw_loss'] * rotate_gw_loss
            + loss_config['rotate_gw_loss_rotated'] * rotate_gw_loss_rotated
        )

        return LossOutput(total_loss, metrics)

def make_rotated_gw_batch(raw_data: RawDomainGroupsT) -> RawDomainGroupsT:
    """Version tournée d'un batch du GW (format shimmer : groupe de domaines ->
    données brutes), utilisée pour les losses `*_rotated`."""

    batch_size = len(raw_data[frozenset({"digit", "image"})]["image"])
    assert all(len(x) == batch_size for group in raw_data.values() for x in group.values()), \
        "tous les groupes doivent avoir la même taille de batch pour partager les angles"
    device = raw_data[frozenset({"digit", "image"})]["image"].device
    angle_deg = torch.rand(batch_size, device=device) * 360.0

    def rotate_group(group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        rotated = {}
        if "image" in group:
            rotated["image"] = rotate_images(group["image"], angle_deg)
        if "digit" in group:
            rotated["digit"] = torch.cat([group["digit"][:, :BASE], angle_to_cos_sin(angle_deg)], dim=-1)
        return rotated

    return {domains: rotate_group(group) for domains, group in raw_data.items()}

def build_gw_encoders_decoders(
    domain_modules: Mapping[str, DomainModule],
    workspace_dim: int,
    encoders_hidden_dim: int | Mapping[str, int],
    encoders_n_layers: int | Mapping[str, int],
    decoders_hidden_dim: int | Mapping[str, int],
    decoders_n_layers: int | Mapping[str, int],
) -> tuple[nn.ModuleDict, nn.ModuleDict]:
    """Crée (non entraînés) un encodeur latent du domaine -> GW et un décodeur
    GW -> latent du domaine pour chaque domaine. Les tailles cachées et nombres
    de couches sont soit communs (int), soit donnés par domaine (dict)."""
    gw_encoders: dict[str, nn.Module] = {}
    gw_decoders: dict[str, nn.Module] = {}
    for name, module in domain_modules.items():
        gw_encoders[name] = GWEncoder(
            module.latent_dim,
            get_from_dict_or_val(encoders_hidden_dim, name, "global_workspace.encoders.hidden_dim"),
            workspace_dim,
            get_from_dict_or_val(encoders_n_layers, name, "global_workspace.encoders.n_layers"),
        )
        gw_decoders[name] = GWDecoder(
            workspace_dim,
            get_from_dict_or_val(decoders_hidden_dim, name, "global_workspace.decoders.hidden_dim"),
            module.latent_dim,
            get_from_dict_or_val(decoders_n_layers, name, "global_workspace.decoders.n_layers"),
        )
    return nn.ModuleDict(gw_encoders), nn.ModuleDict(gw_decoders)

def get_global_workspace_mods(
        config: dict[str, Any],
        domains_configs: list[LoadedDomainConfig],
        ) -> tuple[GWModule, SingleDomainSelection, nn.ModuleDict, MyCustomGWLosses, PixelRotateOperationModule, GwRotateOperationModule]:
    selection_mod = SingleDomainSelection()
    contrastive_fn = ContrastiveLoss(torch.tensor([1 / 0.07]).log(), "mean", False)
    gw_size = config["global_workspace"]["latent_dim"]

    domain_modules = load_domain_modules(domains_configs)
    gw_encoders, gw_decoders = build_gw_encoders_decoders(
        domain_modules,
        gw_size,
        config["global_workspace"]["encoders"]["hidden_dim"],
        config["global_workspace"]["encoders"]["n_layers"],
        config["global_workspace"]["decoders"]["hidden_dim"],
        config["global_workspace"]["decoders"]["n_layers"],
    )

    gw_mod = GWModule(
        domain_modules= domain_modules,
        workspace_dim=gw_size,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,
    )

    rotation_ops_config = config["global_workspace"]["rotation_operations"]

    operation_rotate_pixel_mod = PixelRotateOperationModule(
        latent_dim=rotation_ops_config["pixel"]["latent_dim"],
        degrees=rotation_ops_config["degrees"],
    )

    operation_rotate_gw_mod = GwRotateOperationModule(
        gw_size=gw_size,
        hidden_size=rotation_ops_config["gw"]["hidden_size"],
        latent_dim=rotation_ops_config["gw"]["latent_dim"],
        degrees=rotation_ops_config["degrees"],
    )

    loss_mod = MyCustomGWLosses(
        gw_mod = gw_mod,
        selection_mod=selection_mod,
        domain_mods=domain_modules,
        representation_loss_coefs=config["global_workspace"]["representation_loss_coefficients"],
        contrastive_fn=contrastive_fn,
        operation_rotate_pixel_mod=operation_rotate_pixel_mod,
        operation_rotate_gw_mod=operation_rotate_gw_mod,
        loss_coefs = config["global_workspace"]["loss_coefficients"],
    )

    return gw_mod, selection_mod, loss_mod, operation_rotate_pixel_mod, operation_rotate_gw_mod


def setup_global_workspace(config: dict[str, Any], domains_configs: list[LoadedDomainConfig]) -> MyGlobalWorkspace:
    gw_mod, selection_mod, loss_mod, operation_rotate_pixel_mod, operation_rotate_gw_mod = get_global_workspace_mods(config, domains_configs)

    global_workspace = MyGlobalWorkspace(
        gw_mod=gw_mod,
        selection_mod=selection_mod,
        loss_mod=loss_mod,
        optim_lr=config["training"]["optim"]["lr"],
        operation_rotate_pixel_mod=operation_rotate_pixel_mod,
        operation_rotate_gw_mod=operation_rotate_gw_mod,
    )

    return global_workspace