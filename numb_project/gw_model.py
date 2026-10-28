from typing import Any, Callable, Mapping

from shimmer import ContrastiveLoss, DomainModule, GWDecoder, GWEncoder, GWLosses2Domains, GWModule, GlobalWorkspaceBase, LatentsDomainGroupsT, LossCoefs, LossOutput, ModelModeT, RawDomainGroupT, RawDomainGroupsT, SelectionBase, SingleDomainSelection, combine_loss
from shimmer.modules.losses import GWLosses
import torch
from torch import nn
from torch.optim import AdamW
import torch.nn.functional as F

from numb_project.constants import BASE, DEVICE
from numb_project.domain_module import DigitAngleDomain, MNISTDomain, get_domains_config, load_domain_modules
from numb_project.equivariant_gw import EquivariantGWStructure, EquivariantMLP, field_type
from numb_project.mnist_vae_so2 import SO2VAE
from numb_project.operation_rotation_module import FusedRotateOperationModule, LatentRotateOperationModule, PixelRotateOperationModule
from numb_project.utils import angle_to_cos_sin, rotate_images
from numb_project.utils import get_from_dict_or_val

class MyGlobalWorkspace(GlobalWorkspaceBase):
    """GW entièrement construit à partir de `config` (même format que
    config/config.yaml). La config est enregistrée dans les hyperparamètres du
    checkpoint : `MyGlobalWorkspace.load_from_checkpoint(path)` reconstruit
    exactement les modules entraînés (dont les rotations activées), sans
    dépendre de la config actuelle."""

    def __init__(self, config: dict[str, Any]) -> None:
        gw_mod, selection_mod, loss_mod, rotation_mods = get_global_workspace_mods(config)
        super().__init__(gw_mod, selection_mod, loss_mod, config["training"]["optim"]["lr"])
        self.config = config
        self.rotation_mods = rotation_mods

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
        if "vae_digit" in self.rotation_mods:
            self.log(f"{mode}/vae_digit_image_weight", self.rotation_mods["vae_digit"].image_weight,
                     batch_size=batch_size, add_dataloader_idx=False)

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
        rotation_mods: nn.ModuleDict,
    ) -> None:
        super().__init__(gw_mod, selection_mod, domain_mods, loss_coefs, contrastive_fn)
        self.rotation_mods = rotation_mods
        self.representation_loss_coefs = representation_loss_coefs

    def _rotated_batch(
        self, raw_data: RawDomainGroupsT
    ) -> tuple[LatentsDomainGroupsT, RawDomainGroupsT]:
        raw_rot = make_rotated_gw_batch(raw_data)
        with torch.no_grad():
            latents_rot = {
                domains: {domain: self.domain_mods[domain].encode(x) for domain, x in group.items()}
                for domains, group in raw_rot.items()
            }
        return latents_rot, raw_rot

    def compute_representation_loss(
        self, raw_data: RawDomainGroupsT, domain_latents: LatentsDomainGroupsT
    ) -> torch.Tensor:
        losses = {
            **self.demi_cycle_loss(domain_latents, raw_data),
            **self.cycle_loss(domain_latents, raw_data),
            **self.translation_loss(domain_latents, raw_data),
            **self.contrastive_loss(domain_latents),
        }
        return combine_loss(losses, self.representation_loss_coefs)

    def _losses(self, raw_data: RawDomainGroupsT, domain_latents: LatentsDomainGroupsT) -> dict[str, torch.Tensor]:
        """Losses d'une version du batch, nommées comme dans `loss_coefficients`."""
        return {
            "representation_loss": self.compute_representation_loss(raw_data, domain_latents),
            **{f"rotate_{name}_loss": mod.loss(self.gw_mod, raw_data) for name, mod in self.rotation_mods.items()},
        }

    def step(
        self,
        raw_data: RawDomainGroupsT,
        domain_latents: LatentsDomainGroupsT,
        mode: ModelModeT,
    ) -> LossOutput:
        latents_rot, raw_rot = self._rotated_batch(raw_data)

        losses = self._losses(raw_data, domain_latents)
        losses |= {f"{name}_rotated": loss for name, loss in self._losses(raw_rot, latents_rot).items()}

        total_loss = sum(self.loss_coefs[name] * loss for name, loss in losses.items())
        return LossOutput(total_loss, losses)

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

class DigitGWDecoder(GWDecoder):
    """Décodeur GW -> latent du domaine digit : le MLP de GWDecoder, suivi d'un
    softmax sur les BASE premières dimensions (probabilités des chiffres) et
    d'une normalisation de (cos, sin). Toutes les prédictions digit (traductions,
    demi-cycles, cycles, passages par digit des rotations) restent ainsi dans
    l'espace des digits : simplexe des chiffres x cercle des angles."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = super().forward(z)
        return torch.cat([F.softmax(x[:, :BASE], dim=-1), F.normalize(x[:, BASE:], dim=-1)], dim=-1)


def build_gw_encoders_decoders(
    domain_modules: Mapping[str, DomainModule],
    workspace_dim: int,
    encoders_hidden_dim: int | Mapping[str, int],
    encoders_n_layers: int | Mapping[str, int],
    decoders_hidden_dim: int | Mapping[str, int],
    decoders_n_layers: int | Mapping[str, int],
    gw_structure: EquivariantGWStructure | None = None,
) -> tuple[nn.ModuleDict, nn.ModuleDict]:
    """Crée (non entraînés) un encodeur latent du domaine -> GW et un décodeur
    GW -> latent du domaine pour chaque domaine. Les tailles cachées et nombres
    de couches sont soit communs (int), soit donnés par domaine (dict).
    Encodeurs : GWEncoder pour tous les domaines. Décodeurs : DigitGWDecoder
    pour le domaine digit (sortie contrainte), GWDecoder pour les autres.
    Avec `gw_structure` (GW équivariant), le domaine image a un encodeur et un
    décodeur équivariants (`EquivariantMLP`), les autres restent classiques."""
    gw_encoders: dict[str, nn.Module] = {}
    gw_decoders: dict[str, nn.Module] = {}
    for name, module in domain_modules.items():
        if gw_structure is not None and isinstance(module, MNISTDomain):
            assert isinstance(module.vae, SO2VAE), "GW équivariant : le domaine image doit utiliser le VAE so2"
            vae_type = field_type(module.vae.hparams.n_scalars, module.vae.hparams.n_vectors)
            gw_encoders[name] = EquivariantMLP(
                vae_type, gw_structure.field_type,
                get_from_dict_or_val(encoders_hidden_dim, name, "global_workspace.encoders.hidden_dim"),
                get_from_dict_or_val(encoders_n_layers, name, "global_workspace.encoders.n_layers"),
            )
            gw_decoders[name] = EquivariantMLP(
                gw_structure.field_type, vae_type,
                get_from_dict_or_val(decoders_hidden_dim, name, "global_workspace.decoders.hidden_dim"),
                get_from_dict_or_val(decoders_n_layers, name, "global_workspace.decoders.n_layers"),
            )
            continue
        gw_encoders[name] = GWEncoder(
            module.latent_dim,
            get_from_dict_or_val(encoders_hidden_dim, name, "global_workspace.encoders.hidden_dim"),
            workspace_dim,
            get_from_dict_or_val(encoders_n_layers, name, "global_workspace.encoders.n_layers"),
        )
        decoder_cls: type[GWDecoder]
        match module:
            case DigitAngleDomain():
                decoder_cls = DigitGWDecoder
            case _:
                decoder_cls = GWDecoder
        gw_decoders[name] = decoder_cls(
            workspace_dim,
            get_from_dict_or_val(decoders_hidden_dim, name, "global_workspace.decoders.hidden_dim"),
            module.latent_dim,
            get_from_dict_or_val(decoders_n_layers, name, "global_workspace.decoders.n_layers"),
        )
    return nn.ModuleDict(gw_encoders), nn.ModuleDict(gw_decoders)

def build_rotation_modules(
    rotation_config: dict[str, Any], gw_size: int, vae_size: int, digit_size: int
) -> nn.ModuleDict:
    """Une rotation apprise par espace listé dans `rotation_config["enabled"]`,
    parmi : pixels, latent du VAE, état du GW, état du GW passé par digit,
    fusion des deux états GW, vecteur digit, et "vae_digit" (rotations "vae" et
    "digit" fusionnées dans le GW, qui réutilise ces deux modules : ils doivent
    être activés aussi)."""
    common = {"degrees": rotation_config["degrees"], "n_steps": rotation_config["n_steps"]}
    latent_sizes = {"vae": vae_size, "gw": gw_size, "gw_digit": gw_size, "gw_fused": gw_size, "digit": digit_size}

    def build(space: str) -> nn.Module:
        if space == "pixel":
            return PixelRotateOperationModule(latent_dim=rotation_config["pixel"]["latent_dim"], **common)
        assert space in latent_sizes, f"opération de rotation inconnue : {space}"
        return LatentRotateOperationModule(space, latent_sizes[space], rotation_config[space]["hidden_size"],
                                           rotation_config[space]["latent_dim"], **common)

    enabled = rotation_config["enabled"]
    mods = {space: build(space) for space in enabled if space != "vae_digit"}
    if "vae_digit" in enabled:
        assert "vae" in mods and "digit" in mods, "vae_digit réutilise les rotations vae et digit : les activer aussi"
        vae_digit_config = rotation_config["vae_digit"]
        mods["vae_digit"] = FusedRotateOperationModule(
            mods["vae"], mods["digit"], vae_digit_config["image_weight"],
            # absent des configs des checkpoints antérieurs : poids fixe
            vae_digit_config.get("learn_image_weight", False), **common)
    return nn.ModuleDict({space: mods[space] for space in enabled})

def get_global_workspace_mods(
        config: dict[str, Any],
        ) -> tuple[GWModule, SingleDomainSelection, MyCustomGWLosses, nn.ModuleDict]:
    selection_mod = SingleDomainSelection()
    contrastive_fn = ContrastiveLoss(torch.tensor([1 / 0.07]).log(), "mean", False)
    gw_size = config["global_workspace"]["latent_dim"]
    # Clés absentes des configs des checkpoints antérieurs : VAE convolutif, GW classique
    vae = config["domains"].get("image", {}).get("vae", "conv")
    equivariant_config = config["global_workspace"].get("equivariant")
    gw_structure = None if equivariant_config is None else EquivariantGWStructure(gw_size, equivariant_config["n_vectors"])

    domain_modules = load_domain_modules(get_domains_config(["image", "digit"], config["domains"]["digit"], vae))
    gw_encoders, gw_decoders = build_gw_encoders_decoders(
        domain_modules,
        gw_size,
        config["global_workspace"]["encoders"]["hidden_dim"],
        config["global_workspace"]["encoders"]["n_layers"],
        config["global_workspace"]["decoders"]["hidden_dim"],
        config["global_workspace"]["decoders"]["n_layers"],
        gw_structure,
    )

    gw_mod = GWModule(
        domain_modules= domain_modules,
        workspace_dim=gw_size,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,
        fusion_activation_fn=torch.tanh if gw_structure is None else gw_structure.fusion_activation(),
    )
    # Structure de l'état du GW (None = GW classique), cf EquivariantGWStructure.rotate
    gw_mod.gw_structure = gw_structure

    rotation_mods = build_rotation_modules(
        config["global_workspace"]["rotation_operations"], gw_size,
        domain_modules["image"].latent_dim, domain_modules["digit"].latent_dim,
    )

    loss_mod = MyCustomGWLosses(
        gw_mod = gw_mod,
        selection_mod=selection_mod,
        domain_mods=domain_modules,
        representation_loss_coefs=config["global_workspace"]["representation_loss_coefficients"],
        contrastive_fn=contrastive_fn,
        rotation_mods=rotation_mods,
        loss_coefs = config["global_workspace"]["loss_coefficients"],
    )

    return gw_mod, selection_mod, loss_mod, rotation_mods

