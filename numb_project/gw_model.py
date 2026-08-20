from typing import Any, Callable, Mapping

from shimmer import ContrastiveLoss, DomainModule, GWLosses2Domains, GWModule, GlobalWorkspaceBase, LatentsDomainGroupsT, LossCoefs, LossOutput, ModelModeT, RawDomainGroupT, RawDomainGroupsT, SelectionBase, SingleDomainSelection, combine_loss
from shimmer.modules.losses import GWLosses
import torch
from torch import nn
from torch.optim import AdamW
import torch.nn.functional as F

from numb_project.constants import BASE, DEVICE
from numb_project.data_module import rotated_raw_image_groups
from numb_project.domain_module import LoadedDomainConfig, load_domains
from numb_project.operation_arithmetic_module import AddOperationModule, SubOperationModule
from numb_project.operation_module import ChainOperationModule, OperationSelectionModule
from numb_project.operation_rotation_module import GwRotateOperationModule, PixelRotateOperationModule
from numb_project.utils import merge_metrics

class MyGlobalWorkspace(GlobalWorkspaceBase):
    def __init__(
        self,
        gw_mod: GWModule,
        selection_mod: SelectionBase,
        loss_mod: GWLosses,
        operation_mod: nn.ModuleDict,
        operation_selection_mod: OperationSelectionModule,
        task_mod: ChainOperationModule,
        operation_rotate_pixel_mod: PixelRotateOperationModule,
        operation_rotate_gw_mod: GwRotateOperationModule,
        optim_lr: float = 1e-3,
    ) -> None:
        super().__init__(gw_mod, selection_mod, loss_mod, optim_lr)
        self.operation_module = operation_mod
        self.operation_selection_module = operation_selection_mod
        self.task_mod = task_mod
        self.operation_rotate_pixel_module = operation_rotate_pixel_mod
        self.operation_rotate_gw_module = operation_rotate_gw_mod

    def configure_optimizers(self) -> dict[str, AdamW]:
        optimizer = AdamW(
            self.parameters(),
            lr=self.optim_lr,
            weight_decay=self.optim_weight_decay,
        )
        return {"optimizer": optimizer}

    def input_to_gw(self, raw_group: RawDomainGroupT) -> torch.Tensor:
        image = raw_group["image"]
        image_to_latent = self.encode_domain(image, "image")
        gw_state = self.gw_mod.gw_encoders["image"](image_to_latent)

        return gw_state

    def generic_step(
        self,
        batch: RawDomainGroupsT,
        mode: ModelModeT,
    ) -> torch.Tensor:
        domain_latents = self.encode_domains(batch)
        raw_group = batch[frozenset({"digit", "image"})]
                
        init_gw_state = self.input_to_gw(raw_group)

        tasks, right_addend_value, task_targets = self.task_mod.create_chain_add_tasks(raw_group)
        task_predictions, roll_sequence = self.task_mod.forward_chain(init_gw_state, tasks, raw_group["digit"])

        loss_output = self.loss_mod.step(
            batch, domain_latents, mode,
            task_predictions=task_predictions,
            task_targets=task_targets,
            roll_sequence=roll_sequence,
            right_addend_value=right_addend_value,
        )

        batch_size = raw_group["digit"].shape[0]
        for name, metric in loss_output.all.items():
            self.log(f"{mode}/{name}", metric, batch_size=batch_size, add_dataloader_idx=False)

        return loss_output.loss

    def on_train_epoch_start(self) -> None:
        min_temp = 0.1
        max_temp = 1.0
        decay_epochs = 10
        progress = min(self.current_epoch / decay_epochs, 1.0)

        self.operation_selection_module.temperature = max_temp * (min_temp / max_temp) ** progress

class MyCustomGWLosses(GWLosses2Domains):
    def __init__(
        self,
        gw_mod: GWModule,
        selection_mod: SelectionBase,
        domain_mods: dict[str, DomainModule],
        loss_coefs: LossCoefs | Mapping[str, float],
        representation_loss_coefs: LossCoefs | Mapping[str, float],
        contrastive_fn: Callable[[torch.Tensor, torch.Tensor], LossOutput],
        operation_mod: nn.ModuleDict,
        operation_selection_mod: OperationSelectionModule,
        task_mod: ChainOperationModule,
        operation_rotate_pixel_mod: PixelRotateOperationModule,
        operation_rotate_gw_mod: GwRotateOperationModule,
    ) -> None:
        super().__init__(gw_mod, selection_mod, domain_mods, loss_coefs, contrastive_fn)
        self.operation_mod = operation_mod
        self.task_mod = task_mod
        self.operation_rotate_pixel_mod = operation_rotate_pixel_mod
        self.operation_rotate_gw_mod = operation_rotate_gw_mod
        self.representation_loss_coefs = representation_loss_coefs

    def _rotated_image_groups(
        self, domain_latents: LatentsDomainGroupsT, raw_data: RawDomainGroupsT
    ) -> tuple[LatentsDomainGroupsT, RawDomainGroupsT]:
        raw_rot = rotated_raw_image_groups(raw_data)

        single_key = frozenset({"image"})
        pair_key = frozenset({"digit", "image"})

        image_mod = self.domain_mods["image"]
        with torch.no_grad():
            rotated_latent = image_mod.encode(raw_rot[single_key]["image"])

        digit_latent = domain_latents[pair_key]["digit"]

        latents_rot = {
            single_key: {"image": rotated_latent},
            pair_key: {"image": rotated_latent, "digit": digit_latent},
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
        task_predictions: torch.Tensor,
        task_targets: torch.Tensor,
        roll_sequence: torch.Tensor,
        right_addend_value: torch.Tensor,
    ) -> LossOutput:
        loss_config = self.loss_coefs

        latents_rot, raw_rot = self._rotated_image_groups(domain_latents, raw_data)

        representation_loss, metrics = self.compute_representation_loss(raw_data, domain_latents)
        representation_loss_rotated, metrics_rotated = self.compute_representation_rotated_loss(raw_rot, latents_rot)
        metrics = merge_metrics(metrics, metrics_rotated)

        add_loss = self.operation_mod["add"].loss(self.gw_mod, domain_latents)
        sub_loss = self.operation_mod["sub"].loss(self.gw_mod, domain_latents)

        rotate_pixel_loss = self.operation_rotate_pixel_mod.loss(self.gw_mod, raw_data)
        rotate_pixel_loss_rotated = self.operation_rotate_pixel_mod.loss(self.gw_mod, raw_rot)
        rotate_gw_loss = self.operation_rotate_gw_mod.loss(self.gw_mod, raw_data)
        rotate_gw_loss_rotated = self.operation_rotate_gw_mod.loss(self.gw_mod, raw_rot)

        task_loss_output = self.task_mod.loss(task_predictions, task_targets, roll_sequence, right_addend_value)

        metrics["task_loss"] = task_loss_output.loss
        metrics["add_loss"] = add_loss
        metrics["sub_loss"] = sub_loss
        metrics["rotate_pixel_loss"] = rotate_pixel_loss
        metrics["rotate_pixel_loss_rotated"] = rotate_pixel_loss_rotated
        metrics["rotate_gw_loss"] = rotate_gw_loss
        metrics["rotate_gw_loss_rotated"] = rotate_gw_loss_rotated

        total_loss = (
            loss_config['representation_loss'] * representation_loss
            + loss_config['representation_loss_rotated'] * representation_loss_rotated
            + loss_config['add_loss'] * add_loss
            + loss_config['sub_loss'] * sub_loss
            + loss_config['rotate_pixel_loss'] * rotate_pixel_loss
            + loss_config['rotate_pixel_loss_rotated'] * rotate_pixel_loss_rotated
            + loss_config['rotate_gw_loss'] * rotate_gw_loss
            + loss_config['rotate_gw_loss_rotated'] * rotate_gw_loss_rotated
            + loss_config['task_loss'] * task_loss_output.loss
        )

        return LossOutput(total_loss, metrics)

def get_global_workspace_mods(
        config: dict[str, Any],
        domains_configs: list[LoadedDomainConfig],
        ) -> tuple[GWModule, SingleDomainSelection, nn.ModuleDict, OperationSelectionModule, ChainOperationModule, MyCustomGWLosses, PixelRotateOperationModule, GwRotateOperationModule]:
    selection_mod = SingleDomainSelection()
    contrastive_fn = ContrastiveLoss(torch.tensor([1 / 0.07]).log(), "mean", False)
    gw_size = config["global_workspace"]["latent_dim"]
    batch_size = config["training"]["batch_size"]

    domain_modules, gw_encoders, gw_decoders = load_domains(
        domains_configs,
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

    operation_add_mod = AddOperationModule(gw_size, config["global_workspace"]["encoders"]["hidden_dim"]["operation"], gw_size)
    operation_sub_mod = SubOperationModule(gw_size, config["global_workspace"]["encoders"]["hidden_dim"]["operation"], gw_size)

    operation_mod = nn.ModuleDict({
        "add": operation_add_mod,
        "sub": operation_sub_mod,
    })

    rotation_ops_config = config["global_workspace"]["rotation_operations"]

    operation_rotate_pixel_mod = PixelRotateOperationModule(
        latent_dim=rotation_ops_config["pixel"]["latent_dim"],
    )

    # Rotation apprise directement dans l'espace GW (gw_size), à comparer à
    # operation_rotate_pixel_mod (pixels bruts). Reste hors de operation_mod :
    # ChainOperationModule/OperationSelectionModule fixent leur output_size sur
    # les opérations arithmétiques (add/sub) pour la tâche de chaîne digit.
    operation_rotate_gw_mod = GwRotateOperationModule(
        gw_size=gw_size,
        hidden_size=rotation_ops_config["gw"]["hidden_size"],
        latent_dim=rotation_ops_config["gw"]["latent_dim"],
    )

    operation_selection_mod = OperationSelectionModule(input_size=gw_size, output_size=1 + len(operation_mod), hidden_size=128, batch_size=batch_size, device=DEVICE)

    task_mod = ChainOperationModule(20, 0, 9, BASE, gw_mod, operation_selection_mod, operation_mod)

    loss_mod = MyCustomGWLosses(
        gw_mod = gw_mod,
        selection_mod=selection_mod,
        domain_mods=domain_modules,
        representation_loss_coefs=config["global_workspace"]["representation_loss_coefficients"],
        contrastive_fn=contrastive_fn,
        operation_mod = operation_mod,
        operation_selection_mod=operation_selection_mod,
        task_mod=task_mod,
        operation_rotate_pixel_mod=operation_rotate_pixel_mod,
        operation_rotate_gw_mod=operation_rotate_gw_mod,
        loss_coefs = config["global_workspace"]["loss_coefficients"],
    )

    return gw_mod, selection_mod, operation_mod, operation_selection_mod, task_mod, loss_mod, operation_rotate_pixel_mod, operation_rotate_gw_mod


def setup_global_workspace(config: dict[str, Any], domains_configs: list[LoadedDomainConfig]) -> MyGlobalWorkspace:
    gw_mod, selection_mod, operation_mod, operation_selection_mod, task_mod, loss_mod, operation_rotate_pixel_mod, operation_rotate_gw_mod = get_global_workspace_mods(config, domains_configs)

    global_workspace = MyGlobalWorkspace(
        gw_mod=gw_mod,
        selection_mod=selection_mod,
        loss_mod=loss_mod,
        optim_lr=config["training"]["optim"]["lr"],
        operation_mod= operation_mod,
        operation_selection_mod=operation_selection_mod,
        task_mod=task_mod,
        operation_rotate_pixel_mod=operation_rotate_pixel_mod,
        operation_rotate_gw_mod=operation_rotate_gw_mod,
    )

    return global_workspace

def freeze_except_attention(model: MyGlobalWorkspace) -> None:
    for param in model.parameters():
        param.requires_grad = False

    for param in model.operation_selection_module.parameters():
        param.requires_grad = True

def load_pretrained_global_workspace(
    config: dict[str, Any],
    domains_configs: list[LoadedDomainConfig],
    checkpoint_path: str,
    device: str = "cuda",
) -> MyGlobalWorkspace:
    global_workspace = setup_global_workspace(config, domains_configs)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    global_workspace.load_state_dict(state_dict)

    freeze_except_attention(global_workspace)
    global_workspace.to(device)

    return global_workspace