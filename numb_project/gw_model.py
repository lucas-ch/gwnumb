from typing import Any

from shimmer import ContrastiveLoss, GWLosses2Domains, GWModule, GlobalWorkspaceBase, LossOutput, ModelModeT, RawDomainGroupsT, SelectionBase, SingleDomainSelection, combine_loss
from shimmer.modules.losses import GWLosses
import torch
from torch import nn
from torch.optim import AdamW
import torch.nn.functional as F

from numb_project.constants import BASE, DEVICE
from numb_project.domain_module import LoadedDomainConfig, load_domains
from numb_project.operation_module import SubOperationModule, AddOperationModule, ChainOperationModule, OperationSelectionModule

class MyGlobalWorkspace(GlobalWorkspaceBase):
    def __init__(
        self,
        gw_mod: GWModule,
        selection_mod: SelectionBase,
        loss_mod: GWLosses,
        optim_lr: float = 1e-3,
        operation_mod = None,
        operation_selection_mod = None,
        task_mod=None
    ) -> None:
        super().__init__(gw_mod, selection_mod, loss_mod, optim_lr)
        self.operation_module = operation_mod
        self.operation_selection_module = operation_selection_mod
        self.task_mod = task_mod

    def configure_optimizers(self) -> dict[str, AdamW]:
        optimizer = AdamW(
            self.parameters(),
            lr=self.optim_lr,
            weight_decay=self.optim_weight_decay,
        )
        return {"optimizer": optimizer}

    def input_to_gw(self, raw_group):
        image = raw_group["image"]
        image_to_latent = self.encode_domain(image, "image")
        gw_state = self.gw_mod.gw_encoders["image"](image_to_latent)

        return gw_state

    def generic_step(self, batch: RawDomainGroupsT, mode: ModelModeT, start_chain=0, end_chain=10):
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

        return loss_output.loss

    def on_train_epoch_start(self):
        min_temp = 0.1
        max_temp = 1.0
        decay_epochs = 10
        progress = min(self.current_epoch / decay_epochs, 1.0)

        self.operation_selection_module.temperature = max_temp * (min_temp / max_temp) ** progress

class MyCustomGWLosses(GWLosses2Domains):
    def __init__(self, gw_mod, selection_mod, domain_mods, loss_coefs, representation_loss_coefs, contrastive_fn, operation_mod, operation_selection_mod, task_mod) -> None:
        super().__init__(gw_mod, selection_mod, domain_mods, loss_coefs, contrastive_fn)
        self.operation_mod = operation_mod
        self.task_mod = task_mod
        self.representation_loss_coefs = representation_loss_coefs

    def compute_representation_loss(self, raw_data, domain_latents):
        metrics = {}
        metrics.update(self.demi_cycle_loss(domain_latents, raw_data))
        metrics.update(self.cycle_loss(domain_latents, raw_data))
        metrics.update(self.translation_loss(domain_latents, raw_data))
        metrics.update(self.contrastive_loss(domain_latents))
        representation_loss = combine_loss(metrics, self.representation_loss_coefs)

        return representation_loss, metrics

    def step(self, raw_data, domain_latents, mode, task_predictions, task_targets, roll_sequence, right_addend_value) -> LossOutput:
        loss_config = self.loss_coefs

        representation_loss, metrics = self.compute_representation_loss(raw_data, domain_latents)
        add_loss = self.operation_mod["add"].loss(self.gw_mod, domain_latents)
        sub_loss = self.operation_mod["sub"].loss(self.gw_mod, domain_latents)
        task_loss_output = self.task_mod.loss(task_predictions, task_targets, roll_sequence, right_addend_value)
        metrics.update(task_loss_output.metrics)
        metrics["task_loss"] = task_loss_output.loss

        total_loss = loss_config['representation_loss']*representation_loss + loss_config['add_loss']*add_loss + loss_config['sub_loss']*sub_loss + loss_config['task_loss']*task_loss_output.loss

        return LossOutput(total_loss, metrics)

def get_global_workspace_mods(
        config: dict[str, Any],
        domains_configs: list[LoadedDomainConfig],
        ) -> tuple[GWModule, SingleDomainSelection, MyCustomGWLosses]:
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
        "sub": operation_sub_mod
    })

    operation_selection_mod = OperationSelectionModule(input_size=gw_size, output_size=3, hidden_size=128, batch_size=batch_size, device=DEVICE)

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
        loss_coefs = config["global_workspace"]["loss_coefficients"],
    )

    return gw_mod, selection_mod, operation_mod, operation_selection_mod, task_mod, loss_mod


def setup_global_workspace(config: dict[str, Any], domains_configs: list[LoadedDomainConfig]) -> MyGlobalWorkspace:
    gw_mod, selection_mod, operation_mod, operation_selection_mod, task_mod, loss_mod = get_global_workspace_mods(config, domains_configs)

    global_workspace = MyGlobalWorkspace(
        gw_mod=gw_mod,
        selection_mod=selection_mod,
        loss_mod=loss_mod,
        optim_lr=config["training"]["optim"]["lr"],
        operation_mod= operation_mod,
        operation_selection_mod=operation_selection_mod,
        task_mod=task_mod
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