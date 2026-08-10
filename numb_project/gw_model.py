from typing import Any

from shimmer import ContrastiveLoss, GWLosses2Domains, GWModule, GlobalWorkspaceBase, LatentsDomainGroupsT, LossOutput, ModelModeT, RawDomainGroupsT, SelectionBase, SingleDomainSelection, combine_loss
from shimmer.modules.losses import GWLosses
import torch
from torch.optim import AdamW
import torch.nn.functional as F

from numb_project.domain_module import LoadedDomainConfig, OperationModule, load_domains
import random

class MyGlobalWorkspace(GlobalWorkspaceBase):
    def __init__(
        self,
        gw_mod: GWModule,
        selection_mod: SelectionBase,
        loss_mod: GWLosses,
        optim_lr: float = 1e-3,
        operation_mod = None
    ) -> None:
        super().__init__(gw_mod, selection_mod, loss_mod, optim_lr)
        self.operation_module = operation_mod

    def configure_optimizers(self) -> dict[str, AdamW]:
        optimizer = AdamW(
            self.parameters(),
            lr=self.optim_lr,
            weight_decay=self.optim_weight_decay,
        )
        return {"optimizer": optimizer}

class MyCustomGWLosses(GWLosses2Domains):
    def __init__(self, gw_mod, selection_mod, domain_mods, loss_coefs, contrastive_fn, operation_mod) -> None:
        super().__init__(gw_mod, selection_mod, domain_mods, loss_coefs, contrastive_fn)
        self.operation_mod = operation_mod

    def compute_shift_loss(self, digit_one_hot: torch.Tensor, image: torch.Tensor, shift_value: int) -> torch.Tensor:
        target_one_hot = torch.roll(digit_one_hot, shifts=shift_value, dims=1)

        with torch.no_grad():
            gw_input = self.gw_mod.gw_encoders["image"](image)
            gw_target = self.gw_mod.gw_encoders["digit"](target_one_hot)

        batch_size = digit_one_hot.shape[0]
        task = torch.full(
            (batch_size, 1),
            fill_value=float(shift_value),
            device=digit_one_hot.device,
        )

        gw_pred = self.operation_mod(gw_input, task)
        return F.mse_loss(gw_pred, gw_target)

    def compute_shift_cycle_loss(self, digit_one_hot: torch.Tensor):
        with torch.no_grad():
            gw_input = self.gw_mod.gw_encoders["digit"](digit_one_hot)

        batch_size = digit_one_hot.shape[0]
        shift = random.randrange(1,5)

        add = torch.full(
                    (batch_size, 1),
                    fill_value=float(1),
                    device=digit_one_hot.device,
                )

        sub = torch.full(
                    (batch_size, 1),
                    fill_value=float(-1),
                    device=digit_one_hot.device,
                )


        gw_pred = gw_input
        for i in range(shift):
            gw_pred = self.operation_mod(gw_pred, add)

        for i in range(shift):
            gw_pred = self.operation_mod(gw_pred, sub)

        return F.mse_loss(gw_pred, gw_input)

    def step(
        self,
        raw_data: RawDomainGroupsT,
        domain_latents: LatentsDomainGroupsT,
        mode: ModelModeT,
    ) -> LossOutput:

        metrics: dict[str, torch.Tensor] = {}

        metrics.update(self.demi_cycle_loss(domain_latents, raw_data))
        metrics.update(self.cycle_loss(domain_latents, raw_data))
        metrics.update(self.translation_loss(domain_latents, raw_data))
        metrics.update(self.contrastive_loss(domain_latents))

        representation_loss = combine_loss(metrics, self.loss_coefs)

        digit_one_hot = domain_latents[frozenset({'digit', 'image'})]['digit']
        image = domain_latents[frozenset({'digit', 'image'})]['image']

        add_loss = self.compute_shift_loss(digit_one_hot, image, 1)
        sub_loss = self.compute_shift_loss(digit_one_hot, image, -1)
        operation_cycle = self.compute_shift_cycle_loss(digit_one_hot)

        total_loss = representation_loss + 10*(add_loss + sub_loss + operation_cycle)
        
        return LossOutput(total_loss, metrics)


def get_global_workspace_mods(
        config: dict[str, Any],
        domains_configs: list[LoadedDomainConfig]
        ) -> tuple[GWModule, SingleDomainSelection, MyCustomGWLosses]:
    selection_mod = SingleDomainSelection()
    contrastive_fn = ContrastiveLoss(torch.tensor([1 / 0.07]).log(), "mean", False)

    domain_modules, gw_encoders, gw_decoders = load_domains(
        domains_configs,
        config["global_workspace"]["latent_dim"],
        config["global_workspace"]["encoders"]["hidden_dim"],
        config["global_workspace"]["encoders"]["n_layers"],
        config["global_workspace"]["decoders"]["hidden_dim"],
        config["global_workspace"]["decoders"]["n_layers"],
    )

    gw_mod = GWModule(
        domain_modules= domain_modules,
        workspace_dim=config["global_workspace"]["latent_dim"],
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,
        fusion_activation_fn=torch.tanh
    )

    operation_mod = OperationModule(10, 1, 128)

    loss_mod = MyCustomGWLosses(
        gw_mod = gw_mod,
        selection_mod=selection_mod,
        domain_mods=domain_modules,
        loss_coefs=config["global_workspace"]["loss_coefficients"],
        contrastive_fn=contrastive_fn,
        operation_mod = operation_mod
    )

    return gw_mod, selection_mod, operation_mod, loss_mod


def setup_global_workspace(config: dict[str, Any], domains_configs: list[LoadedDomainConfig]) -> MyGlobalWorkspace:
    gw_mod, selection_mod, operation_mod, loss_mod = get_global_workspace_mods(config, domains_configs)

    global_workspace = MyGlobalWorkspace(
        gw_mod=gw_mod,
        selection_mod=selection_mod,
        loss_mod=loss_mod,
        optim_lr=config["training"]["optim"]["lr"],
        operation_mod= operation_mod
    )

    return global_workspace
