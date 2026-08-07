from shimmer import ContrastiveLoss, GWLosses2Domains, GWModule, GlobalWorkspaceBase, LatentsDomainGroupsT, LossOutput, ModelModeT, RawDomainGroupsT, SelectionBase, SingleDomainSelection, combine_loss
from shimmer.modules.losses import GWLosses
import torch
from torch.optim import AdamW

from numb_project.domain_module import LoadedDomainConfig, load_pretrained_domains


class MyGlobalWorkspace(GlobalWorkspaceBase):
    def __init__(
        self,
        gw_mod: GWModule,
        selection_mod: SelectionBase,
        loss_mod: GWLosses,
        optim_lr: float = 1e-3,
    ):
        super().__init__(gw_mod, selection_mod, loss_mod, optim_lr)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.optim_lr,
            weight_decay=self.optim_weight_decay,
        )
        return {"optimizer": optimizer}

class MyCustomGWLosses(GWLosses2Domains):
    def __init__(self, gw_mod, selection_mod, domain_mods, loss_coefs, contrastive_fn):
        super().__init__(gw_mod, selection_mod, domain_mods, loss_coefs, contrastive_fn)

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
        
        return LossOutput(combine_loss(metrics, self.loss_coefs), metrics)


def get_global_workspace_params(config):
    selection_mod = SingleDomainSelection()
    contrastive_fn = ContrastiveLoss(torch.tensor([1 / 0.07]).log(), "mean", False)

    image_domain_config = LoadedDomainConfig(domain_type="image")
    digit_domain_config = LoadedDomainConfig(domain_type="digit")

    domain_modules, gw_encoders, gw_decoders = load_pretrained_domains(
        [image_domain_config, digit_domain_config],
        config["global_workspace"]["latent_dim"],
        config["global_workspace"]["encoders"]["hidden_dim"],
        config["global_workspace"]["encoders"]["n_layers"],
        config["global_workspace"]["decoders"]["hidden_dim"],
        config["global_workspace"]["decoders"]["n_layers"],
    )

    gw_mod = GWModule(
        domain_modules= domain_modules,
        workspace_dim=12,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,
        fusion_activation_fn=torch.tanh
    )

    loss_mod = MyCustomGWLosses(
        gw_mod = gw_mod,
        selection_mod=selection_mod,
        domain_mods=domain_modules,
        loss_coefs=config["global_workspace"]["loss_coefficients"],
        contrastive_fn=contrastive_fn
    )

    return gw_mod, selection_mod, loss_mod


def setup_global_workspace(config):
    gw_mod, selection_mod, loss_mod = get_global_workspace_params(config)

    global_workspace = MyGlobalWorkspace(
        gw_mod=gw_mod,
        selection_mod=selection_mod,
        loss_mod=loss_mod,
        optim_lr=config["training"]["optim"]["lr"]
    )

    return global_workspace
