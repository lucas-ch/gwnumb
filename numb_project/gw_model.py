from typing import Any

from shimmer import ContrastiveLoss, GWLosses2Domains, GWModule, GlobalWorkspaceBase, LossOutput, ModelModeT, RawDomainGroupsT, SelectionBase, SingleDomainSelection, combine_loss
from shimmer.modules.losses import GWLosses
from shimmer.utils import groups_batch_size
import torch
from torch.optim import AdamW
import torch.nn.functional as F

from numb_project.constants import BASE, DEVICE
from numb_project.domain_module import AttentionModule, LoadedDomainConfig, OperationModule, load_domains
import random

def make_chain_task(
    digit_one_hot: torch.Tensor, base: int = BASE, start_chain = 0, end_chain = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = digit_one_hot.shape[0]
    device = digit_one_hot.device

    digit_idx = digit_one_hot.argmax(dim=1)
    right_addend_value = torch.randint(start_chain, end_chain, (batch_size,), device=device)

    target_idx = (digit_idx + right_addend_value) % base

    right_addend_onehot = F.one_hot(right_addend_value, num_classes=base).float()
    target_one_hot = F.one_hot(target_idx, num_classes=base).float()

    return right_addend_onehot, target_one_hot

class MyGlobalWorkspace(GlobalWorkspaceBase):
    def __init__(
        self,
        gw_mod: GWModule,
        selection_mod: SelectionBase,
        loss_mod: GWLosses,
        optim_lr: float = 1e-3,
        operation_mod = None,
        attention_mod = None,
    ) -> None:
        super().__init__(gw_mod, selection_mod, loss_mod, optim_lr)
        self.operation_module = operation_mod
        self.attention_module = attention_mod

    def configure_optimizers(self) -> dict[str, AdamW]:
        optimizer = AdamW(
            self.parameters(),
            lr=self.optim_lr,
            weight_decay=self.optim_weight_decay,
        )
        return {"optimizer": optimizer}

    def forward_chain(self, image, right_addend_onehot, digit_one_hot, chain_length=BASE*2):
        batch_size = image.shape[0]
        h0 = torch.zeros(batch_size, 128, device=DEVICE)
        c0 = torch.zeros(batch_size, 128, device=DEVICE)
        hc = (h0, c0)
        z = torch.zeros(batch_size, 10, device=image.device)
        z[:, 0] = 1.0
        gw_state = self.gw_mod.gw_encoders["digit"](z)


        outputs = []

        for t in range(chain_length):
            attn_t, hc = self.attention_module(right_addend_onehot, hc)
            a_perc, a_op, a_out = attn_t[:, 0:1], attn_t[:, 1:2], attn_t[:, 2:3]

            if t == 0:
                image_to_latent = self.encode_domain(image, "image")
                perception_to_gw = self.gw_mod.gw_encoders["image"](image_to_latent)
                a_addl_raw = torch.argmax(digit_one_hot, dim=1)
                a_addr = torch.argmax(right_addend_onehot, dim=1)

                decoded_digit = self.gw_mod.gw_decoders["digit"](perception_to_gw)
                a_addl_decoded = torch.argmax(decoded_digit, dim=1)
            else:
                perception_to_gw = 0.0

            task = torch.ones(batch_size, 1, device=image.device)

            a_gw_state = torch.argmax(self.gw_mod.gw_decoders["digit"](gw_state), dim = 1)
            op_to_gw = self.operation_module(gw_state, task)
            a_digit_post_ope = torch.argmax(self.gw_mod.gw_decoders["digit"](op_to_gw), dim = 1)

            digit_pred = self.gw_mod.gw_decoders["digit"](gw_state)
            out_to_gw = self.gw_mod.gw_encoders["digit"](digit_pred)

            gw_state = a_perc * perception_to_gw + a_op * op_to_gw + a_out * out_to_gw
            outputs.append(digit_pred)

        return torch.stack(outputs, dim=1)

    def generic_step(self, batch: RawDomainGroupsT, mode: ModelModeT, start_chain=0, end_chain=10):
        domain_latents = self.encode_domains(batch)
        batch_size = groups_batch_size(domain_latents)

        raw_group = batch[frozenset({"digit", "image"})]
        image = raw_group["image"]
        digit_one_hot = raw_group["digit"]

        right_addend_onehot, target = make_chain_task(digit_one_hot, base=BASE, start_chain=start_chain, end_chain=end_chain)
        cumulative_preds = self.forward_chain(image, right_addend_onehot, digit_one_hot)

        loss_output = self.loss_mod.step(
            batch, domain_latents, mode,
            cumulative_preds=cumulative_preds,
            target=target,
        )

        for name, metric in loss_output.all.items():
            self.log(f"{mode}/{name}", metric, batch_size=batch_size, add_dataloader_idx=False)

        total_loss = loss_output.loss

        return total_loss

    def on_train_epoch_start(self):
        min_temp = 0.01
        max_temp = 0.5
        decay_epochs = 20
        progress = min(self.current_epoch / decay_epochs, 1.0)

        self.attention_module.temperature = max_temp * (min_temp / max_temp) ** progress

class MyCustomGWLosses(GWLosses2Domains):
    def __init__(self, gw_mod, selection_mod, domain_mods, loss_coefs, contrastive_fn, operation_mod, attention_mod) -> None:
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

    def compute_chain_loss(self, cumulative_preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        chain_length = cumulative_preds.size(1)
        losses = torch.zeros(chain_length, device=cumulative_preds.device)
        for i in range(chain_length):
            losses[i] = F.mse_loss(cumulative_preds[:, i, :], target)
        return torch.mean(losses[1:]) + losses[-1]

    def step(self, raw_data, domain_latents, mode, cumulative_preds=None, target=None) -> LossOutput:
        metrics = {}
        metrics.update(self.demi_cycle_loss(domain_latents, raw_data))
        metrics.update(self.cycle_loss(domain_latents, raw_data))
        metrics.update(self.translation_loss(domain_latents, raw_data))
        metrics.update(self.contrastive_loss(domain_latents))
        representation_loss = combine_loss(metrics, self.loss_coefs)

        digit_one_hot = domain_latents[frozenset({"digit", "image"})]["digit"]
        image = domain_latents[frozenset({"digit", "image"})]["image"]

        add_loss = self.compute_shift_loss(digit_one_hot, image, 1)
        sub_loss = self.compute_shift_loss(digit_one_hot, image, -1)
        operation_cycle = self.compute_shift_cycle_loss(digit_one_hot)

        total_loss = representation_loss + 10 * (add_loss + sub_loss + operation_cycle)

        if cumulative_preds is not None and target is not None:
            chain_loss = self.compute_chain_loss(cumulative_preds, target)
            metrics["chain_loss"] = chain_loss
            total_loss = total_loss + chain_loss

        return LossOutput(total_loss, metrics)

def get_global_workspace_mods(
        config: dict[str, Any],
        domains_configs: list[LoadedDomainConfig]
        ) -> tuple[GWModule, SingleDomainSelection, MyCustomGWLosses]:
    selection_mod = SingleDomainSelection()
    contrastive_fn = ContrastiveLoss(torch.tensor([1 / 0.07]).log(), "mean", False)
    gw_size = config["global_workspace"]["latent_dim"]

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
        fusion_activation_fn=torch.tanh
    )

    operation_mod = OperationModule(gw_size, 1, config["global_workspace"]["encoders"]["hidden_dim"]["operation"])
    attention_mod = AttentionModule()

    loss_mod = MyCustomGWLosses(
        gw_mod = gw_mod,
        selection_mod=selection_mod,
        domain_mods=domain_modules,
        loss_coefs=config["global_workspace"]["loss_coefficients"],
        contrastive_fn=contrastive_fn,
        operation_mod = operation_mod,
        attention_mod=attention_mod
    )

    return gw_mod, selection_mod, operation_mod, attention_mod, loss_mod


def setup_global_workspace(config: dict[str, Any], domains_configs: list[LoadedDomainConfig]) -> MyGlobalWorkspace:
    gw_mod, selection_mod, operation_mod, attention_mod, loss_mod = get_global_workspace_mods(config, domains_configs)

    global_workspace = MyGlobalWorkspace(
        gw_mod=gw_mod,
        selection_mod=selection_mod,
        loss_mod=loss_mod,
        optim_lr=config["training"]["optim"]["lr"],
        operation_mod= operation_mod,
        attention_mod=attention_mod
    )

    return global_workspace
