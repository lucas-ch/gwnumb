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

    def forward_chain(self, image, right_addend_onehot, digit_one_hot, chain_length=10):
        batch_size = image.shape[0]
        h0 = torch.zeros(batch_size, 32, device=DEVICE)
        c0 = torch.zeros(batch_size, 32, device=DEVICE)
        hc = (h0, c0)

        image_to_latent = self.encode_domain(image, "image")
        gw_state = self.gw_mod.gw_encoders["image"](image_to_latent)
        task = torch.ones(batch_size, 1, device=image.device)

        outputs = []
        a_roll_sequence = []

        digit_pred = self.gw_mod.gw_decoders["digit"](gw_state)
        outputs.append(digit_pred)
        for t in range(chain_length):
            step_scalar = torch.full((batch_size, 1), float(t) / chain_length, device=image.device)
            a_roll, hc = self.attention_module(right_addend_onehot, hc, step_embedding=step_scalar)
            a_roll_sequence.append(a_roll)

            op_result = self.operation_module(gw_state, task)

            a_addl = torch.argmax(digit_one_hot, dim=1)
            a_addr = torch.argmax(right_addend_onehot, dim=1)
            a_ground_truth = a_addl + a_addr
            a_gw_state = torch.argmax(self.gw_mod.gw_decoders["digit"](gw_state), dim = 1)

            gw_state = a_roll * op_result + (1 - a_roll) * gw_state

            digit_pred = self.gw_mod.gw_decoders["digit"](gw_state)
            outputs.append(digit_pred)

        cumulative_preds = torch.stack(outputs, dim=1)          # (batch, chain_length, base)
        a_roll_sequence = torch.stack(a_roll_sequence, dim=1)    # (batch, chain_length, 1)

        return cumulative_preds, a_roll_sequence

    def generic_step(self, batch: RawDomainGroupsT, mode: ModelModeT,
                    start_chain=0, end_chain=10, chain_length=10):
        domain_latents = self.encode_domains(batch)
        batch_size = groups_batch_size(domain_latents)

        raw_group = batch[frozenset({"digit", "image"})]
        image = raw_group["image"]
        digit_one_hot = raw_group["digit"]

        right_addend_onehot, target = make_chain_task(
            digit_one_hot, base=BASE, start_chain=start_chain, end_chain=end_chain
        )
        cumulative_preds, a_roll_sequence = self.forward_chain(
            image, right_addend_onehot, digit_one_hot, chain_length=chain_length
        )

        left_digit_idx = digit_one_hot.argmax(dim=1)
        right_addend_value = right_addend_onehot.argmax(dim=1)

        loss_output = self.loss_mod.step(
            batch, domain_latents, mode,
            cumulative_preds=cumulative_preds,
            a_roll_sequence=a_roll_sequence,
            left_digit_idx=left_digit_idx,
            target=target,
            right_addend_value=right_addend_value,
            base=BASE,
        )

        for name, metric in loss_output.all.items():
            self.log(f"{mode}/{name}", metric, batch_size=batch_size, add_dataloader_idx=False)

        return loss_output.loss

    def on_train_epoch_start(self):
        min_temp = 0.1
        max_temp = 1.0
        decay_epochs = 5
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

    def compute_halting_loss(
        self, a_roll_sequence: torch.Tensor, right_addend_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Supervise le choix roll/pas-roll à chaque pas.
        a_roll_sequence: (batch, chain_length, 1)
        right_addend_value: (batch,) -- nombre de +1 attendus
        Cible : rouler (1) tant que t < right_addend_value, sinon ne pas rouler (0).
        """
        batch_size, chain_length, _ = a_roll_sequence.shape
        device = a_roll_sequence.device

        t_range = torch.arange(chain_length, device=device).unsqueeze(0)
        target = (t_range < right_addend_value.unsqueeze(1)).float()
        target = target.unsqueeze(-1)

        return F.binary_cross_entropy(a_roll_sequence, target)

    def compute_final_prediction_loss(
        self, cumulative_preds: torch.Tensor, target_one_hot: torch.Tensor
    ) -> torch.Tensor:
        """Supervise uniquement la prédiction au dernier pas, contre la cible finale (one-hot)."""
        return F.mse_loss(cumulative_preds[:, -1, :], target_one_hot)

    def step(self, raw_data, domain_latents, mode, cumulative_preds=None,
            a_roll_sequence=None, left_digit_idx=None, target=None,
            right_addend_value=None, base=BASE):
        total_loss = 0
        metrics = {}

        if False:
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

        if cumulative_preds is not None:
            halting_loss = self.compute_halting_loss(a_roll_sequence, right_addend_value)
            final_loss = self.compute_final_prediction_loss(cumulative_preds, target)  # one-hot, pas idx

            metrics["halting_loss"] = halting_loss
            metrics["final_loss"] = final_loss

            total_loss = total_loss + halting_loss + final_loss

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
    attention_mod = AttentionModule(output_size=2)

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

def freeze_except_attention(model: MyGlobalWorkspace) -> None:
    for param in model.parameters():
        param.requires_grad = False

    for param in model.attention_module.parameters():
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