from typing import Any

import pytest
import torch
from shimmer import ContrastiveLoss, DomainModule, GWDecoder, GWEncoder, GWModule, LossOutput, SingleDomainSelection
from torch import nn
import torch.nn.functional as F

from numb_project.constants import BASE
from numb_project.domain_module import IdentityDomain, LoadedDomainConfig
from numb_project.gw_model import (
    MyCustomGWLosses,
    MyGlobalWorkspace,
    freeze_except_attention,
    get_global_workspace_mods,
    load_pretrained_global_workspace,
    setup_global_workspace,
)
from numb_project.operation_module import AddOperationModule, ChainOperationModule, OperationSelectionModule, RotateOperationModule, SubOperationModule

GW_SIZE = BASE  # OperationSelectionModule's LSTMCell expects task one-hots of width BASE as its GW-sized input.
IMAGE_SHAPE = (1, 4, 4)  # spatial shape so RotateOperationModule can rotate it like a real image.
IMAGE_DIM = IMAGE_SHAPE[0] * IMAGE_SHAPE[1] * IMAGE_SHAPE[2]
HIDDEN_DIM = 8
CHAIN_LENGTH = 2
BATCH_SIZE = 3


class SpatialIdentityDomain(DomainModule):
    """Identity domain module whose latent is the flattened image, so it can
    stand in for a real image domain (encode/decode round-trip a spatial
    tensor) without needing a pretrained VAE checkpoint."""

    def __init__(self, shape: tuple[int, int, int]) -> None:
        super().__init__(shape[0] * shape[1] * shape[2])
        self.shape = shape

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z.reshape(z.shape[0], *self.shape)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, raw_target: Any) -> LossOutput:
        return LossOutput(F.mse_loss(pred, target, reduction="mean"))

REPRESENTATION_LOSS_COEFS = {
    "demi_cycles": 1.0,
    "cycles": 1.0,
    "translations": 1.0,
    "contrastives": 1.0,
}
LOSS_COEFS = {
    "representation_loss": 1.0,
    "add_loss": 1.0,
    "sub_loss": 1.0,
    "rotate_loss": 1.0,
    "task_loss": 1.0,
}


def make_config(batch_size: int = BATCH_SIZE, lr: float = 1e-3) -> dict[str, Any]:
    hidden_dim = {"operation": HIDDEN_DIM, "image": HIDDEN_DIM, "digit": HIDDEN_DIM}
    return {
        "global_workspace": {
            "latent_dim": GW_SIZE,
            "encoders": {"hidden_dim": hidden_dim, "n_layers": 1},
            "decoders": {"hidden_dim": hidden_dim, "n_layers": 1},
            "representation_loss_coefficients": REPRESENTATION_LOSS_COEFS,
            "loss_coefficients": LOSS_COEFS,
        },
        "training": {"batch_size": batch_size, "optim": {"lr": lr}},
    }


def make_workspace() -> tuple[MyGlobalWorkspace, dict]:
    """Builds a minimal but fully-wired MyGlobalWorkspace with 'image'/'digit'
    domains, using IdentityDomain for both so the graph doesn't depend on a
    pretrained VAE checkpoint."""
    domain_modules = {"image": SpatialIdentityDomain(IMAGE_SHAPE), "digit": IdentityDomain(BASE)}
    gw_encoders = {
        "image": GWEncoder(IMAGE_DIM, HIDDEN_DIM, GW_SIZE, 1),
        "digit": GWEncoder(BASE, HIDDEN_DIM, GW_SIZE, 1),
    }
    gw_decoders = {
        "image": GWDecoder(GW_SIZE, HIDDEN_DIM, IMAGE_DIM, 1),
        "digit": GWDecoder(GW_SIZE, HIDDEN_DIM, BASE, 1),
    }
    gw_mod = GWModule(domain_modules, GW_SIZE, gw_encoders, gw_decoders)
    selection_mod = SingleDomainSelection()
    contrastive_fn = ContrastiveLoss(torch.tensor([1 / 0.07]).log(), "mean", False)

    operation_mod = nn.ModuleDict({
        "add": AddOperationModule(GW_SIZE, HIDDEN_DIM, GW_SIZE),
        "sub": SubOperationModule(GW_SIZE, HIDDEN_DIM, GW_SIZE),
        "rotate": RotateOperationModule(GW_SIZE, HIDDEN_DIM, GW_SIZE),
    })
    operation_selection_mod = OperationSelectionModule(
        input_size=GW_SIZE, output_size=1 + len(operation_mod), hidden_size=HIDDEN_DIM, batch_size=BATCH_SIZE, device="cpu"
    )
    task_mod = ChainOperationModule(
        CHAIN_LENGTH, 0, 9, BASE, gw_mod, operation_selection_mod, operation_mod
    )

    loss_mod = MyCustomGWLosses(
        gw_mod=gw_mod,
        selection_mod=selection_mod,
        domain_mods=domain_modules,
        loss_coefs=LOSS_COEFS,
        representation_loss_coefs=REPRESENTATION_LOSS_COEFS,
        contrastive_fn=contrastive_fn,
        operation_mod=operation_mod,
        operation_selection_mod=operation_selection_mod,
        task_mod=task_mod,
    )

    workspace = MyGlobalWorkspace(
        gw_mod=gw_mod,
        selection_mod=selection_mod,
        loss_mod=loss_mod,
        optim_lr=1e-3,
        operation_mod=operation_mod,
        operation_selection_mod=operation_selection_mod,
        task_mod=task_mod,
    )

    image = torch.randn(BATCH_SIZE, *IMAGE_SHAPE)
    digit = nn.functional.one_hot(torch.randint(0, BASE, (BATCH_SIZE,)), num_classes=BASE).float()
    batch = {
        frozenset(["image"]): {"image": image},
        frozenset(["digit"]): {"digit": digit},
        frozenset(["image", "digit"]): {"image": image, "digit": digit},
    }

    return workspace, batch


@pytest.fixture
def workspace_and_batch() -> tuple[MyGlobalWorkspace, dict]:
    return make_workspace()


class TestMyGlobalWorkspace:
    def test_init_stores_operation_submodules(self, workspace_and_batch) -> None:
        workspace, _ = workspace_and_batch

        assert isinstance(workspace.operation_module, nn.ModuleDict)
        assert isinstance(workspace.operation_selection_module, OperationSelectionModule)
        assert isinstance(workspace.task_mod, ChainOperationModule)
        assert workspace.gw_mod is workspace.loss_mod.gw_mod

    def test_configure_optimizers_returns_adamw(self, workspace_and_batch) -> None:
        workspace, _ = workspace_and_batch

        result = workspace.configure_optimizers()

        assert set(result.keys()) == {"optimizer"}
        assert isinstance(result["optimizer"], torch.optim.AdamW)
        assert result["optimizer"].defaults["lr"] == workspace.optim_lr

    def test_input_to_gw_encodes_image_into_workspace_dim(self, workspace_and_batch) -> None:
        workspace, batch = workspace_and_batch
        raw_group = batch[frozenset(["image", "digit"])]

        gw_state = workspace.input_to_gw(raw_group)

        assert gw_state.shape == (BATCH_SIZE, GW_SIZE)

    def test_generic_step_returns_finite_scalar_loss(self, workspace_and_batch) -> None:
        workspace, batch = workspace_and_batch

        loss = workspace.generic_step(batch, mode="train")

        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_on_train_epoch_start_decays_temperature(self, workspace_and_batch) -> None:
        workspace, _ = workspace_and_batch
        workspace.operation_selection_module.temperature = 1.0
        # current_epoch defaults to 0 without an attached trainer, matching the
        # very first call at the start of training (no decay yet).

        workspace.on_train_epoch_start()

        assert workspace.operation_selection_module.temperature == pytest.approx(1.0)


class TestMyCustomGWLosses:
    def test_init_stores_custom_submodules(self, workspace_and_batch) -> None:
        workspace, _ = workspace_and_batch
        loss_mod = workspace.loss_mod

        assert loss_mod.operation_mod is workspace.operation_module
        assert loss_mod.task_mod is workspace.task_mod
        assert loss_mod.representation_loss_coefs == REPRESENTATION_LOSS_COEFS

    def test_compute_representation_loss_returns_scalar_and_metrics(self, workspace_and_batch) -> None:
        workspace, batch = workspace_and_batch
        domain_latents = workspace.encode_domains(batch)

        loss, metrics = workspace.loss_mod.compute_representation_loss(batch, domain_latents)

        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert "demi_cycles" in metrics

    def test_step_combines_representation_and_task_losses(self, workspace_and_batch) -> None:
        workspace, batch = workspace_and_batch
        domain_latents = workspace.encode_domains(batch)
        raw_group = batch[frozenset(["image", "digit"])]
        init_gw_state = workspace.input_to_gw(raw_group)
        tasks, right_addend_value, task_targets = workspace.task_mod.create_chain_add_tasks(raw_group)
        task_predictions, roll_sequence = workspace.task_mod.forward_chain(init_gw_state, tasks, raw_group["digit"])

        loss_output = workspace.loss_mod.step(
            batch, domain_latents, "train",
            task_predictions=task_predictions,
            task_targets=task_targets,
            roll_sequence=roll_sequence,
            right_addend_value=right_addend_value,
        )

        assert torch.isfinite(loss_output.loss)
        assert "task_loss" in loss_output.metrics


class TestModuleFunctions:
    def test_get_global_workspace_mods_wires_all_components(self) -> None:
        config = make_config()
        domains_configs = [LoadedDomainConfig(domain_type="digit")]

        gw_mod, selection_mod, operation_mod, operation_selection_mod, task_mod, loss_mod = get_global_workspace_mods(
            config, domains_configs
        )

        assert isinstance(gw_mod, GWModule)
        assert isinstance(selection_mod, SingleDomainSelection)
        assert set(operation_mod.keys()) == {"add", "sub", "rotate"}
        assert isinstance(operation_selection_mod, OperationSelectionModule)
        assert isinstance(task_mod, ChainOperationModule)
        assert isinstance(loss_mod, MyCustomGWLosses)
        assert set(gw_mod.domain_mods.keys()) == {"digit"}

    def test_setup_global_workspace_returns_configured_workspace(self) -> None:
        config = make_config(lr=5e-4)
        domains_configs = [LoadedDomainConfig(domain_type="digit")]

        workspace = setup_global_workspace(config, domains_configs)

        assert isinstance(workspace, MyGlobalWorkspace)
        assert workspace.optim_lr == 5e-4

    def test_freeze_except_attention_only_leaves_operation_selection_trainable(self, workspace_and_batch) -> None:
        workspace, _ = workspace_and_batch

        freeze_except_attention(workspace)

        assert all(not p.requires_grad for p in workspace.gw_mod.parameters())
        assert all(p.requires_grad for p in workspace.operation_selection_module.parameters())

    def test_load_pretrained_global_workspace_restores_weights_and_freezes(self, tmp_path) -> None:
        config = make_config()
        domains_configs = [LoadedDomainConfig(domain_type="digit")]
        source_workspace = setup_global_workspace(config, domains_configs)

        checkpoint_path = tmp_path / "workspace.ckpt"
        torch.save({"state_dict": source_workspace.state_dict()}, checkpoint_path)

        loaded_workspace = load_pretrained_global_workspace(
            config, domains_configs, str(checkpoint_path), device="cpu"
        )

        assert isinstance(loaded_workspace, MyGlobalWorkspace)
        assert all(not p.requires_grad for p in loaded_workspace.gw_mod.parameters())
        assert all(p.requires_grad for p in loaded_workspace.operation_selection_module.parameters())
