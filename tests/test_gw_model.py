from typing import Any

import pytest
import torch
from shimmer import ContrastiveLoss, DomainModule, GWModule, LossOutput, SingleDomainSelection
from torch import nn
import torch.nn.functional as F

from numb_project.constants import BASE, DIGIT_DIM
from numb_project.domain_module import DigitAngleDomain, LoadedDomainConfig
from numb_project.gw_model import (
    MyCustomGWLosses,
    MyGlobalWorkspace,
    build_gw_encoders_decoders,
    build_rotation_modules,
    get_global_workspace_mods,
    make_rotated_gw_batch,
    setup_global_workspace,
)
from numb_project.operation_rotation_module import LatentRotateOperationModule, PixelRotateOperationModule
from numb_project.utils import angle_to_cos_sin

GW_SIZE = 10
IMAGE_SHAPE = (1, 28, 28)  # matches MNIST_IMAGE_SHAPE : PixelRotateOperationModule's CNN is sized for 28x28 input.
IMAGE_DIM = IMAGE_SHAPE[0] * IMAGE_SHAPE[1] * IMAGE_SHAPE[2]
HIDDEN_DIM = 8
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
    "representation_loss_rotated": 1.0,
    "rotate_pixel_loss": 1.0,
    "rotate_pixel_loss_rotated": 1.0,
    "rotate_vae_loss": 1.0,
    "rotate_vae_loss_rotated": 1.0,
    "rotate_gw_loss": 1.0,
    "rotate_gw_loss_rotated": 1.0,
}
ROTATION_OPERATIONS_CONFIG = {
    "degrees": 30.0,
    "n_steps": 2,
    "pixel": {"latent_dim": HIDDEN_DIM},
    "vae": {"hidden_size": HIDDEN_DIM, "latent_dim": HIDDEN_DIM},
    "gw": {"hidden_size": HIDDEN_DIM, "latent_dim": HIDDEN_DIM},
}


def make_config(batch_size: int = BATCH_SIZE, lr: float = 1e-3) -> dict[str, Any]:
    hidden_dim = {"image": HIDDEN_DIM, "digit": HIDDEN_DIM}
    return {
        "global_workspace": {
            "latent_dim": GW_SIZE,
            "encoders": {"hidden_dim": hidden_dim, "n_layers": 1},
            "decoders": {"hidden_dim": hidden_dim, "n_layers": 1},
            "representation_loss_coefficients": REPRESENTATION_LOSS_COEFS,
            "loss_coefficients": LOSS_COEFS,
            "rotation_operations": ROTATION_OPERATIONS_CONFIG,
        },
        "training": {"batch_size": batch_size, "optim": {"lr": lr}},
    }


def make_batch() -> dict:
    image = torch.rand(BATCH_SIZE, *IMAGE_SHAPE)
    one_hot = F.one_hot(torch.randint(0, BASE, (BATCH_SIZE,)), num_classes=BASE).float()
    digit = torch.cat([one_hot, angle_to_cos_sin(torch.zeros(BATCH_SIZE))], dim=-1)
    return {
        frozenset(["image"]): {"image": torch.rand(BATCH_SIZE, *IMAGE_SHAPE)},
        frozenset(["digit"]): {"digit": digit.clone()},
        frozenset(["image", "digit"]): {"image": image, "digit": digit},
    }


def make_domain_modules() -> nn.ModuleDict:
    """Domaines 'image'/'digit' sans checkpoint : identité sur l'image aplatie."""
    return nn.ModuleDict({"image": SpatialIdentityDomain(IMAGE_SHAPE), "digit": DigitAngleDomain()})


def make_workspace() -> tuple[MyGlobalWorkspace, dict]:
    """Builds a minimal but fully-wired MyGlobalWorkspace with 'image'/'digit'
    domains, using an identity image domain so the graph doesn't depend on a
    pretrained VAE checkpoint."""
    domain_modules = make_domain_modules()
    gw_encoders, gw_decoders = build_gw_encoders_decoders(domain_modules, GW_SIZE, HIDDEN_DIM, 1, HIDDEN_DIM, 1)
    gw_mod = GWModule(domain_modules, GW_SIZE, gw_encoders, gw_decoders)
    selection_mod = SingleDomainSelection()
    contrastive_fn = ContrastiveLoss(torch.tensor([1 / 0.07]).log(), "mean", False)

    rotation_mods = build_rotation_modules(ROTATION_OPERATIONS_CONFIG, GW_SIZE, IMAGE_DIM)

    loss_mod = MyCustomGWLosses(
        gw_mod=gw_mod,
        selection_mod=selection_mod,
        domain_mods=domain_modules,
        loss_coefs=LOSS_COEFS,
        representation_loss_coefs=REPRESENTATION_LOSS_COEFS,
        contrastive_fn=contrastive_fn,
        rotation_mods=rotation_mods,
    )

    workspace = MyGlobalWorkspace(
        gw_mod=gw_mod,
        selection_mod=selection_mod,
        loss_mod=loss_mod,
        optim_lr=1e-3,
        rotation_mods=rotation_mods,
    )
    # generic_step logs metrics, which needs an attached trainer
    workspace.log = lambda *args, **kwargs: None

    return workspace, make_batch()


@pytest.fixture
def workspace_and_batch() -> tuple[MyGlobalWorkspace, dict]:
    return make_workspace()


class TestMyGlobalWorkspace:
    def test_init_stores_rotation_submodules(self, workspace_and_batch) -> None:
        workspace, _ = workspace_and_batch

        assert set(workspace.rotation_mods.keys()) == {"pixel", "vae", "gw"}
        assert isinstance(workspace.rotation_mods["pixel"], PixelRotateOperationModule)
        assert workspace.rotation_mods["vae"].space == "vae"
        assert workspace.rotation_mods["gw"].space == "gw"
        assert workspace.gw_mod is workspace.loss_mod.gw_mod

    def test_configure_optimizers_returns_adamw(self, workspace_and_batch) -> None:
        workspace, _ = workspace_and_batch

        result = workspace.configure_optimizers()

        assert set(result.keys()) == {"optimizer"}
        assert isinstance(result["optimizer"], torch.optim.AdamW)
        assert result["optimizer"].defaults["lr"] == workspace.optim_lr

    def test_generic_step_returns_finite_scalar_loss(self, workspace_and_batch) -> None:
        workspace, batch = workspace_and_batch

        loss = workspace.generic_step(batch, mode="train")

        assert loss.ndim == 0
        assert torch.isfinite(loss)


class TestMyCustomGWLosses:
    def test_init_stores_custom_submodules(self, workspace_and_batch) -> None:
        workspace, _ = workspace_and_batch
        loss_mod = workspace.loss_mod

        assert loss_mod.rotation_mods is workspace.rotation_mods
        assert loss_mod.representation_loss_coefs == REPRESENTATION_LOSS_COEFS

    def test_compute_representation_loss_returns_finite_scalar(self, workspace_and_batch) -> None:
        workspace, batch = workspace_and_batch
        domain_latents = workspace.encode_domains(batch)

        loss = workspace.loss_mod.compute_representation_loss(batch, domain_latents)

        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_step_logs_the_configured_losses(self, workspace_and_batch) -> None:
        workspace, batch = workspace_and_batch
        domain_latents = workspace.encode_domains(batch)

        loss_output = workspace.loss_mod.step(batch, domain_latents, "train")

        assert torch.isfinite(loss_output.loss)
        assert set(loss_output.metrics) == set(LOSS_COEFS)

    def test_step_total_loss_is_weighted_sum_of_configured_losses(self, workspace_and_batch) -> None:
        workspace, batch = workspace_and_batch
        workspace.loss_mod.loss_coefs = {name: 0.5 + i for i, name in enumerate(LOSS_COEFS)}
        domain_latents = workspace.encode_domains(batch)

        torch.manual_seed(0)  # même angles pour le batch tourné dans les deux calculs
        loss_output = workspace.loss_mod.step(batch, domain_latents, "train")

        expected = sum(coef * loss_output.metrics[name] for name, coef in workspace.loss_mod.loss_coefs.items())
        assert torch.isclose(loss_output.loss, expected)


class TestMakeRotatedGwBatch:
    def test_rotates_every_group_with_shared_angles(self) -> None:
        batch = make_batch()

        rotated = make_rotated_gw_batch(batch)

        assert set(rotated.keys()) == set(batch.keys())
        pair_angles = rotated[frozenset(["image", "digit"])]["digit"][:, BASE:]
        digit_angles = rotated[frozenset(["digit"])]["digit"][:, BASE:]
        assert torch.allclose(pair_angles, digit_angles)
        # (cos, sin) stays on the unit circle
        assert torch.allclose(pair_angles.norm(dim=-1), torch.ones(BATCH_SIZE))

    def test_keeps_digit_identity_and_rotates_images(self) -> None:
        batch = make_batch()

        rotated = make_rotated_gw_batch(batch)

        pair, rotated_pair = batch[frozenset(["image", "digit"])], rotated[frozenset(["image", "digit"])]
        assert torch.equal(rotated_pair["digit"][:, :BASE], pair["digit"][:, :BASE])
        assert rotated_pair["image"].shape == pair["image"].shape
        assert not torch.equal(rotated_pair["image"], pair["image"])

    def test_rejects_groups_of_different_batch_sizes(self) -> None:
        batch = make_batch()
        batch[frozenset(["digit"])]["digit"] = batch[frozenset(["digit"])]["digit"][:1]

        with pytest.raises(AssertionError):
            make_rotated_gw_batch(batch)


class TestModuleFunctions:
    def test_build_gw_encoders_decoders_matches_domain_latent_dims(self) -> None:
        domain_modules = {"image": SpatialIdentityDomain(IMAGE_SHAPE), "digit": DigitAngleDomain()}

        gw_encoders, gw_decoders = build_gw_encoders_decoders(
            domain_modules, GW_SIZE, {"image": HIDDEN_DIM, "digit": 4}, 1, HIDDEN_DIM, 2
        )

        assert set(gw_encoders.keys()) == set(gw_decoders.keys()) == {"image", "digit"}
        assert gw_encoders["image"](torch.rand(2, IMAGE_DIM)).shape == (2, GW_SIZE)
        assert gw_encoders["digit"](torch.rand(2, DIGIT_DIM)).shape == (2, GW_SIZE)
        assert gw_decoders["image"](torch.rand(2, GW_SIZE)).shape == (2, IMAGE_DIM)
        assert gw_decoders["digit"](torch.rand(2, GW_SIZE)).shape == (2, DIGIT_DIM)

    def test_get_global_workspace_mods_wires_all_components(self, monkeypatch) -> None:
        monkeypatch.setattr("numb_project.gw_model.load_domain_modules", lambda configs: make_domain_modules())

        gw_mod, selection_mod, loss_mod, rotation_mods = get_global_workspace_mods(make_config(), [])

        assert isinstance(gw_mod, GWModule)
        assert isinstance(selection_mod, SingleDomainSelection)
        assert isinstance(loss_mod, MyCustomGWLosses)
        assert loss_mod.rotation_mods is rotation_mods
        assert {name: mod.degrees for name, mod in rotation_mods.items()} == {"pixel": 30.0, "vae": 30.0, "gw": 30.0}
        assert all(mod.n_steps == 2 for mod in rotation_mods.values())
        assert set(gw_mod.domain_mods.keys()) == {"image", "digit"}
        # les domaines doivent être des sous-modules du GW (state_dict, .to(device))
        assert any(name.startswith("domain_mods.digit.") for name in gw_mod.state_dict())

    def test_build_rotation_modules_sizes_each_latent_space(self) -> None:
        rotation_mods = build_rotation_modules(ROTATION_OPERATIONS_CONFIG, gw_size=GW_SIZE, vae_size=20)

        assert rotation_mods["vae"](torch.randn(2, 20)).shape == (2, 20)
        assert rotation_mods["gw"](torch.randn(2, GW_SIZE)).shape == (2, GW_SIZE)

    def test_setup_global_workspace_returns_configured_workspace(self, monkeypatch) -> None:
        monkeypatch.setattr("numb_project.gw_model.load_domain_modules", lambda configs: make_domain_modules())

        workspace = setup_global_workspace(make_config(lr=5e-4), [])

        assert isinstance(workspace, MyGlobalWorkspace)
        assert workspace.optim_lr == 5e-4
