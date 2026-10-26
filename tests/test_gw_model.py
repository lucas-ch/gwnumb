from typing import Any

import pytest
import torch
from shimmer import ContrastiveLoss, DomainModule, GWDecoder, GWModule, LossOutput, SingleDomainSelection
from torch import nn
import torch.nn.functional as F

from numb_project.constants import BASE, DIGIT_DIM
from numb_project.domain_module import DigitAngleDomain
from numb_project.gw_model import (
    MyCustomGWLosses,
    MyGlobalWorkspace,
    DigitGWDecoder,
    build_gw_encoders_decoders,
    build_rotation_modules,
    get_global_workspace_mods,
    make_rotated_gw_batch,
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
    "rotate_gw_digit_loss": 1.0,
    "rotate_gw_digit_loss_rotated": 1.0,
    "rotate_gw_fused_loss": 1.0,
    "rotate_gw_fused_loss_rotated": 1.0,
    "rotate_digit_loss": 1.0,
    "rotate_digit_loss_rotated": 1.0,
}
ROTATION_OPERATIONS_CONFIG = {
    "enabled": ["pixel", "vae", "gw", "gw_digit", "gw_fused", "digit"],
    "degrees": 30.0,
    "n_steps": 2,
    "pixel": {"latent_dim": HIDDEN_DIM},
    "vae": {"hidden_size": HIDDEN_DIM, "latent_dim": HIDDEN_DIM},
    "gw": {"hidden_size": HIDDEN_DIM, "latent_dim": HIDDEN_DIM},
    "gw_digit": {"hidden_size": HIDDEN_DIM, "latent_dim": HIDDEN_DIM},
    "gw_fused": {"hidden_size": HIDDEN_DIM, "latent_dim": HIDDEN_DIM},
    "digit": {"hidden_size": HIDDEN_DIM, "latent_dim": HIDDEN_DIM},
}


def make_config(batch_size: int = BATCH_SIZE, lr: float = 1e-3, enabled: list[str] | None = None) -> dict[str, Any]:
    hidden_dim = {"image": HIDDEN_DIM, "digit": HIDDEN_DIM}
    rotation_config = ROTATION_OPERATIONS_CONFIG | ({"enabled": enabled} if enabled is not None else {})
    return {
        "domains": {"digit": {"angle_loss_weight": 1.0}},
        "global_workspace": {
            "latent_dim": GW_SIZE,
            "encoders": {"hidden_dim": hidden_dim, "n_layers": 1},
            "decoders": {"hidden_dim": hidden_dim, "n_layers": 1},
            "representation_loss_coefficients": REPRESENTATION_LOSS_COEFS,
            "loss_coefficients": LOSS_COEFS,
            "rotation_operations": rotation_config,
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


def make_workspace(monkeypatch: pytest.MonkeyPatch, **config_kwargs: Any) -> MyGlobalWorkspace:
    """Builds a minimal but fully-wired MyGlobalWorkspace with 'image'/'digit'
    domains, using an identity image domain so the graph doesn't depend on a
    pretrained VAE checkpoint."""
    monkeypatch.setattr("numb_project.gw_model.load_domain_modules", lambda configs: make_domain_modules())
    workspace = MyGlobalWorkspace(make_config(**config_kwargs))
    # generic_step logs metrics, which needs an attached trainer
    workspace.log = lambda *args, **kwargs: None
    return workspace


@pytest.fixture
def workspace_and_batch(monkeypatch) -> tuple[MyGlobalWorkspace, dict]:
    return make_workspace(monkeypatch), make_batch()


class TestMyGlobalWorkspace:
    def test_init_stores_rotation_submodules(self, workspace_and_batch) -> None:
        workspace, _ = workspace_and_batch

        assert set(workspace.rotation_mods.keys()) == {"pixel", "vae", "gw", "gw_digit", "gw_fused", "digit"}
        assert isinstance(workspace.rotation_mods["pixel"], PixelRotateOperationModule)
        assert all(workspace.rotation_mods[space].space == space for space in ("vae", "gw", "gw_digit", "gw_fused", "digit"))
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

    def test_representation_loss_can_keep_a_single_cycle_direction(self, workspace_and_batch) -> None:
        # config.yaml désactive le cycle image -> digit -> image avec cycles: 0.0
        workspace, batch = workspace_and_batch
        loss_mod = workspace.loss_mod
        domain_latents = workspace.encode_domains(batch)
        loss_mod.representation_loss_coefs = {"cycles": 0.0, "cycle_digit_through_image": 1.0}

        loss = loss_mod.compute_representation_loss(batch, domain_latents)

        cycles = loss_mod.cycle_loss(domain_latents, batch)
        assert torch.isclose(loss, cycles["cycle_digit_through_image"])
        assert not torch.isclose(loss, cycles["cycle_image_through_digit"])

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

        gw_mod, selection_mod, loss_mod, rotation_mods = get_global_workspace_mods(make_config())

        assert isinstance(gw_mod, GWModule)
        assert isinstance(selection_mod, SingleDomainSelection)
        assert isinstance(loss_mod, MyCustomGWLosses)
        assert loss_mod.rotation_mods is rotation_mods
        assert {name: mod.degrees for name, mod in rotation_mods.items()} == {"pixel": 30.0, "vae": 30.0, "gw": 30.0, "gw_digit": 30.0, "gw_fused": 30.0, "digit": 30.0}
        assert all(mod.n_steps == 2 for mod in rotation_mods.values())
        assert set(gw_mod.domain_mods.keys()) == {"image", "digit"}
        # les domaines doivent être des sous-modules du GW (state_dict, .to(device))
        assert any(module is gw_mod.domain_mods["digit"] for module in gw_mod.modules())

    def test_build_rotation_modules_sizes_each_latent_space(self) -> None:
        rotation_mods = build_rotation_modules(ROTATION_OPERATIONS_CONFIG, gw_size=GW_SIZE, vae_size=20, digit_size=DIGIT_DIM)

        assert rotation_mods["vae"](torch.randn(2, 20)).shape == (2, 20)
        assert rotation_mods["gw"](torch.randn(2, GW_SIZE)).shape == (2, GW_SIZE)
        assert rotation_mods["gw_digit"](torch.randn(2, GW_SIZE)).shape == (2, GW_SIZE)
        assert rotation_mods["gw_fused"](torch.randn(2, GW_SIZE)).shape == (2, GW_SIZE)
        assert rotation_mods["digit"](torch.randn(2, DIGIT_DIM)).shape == (2, DIGIT_DIM)

    def test_build_rotation_modules_only_builds_enabled_spaces(self) -> None:
        config = ROTATION_OPERATIONS_CONFIG | {"enabled": ["gw", "gw_fused"]}

        rotation_mods = build_rotation_modules(config, gw_size=GW_SIZE, vae_size=20, digit_size=DIGIT_DIM)

        assert list(rotation_mods.keys()) == ["gw", "gw_fused"]


class TestConfigInCheckpoint:
    def test_workspace_is_built_from_config(self, monkeypatch) -> None:
        workspace = make_workspace(monkeypatch, lr=5e-4)

        assert workspace.optim_lr == 5e-4
        assert workspace.config == make_config(lr=5e-4)

    def test_disabled_rotations_have_no_module_nor_loss(self, monkeypatch) -> None:
        workspace = make_workspace(monkeypatch, enabled=["gw"])
        batch = make_batch()

        loss_output = workspace.loss_mod.step(batch, workspace.encode_domains(batch), "train")

        assert set(workspace.rotation_mods.keys()) == {"gw"}
        assert {name for name in loss_output.metrics if name.startswith("rotate_")} == {"rotate_gw_loss", "rotate_gw_loss_rotated"}

    def test_checkpoint_reloads_with_its_own_config(self, monkeypatch, tmp_path) -> None:
        workspace = make_workspace(monkeypatch, enabled=["pixel", "gw_fused"])
        checkpoint_path = tmp_path / "gw.ckpt"
        torch.save({
            "state_dict": workspace.state_dict(),
            MyGlobalWorkspace.CHECKPOINT_HYPER_PARAMS_KEY: dict(workspace.hparams),
            "pytorch-lightning_version": "2.0.0",
        }, checkpoint_path)

        # pas de config passée au chargement : celle du checkpoint est utilisée
        reloaded = MyGlobalWorkspace.load_from_checkpoint(checkpoint_path, weights_only=False)

        assert reloaded.config == workspace.config
        assert set(reloaded.rotation_mods.keys()) == {"pixel", "gw_fused"}
        for name, value in workspace.state_dict().items():
            assert torch.equal(reloaded.state_dict()[name], value)




def test_build_gw_encoders_decoders_constrains_only_the_digit_decoder() -> None:
    _, gw_decoders = build_gw_encoders_decoders(make_domain_modules(), GW_SIZE, HIDDEN_DIM, 1, HIDDEN_DIM, 1)

    digit = gw_decoders["digit"](torch.randn(4, GW_SIZE) * 5)

    assert isinstance(gw_decoders["digit"], DigitGWDecoder)
    assert type(gw_decoders["image"]) is GWDecoder
    assert (digit[:, :BASE] >= 0).all()
    assert torch.allclose(digit[:, :BASE].sum(dim=-1), torch.ones(4))
    assert torch.allclose(digit[:, BASE:].norm(dim=-1), torch.ones(4))
