import torch
import torch.nn.functional as F
from shimmer import GWModule

from numb_project.constants import BASE, DIGIT_DIM
from numb_project.domain_module import DigitAngleDomain
from numb_project.gw_model import build_gw_encoders_decoders
from numb_project.operation_rotation_module import (
    LatentRotateOperationModule,
    PixelRotateOperationModule,
    chained_rotation_targets,
)
from numb_project.utils import angle_to_cos_sin, clean_digit, rotate_images
from tests.test_gw_model import GW_SIZE, HIDDEN_DIM, IMAGE_DIM, IMAGE_SHAPE, SpatialIdentityDomain

BATCH_SIZE = 3


def make_raw_data() -> dict:
    one_hot = F.one_hot(torch.randint(0, BASE, (BATCH_SIZE,)), num_classes=BASE).float()
    digit = torch.cat([one_hot, angle_to_cos_sin(torch.zeros(BATCH_SIZE))], dim=-1)
    return {frozenset(["image", "digit"]): {"image": torch.rand(BATCH_SIZE, *IMAGE_SHAPE), "digit": digit}}


def make_gw_mod() -> GWModule:
    domain_modules = {"image": SpatialIdentityDomain(IMAGE_SHAPE), "digit": DigitAngleDomain()}
    gw_encoders, gw_decoders = build_gw_encoders_decoders(domain_modules, GW_SIZE, HIDDEN_DIM, 1, HIDDEN_DIM, 1)
    return GWModule(domain_modules, GW_SIZE, gw_encoders, gw_decoders)


def test_chained_rotation_targets_rotates_by_multiples_of_degrees() -> None:
    image = torch.rand(BATCH_SIZE, *IMAGE_SHAPE)

    targets = chained_rotation_targets(image, 20.0, 3)

    assert len(targets) == 3
    expected = rotate_images(image, torch.full((BATCH_SIZE,), 60.0))
    assert torch.allclose(targets[2], expected)


def test_pixel_loss_with_one_step_is_single_rotation_mse() -> None:
    module = PixelRotateOperationModule(latent_dim=HIDDEN_DIM, n_steps=1)
    raw_data = make_raw_data()
    image = raw_data[frozenset(["image", "digit"])]["image"]

    loss = module.loss(None, raw_data)

    expected = F.mse_loss(module(image), rotate_images(image, torch.full((BATCH_SIZE,), module.degrees)))
    assert torch.isclose(loss, expected)


def test_pixel_loss_averages_chained_steps_and_backpropagates_through_chain() -> None:
    module = PixelRotateOperationModule(latent_dim=HIDDEN_DIM, n_steps=3)
    raw_data = make_raw_data()
    image = raw_data[frozenset(["image", "digit"])]["image"]

    loss = module.loss(None, raw_data)

    preds, pred = [], image
    for _ in range(3):
        pred = module(pred)
        preds.append(pred)
    targets = chained_rotation_targets(image, module.degrees, 3)
    expected = torch.stack([F.mse_loss(p, t) for p, t in zip(preds, targets)]).mean()
    assert torch.isclose(loss, expected)
    loss.backward()
    assert all(p.grad is not None for p in module.parameters())


def test_latent_losses_with_chained_steps_train_only_the_rotation_module() -> None:
    for space, size in [("gw", GW_SIZE), ("vae", IMAGE_DIM), ("gw_digit", GW_SIZE), ("gw_fused", GW_SIZE), ("digit", DIGIT_DIM)]:
        gw_mod = make_gw_mod()
        module = LatentRotateOperationModule(space, size, hidden_size=HIDDEN_DIM, latent_dim=HIDDEN_DIM, n_steps=3)

        loss = module.loss(gw_mod, make_raw_data())
        loss.backward()

        assert torch.isfinite(loss)
        assert all(p.grad is not None for p in module.parameters())
        # la loss de rotation n'entraîne pas le GW
        assert all(p.grad is None for p in gw_mod.gw_encoders.parameters())
        assert all(p.grad is None for p in gw_mod.gw_decoders.parameters())


def test_latent_module_encodes_images_in_its_space() -> None:
    gw_mod = make_gw_mod()
    image = make_raw_data()[frozenset(["image", "digit"])]["image"]

    gw_state = LatentRotateOperationModule("gw", GW_SIZE).encode(gw_mod, image)
    vae_latent = LatentRotateOperationModule("vae", IMAGE_DIM).encode(gw_mod, image)

    assert gw_state.shape == (BATCH_SIZE, GW_SIZE)
    assert gw_state.abs().max() <= 1
    assert torch.equal(vae_latent, gw_mod.domain_mods["image"].encode(image))


def test_symbolic_spaces_go_through_the_digit_domain() -> None:
    gw_mod = make_gw_mod()
    image = make_raw_data()[frozenset(["image", "digit"])]["image"]

    digit = LatentRotateOperationModule("digit", DIGIT_DIM).encode(gw_mod, image)
    gw_digit = LatentRotateOperationModule("gw_digit", GW_SIZE).encode(gw_mod, image)

    image_state = torch.tanh(gw_mod.gw_encoders["image"](gw_mod.domain_mods["image"].encode(image)))
    expected_digit = clean_digit(gw_mod.gw_decoders["digit"](image_state))
    assert torch.allclose(digit, expected_digit)
    assert torch.allclose(gw_digit, torch.tanh(gw_mod.gw_encoders["digit"](expected_digit)))


def test_fused_space_averages_gw_and_gw_digit_states() -> None:
    gw_mod = make_gw_mod()
    image = make_raw_data()[frozenset(["image", "digit"])]["image"]

    fused = LatentRotateOperationModule("gw_fused", GW_SIZE).encode(gw_mod, image)

    gw_state = LatentRotateOperationModule("gw", GW_SIZE).encode(gw_mod, image)
    gw_digit = LatentRotateOperationModule("gw_digit", GW_SIZE).encode(gw_mod, image)
    assert torch.allclose(fused, 0.5 * (gw_state + gw_digit))


def test_clean_digit_keeps_only_digit_and_angle() -> None:
    digits = torch.tensor([[0.1, 0.9, -0.2, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.5]])

    cleaned = clean_digit(digits)

    assert torch.equal(cleaned[:, :BASE], F.one_hot(torch.tensor([1]), BASE).float())
    assert torch.allclose(cleaned[:, BASE:], torch.tensor([[0.0, 1.0]]))


def test_only_gw_modules_outputs_are_bounded_by_tanh() -> None:
    x = torch.randn(BATCH_SIZE, GW_SIZE) * 5
    gw_module = LatentRotateOperationModule("gw", GW_SIZE, hidden_size=HIDDEN_DIM, latent_dim=HIDDEN_DIM)
    vae_module = LatentRotateOperationModule("vae", GW_SIZE, hidden_size=HIDDEN_DIM, latent_dim=HIDDEN_DIM)
    torch.nn.init.constant_(vae_module.decoder[-1].bias, 3.0)

    gw_digit_module = LatentRotateOperationModule("gw_digit", GW_SIZE, hidden_size=HIDDEN_DIM, latent_dim=HIDDEN_DIM)
    gw_fused_module = LatentRotateOperationModule("gw_fused", GW_SIZE, hidden_size=HIDDEN_DIM, latent_dim=HIDDEN_DIM)
    digit_module = LatentRotateOperationModule("digit", GW_SIZE, hidden_size=HIDDEN_DIM, latent_dim=HIDDEN_DIM)
    torch.nn.init.constant_(vae_module.decoder[-1].bias, 3.0)
    torch.nn.init.constant_(digit_module.decoder[-1].bias, 3.0)

    assert gw_module(torch.tanh(x)).abs().max() <= 1
    assert gw_digit_module(torch.tanh(x)).abs().max() <= 1
    assert gw_fused_module(torch.tanh(x)).abs().max() <= 1
    assert vae_module(x).abs().max() > 1
    assert digit_module(x).abs().max() > 1
