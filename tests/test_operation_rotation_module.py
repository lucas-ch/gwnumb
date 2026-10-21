import torch
import torch.nn.functional as F
from shimmer import GWModule

from numb_project.constants import BASE
from numb_project.domain_module import DigitAngleDomain
from numb_project.gw_model import build_gw_encoders_decoders
from numb_project.operation_rotation_module import (
    GwRotateOperationModule,
    PixelRotateOperationModule,
    chained_rotation_targets,
)
from numb_project.utils import angle_to_cos_sin, rotate_images
from tests.test_gw_model import GW_SIZE, HIDDEN_DIM, IMAGE_SHAPE, SpatialIdentityDomain

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


def test_gw_loss_with_chained_steps_trains_only_the_rotation_module() -> None:
    gw_mod = make_gw_mod()
    module = GwRotateOperationModule(gw_size=GW_SIZE, hidden_size=HIDDEN_DIM, latent_dim=HIDDEN_DIM, n_steps=3)

    loss = module.loss(gw_mod, make_raw_data())
    loss.backward()

    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in module.parameters())
    # la loss de rotation n'entraîne pas le GW
    assert all(p.grad is None for p in gw_mod.gw_encoders.parameters())


def test_gw_module_output_stays_in_tanh_range() -> None:
    module = GwRotateOperationModule(gw_size=GW_SIZE, hidden_size=HIDDEN_DIM, latent_dim=HIDDEN_DIM)
    out = module(torch.tanh(torch.randn(BATCH_SIZE, GW_SIZE) * 5))

    assert out.abs().max() <= 1
