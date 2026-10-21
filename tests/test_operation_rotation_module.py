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
    digit_recognition,
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


def test_gw_loss_and_metrics_with_chained_steps_are_finite() -> None:
    gw_mod = make_gw_mod()
    module = GwRotateOperationModule(gw_size=GW_SIZE, hidden_size=HIDDEN_DIM, latent_dim=HIDDEN_DIM, n_steps=3)
    raw_data = make_raw_data()

    loss = module.loss(gw_mod, raw_data)
    loss.backward()
    metrics = module.metrics(gw_mod, raw_data)

    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in module.parameters())
    # la loss de rotation n'entraîne pas le GW
    assert all(p.grad is None for p in gw_mod.gw_encoders.parameters())
    assert set(metrics) == {"pixel_mse", "vae_floor_pixel_mse", "digit_accuracy", "angle_error_deg"}
    assert all(torch.isfinite(v) for v in metrics.values())


def test_gw_residual_module_starts_as_identity() -> None:
    module = GwRotateOperationModule(gw_size=GW_SIZE, hidden_size=HIDDEN_DIM, latent_dim=HIDDEN_DIM, residual=True)
    gw_state = torch.tanh(torch.randn(BATCH_SIZE, GW_SIZE))

    assert torch.allclose(module(gw_state), gw_state, atol=1e-5)


def test_gw_module_output_stays_in_tanh_range() -> None:
    for residual in (False, True):
        module = GwRotateOperationModule(gw_size=GW_SIZE, hidden_size=HIDDEN_DIM, latent_dim=HIDDEN_DIM, residual=residual)
        out = module(torch.tanh(torch.randn(BATCH_SIZE, GW_SIZE) * 5))

        assert out.abs().max() <= 1


def test_gw_residual_module_trains_its_last_layer() -> None:
    gw_mod = make_gw_mod()
    module = GwRotateOperationModule(
        gw_size=GW_SIZE, hidden_size=HIDDEN_DIM, latent_dim=HIDDEN_DIM, n_steps=2, residual=True
    )

    module.loss(gw_mod, make_raw_data()).backward()

    # couche finale initialisée à zéro : elle reçoit bien un gradient
    assert module.decoder[-1].weight.grad.abs().sum() > 0


def test_digit_recognition_is_perfect_on_exact_digit_vectors() -> None:
    from types import SimpleNamespace

    labels = torch.tensor([3, 7, 1])
    angles = torch.tensor([10.0, 200.0, 350.0])
    digits = torch.cat([F.one_hot(labels, BASE).float(), angle_to_cos_sin(angles)], dim=-1)
    # décodeur digit identité : l'"état GW" est directement le vecteur digit
    gw_mod = SimpleNamespace(gw_decoders={"digit": lambda x: x})

    accuracy, angle_error = digit_recognition(gw_mod, digits, labels, angles)
    _, shifted_error = digit_recognition(gw_mod, digits, labels, angles + 30.0)

    assert accuracy == 1.0
    assert angle_error < 1e-3
    assert torch.isclose(shifted_error, torch.tensor(30.0), atol=1e-3)


def test_pixel_metrics_report_digit_recognition_over_chained_steps() -> None:
    gw_mod = make_gw_mod()
    module = PixelRotateOperationModule(latent_dim=HIDDEN_DIM, n_steps=3)

    metrics = module.metrics(gw_mod, make_raw_data())

    assert set(metrics) == {"digit_accuracy", "angle_error_deg"}
    assert 0 <= metrics["digit_accuracy"] <= 1
    assert 0 <= metrics["angle_error_deg"] <= 180
