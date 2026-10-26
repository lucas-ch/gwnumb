import torch
import torch.nn.functional as F

from numb_project.constants import BASE, DIGIT_DIM
from numb_project.domain_module import (
    DigitAngleDomain,
    LoadedDomainConfig,
    MNISTDomain,
    get_domains_config,
    load_domain_module,
    load_domain_modules,
)
from numb_project.mnist_vae import VAE



def test_mnist_domain_wraps_vae_for_encode_decode_roundtrip() -> None:
    vae = VAE(latent_dim=4)
    domain = MNISTDomain(vae)
    x = torch.rand(2, 1, 28, 28)

    z = domain.encode(x)
    x_hat = domain.decode(z)

    assert domain.latent_dim == 4
    assert z.shape == (2, 4)
    assert x_hat.shape == (2, 1, 28, 28)
    assert all(not p.requires_grad for p in domain.vae.parameters())


def test_get_domains_config_builds_one_config_per_domain() -> None:
    configs = get_domains_config(["image", "digit"])

    domain_types = {config.domain_type for config in configs}
    assert domain_types == {"image", "digit"}
    assert len(configs) == 2


def test_load_domain_module_digit_returns_digit_angle_domain() -> None:
    module = load_domain_module(LoadedDomainConfig(domain_type="digit", args={"angle_loss_weight": 0.5}))

    assert isinstance(module, DigitAngleDomain)
    assert module.latent_dim == DIGIT_DIM
    assert module.angle_loss_weight == 0.5


def test_load_domain_modules_keys_modules_by_domain_type() -> None:
    modules = load_domain_modules([LoadedDomainConfig(domain_type="digit")])

    assert isinstance(modules, torch.nn.ModuleDict)
    assert set(modules.keys()) == {"digit"}
    assert isinstance(modules["digit"], DigitAngleDomain)


def test_digit_angle_domain_loss_is_cross_entropy_on_digit_plus_weighted_angle_mse() -> None:
    domain = DigitAngleDomain(angle_loss_weight=2.0)
    logits = torch.randn(4, DIGIT_DIM)
    pred = torch.cat([F.softmax(logits[:, :BASE], dim=-1), logits[:, BASE:]], dim=-1)
    labels = torch.randint(0, BASE, (4,))
    target = torch.cat([F.one_hot(labels, BASE).float(), torch.randn(4, 2)], dim=-1)

    loss = domain.compute_loss(pred, target, raw_target=None)

    digit_loss = F.cross_entropy(logits[:, :BASE], labels)
    angle_loss = F.mse_loss(pred[:, BASE:], target[:, BASE:])
    assert torch.isclose(loss.loss, digit_loss + 2.0 * angle_loss)


def test_digit_loss_stays_finite_when_a_probability_underflows() -> None:
    domain = DigitAngleDomain()
    pred = torch.cat([F.one_hot(torch.tensor([0]), BASE).float(), torch.tensor([[1.0, 0.0]])], dim=-1)
    target = torch.cat([F.one_hot(torch.tensor([1]), BASE).float(), torch.tensor([[1.0, 0.0]])], dim=-1)

    assert torch.isfinite(domain.compute_loss(pred, target, raw_target=None).loss)
