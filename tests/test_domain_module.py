import math

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

    digit_loss = F.cross_entropy(logits[:, :BASE], labels) / math.log(BASE)
    angle_loss = F.mse_loss(pred[:, BASE:], target[:, BASE:])
    assert torch.isclose(loss.loss, digit_loss + 2.0 * angle_loss)


def test_digit_loss_stays_finite_when_a_probability_underflows() -> None:
    domain = DigitAngleDomain()
    pred = torch.cat([F.one_hot(torch.tensor([0]), BASE).float(), torch.tensor([[1.0, 0.0]])], dim=-1)
    target = torch.cat([F.one_hot(torch.tensor([1]), BASE).float(), torch.tensor([[1.0, 0.0]])], dim=-1)

    assert torch.isfinite(domain.compute_loss(pred, target, raw_target=None).loss)


def test_mnist_domain_loss_is_mse_normalized_by_latent_variance() -> None:
    domain = MNISTDomain(VAE(latent_dim=4), latent_variance=0.5)
    pred, target = torch.randn(3, 4), torch.randn(3, 4)

    loss = domain.compute_loss(pred, target, raw_target=None)

    assert torch.isclose(loss.loss, F.mse_loss(pred, target) / 0.5)
    assert "latent_variance" in domain.state_dict()


def test_mnist_domain_loss_is_one_when_predicting_the_mean() -> None:
    target = torch.randn(10000, 4) * 3
    domain = MNISTDomain(VAE(latent_dim=4), latent_variance=target.var(dim=0).mean().item())

    loss = domain.compute_loss(target.mean(dim=0).expand_as(target), target, raw_target=None)

    assert torch.isclose(loss.loss, torch.tensor(1.0), atol=1e-3)


def test_digit_loss_is_one_for_uniform_probabilities_and_random_angles() -> None:
    domain = DigitAngleDomain()
    n = 20000
    angles = torch.rand(n, 2) * 2 * torch.pi
    target = torch.cat([F.one_hot(torch.randint(0, BASE, (n,)), BASE).float(),
                        torch.stack([angles[:, 0].cos(), angles[:, 0].sin()], dim=-1)], dim=-1)
    pred = torch.cat([torch.full((n, BASE), 1 / BASE),
                      torch.stack([angles[:, 1].cos(), angles[:, 1].sin()], dim=-1)], dim=-1)

    loss = domain.compute_loss(pred, target, raw_target=None)

    # cross entropy normalisée = 1, MSE angle ~ 1
    assert torch.isclose(loss.loss, torch.tensor(2.0), atol=0.05)
