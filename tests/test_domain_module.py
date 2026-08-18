import torch
import torch.nn.functional as F

from numb_project.constants import BASE
from numb_project.domain_module import (
    IdentityDomain,
    LoadedDomainConfig,
    MNISTDomain,
    get_domains_config,
    load_pretrained_module,
)
from numb_project.mnist_vae import VAE


def test_identity_domain_encode_decode_transform_are_identity() -> None:
    domain = IdentityDomain(size=BASE)
    x = torch.randn(3, BASE)

    assert torch.equal(domain.encode(x), x)
    assert torch.equal(domain.decode(x), x)
    assert torch.equal(domain.transform(x), x)
    assert torch.equal(domain.forward(x), x)


def test_identity_domain_compute_loss_returns_mse() -> None:
    domain = IdentityDomain(size=BASE)
    pred = torch.randn(4, BASE)
    target = torch.randn(4, BASE)

    loss = domain.compute_loss(pred, target, raw_target=None)

    assert torch.isclose(loss.loss, F.mse_loss(pred, target, reduction="mean"))


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


def test_load_pretrained_module_digit_returns_identity_domain() -> None:
    module = load_pretrained_module(LoadedDomainConfig(domain_type="digit"))

    assert isinstance(module, IdentityDomain)
    assert module.size == BASE
