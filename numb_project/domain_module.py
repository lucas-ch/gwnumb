from math import log
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel
from shimmer import DomainModule, LossOutput
import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from torchvision import datasets, transforms

from numb_project.mnist_vae import VAE
from numb_project.mnist_vae_equivariant import EquivariantVAE
from numb_project.mnist_vae_so2 import SO2VAE
from numb_project.constants import BASE, DIGIT_DIM, EQUIVARIANT_VAE_CHECKPOINT_FILE, SO2_VAE_CHECKPOINT_FILE, VAE_CHECKPOINT_FILE

# VAE du domaine image, choisi par `domains.image.vae` dans la config :
# classe et checkpoint
VAES: dict[str, tuple[type[VAE], str]] = {
    "conv": (VAE, VAE_CHECKPOINT_FILE),                         # mnist_vae.py
    "c4": (EquivariantVAE, EQUIVARIANT_VAE_CHECKPOINT_FILE),    # mnist_vae_equivariant.py
    "so2": (SO2VAE, SO2_VAE_CHECKPOINT_FILE),                   # mnist_vae_so2.py
}


class DigitAngleDomain(DomainModule):
    """Domaine digit : one-hot du chiffre (BASE dimensions) suivi de (cos, sin)
    de l'angle de l'image. Le latent est le vecteur lui-même (encode/decode =
    identité).

    Les prédictions du GW sortent de `DigitGWDecoder` (gw_model.py) :
    probabilités des chiffres et (cos, sin) de norme 1.

    Loss en deux parties, pour que l'angle (2 dimensions sur 12) ne soit pas
    noyé dans la moyenne : cross entropy sur le chiffre (NLL des probabilités
    prédites, la cible est le one-hot) et MSE sur (cos, sin). Chaque partie est
    normalisée pour valoir 0 si la prédiction est parfaite et ~1 au hasard :
    la cross entropy est divisée par ln(BASE) (probabilités uniformes), la MSE
    sur (cos, sin) vaut déjà 1 pour un angle aléatoire. Même convention que
    `MNISTDomain`, pour que les losses des deux domaines s'additionnent."""

    def __init__(self, angle_loss_weight: float = 1.0) -> None:
        super().__init__(DIGIT_DIM)
        self.angle_loss_weight = angle_loss_weight

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, raw_target: Any) -> LossOutput:
        # plancher au plus petit float normal : log(0) seulement si le softmax
        # sous-déborde (écart de logits > ~87), sinon gradient exact (p - cible)
        log_probs = pred[:, :BASE].clamp_min(torch.finfo(pred.dtype).tiny).log()
        digit_loss = -(target[:, :BASE] * log_probs).sum(dim=-1).mean() / log(BASE)
        angle_loss = F.mse_loss(pred[:, BASE:], target[:, BASE:], reduction="mean")
        return LossOutput(digit_loss + self.angle_loss_weight * angle_loss)

def mnist_latent_variance(vae: VAE, n_images: int = 10000, batch_size: int = 1000) -> float:
    """Variance des latents du VAE (mu), moyennée sur les dimensions, sur les
    `n_images` premières images d'entraînement de MNIST. Encodage par lots : le
    VAE SO(2) ne tient pas en mémoire GPU sur 10 000 images d'un coup."""
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    images = torch.stack([dataset[i][0] for i in range(n_images)]).to(vae.device)
    with torch.no_grad():
        mu = torch.cat([vae.encoder(batch)[0] for batch in images.split(batch_size)])
    return mu.var(dim=0).mean().item()


class MNISTDomain(DomainModule):
    """Domaine image : latent = mu du VAE (gelé). La loss est la MSE sur le
    latent divisée par `latent_variance` (variance des latents, cf
    `mnist_latent_variance`) : 0 si la prédiction est parfaite, ~1 en prédisant
    la moyenne, même convention que `DigitAngleDomain`. La variance est un
    buffer : elle est enregistrée dans le checkpoint."""

    def __init__(self, vae: VAE, latent_variance: float = 1.0) -> None:
        latent_dim = vae.hparams.latent_dim
        super().__init__(latent_dim)

        self.vae = vae
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        self.register_buffer("latent_variance", torch.tensor(latent_variance))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mu, _ = self.vae.encoder(x)

        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.vae.decoder(z)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: Sequence[torch.Tensor]) -> list[torch.Tensor]:  # type: ignore
        return self.decode(self.encode(x))

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, raw_target: Any) -> LossOutput:
        return LossOutput(F.mse_loss(pred, target, reduction="mean") / self.latent_variance)

class LoadedDomainConfig(BaseModel):
    checkpoint_path: Path = Path("")
    domain_type: str
    args: Mapping[str, Any] = {}

def get_domains_config(
    domains: list[str], digit_args: Mapping[str, Any] | None = None, vae: str = "conv"
) -> list[LoadedDomainConfig]:
    """`vae` : VAE du domaine image, clé de `VAES`."""
    domains_config = []
    if 'image' in domains:
        domains_config.append(LoadedDomainConfig(domain_type="image", checkpoint_path=VAES[vae][1], args={"vae": vae}))
    if 'digit' in domains:
        domains_config.append(LoadedDomainConfig(domain_type="digit", args=digit_args or {}))

    return domains_config

def load_domain_module(domain: LoadedDomainConfig) -> DomainModule:
    module: DomainModule
    match domain.domain_type:
        case "image":
            vae = VAES[domain.args.get("vae", "conv")][0].load_from_checkpoint(domain.checkpoint_path)
            vae.eval()
            vae.freeze()

            module = MNISTDomain(vae, mnist_latent_variance(vae))

        case "digit":
            module = DigitAngleDomain(**domain.args)
    return module

def load_domain_modules(domains: Sequence[LoadedDomainConfig]) -> nn.ModuleDict:
    # ModuleDict (et non dict) : les domaines doivent être des sous-modules du GW
    # pour être dans son state_dict et suivre ses .to(device)
    return nn.ModuleDict({domain.domain_type: load_domain_module(domain) for domain in domains})
