from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel
from shimmer import DomainModule, LossOutput
import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F

from numb_project.mnist_vae import VAE
from numb_project.constants import BASE, DIGIT_DIM, VAE_CHECKPOINT_FILE


class DigitAngleDomain(DomainModule):
    """Domaine digit : one-hot du chiffre (BASE dimensions) suivi de (cos, sin)
    de l'angle de l'image. Le latent est le vecteur lui-même (encode/decode =
    identité).

    Les prédictions du GW sortent de `DigitGWDecoder` (gw_model.py) :
    probabilités des chiffres et (cos, sin) de norme 1.

    Loss en deux parties, pour que l'angle (2 dimensions sur 12) ne soit pas
    noyé dans la moyenne : cross entropy sur le chiffre (NLL des probabilités
    prédites, la cible est le one-hot) et MSE sur (cos, sin)."""

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
        digit_loss = -(target[:, :BASE] * log_probs).sum(dim=-1).mean()
        angle_loss = F.mse_loss(pred[:, BASE:], target[:, BASE:], reduction="mean")
        return LossOutput(digit_loss + self.angle_loss_weight * angle_loss)

class MNISTDomain(DomainModule):
    def __init__(self, vae: VAE) -> None:
        latent_dim = vae.encoder.fc_mu.out_features
        super().__init__(latent_dim)

        self.vae = vae
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

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
        return LossOutput(F.mse_loss(pred, target, reduction="mean"))

class LoadedDomainConfig(BaseModel):
    checkpoint_path: Path = Path("")
    domain_type: str
    args: Mapping[str, Any] = {}

def get_domains_config(domains: list[str], digit_args: Mapping[str, Any] | None = None) -> list[LoadedDomainConfig]:
    domains_config = []
    if 'image' in domains:
        domains_config.append(LoadedDomainConfig(domain_type="image", checkpoint_path=VAE_CHECKPOINT_FILE))
    if 'digit' in domains:
        domains_config.append(LoadedDomainConfig(domain_type="digit", args=digit_args or {}))

    return domains_config

def load_domain_module(domain: LoadedDomainConfig) -> DomainModule:
    module: DomainModule
    match domain.domain_type:
        case "image":
            vae = VAE.load_from_checkpoint(domain.checkpoint_path)
            vae.eval()
            vae.freeze()

            module = MNISTDomain(vae)

        case "digit":
            module = DigitAngleDomain(**domain.args)
    return module

def load_domain_modules(domains: Sequence[LoadedDomainConfig]) -> nn.ModuleDict:
    # ModuleDict (et non dict) : les domaines doivent être des sous-modules du GW
    # pour être dans son state_dict et suivre ses .to(device)
    return nn.ModuleDict({domain.domain_type: load_domain_module(domain) for domain in domains})
