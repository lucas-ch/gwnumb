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


class IdentityDomain(DomainModule):
    def __init__(self, size: int) -> None:
        super().__init__(size)

        self.size = size
        self.encoder = Linear(self.size, self.size, bias=False)
        self.decoder = Linear(self.size, self.size, bias=False)

        self.encoder.weight.data = torch.eye(self.size)
        self.decoder.weight.data = torch.eye(self.size)   
        for param in list(self.encoder.parameters()) + list(self.decoder.parameters()):
            param.requires_grad = False
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return self.decode(self.encode(x))
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, raw_target: Any) -> LossOutput:
        return LossOutput(F.mse_loss(pred, target, reduction="mean"))

class DigitAngleDomain(IdentityDomain):
    """Domaine digit : one-hot du chiffre (BASE dimensions) suivi de (cos, sin)
    de l'angle de l'image. Les deux parties ont chacune leur MSE, pour que
    l'angle (2 dimensions sur 12) ne soit pas noyé dans la moyenne."""

    def __init__(self, angle_loss_weight: float = 1.0) -> None:
        super().__init__(DIGIT_DIM)
        self.angle_loss_weight = angle_loss_weight

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, raw_target: Any) -> LossOutput:
        digit_loss = F.mse_loss(pred[:, :BASE], target[:, :BASE], reduction="mean")
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
