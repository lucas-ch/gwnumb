from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel
from shimmer import DomainModule, GWDecoder, GWEncoder, LossOutput
import torch
from torch import nn
from torch.nn import Linear, Module
import torch.nn.functional as F

from numb_project.mnist_vae import VAE
from numb_project.utils import get_from_dict_or_val

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
    
    def compute_loss(self, pred, target, raw_target):
        return LossOutput(F.mse_loss(pred, target, reduction="mean"))

class MNISTDomain(DomainModule):
    def __init__(self, latent_dim: int, vae: VAE) -> None:
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
        """Vecteur latent 12D -> image MNIST reconstruite."""
        with torch.no_grad():
            return self.vae.decoder(z)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: Sequence[torch.Tensor]) -> list[torch.Tensor]:  # type: ignore
        return self.decode(self.encode(x))

    def compute_loss(self, pred, target, raw_target):
        return LossOutput(F.mse_loss(pred, target, reduction="mean"))

class LoadedDomainConfig(BaseModel):
    checkpoint_path: Path = ""
    domain_type: str
    args: Mapping[str, Any] = {}


def load_pretrained_module(domain: LoadedDomainConfig) -> DomainModule:
    module: DomainModule
    match domain.domain_type:
        case "image":
            CKPT_PATH = "/home/lucas/gwnumb/checkpoints/mnist.ckpt"

            vae = VAE.load_from_checkpoint(CKPT_PATH, map_location="cpu")
            vae.eval()
            vae.freeze()

            module = MNISTDomain(12, vae)

        case "digit":
            module = IdentityDomain(10)
    return module

def load_pretrained_domain(
    domain: LoadedDomainConfig,
    workspace_dim: int,
    encoders_hidden_dim: int | Mapping[str, int],
    encoders_n_layers: int | Mapping[str, int],
    decoders_hidden_dim: int | Mapping[str, int],
    decoders_n_layers: int | Mapping[str, int],
) -> tuple[DomainModule, Module, Module]:
    module = load_pretrained_module(domain)
    encoder_hidden_dim = get_from_dict_or_val(
        encoders_hidden_dim, domain.domain_type, "global_workspace.encoders.hidden_dim"
    )
    decoder_hidden_dim = get_from_dict_or_val(
        decoders_hidden_dim, domain.domain_type, "global_workspace.decoders.hidden_dim"
    )

    encoder_n_layers = get_from_dict_or_val(
        encoders_n_layers, domain.domain_type, "global_workspace.encoder.n_layers"
    )

    decoder_n_layers = get_from_dict_or_val(
        decoders_n_layers, domain.domain_type, "global_workspace.decoders.n_layers"
    )

    gw_encoder = GWEncoder(
        module.latent_dim, encoder_hidden_dim, workspace_dim, encoder_n_layers
            )
    gw_decoder = GWDecoder(
                    workspace_dim, decoder_hidden_dim, module.latent_dim, decoder_n_layers
                )

    return module, gw_encoder, gw_decoder

def load_pretrained_domains(
    domains: Sequence[LoadedDomainConfig],
    workspace_dim: int,
    encoders_hidden_dim: int | Mapping[str, int],
    encoders_n_layers: int | Mapping[str, int],
    decoders_hidden_dim: int | Mapping[str, int],
    decoders_n_layers: int | Mapping[str, int],
    is_linear: bool = False,
    bias: bool = False,
) -> tuple[nn.ModuleDict, nn.ModuleDict, nn.ModuleDict]:
    modules: dict[str, DomainModule] = {}
    gw_encoders: dict[str, Module] = {}
    gw_decoders: dict[str, Module] = {}
    for domain in domains:
        model, encoder, decoder = load_pretrained_domain(
            domain,
            workspace_dim,
            encoders_hidden_dim,
            encoders_n_layers,
            decoders_hidden_dim,
            decoders_n_layers,
        )
        modules[domain.domain_type] = model
        gw_encoders[domain.domain_type] = encoder
        gw_decoders[domain.domain_type] = decoder

    return nn.ModuleDict(modules), nn.ModuleDict(gw_encoders), nn.ModuleDict(gw_decoders)