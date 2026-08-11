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
from numb_project.constants import BASE, VAE_CHECKPOINT_FILE


class IdentityDomain(DomainModule):
    def __init__(self, size) -> None:
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

    def compute_loss(self, pred, target, raw_target):
        return LossOutput(F.mse_loss(pred, target, reduction="mean"))

class OperationModule(nn.Module):
    def __init__(self, gw_size, task_size, hidden_size):
        super().__init__()

        self.transfo = nn.Sequential(
            nn.Linear(gw_size + task_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, gw_size),
        )

    def forward(self, x, task):
        combined = torch.cat([x, task], dim=1)
        return self.transfo(combined)

class AttentionModule(nn.Module):
    def __init__(self, gw_size=10, output_size=3, hidden_size=128, temperature=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_cell = nn.LSTMCell(input_size=gw_size, hidden_size=hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.temperature = temperature

    def forward(self, x: torch.Tensor, hc: tuple[torch.Tensor, torch.Tensor]):
        h, c = self.memory_cell(x, hc)
        logits = self.output(h)

        attention = torch.softmax(logits / self.temperature, dim=-1)

        return attention, (h, c)

class LoadedDomainConfig(BaseModel):
    checkpoint_path: Path = ""
    domain_type: str
    args: Mapping[str, Any] = {}

def get_domains_config(domains: list[str]):
    domains_config = []
    if 'image' in domains:
        domains_config.append(LoadedDomainConfig(domain_type="image", checkpoint_path=VAE_CHECKPOINT_FILE))
    if 'digit' in domains:
        domains_config.append(LoadedDomainConfig(domain_type="digit"))

    return domains_config

def load_pretrained_module(domain: LoadedDomainConfig) -> DomainModule:
    module: DomainModule
    match domain.domain_type:
        case "image":
            CKPT_PATH = domain.checkpoint_path
            vae = VAE.load_from_checkpoint(CKPT_PATH)
            vae.eval()
            vae.freeze()

            module = MNISTDomain(vae)

        case "digit":
            module = IdentityDomain(BASE)
    return module

def load_domain(
    domain: LoadedDomainConfig,
    workspace_dim: int,
    encoders_hidden_dim: int | Mapping[str, int],
    encoders_n_layers: int | Mapping[str, int],
    decoders_hidden_dim: int | Mapping[str, int],
    decoders_n_layers: int | Mapping[str, int]) -> tuple[DomainModule, Module, Module]:
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

def load_domains(
    domains: Sequence[LoadedDomainConfig],
    workspace_dim: int,
    encoders_hidden_dim: int | Mapping[str, int],
    encoders_n_layers: int | Mapping[str, int],
    decoders_hidden_dim: int | Mapping[str, int],
    decoders_n_layers: int | Mapping[str, int]) -> tuple[nn.ModuleDict, nn.ModuleDict, nn.ModuleDict]:
        modules: dict[str, DomainModule] = {}
        gw_encoders: dict[str, Module] = {}
        gw_decoders: dict[str, Module] = {}
        for domain in domains:
            model, encoder, decoder = load_domain(
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