"""Global workspace équivariant à SO(2), branché sur le VAE SO(2)
(mnist_vae_so2.py).

L'état du GW reste un vecteur (B, latent_dim), compatible avec shimmer, mais
structuré comme le latent du VAE : n_scalars champs scalaires (invariants) puis
n_vectors champs vectoriels de fréquence 1 (2 coordonnées, tournés de θ quand
l'image tourne de θ).

- Encodeur/décodeur GW image (latent VAE <-> GW) : MLP équivariants (escnn,
  sans espace de base) : encoder une image tournée donne l'état GW tourné.
- Fusion : `EquivariantTanh` remplace le tanh coordonnée par coordonnée de
  shimmer, qui ne respecte pas les vecteurs.
- Encodeur/décodeur digit : MLP classiques (cf gw_model.py), qui doivent
  apprendre à se brancher sur cette structure via les losses d'alignement."""
import torch
from escnn import gspaces
from escnn import nn as enn
from torch import nn

from numb_project.mnist_vae_so2 import MAX_FREQUENCY, fourier_elu, latent_types, make_gspace, rotate_fields


def vector_gspace() -> gspaces.GSpace:
    """SO(2) agissant sur des vecteurs (pas d'espace de base), même groupe que
    les convolutions du VAE SO(2)."""
    return gspaces.no_base_space(make_gspace().fibergroup)


def field_type(n_scalars: int, n_vectors: int) -> enn.FieldType:
    """n_scalars scalaires puis n_vectors vecteurs de fréquence 1 : même
    structure que le latent du VAE SO(2)."""
    return latent_types(vector_gspace(), n_scalars, n_vectors)[0]


class EquivariantMLP(nn.Module):
    """Équivalent équivariant de `GWDecoder`/`GWEncoder` de shimmer : Linear,
    activation, n_layers x (Linear, activation), Linear. Les couches cachées
    sont des champs de Fourier (fréquences 0..MAX_FREQUENCY) avec FourierELU ;
    `hidden_dim` est arrondi à un multiple de leur dimension."""

    def __init__(self, in_type: enn.FieldType, out_type: enn.FieldType, hidden_dim: int, n_layers: int):
        super().__init__()
        self.in_type, self.out_type = in_type, out_type
        channels = max(1, hidden_dim // (2 * MAX_FREQUENCY + 1))
        acts = [fourier_elu(in_type.gspace, channels) for _ in range(n_layers + 1)]
        layers: list[enn.EquivariantModule] = [enn.Linear(in_type, acts[0].in_type), acts[0]]
        for previous, act in zip(acts, acts[1:]):
            layers += [enn.Linear(previous.in_type, act.in_type), act]
        layers.append(enn.Linear(acts[-1].in_type, out_type))
        self.net = enn.SequentialModule(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(enn.GeometricTensor(x, self.in_type)).tensor


class EquivariantTanh(nn.Module):
    """Activation de fusion du GW, équivariante : tanh sur les scalaires, et
    pour chaque vecteur v, v · tanh(|v|) / |v| (direction conservée, norme
    écrasée dans [0, 1[). Comme le tanh de shimmer, l'état reste dans ]-1, 1[."""

    def __init__(self, n_scalars: int, n_vectors: int, eps: float = 1e-8):
        super().__init__()
        self.n_scalars, self.n_vectors, self.eps = n_scalars, n_vectors, eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scalars = torch.tanh(x[:, :self.n_scalars])
        vectors = x[:, self.n_scalars:].reshape(x.shape[0], self.n_vectors, 2)
        norm = (vectors.pow(2).sum(-1, keepdim=True) + self.eps).sqrt()
        vectors = vectors * torch.tanh(norm) / norm
        return torch.cat([scalars, vectors.flatten(1)], dim=-1)


class EquivariantGWStructure:
    """Structure de l'état du GW : `latent_dim` = n_scalars + 2 n_vectors."""

    def __init__(self, latent_dim: int, n_vectors: int):
        self.n_scalars = latent_dim - 2 * n_vectors
        self.n_vectors = n_vectors
        assert self.n_scalars >= 0, "latent_dim doit être >= 2 n_vectors"
        self.field_type = field_type(self.n_scalars, n_vectors)

    def fusion_activation(self) -> EquivariantTanh:
        return EquivariantTanh(self.n_scalars, self.n_vectors)

    def rotate(self, gw_state: torch.Tensor, angle_deg: float) -> torch.Tensor:
        """Action de `rotate_images(·, angle_deg)` sur un état du GW."""
        return rotate_fields(gw_state, self.field_type, angle_deg)
