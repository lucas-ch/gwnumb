"""VAE équivariant à toutes les rotations (groupe continu SO(2)), construit avec
escnn. Version « complète » de mnist_vae_equivariant.py, qui ne l'est qu'aux
rotations de 90° (C4).

Les caractéristiques cachées sont des champs de Fourier de SO(2) à bande
limitée (fréquences 0 à MAX_FREQUENCY), avec une non-linéarité FourierELU :
ELU appliquée sur N_GRID rotations échantillonnées puis reprojetée sur les
fréquences, ce qui commute avec toutes les rotations (aux erreurs
d'échantillonnage près).

Le latent est fait de :
- N_SCALARS champs scalaires (fréquence 0), invariants par rotation : le
  « contenu » de l'image (chiffre, style) ;
- N_VECTORS champs vectoriels (fréquence 1, 2 coordonnées) : tourner l'image
  d'un angle θ tourne chacun de ces vecteurs de θ.
logvar a une seule valeur (invariante) par champ, partagée par les 2
coordonnées d'un vecteur : la covariance d'un champ vectoriel est isotrope,
donc le posterior N(mu, sigma²) est équivariant et le prior N(0, I) invariant.

Sur une grille de pixels, l'équivariance n'est exacte que pour 90°, 180° et
270° ; pour les autres angles elle est approchée (filtres steerables à
fréquence coupée, masque circulaire en entrée, pooling anti-aliasé)."""
import torch
import torch.nn as nn
from escnn import gspaces
from escnn import nn as enn
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from numb_project.constants import SO2_VAE_CHECKPOINT_FILE
from numb_project.mnist_vae import VAE

N_SCALARS = 8
N_VECTORS = 6
LATENT_DIM = N_SCALARS + 2 * N_VECTORS  # 20, comme les autres VAE
MAX_FREQUENCY = 3  # fréquence max des champs cachés et des filtres
N_GRID = 16        # rotations échantillonnées par FourierELU


def make_gspace() -> gspaces.GSpace2D:
    return gspaces.rot2dOnR2(N=-1, maximum_frequency=MAX_FREQUENCY)


def fourier_elu(gspace: gspaces.GSpace, channels: int) -> enn.FourierELU:
    """Non-linéarité sur `channels` champs de Fourier (fréquences 0..MAX_FREQUENCY,
    dimension 2 * MAX_FREQUENCY + 1 chacun) ; son `in_type` sert de type caché."""
    irreps = gspace.fibergroup.bl_irreps(MAX_FREQUENCY)
    return enn.FourierELU(gspace, channels, irreps=irreps, N=N_GRID, inplace=True)


def blur_pool(field_type: enn.FieldType) -> list[enn.EquivariantModule]:
    """Sous-échantillonnage par 2 anti-aliasé : flou gaussien (stride 1, noyau
    impair) puis moyenne 2x2. Les deux respectent la symétrie de la grille paire
    (28, 14), contrairement à un pooling à stride 2 qui garde les indices pairs."""
    return [
        enn.PointwiseAvgPoolAntialiased2D(field_type, sigma=0.66, stride=1),
        enn.PointwiseAvgPool2D(field_type, kernel_size=2),
    ]


def latent_types(gspace: gspaces.GSpace, n_scalars: int, n_vectors: int):
    """(type de mu, type de logvar) : mu = scalaires + vecteurs de fréquence 1,
    logvar = un scalaire par champ."""
    mu_type = enn.FieldType(gspace, [gspace.trivial_repr] * n_scalars + [gspace.irrep(1)] * n_vectors)
    logvar_type = enn.FieldType(gspace, [gspace.trivial_repr] * (n_scalars + n_vectors))
    return mu_type, logvar_type


def rotate_fields(z: torch.Tensor, field_type: enn.FieldType, angle_deg: float) -> torch.Tensor:
    """Action de `rotate_images(·, angle_deg)` sur des vecteurs (B, field_type.size)
    de type `field_type`. Les angles positifs de rotate_images tournent dans le
    sens horaire, ceux d'escnn dans le sens trigonométrique."""
    g = field_type.fibergroup.element(-angle_deg * torch.pi / 180)
    rho = torch.as_tensor(field_type.representation(g), dtype=z.dtype, device=z.device)
    return z @ rho.T


class SO2Encoder(nn.Module):
    """Image (B, 1, 28, 28) -> (mu, logvar), chacun (B, latent_dim). Même
    structure que `Encoder` : 2 étages convolutifs (28 -> 14 -> 7), une couche
    cachée, puis mu et logvar."""

    def __init__(self, n_scalars=N_SCALARS, n_vectors=N_VECTORS, channels=(8, 16, 32)):
        super().__init__()
        self.gspace = make_gspace()
        self.in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])
        act1, act2, act_hidden = (fourier_elu(self.gspace, c) for c in channels)
        self.conv = enn.SequentialModule(
            enn.MaskModule(self.in_type, 28, margin=1),  # coins hors du disque -> 0
            enn.R2Conv(self.in_type, act1.in_type, kernel_size=5, padding=2),
            act1,
            *blur_pool(act1.in_type),                                              # -> 14x14
            enn.R2Conv(act1.in_type, act2.in_type, kernel_size=5, padding=2),
            act2,
            *blur_pool(act2.in_type),                                              # -> 7x7
            enn.R2Conv(act2.in_type, act_hidden.in_type, kernel_size=7),           # -> 1x1, rôle de la couche dense
            act_hidden,
        )
        self.mu_type, self.logvar_type = latent_types(self.gspace, n_scalars, n_vectors)
        self.fc_mu = enn.R2Conv(act_hidden.in_type, self.mu_type, kernel_size=1)
        self.fc_logvar = enn.R2Conv(act_hidden.in_type, self.logvar_type, kernel_size=1)
        # Coordonnée de mu -> champ auquel elle appartient (pour étendre logvar)
        field_sizes = torch.tensor([r.size for r in self.mu_type.representations])
        self.register_buffer("field_index", torch.repeat_interleave(torch.arange(len(field_sizes)), field_sizes))

    def forward(self, x):
        h = self.conv(enn.GeometricTensor(x, self.in_type))
        mu = self.fc_mu(h).tensor.flatten(1)
        logvar = self.fc_logvar(h).tensor.flatten(1)[:, self.field_index]
        return mu, logvar


class SO2Decoder(nn.Module):
    """Latent (B, latent_dim) -> image (B, 1, 28, 28). Miroir de l'encodeur :
    couche cachée, 1x1 -> 7x7, puis 7 -> 14 -> 28."""

    def __init__(self, n_scalars=N_SCALARS, n_vectors=N_VECTORS, channels=(8, 16, 32)):
        super().__init__()
        self.gspace = make_gspace()
        self.latent_type, _ = latent_types(self.gspace, n_scalars, n_vectors)
        act1, act2, act_hidden = (fourier_elu(self.gspace, c) for c in channels)
        act2_bis = fourier_elu(self.gspace, channels[1])
        out_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])
        self.net = enn.SequentialModule(
            enn.R2Conv(self.latent_type, act_hidden.in_type, kernel_size=1),
            act_hidden,
            enn.R2ConvTransposed(act_hidden.in_type, act2.in_type, kernel_size=7),  # 1x1 -> 7x7
            act2,
            enn.R2Upsampling(act2.in_type, scale_factor=2),                          # -> 14x14
            enn.R2Conv(act2.in_type, act1.in_type, kernel_size=5, padding=2),
            act1,
            enn.R2Upsampling(act1.in_type, scale_factor=2),                          # -> 28x28
            enn.R2Conv(act1.in_type, act2_bis.in_type, kernel_size=5, padding=2),
            act2_bis,
            enn.R2Conv(act2_bis.in_type, out_type, kernel_size=5, padding=2),
        )

    def forward(self, z):
        h = enn.GeometricTensor(z.view(z.size(0), -1, 1, 1), self.latent_type)
        return torch.sigmoid(self.net(h).tensor)  # pixels dans [0, 1]


class SO2VAE(VAE):
    """`VAE` (même loss, même entraînement avec des rotations aléatoires de
    0 à 360°) avec un encodeur et un décodeur équivariants à SO(2)."""

    def __init__(self, n_scalars=N_SCALARS, n_vectors=N_VECTORS, lr=3e-4, kld_weight=1.0):
        super().__init__(n_scalars + 2 * n_vectors, lr, kld_weight)
        self.encoder = SO2Encoder(n_scalars, n_vectors)
        self.decoder = SO2Decoder(n_scalars, n_vectors)

    def rotate_latent(self, z: torch.Tensor, angle_deg: float) -> torch.Tensor:
        """Action sur le latent de `rotate_images(·, angle_deg)` : scalaires
        inchangés, vecteurs tournés. encoder(rotate_images(x, a)) ≈
        rotate_latent(encoder(x), a). Les angles positifs de rotate_images
        tournent dans le sens horaire, ceux d'escnn dans le sens trigonométrique."""
        return rotate_fields(z, self.encoder.mu_type, angle_deg)


def main():
    transform = transforms.ToTensor()  # normalise déjà dans [0, 1], la rotation est faite dans generic_step

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

    model = SO2VAE(n_scalars=N_SCALARS, n_vectors=N_VECTORS, lr=3e-4, kld_weight=0.75)

    checkpoint_dir, checkpoint_name = SO2_VAE_CHECKPOINT_FILE.rsplit("/", 1)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_last=True,
        mode="min",
        dirpath=checkpoint_dir,
        filename=checkpoint_name.removesuffix(".ckpt"))

    trainer = Trainer(
        max_epochs=1000,
        callbacks=[checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
