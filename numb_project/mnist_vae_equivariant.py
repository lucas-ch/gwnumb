"""VAE équivariant aux rotations de 90° (groupe C4), construit avec escnn, à
comparer au VAE convolutif de mnist_vae.py (même latent de 20 dimensions, même
entraînement).

Le latent est fait de LATENT_DIM / 4 champs « réguliers » de C4 : chaque champ a
4 coordonnées, et tourner l'image de 90° permute circulairement ces 4
coordonnées. Une permutation garde diagonale la covariance du VAE (mu et logvar
sont des champs réguliers) et laisse invariant le prior N(0, I).

L'équivariance est exacte pour 90°, 180° et 270° : ces rotations envoient la
grille de pixels sur elle-même, et toutes les couches (convolutions à noyau
impair, pooling 2x2 sur 28 et 14 pixels, upsampling) respectent la symétrie de
la grille. Pour les autres angles, elle n'est qu'approchée."""
import torch
import torch.nn as nn
from escnn import gspaces
from escnn import nn as enn
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from numb_project.constants import EQUIVARIANT_VAE_CHECKPOINT_FILE
from numb_project.mnist_vae import VAE

LATENT_DIM = 20
N_ROTATIONS = 4  # C4 : rotations de 0, 90, 180 et 270°


class EquivariantEncoder(nn.Module):
    """Image (B, 1, 28, 28) -> (mu, logvar), chacun (B, latent_dim), champs
    réguliers de C4. Même structure que `Encoder` : 2 étages convolutifs
    (28 -> 14 -> 7), une couche cachée, puis mu et logvar."""

    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=256):
        super().__init__()
        assert latent_dim % N_ROTATIONS == 0, "latent_dim doit être un multiple de 4 (champs réguliers de C4)"
        self.gspace = gspaces.rot2dOnR2(N=N_ROTATIONS)
        self.in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])

        def regular(n_channels: int) -> enn.FieldType:
            return enn.FieldType(self.gspace, [self.gspace.regular_repr] * (n_channels // N_ROTATIONS))

        h1, h2, hidden = regular(32), regular(64), regular(hidden_dim)
        self.latent_type = regular(latent_dim)
        self.conv = enn.SequentialModule(
            enn.R2Conv(self.in_type, h1, kernel_size=5, padding=2),
            enn.ReLU(h1),
            enn.PointwiseAvgPool2D(h1, kernel_size=2),  # -> 14x14
            enn.R2Conv(h1, h2, kernel_size=5, padding=2),
            enn.ReLU(h2),
            enn.PointwiseAvgPool2D(h2, kernel_size=2),  # -> 7x7
            enn.R2Conv(h2, hidden, kernel_size=7),      # -> 1x1, rôle de la couche dense
            enn.ReLU(hidden),
        )
        self.fc_mu = enn.R2Conv(hidden, self.latent_type, kernel_size=1)
        self.fc_logvar = enn.R2Conv(hidden, self.latent_type, kernel_size=1)

    def forward(self, x):
        h = self.conv(enn.GeometricTensor(x, self.in_type))
        mu = self.fc_mu(h).tensor.flatten(1)
        logvar = self.fc_logvar(h).tensor.flatten(1)
        return mu, logvar


class EquivariantDecoder(nn.Module):
    """Latent (B, latent_dim), champs réguliers de C4 -> image (B, 1, 28, 28).
    Miroir de l'encodeur : couche cachée, 1x1 -> 7x7, puis 7 -> 14 -> 28."""

    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=256):
        super().__init__()
        self.gspace = gspaces.rot2dOnR2(N=N_ROTATIONS)

        def regular(n_channels: int) -> enn.FieldType:
            return enn.FieldType(self.gspace, [self.gspace.regular_repr] * (n_channels // N_ROTATIONS))

        self.latent_type = regular(latent_dim)
        hidden, h2, h1 = regular(hidden_dim), regular(64), regular(32)
        out_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])
        self.net = enn.SequentialModule(
            enn.R2Conv(self.latent_type, hidden, kernel_size=1),
            enn.ReLU(hidden),
            enn.R2ConvTransposed(hidden, h2, kernel_size=7),  # 1x1 -> 7x7
            enn.ReLU(h2),
            enn.R2Upsampling(h2, scale_factor=2),             # -> 14x14
            enn.R2Conv(h2, h1, kernel_size=5, padding=2),
            enn.ReLU(h1),
            enn.R2Upsampling(h1, scale_factor=2),             # -> 28x28
            enn.R2Conv(h1, out_type, kernel_size=5, padding=2),
        )

    def forward(self, z):
        h = enn.GeometricTensor(z.view(z.size(0), -1, 1, 1), self.latent_type)
        return torch.sigmoid(self.net(h).tensor)  # pixels dans [0, 1]


class EquivariantVAE(VAE):
    """`VAE` (même loss, même entraînement avec des rotations aléatoires de
    0 à 360°) avec un encodeur et un décodeur équivariants à C4."""

    def __init__(self, latent_dim=LATENT_DIM, lr=3e-4, kld_weight=1.0):
        super().__init__(latent_dim, lr, kld_weight)
        self.encoder = EquivariantEncoder(latent_dim)
        self.decoder = EquivariantDecoder(latent_dim)

    def rotate_latent(self, z: torch.Tensor, angle_deg: float) -> torch.Tensor:
        """Action sur le latent de `rotate_images(·, angle_deg)`, pour un
        multiple de 90° : décalage circulaire des 4 coordonnées de chaque champ
        régulier. encoder(rotate_images(x, a)) = rotate_latent(encoder(x), a)."""
        assert angle_deg % 90 == 0, "C4 : seuls les multiples de 90° agissent sur le latent"
        k = -int(angle_deg // 90)  # rotate_images tourne dans le sens horaire
        return z.view(z.size(0), -1, N_ROTATIONS).roll(k, dims=-1).flatten(1)


def main():
    transform = transforms.ToTensor()  # normalise déjà dans [0, 1], la rotation est faite dans generic_step

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

    model = EquivariantVAE(latent_dim=LATENT_DIM, lr=3e-4, kld_weight=0.75)

    checkpoint_dir, checkpoint_name = EQUIVARIANT_VAE_CHECKPOINT_FILE.rsplit("/", 1)
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
