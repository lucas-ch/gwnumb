import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from numb_project.data_module import rotate_image_batch

LATENT_DIM = 20

class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # -> 32x14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> 64x7x7
            nn.ReLU(),
        )
        self.flatten_dim = 64 * 7 * 7
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64 * 7 * 7),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> 32x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # -> 1x28x28
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 64, 7, 7)
        x_hat = self.deconv(h)
        return torch.sigmoid(x_hat)  # pixels dans [0, 1]


class VAE(LightningModule):
    def __init__(self, latent_dim=LATENT_DIM, lr=3e-4, kld_weight=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def generic_step(self, batch, batch_idx):
        x, _ = batch  # MNIST retourne (image, label), on ignore le label
        # Même rotation (bilinéaire, sur GPU) que celle appliquée dans le GW
        angle_deg = torch.rand(x.size(0), device=x.device) * 360.0
        x = rotate_image_batch(x, angle_deg)
        x_hat, mu, logvar = self(x)

        recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        loss = recon_loss + self.hparams.kld_weight * kld_loss
        return loss, recon_loss, kld_loss

    def training_step(self, batch, batch_idx):
        loss, recon_loss, kld_loss = self.generic_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kld_loss", kld_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon_loss, kld_loss = self.generic_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def main():
    transform = transforms.ToTensor()  # normalise déjà dans [0, 1], la rotation est faite dans generic_step

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

    model = VAE(latent_dim=LATENT_DIM, lr=3e-4, kld_weight=0.75)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_last=True,
        mode="min",
        dirpath="/home/lucas/gwnumb/checkpoints",
        filename=f"mnist-{LATENT_DIM}D")

    trainer = Trainer(
        max_epochs=1000,
        callbacks=[checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()