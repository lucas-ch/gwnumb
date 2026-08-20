from shimmer import GWModuleBase, RawDomainGroupsT
from torch import nn
import torch
import torch.nn.functional as F

from numb_project.data_module import rotate_image_batch


class PixelRotateOperationModule(nn.Module):
    """Rotation apprise directement sur l'image (CNN), pour un angle fixe
    (`degrees`) : pas de conditionnement sur l'angle, le réseau apprend une
    seule transformation constante image -> image pivotée de `degrees`."""

    def __init__(
        self,
        latent_dim: int = 64,
        channels: tuple[int, int] = (32, 64),
        degrees: float = 20.0,
    ):
        super().__init__()
        c1, c2 = channels
        self.degrees = degrees

        self.enc = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.enc_fc = nn.Linear(c2 * 7 * 7, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, c2 * 7 * 7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c2, c1, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # 7->14
            nn.ConvTranspose2d(c1, 1, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),  # 14->28
        )
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        z = self.enc_fc(z.flatten(1))

        out = self.dec_fc(z).view(-1, self.channels[1], 7, 7)
        return self.dec(out)

    def loss(
        self,
        gw_mod: GWModuleBase,
        raw_data: RawDomainGroupsT,
    ) -> torch.Tensor:
        image = raw_data[frozenset({"digit", "image"})]['image']
        batch_size = image.shape[0]

        angle_deg = torch.full((batch_size,), self.degrees, device=image.device)
        with torch.no_grad():
            target = rotate_image_batch(image, angle_deg)

        pred = self(image)

        return F.mse_loss(pred, target)


class GwRotateOperationModule(nn.Module):
    """Rotation apprise directement dans l'espace du global workspace (10D par
    défaut), pour un angle fixe. Même structure que PixelRotateOperationModule
    (encodeur -> goulot d'étranglement `latent_dim` -> décodeur), mais avec des
    couches denses plutôt que des convolutions puisque l'espace GW n'a pas de
    structure spatiale ; pas de Sigmoid final non plus, les coordonnées GW
    n'étant pas bornées comme des pixels."""

    def __init__(
        self,
        gw_size: int = 10,
        hidden_size: int = 128,
        latent_dim: int = 64,
        degrees: float = 20.0,
    ):
        super().__init__()
        self.degrees = degrees

        self.encoder = nn.Sequential(
            nn.Linear(gw_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, gw_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def loss(
        self,
        gw_mod: GWModuleBase,
        raw_data: RawDomainGroupsT,
    ) -> torch.Tensor:
        image = raw_data[frozenset({"digit", "image"})]['image']
        batch_size = image.shape[0]

        angle_deg = torch.full((batch_size,), self.degrees, device=image.device)
        rotated_image = rotate_image_batch(image, angle_deg)

        with torch.no_grad():
            image_latent = gw_mod.domain_mods["image"].encode(image)
            rotated_latent = gw_mod.domain_mods["image"].encode(rotated_image)
            input = gw_mod.gw_encoders["image"](image_latent)
            target = gw_mod.gw_encoders["image"](rotated_latent)

        pred = self(input)

        return F.mse_loss(pred, target)
