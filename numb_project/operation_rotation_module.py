from shimmer import GWModule, GWModuleBase, RawDomainGroupsT
from torch import nn
import torch
import torch.nn.functional as F

from numb_project.constants import BASE
from numb_project.utils import angle_to_cos_sin, rotate_images


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
            nn.ConvTranspose2d(c2, c1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),  # 7->14
            nn.ConvTranspose2d(c1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # 14->28
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
            target = rotate_images(image, angle_deg)

        pred = self(image)

        return F.mse_loss(pred, target)


class GwRotateOperationModule(nn.Module):
    """Rotation apprise directement dans l'espace du global workspace (10D par
    défaut), pour un angle fixe. Même structure que PixelRotateOperationModule
    (encodeur -> goulot d'étranglement `latent_dim` -> décodeur), mais avec des
    couches denses plutôt que des convolutions puisque l'espace GW n'a pas de
    structure spatiale. Tanh final : l'état GW est tanh(encodeur GW), donc dans
    [-1, 1]."""

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
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    @staticmethod
    def image_to_gw_state(gw_mod: GWModule, image: torch.Tensor) -> torch.Tensor:
        # État GW réel d'une image seule : tanh de l'encodage GW (cf GWModule.fuse)
        image_latent = gw_mod.domain_mods["image"].encode(image)
        return gw_mod.fusion_activation_fn(gw_mod.gw_encoders["image"](image_latent))

    def loss(
        self,
        gw_mod: GWModule,
        raw_data: RawDomainGroupsT,
    ) -> torch.Tensor:
        image = raw_data[frozenset({"digit", "image"})]['image']
        batch_size = image.shape[0]

        angle_deg = torch.full((batch_size,), self.degrees, device=image.device)
        rotated_image = rotate_images(image, angle_deg)

        with torch.no_grad():
            input = self.image_to_gw_state(gw_mod, image)
            target = self.image_to_gw_state(gw_mod, rotated_image)

        pred = self(input)

        return F.mse_loss(pred, target)

    @torch.no_grad()
    def metrics(
        self,
        gw_mod: GWModule,
        raw_data: RawDomainGroupsT,
    ) -> dict[str, torch.Tensor]:
        """Métriques pour comparer avec PixelRotateOperationModule (non utilisées
        dans la loss) :
         - pixel_mse : la prédiction GW est décodée jusqu'aux pixels (décodeur GW
           puis décodeur du VAE) et comparée à l'image tournée, comme rotate_pixel_loss ;
         - vae_floor_pixel_mse : MSE de l'image tournée passée par le VAE seul,
           plancher que la voie GW ne peut pas battre ;
         - angle_error_deg : la prédiction est décodée vers le domaine digit et
           l'angle lu dans (cos, sin) est comparé à l'angle de départ + `degrees`."""
        pair_key = frozenset({"digit", "image"})
        image = raw_data[pair_key]['image']
        digit = raw_data[pair_key]['digit']
        image_mod = gw_mod.domain_mods["image"]

        angle_deg = torch.full((image.shape[0],), self.degrees, device=image.device)
        rotated_image = rotate_images(image, angle_deg)

        pred = self(self.image_to_gw_state(gw_mod, image))
        pred_pixels = image_mod.decode(gw_mod.gw_decoders["image"](pred))
        vae_pixels = image_mod.decode(image_mod.encode(rotated_image))

        start_angle = torch.rad2deg(torch.atan2(digit[:, BASE + 1], digit[:, BASE]))
        target_cos_sin = angle_to_cos_sin(start_angle + self.degrees)
        pred_cos_sin = gw_mod.gw_decoders["digit"](pred)[:, BASE:]
        # Angle entre les vecteurs (cos, sin) prédit et cible, dans [0, 180]
        angle_error = torch.rad2deg(torch.atan2(
            pred_cos_sin[:, 1] * target_cos_sin[:, 0] - pred_cos_sin[:, 0] * target_cos_sin[:, 1],
            (pred_cos_sin * target_cos_sin).sum(dim=-1),
        )).abs()

        return {
            "pixel_mse": F.mse_loss(pred_pixels, rotated_image),
            "vae_floor_pixel_mse": F.mse_loss(vae_pixels, rotated_image),
            "angle_error_deg": angle_error.mean(),
        }
