from shimmer import GWModule, GWModuleBase, RawDomainGroupsT
from torch import nn
import torch
import torch.nn.functional as F

from numb_project.constants import BASE
from numb_project.utils import angle_to_cos_sin, rotate_images


def image_to_gw_state(gw_mod: GWModule, image: torch.Tensor) -> torch.Tensor:
    # État GW réel d'une image seule : tanh de l'encodage GW (cf GWModule.fuse)
    image_latent = gw_mod.domain_mods["image"].encode(image)
    return gw_mod.fusion_activation_fn(gw_mod.gw_encoders["image"](image_latent))


def chained_rotation_targets(image: torch.Tensor, degrees: float, n_steps: int) -> list[torch.Tensor]:
    """Cibles d'une rotation de `degrees` enchaînée `n_steps` fois : l'image
    tournée de k x `degrees` pour k = 1..n_steps, chaque cible étant calculée en
    une seule rotation pour ne pas accumuler l'erreur d'interpolation."""
    return [
        rotate_images(image, torch.full((image.shape[0],), degrees * step, device=image.device))
        for step in range(1, n_steps + 1)
    ]


class PixelRotateOperationModule(nn.Module):
    """Rotation apprise directement sur l'image (CNN), pour un angle fixe
    (`degrees`)."""

    def __init__(
        self,
        latent_dim: int = 64,
        channels: tuple[int, int] = (32, 64),
        degrees: float = 20.0,
        n_steps: int = 1,
    ):
        super().__init__()
        c1, c2 = channels
        self.degrees = degrees
        self.n_steps = n_steps

        self.enc = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.enc_fc = nn.Linear(c2 * 7 * 7, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, c2 * 7 * 7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1),  # 7->14
            nn.ReLU(),
            nn.ConvTranspose2d(c1, 1, kernel_size=4, stride=2, padding=1),   # 14->28
            nn.Sigmoid(),
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
        # Le module est appliqué n_steps fois d'affilée : chaque étape k est
        # comparée à l'image tournée de k x degrees, et le gradient traverse
        # toute la chaîne pour que le module apprenne à corriger sa propre dérive.
        image = raw_data[frozenset({"digit", "image"})]['image']

        pred = image
        losses = []
        for target in chained_rotation_targets(image, self.degrees, self.n_steps):
            pred = self(pred)
            losses.append(F.mse_loss(pred, target))

        return torch.stack(losses).mean()

class GwRotateOperationModule(nn.Module):
    """Rotation apprise directement dans l'espace du global workspace (10D par
    défaut), pour un angle fixe.

    Avec `residual`, le MLP ne prédit que le déplacement de l'état GW, ajouté
    avant le tanh : sortie = tanh(atanh(x) + MLP(x)). Ce qui ne doit pas changer
    avec la rotation est recopié tel quel au lieu d'être reconstruit, et la
    dernière couche initialisée à zéro fait démarrer le module à l'identité.
    Sans `residual` : sortie = tanh(MLP(x))."""

    # atanh(±1) est infini : les états GW (sortie de tanh) sont bornés juste avant
    ATANH_CLAMP = 0.999

    def __init__(
        self,
        gw_size: int = 10,
        hidden_size: int = 128,
        latent_dim: int = 64,
        degrees: float = 20.0,
        n_steps: int = 1,
        residual: bool = False,
    ):
        super().__init__()
        self.degrees = degrees
        self.n_steps = n_steps
        self.residual = residual

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
        if residual:
            nn.init.zeros_(self.decoder[-1].weight)
            nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.decoder(self.encoder(x))
        if self.residual:
            out = torch.atanh(x.clamp(-self.ATANH_CLAMP, self.ATANH_CLAMP)) + out
        return torch.tanh(out)

    def loss(
        self,
        gw_mod: GWModule,
        raw_data: RawDomainGroupsT,
    ) -> torch.Tensor:
        # Même principe que PixelRotateOperationModule.loss : n_steps rotations
        # enchaînées dans l'espace GW, chaque étape comparée à l'état GW de
        # l'image tournée de k x degrees.
        image = raw_data[frozenset({"digit", "image"})]['image']

        with torch.no_grad():
            gw_state = image_to_gw_state(gw_mod, image)
            targets = [
                image_to_gw_state(gw_mod, rotated)
                for rotated in chained_rotation_targets(image, self.degrees, self.n_steps)
            ]

        pred = gw_state
        losses = []
        for target in targets:
            pred = self(pred)
            losses.append(F.mse_loss(pred, target))

        return torch.stack(losses).mean()