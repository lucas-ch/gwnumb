from shimmer import GWModule, GWModuleBase, RawDomainGroupsT
from torch import nn
import torch
import torch.nn.functional as F

from numb_project.utils import rotate_images


def image_to_gw_state(gw_mod: GWModule, image: torch.Tensor) -> torch.Tensor:
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
        self.channels = channels
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

        pred = image
        losses = []
        for target in chained_rotation_targets(image, self.degrees, self.n_steps):
            pred = self(pred)
            losses.append(F.mse_loss(pred, target))

        return torch.stack(losses).mean()


class LatentRotateOperationModule(nn.Module):
    """Rotation apprise dans un espace latent, pour un angle fixe : MLP encodeur
    -> couche intermédiaire de taille `latent_dim` -> décodeur. L'espace est
    choisi par `space` :
     - "gw"  : état du global workspace (tanh de l'encodage GW), dans [-1, 1] ;
     - "vae" : latent du VAE (mu), non borné, pour comparer avec une rotation
       faite hors du GW.
    Entraînement identique dans les deux cas, seul l'encodage des images change."""

    def __init__(
        self,
        space: str,
        size: int,
        hidden_size: int = 128,
        latent_dim: int = 64,
        degrees: float = 20.0,
        n_steps: int = 1,
    ):
        super().__init__()
        assert space in ("gw", "vae"), f"espace inconnu : {space}"
        self.space = space
        self.degrees = degrees
        self.n_steps = n_steps

        self.encoder = nn.Sequential(
            nn.Linear(size, hidden_size),
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
            nn.Linear(hidden_size, size),
        )

    def encode(self, gw_mod: GWModule, image: torch.Tensor) -> torch.Tensor:
        """Image -> point de l'espace où se fait la rotation."""
        if self.space == "gw":
            return image_to_gw_state(gw_mod, image)
        return gw_mod.domain_mods["image"].encode(image)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.decoder(self.encoder(x))
        return torch.tanh(out) if self.space == "gw" else out

    def loss(
        self,
        gw_mod: GWModule,
        raw_data: RawDomainGroupsT,
    ) -> torch.Tensor:
        # Même principe que PixelRotateOperationModule.loss : n_steps rotations
        # enchaînées dans l'espace latent, chaque étape comparée à l'encodage
        # de l'image tournée de k x degrees.
        image = raw_data[frozenset({"digit", "image"})]['image']

        with torch.no_grad():
            state = self.encode(gw_mod, image)
            targets = [
                self.encode(gw_mod, rotated)
                for rotated in chained_rotation_targets(image, self.degrees, self.n_steps)
            ]

        pred = state
        losses = []
        for target in targets:
            pred = self(pred)
            losses.append(F.mse_loss(pred, target))

        return torch.stack(losses).mean()
