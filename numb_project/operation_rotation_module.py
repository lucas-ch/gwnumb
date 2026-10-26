from shimmer import GWModule, GWModuleBase, RawDomainGroupsT
from torch import nn
import torch
import torch.nn.functional as F

from numb_project.utils import rotate_digit_angle, rotate_images

def image_to_gw_state(gw_mod: GWModule, image: torch.Tensor) -> torch.Tensor:
    image_latent = gw_mod.domain_mods["image"].encode(image)
    return gw_mod.fusion_activation_fn(gw_mod.gw_encoders["image"](image_latent))


def digit_to_gw_state(gw_mod: GWModule, digit: torch.Tensor) -> torch.Tensor:
    return gw_mod.fusion_activation_fn(gw_mod.gw_encoders["digit"](digit))


def chained_rotation_targets(image: torch.Tensor, degrees: float, n_steps: int) -> list[torch.Tensor]:
    """Cibles d'une rotation de `degrees` enchaînée `n_steps` fois : l'image
    tournée de k x `degrees` pour k = 1..n_steps, chaque cible étant calculée en
    une seule rotation pour ne pas accumuler l'erreur d'interpolation."""
    return [
        rotate_images(image, torch.full((image.shape[0],), degrees * step, device=image.device))
        for step in range(1, n_steps + 1)
    ]


def chained_digit_targets(digit: torch.Tensor, degrees: float, n_steps: int) -> list[torch.Tensor]:
    """Équivalent de `chained_rotation_targets` dans l'espace digit : rotation
    exacte (chiffre inchangé, angle + k x `degrees`) pour k = 1..n_steps."""
    return [rotate_digit_angle(digit, degrees * step) for step in range(1, n_steps + 1)]


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
    choisi par `space` (du plus riche au plus symbolique) :
     - "vae"      : latent du VAE (mu), non borné, rotation faite hors du GW ;
     - "gw"       : état du global workspace (tanh de l'encodage GW), dans [-1, 1] ;
     - "gw_digit" : encodage GW du vrai digit, qui ne contient que chiffre et angle ;
     - "gw_fused" : fusion des deux précédents, 0.5 (gw + gw_digit) ;
     - "digit"    : vrai vecteur digit (one-hot du chiffre + (cos, sin) de l'angle).
    Entraînement identique dans tous les cas, seul l'encodage change (cf
    `encode`). Les cibles sont l'encodage de la vérité tournée : image tournée
    côté image, digit tourné exactement (angle + k x degrees) côté digit, pour
    que la cible digit ne dépende pas de la perception du GW."""

    SPACES = ("vae", "gw", "gw_digit", "gw_fused", "digit")
    BOUNDED_SPACES = ("gw", "gw_digit", "gw_fused")  # états GW (ou leur moyenne), bornés par le tanh

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
        assert space in self.SPACES, f"espace inconnu : {space}"
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

    def encode(self, gw_mod: GWModule, image: torch.Tensor, digit: torch.Tensor) -> torch.Tensor:
        """(image, vrai digit) -> point de l'espace où se fait la rotation."""
        match self.space:
            case "vae":
                return gw_mod.domain_mods["image"].encode(image)
            case "gw":
                return image_to_gw_state(gw_mod, image)
            case "gw_digit":
                return digit_to_gw_state(gw_mod, digit)
            case "gw_fused":
                return 0.5 * (image_to_gw_state(gw_mod, image) + digit_to_gw_state(gw_mod, digit))
            case "digit":
                return digit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.decoder(self.encoder(x))
        return torch.tanh(out) if self.space in self.BOUNDED_SPACES else out

    def loss(
        self,
        gw_mod: GWModule,
        raw_data: RawDomainGroupsT,
    ) -> torch.Tensor:
        # Même principe que PixelRotateOperationModule.loss : n_steps rotations
        # enchaînées dans l'espace latent, chaque étape comparée à l'encodage
        # de l'image et du digit tournés de k x degrees.
        group = raw_data[frozenset({"digit", "image"})]
        image, digit = group['image'], group['digit']

        with torch.no_grad():
            state = self.encode(gw_mod, image, digit)
            targets = [
                self.encode(gw_mod, rotated_image, rotated_digit)
                for rotated_image, rotated_digit in zip(
                    chained_rotation_targets(image, self.degrees, self.n_steps),
                    chained_digit_targets(digit, self.degrees, self.n_steps),
                )
            ]

        pred = state
        losses = []
        for target in targets:
            pred = self(pred)
            losses.append(F.mse_loss(pred, target))

        return torch.stack(losses).mean()
