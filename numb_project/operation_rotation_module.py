from shimmer import GWModule, GWModuleBase, RawDomainGroupsT
from torch import nn
from torch.func import functional_call
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
     - "gw"       : état du global workspace (activation de fusion de l'encodage GW), dans ]-1, 1[ ;
     - "gw_digit" : encodage GW du vrai digit, qui ne contient que chiffre et angle ;
     - "gw_fused" : fusion des deux précédents, 0.5 (gw + gw_digit) ;
     - "digit"    : vrai vecteur digit (one-hot du chiffre + (cos, sin) de l'angle).
    Entraînement identique dans tous les cas, seul l'encodage change (cf
    `encode`). Les cibles sont l'encodage de la vérité tournée : image tournée
    côté image, digit tourné exactement (angle + k x degrees) côté digit, pour
    que la cible digit ne dépende pas de la perception du GW."""

    SPACES = ("vae", "gw", "gw_digit", "gw_fused", "digit")
    BOUNDED_SPACES = ("gw", "gw_digit", "gw_fused")  # états GW (ou leur moyenne), bornés par l'activation de fusion

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


def frozen_call(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """`module(x)` avec ses paramètres détachés : le gradient traverse le module
    (vers ce qui produit `x`) sans mettre à jour ses poids."""
    params = {name: param.detach() for name, param in module.named_parameters()}
    return functional_call(module, params, (x,))


class FusedRotateOperationModule(nn.Module):
    """Rotation faite séparément dans le latent du VAE (module "vae") et dans
    l'espace digit (module "digit"), puis fusionnée dans le GW :
        gw = tanh(image_weight · E_image(R_vae(v)) + (1 - image_weight) · E_digit(R_digit(d)))
    puis décodée dans les deux domaines. Les deux modules de rotation sont
    partagés avec les variantes "vae" et "digit" et ne sont entraînés que par
    leur propre loss : ils sont gelés pour cette loss (cf `frozen_call`), les
    rotations ne dépendent donc que de leur espace. C'est le GW (encodeurs et
    décodeurs) que cette loss entraîne, à combiner les deux rotations.

    `image_weight` est fixe, ou appris par cette loss si `learn_image_weight`
    (paramétré par sigmoid(logit) pour rester dans ]0, 1[, initialisé à
    `image_weight`).

    Rotations enchaînées : chaque étape repart des décodages du GW fusionné
    (latent VAE, digit décodé), le GW servant de point de synchronisation
    entre les deux domaines. Rien n'est nettoyé (pas de `clean_digit`) : tout
    est différentiable et le gradient traverse toute la chaîne."""

    def __init__(
        self,
        vae_rotation: "LatentRotateOperationModule",
        digit_rotation: "LatentRotateOperationModule",
        image_weight: float = 0.5,
        learn_image_weight: bool = False,
        degrees: float = 20.0,
        n_steps: int = 1,
    ):
        super().__init__()
        assert vae_rotation.space == "vae" and digit_rotation.space == "digit"
        assert 0 < image_weight < 1, "image_weight doit être dans ]0, 1["
        self.vae_rotation = vae_rotation
        self.digit_rotation = digit_rotation
        logit = torch.logit(torch.tensor(image_weight))
        if learn_image_weight:
            self.image_weight_logit = nn.Parameter(logit)
        else:
            # non persistant : un poids fixe n'apparaît pas dans le checkpoint
            self.register_buffer("image_weight_logit", logit, persistent=False)
        self.degrees = degrees
        self.n_steps = n_steps

    @property
    def image_weight(self) -> torch.Tensor:
        """Poids de la branche vae dans la fusion (celle de digit vaut 1 - image_weight)."""
        return torch.sigmoid(self.image_weight_logit)

    def fuse_and_decode(
        self, gw_mod: GWModule, vae_latent: torch.Tensor, digit: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(latent VAE, digit) -> fusion dans le GW (poids image_weight /
        1 - image_weight) -> (latent VAE, digit) décodés."""
        encodings = {"image": gw_mod.gw_encoders["image"](vae_latent), "digit": gw_mod.gw_encoders["digit"](digit)}
        weights = self.image_weight.expand(digit.shape[0])
        gw_state = gw_mod.fuse(encodings, {"image": weights, "digit": 1 - weights})
        return gw_mod.gw_decoders["image"](gw_state), gw_mod.gw_decoders["digit"](gw_state)

    def step(self, gw_mod: GWModule, vae_latent: torch.Tensor, digit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Une rotation : (latent VAE, digit) -> (latent VAE, digit) décodés du
        GW fusionné. Modules de rotation gelés, GW entraîné."""
        return self.fuse_and_decode(gw_mod, frozen_call(self.vae_rotation, vae_latent),
                                    frozen_call(self.digit_rotation, digit))

    def loss(
        self,
        gw_mod: GWModule,
        raw_data: RawDomainGroupsT,
    ) -> torch.Tensor:
        # À chaque étape : loss du domaine image (MSE normalisée) sur le latent
        # VAE décodé + loss du domaine digit sur le digit décodé, vs la vérité
        # tournée de k x degrees (VAE de l'image tournée, vrai digit tourné).
        group = raw_data[frozenset({"digit", "image"})]
        image, digit = group['image'], group['digit']
        image_mod, digit_mod = gw_mod.domain_mods["image"], gw_mod.domain_mods["digit"]

        with torch.no_grad():
            vae_latent = image_mod.encode(image)
            vae_targets = [image_mod.encode(rotated)
                           for rotated in chained_rotation_targets(image, self.degrees, self.n_steps)]
        digit_targets = chained_digit_targets(digit, self.degrees, self.n_steps)

        losses = []
        for vae_target, digit_target in zip(vae_targets, digit_targets):
            vae_latent, digit = self.step(gw_mod, vae_latent, digit)
            losses.append(image_mod.compute_loss(vae_latent, vae_target, None).loss
                          + digit_mod.compute_loss(digit, digit_target, None).loss)

        return torch.stack(losses).mean()
