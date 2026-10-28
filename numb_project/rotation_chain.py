"""Contrôleur qui enchaîne des rotations mentales de 20° pour redresser une image.

Le GW (checkpoint `GW_CHECKPOINT`) est gelé ; seul le contrôleur est entraîné.
L'état mental est celui de la voie « VAE + digit séparés » : (latent VAE,
digit perçu), chacun tourné par sa rotation apprise (module vae_digit). Le
contrôleur l'observe à travers le GW (fusion des deux) et choisit à chaque pas
entre deux actions :
 - stop : termine la séquence (définitivement, cf `alive`) ;
 - rotation de 20° des deux composantes de l'état.
Loss : écart (MSE) entre l'état du GW à la fin de la séquence et la projection
dans le GW de l'image redressée seule (encodeur image, sans digit ni fusion).

Deux façons d'entraîner les choix discrets (`estimator`) :
 - "gumbel" : Gumbel-softmax « hard » (straight-through), le gradient de la
   loss traverse les rotations jusqu'aux choix ;
 - "reinforce" : actions tirées de la politique, la loss sert de récompense
   (gradient de politique avec la moyenne du batch comme baseline)."""
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from numb_project.constants import BASE, CHAIN_CHECKPOINT_FOLDER, GW_CHECKPOINT_FOLDER
from numb_project.gw_model import MyGlobalWorkspace
from numb_project.operation_rotation_module import image_to_gw_state
from numb_project.utils import clean_digit, cos_sin_to_angle, rotate_images

GW_CHECKPOINT = f"{GW_CHECKPOINT_FOLDER}/symmetry.ckpt"
STOP, ROTATE = 0, 1  # indices des actions


class RotationController(nn.Module):
    """LSTM : état du GW au pas t -> logits des actions (stop, rotation)."""

    def __init__(self, gw_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(gw_dim, hidden_size)
        self.output = nn.Linear(hidden_size, 2)

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(batch_size, self.hidden_size, device=device)
        return zeros, zeros.clone()

    def forward(
        self, gw_state: torch.Tensor, hc: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h, c = self.cell(gw_state, hc)
        return self.output(h), (h, c)


class UprightChainModule(LightningModule):
    """Apprend à redresser une image tournée de m x 20° (m dans 0..17) en
    enchaînant au plus `max_steps` rotations mentales de 20°.

    Actions discrètes : tirées selon `estimator` à l'entraînement, argmax à
    l'évaluation. `alive` vaut 1 tant que stop n'a pas été choisi, 0 ensuite :
    une rotation n'est appliquée que si alive = 1."""

    def __init__(self, gw_checkpoint: str = GW_CHECKPOINT, max_steps: int = 18, hidden_size: int = 64,
                 lr: float = 1e-3, estimator: str = "reinforce", tau: float = 1.0, entropy_weight: float = 0.01):
        super().__init__()
        self.save_hyperparameters()
        self.gw = MyGlobalWorkspace.load_from_checkpoint(gw_checkpoint)
        self.gw.freeze()
        self.fused = self.gw.rotation_mods["vae_digit"]
        self.controller = RotationController(self.gw.gw_mod.workspace_dim, hidden_size)

    def perceive(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Image -> état mental de départ (latent VAE, digit propre perçu par le GW)."""
        vae_latent = self.gw.gw_mod.domain_mods["image"].encode(images)
        digit = clean_digit(self.gw.gw_mod.gw_decoders["digit"](image_to_gw_state(self.gw.gw_mod, images)))
        return vae_latent, digit

    def gw_state(self, vae_latent: torch.Tensor, digit: torch.Tensor) -> torch.Tensor:
        return self.fused.fuse(self.gw.gw_mod, vae_latent, digit)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Enchaîne les actions depuis `images`. Renvoie l'état final du GW,
        les actions (B, max_steps, 2), le nombre de rotations effectuées,
        pour REINFORCE la log-probabilité et l'entropie des choix faits avant
        stop, et la trajectoire : états du GW (B, max_steps + 1, D, avant chaque
        choix puis à la fin) et probabilité de stop à chaque pas (B, max_steps)."""
        vae_latent, digit = self.perceive(images)
        hc = self.controller.initial_state(images.shape[0], images.device)
        alive = torch.ones(images.shape[0], 1, device=images.device)
        actions, n_rotations = [], torch.zeros(images.shape[0], device=images.device)
        log_prob = entropy = torch.zeros(images.shape[0], device=images.device)
        states, stop_probs = [], []
        for _ in range(self.hparams.max_steps):
            states.append(self.gw_state(vae_latent, digit))
            logits, hc = self.controller(states[-1], hc)
            stop_probs.append(logits.softmax(-1)[:, STOP])
            if not self.training:
                action = F.one_hot(logits.argmax(-1), 2).float()
            elif self.hparams.estimator == "gumbel":
                action = F.gumbel_softmax(logits, tau=self.hparams.tau, hard=True)
            else:
                policy = torch.distributions.Categorical(logits=logits)
                sample = policy.sample()
                action = F.one_hot(sample, 2).float()
                log_prob = log_prob + alive.squeeze(1) * policy.log_prob(sample)
                entropy = entropy + alive.squeeze(1) * policy.entropy()
            rotate = alive * action[:, ROTATE:ROTATE + 1]
            vae_latent = vae_latent + rotate * (self.fused.vae_rotation(vae_latent) - vae_latent)
            digit = digit + rotate * (self.fused.digit_rotation(digit) - digit)
            alive = alive * (1 - action[:, STOP:STOP + 1])
            actions.append(action)
            n_rotations = n_rotations + rotate.squeeze(1)
        states.append(self.gw_state(vae_latent, digit))
        return {"gw_state": states[-1], "actions": torch.stack(actions, 1),
                "n_rotations": n_rotations, "log_prob": log_prob, "entropy": entropy,
                "states": torch.stack(states, 1), "stop_probs": torch.stack(stop_probs, 1)}

    def generic_step(self, images: torch.Tensor, mode: str) -> torch.Tensor:
        degrees = self.gw.config["global_workspace"]["rotation_operations"]["degrees"]
        n_angles = round(360 / degrees)
        m = torch.randint(0, n_angles, (images.shape[0],), device=images.device)
        rotated = rotate_images(images, m * degrees)
        with torch.no_grad():
            target = image_to_gw_state(self.gw.gw_mod, images)
        out = self(rotated)
        per_image = (out["gw_state"] - target).pow(2).mean(1)
        loss = per_image.mean()
        if self.training and self.hparams.estimator == "reinforce":
            advantage = (per_image - per_image.mean()).detach()
            train_loss = (advantage * out["log_prob"]).mean() - self.hparams.entropy_weight * out["entropy"].mean()
        else:
            train_loss = loss

        # Suivi : rotations attendues (rotate_images(·, +degrees) à chaque pas)
        expected = (-m) % n_angles
        step_error = (out["n_rotations"].round().long() - expected) % n_angles
        step_error = torch.minimum(step_error, n_angles - step_error).float()
        final_angle = cos_sin_to_angle(self.gw.gw_mod.gw_decoders["digit"](out["gw_state"])[:, BASE:])
        angle_error = torch.minimum(final_angle, 360 - final_angle)
        self.log_dict({f"{mode}/loss": loss, f"{mode}/exact": (step_error == 0).float().mean(),
                       f"{mode}/step_error": step_error.mean(), f"{mode}/angle_error": angle_error.mean(),
                       f"{mode}/n_rotations": out["n_rotations"].mean()}, prog_bar=mode == "val")
        return train_loss

    def training_step(self, batch, batch_idx):
        return self.generic_step(batch[0], "train")

    def validation_step(self, batch, batch_idx):
        return self.generic_step(batch[0], "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.controller.parameters(), lr=self.hparams.lr)


def main():
    transform = transforms.ToTensor()
    train_loader = DataLoader(datasets.MNIST("./data", train=True, download=True, transform=transform),
                              batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(datasets.MNIST("./data", train=False, download=True, transform=transform),
                            batch_size=256, num_workers=4)

    model = UprightChainModule()
    trainer = Trainer(
        max_epochs=20,
        callbacks=[ModelCheckpoint(dirpath=CHAIN_CHECKPOINT_FOLDER, filename="upright", monitor="val/loss", mode="min")],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
