"""Fonctions communes aux notebooks d'évaluation (test*.ipynb) : chargement des
modèles, données de test, passages entre les espaces, lecture du domaine digit,
affichage et analyse des rotations enchaînées."""
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from cfg_tools import load_config_files
from torchvision import datasets, transforms

from numb_project.constants import BASE, CONFIG_FILE, CONFIG_FOLDER, DEVICE
from numb_project.domain_module import get_domains_config
from numb_project.gw_model import MyGlobalWorkspace, get_global_workspace_mods
from numb_project.mnist_vae import VAE
from numb_project.utils import angle_to_cos_sin, cos_sin_to_angle, rotate_images


# --- Chargement des modèles ---

def load_global_workspace(checkpoint_path: str) -> MyGlobalWorkspace:
    """GW construit avec la config actuelle, poids du checkpoint, en mode eval."""
    config = load_config_files(CONFIG_FOLDER, use_cli=False, load_files=[CONFIG_FILE])[0]
    domain_configs = get_domains_config(["image", "digit"], config["domains"]["digit"])
    gw_mod, selection_mod, loss_mod, rotate_pixel_mod, rotate_gw_mod = get_global_workspace_mods(config, domain_configs)
    gw = MyGlobalWorkspace.load_from_checkpoint(
        checkpoint_path,
        gw_mod=gw_mod,
        selection_mod=selection_mod,
        loss_mod=loss_mod,
        operation_rotate_pixel_mod=rotate_pixel_mod,
        operation_rotate_gw_mod=rotate_gw_mod,
        map_location=DEVICE,
        weights_only=False,
    )
    return gw.eval()


def load_vae(checkpoint_path: str) -> VAE:
    vae = VAE.load_from_checkpoint(checkpoint_path, map_location=DEVICE)
    vae.freeze()
    return vae


def vae_reconstruct(vae: VAE, images: torch.Tensor) -> torch.Tensor:
    mu, _ = vae.encoder(images)
    return vae.decoder(mu)


# --- Données de test ---

def mnist_batch(n_images: int, train: bool = False, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """`n_images` images MNIST (test set par défaut) et leurs chiffres, tirées au
    hasard mais toujours les mêmes pour un `seed` donné."""
    dataset = datasets.MNIST(root="./data", train=train, download=True, transform=transforms.ToTensor())
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:n_images]
    images = torch.stack([dataset[i][0] for i in indices]).to(DEVICE)
    labels = torch.tensor([dataset[i][1] for i in indices], device=DEVICE)
    return images, labels


def random_angles(n_images: int, seed: int = 0) -> torch.Tensor:
    return torch.rand(n_images, device=DEVICE, generator=torch.Generator(device=DEVICE).manual_seed(seed)) * 360


def rotate_by(images: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Tourne toutes les images du même angle."""
    return rotate_images(images, torch.full((images.shape[0],), angle_deg, device=images.device))


# --- Passages entre les espaces (latents des domaines <-> état GW) ---

def to_gw(gw: MyGlobalWorkspace, domain: str, latent: torch.Tensor) -> torch.Tensor:
    # État GW réel : tanh de l'encodage GW (cf GWModule.fuse)
    return gw.gw_mod.fusion_activation_fn(gw.gw_mod.gw_encoders[domain](latent))


def from_gw(gw: MyGlobalWorkspace, domain: str, gw_state: torch.Tensor) -> torch.Tensor:
    return gw.gw_mod.gw_decoders[domain](gw_state)


def image_to_gw(gw: MyGlobalWorkspace, images: torch.Tensor) -> torch.Tensor:
    return to_gw(gw, "image", gw.gw_mod.domain_mods["image"].encode(images))


def gw_to_image(gw: MyGlobalWorkspace, gw_state: torch.Tensor) -> torch.Tensor:
    return gw.gw_mod.domain_mods["image"].decode(from_gw(gw, "image", gw_state))


def demi_cycle(gw: MyGlobalWorkspace, domain: str, latent: torch.Tensor) -> torch.Tensor:
    return from_gw(gw, domain, to_gw(gw, domain, latent))


def cycle(gw: MyGlobalWorkspace, domain: str, through: str, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Renvoie (latent intermédiaire dans `through`, latent revenu dans `domain`)."""
    intermediate = from_gw(gw, through, to_gw(gw, domain, latent))
    return intermediate, from_gw(gw, domain, to_gw(gw, through, intermediate))


# --- Domaine digit : one-hot du chiffre + (cos, sin) de l'angle ---

def digit_vector(labels: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
    return torch.cat([F.one_hot(labels, BASE).float(), angle_to_cos_sin(angles_deg)], dim=-1)


def read_digit(digits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Chiffre (argmax du one-hot) et angle en degrés lus dans un vecteur digit."""
    return digits[:, :BASE].argmax(dim=-1), cos_sin_to_angle(digits[:, BASE:])


def angle_error(pred_deg: torch.Tensor, target_deg: torch.Tensor) -> torch.Tensor:
    """Écart angulaire en degrés, dans [0, 180]."""
    return ((pred_deg - target_deg + 180) % 360 - 180).abs()


def recognition(digits: torch.Tensor, labels: torch.Tensor, angles_deg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Par image : chiffre reconnu (1.0 ou 0.0) et erreur d'angle (°)."""
    pred_labels, pred_angles = read_digit(digits)
    return (pred_labels == labels).float(), angle_error(pred_angles, angles_deg)


def format_recognition(digits: torch.Tensor, labels: torch.Tensor, angles_deg: torch.Tensor) -> str:
    correct, err = recognition(digits, labels, angles_deg)
    return (f"précision chiffre = {correct.mean():6.1%}, "
            f"erreur d'angle moyenne = {err.mean():5.1f}° (médiane {err.median():5.1f}°)")


# --- Affichage ---

def show_rows(rows: list[tuple], n_show: int = 8) -> None:
    """Une ligne d'images par élément de `rows` : (label, images) ou
    (label, images, titres des images)."""
    fig, axes = plt.subplots(len(rows), n_show, figsize=(n_show * 1.2, 1.4 * len(rows)), squeeze=False)
    for r, (label, images, *titles) in enumerate(rows):
        for i in range(n_show):
            axes[r, i].imshow(images[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[r, i].set_xticks([])
            axes[r, i].set_yticks([])
            if titles:
                axes[r, i].set_title(titles[0][i], fontsize=7)
        axes[r, 0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
    plt.tight_layout()
    plt.show()


def show_image(image: torch.Tensor, title: str) -> None:
    plt.figure(figsize=(2.5, 2.5))
    plt.imshow(image[0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")
    plt.show()


# --- Rotations enchaînées : voie pixel vs voie GW ---

ROTATION_PATHS = ("pixel", "GW", "vraie image tournée")


def chained_rotations(
    gw: MyGlobalWorkspace, images: torch.Tensor, labels: torch.Tensor, start_angles: torch.Tensor, max_n: int = 12
) -> dict[str, dict[str, torch.Tensor]]:
    """Applique 1 à `max_n` fois les rotations apprises et mesure, pour chaque voie,
    chaque n et chaque image : "mse" (image produite vs vraie rotation),
    "correct" (chiffre reconnu) et "angle_error" (°), tenseurs (max_n, images).
     - pixel : module de rotation pixel, image produite lue via image -> GW -> digit ;
     - GW : module de rotation GW, état GW lu par le décodeur digit, image par GW -> image ;
     - vraie image tournée : référence, vraie rotation passée par image -> GW (-> image).
       Sa MSE est donc le plancher de la voie GW (perte du demi-cycle)."""
    ope_pixel, ope_gw = gw.operation_rotate_pixel_module, gw.operation_rotate_gw_module
    results = {path: {"mse": [], "correct": [], "angle_error": []} for path in ROTATION_PATHS}
    pixel_images, gw_state = images, image_to_gw(gw, images)
    for n in range(1, max_n + 1):
        pixel_images, gw_state = ope_pixel(pixel_images), ope_gw(gw_state)
        target = rotate_by(images, ope_gw.degrees * n)
        target_state = image_to_gw(gw, target)
        # (image produite, état GW correspondant) pour chaque voie
        outputs = {
            "pixel": (pixel_images, image_to_gw(gw, pixel_images)),
            "GW": (gw_to_image(gw, gw_state), gw_state),
            "vraie image tournée": (gw_to_image(gw, target_state), target_state),
        }
        for path, (produced, state) in outputs.items():
            correct, err = recognition(from_gw(gw, "digit", state), labels, start_angles + ope_gw.degrees * n)
            results[path]["mse"].append(((produced - target) ** 2).mean(dim=(1, 2, 3)))
            results[path]["correct"].append(correct)
            results[path]["angle_error"].append(err)
    return {path: {name: torch.stack(values).cpu() for name, values in r.items()} for path, r in results.items()}


def plot_chained_rotations(results: dict[str, dict[str, torch.Tensor]], title: str) -> None:
    """Moyennes sur les images, en fonction du nombre de rotations enchaînées."""
    max_n = results["GW"]["mse"].shape[0]
    ns = range(1, max_n + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, (metric, ylabel, chance) in zip(axes, [
        ("mse", "MSE en pixels (vs vraie rotation)", None),
        ("correct", "précision sur le chiffre", 0.1),
        ("angle_error", "erreur d'angle moyenne (°)", 90),
    ]):
        for path in ROTATION_PATHS:
            ax.plot(ns, results[path][metric].mean(dim=1), "--" if path == "vraie image tournée" else "-o", label=path)
        if chance is not None:
            ax.axhline(chance, color="gray", linestyle=":", label="hasard")
        ax.set_xlabel("nombre de rotations enchaînées")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

    for n in range(max_n):
        print(f"n={n + 1:>2}: " + "  |  ".join(
            f"{path} : MSE {results[path]['mse'][n].mean():.4f}, chiffre {results[path]['correct'][n].mean():5.1%}, "
            f"angle {results[path]['angle_error'][n].mean():5.1f}°"
            for path in ROTATION_PATHS
        ))


def plot_per_digit(results: dict[str, dict[str, torch.Tensor]], labels: torch.Tensor, title: str) -> None:
    """Précision et erreur d'angle par chiffre (lignes) et par nombre de rotations (colonnes)."""
    labels = labels.cpu()
    max_n = results["GW"]["correct"].shape[0]
    fig, axes = plt.subplots(2, len(ROTATION_PATHS), figsize=(6 * len(ROTATION_PATHS), 9))
    for row, (metric, label, cmap, vmax, fmt) in enumerate([
        ("correct", "précision sur le chiffre", "RdYlGn", 1, "{:.0%}"),
        ("angle_error", "erreur d'angle moyenne (°)", "RdYlGn_r", 90, "{:.0f}"),
    ]):
        for col, path in enumerate(ROTATION_PATHS):
            per_digit = torch.stack([results[path][metric][:, labels == d].mean(dim=1) for d in range(BASE)])
            ax = axes[row, col]
            im = ax.imshow(per_digit.numpy(), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
            for d in range(BASE):
                for n in range(max_n):
                    ax.text(n, d, fmt.format(per_digit[d, n].item()), ha="center", va="center", fontsize=6)
            ax.set_xticks(range(max_n), [str(n + 1) for n in range(max_n)])
            ax.set_yticks(range(BASE))
            ax.set_xlabel("nombre de rotations enchaînées")
            ax.set_ylabel("chiffre")
            ax.set_title(f"{path} : {label}", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def show_chained_rotations(gw: MyGlobalWorkspace, images: torch.Tensor,
                           n_values: tuple[int, ...] = (1, 4, 8, 12), n_show: int = 6) -> None:
    """Images après n rotations : vraie rotation, voie pixel, voie GW."""
    ope_pixel, ope_gw = gw.operation_rotate_pixel_module, gw.operation_rotate_gw_module
    images = images[:n_show]
    rows = [("départ", images)]
    pixel_images, gw_state = images, image_to_gw(gw, images)
    for n in range(1, max(n_values) + 1):
        pixel_images, gw_state = ope_pixel(pixel_images), ope_gw(gw_state)
        if n in n_values:
            rows += [
                (f"réel n={n}", rotate_by(images, ope_gw.degrees * n)),
                (f"pixel n={n}", pixel_images),
                (f"GW n={n}", gw_to_image(gw, gw_state)),
            ]
    show_rows(rows, n_show=n_show)
