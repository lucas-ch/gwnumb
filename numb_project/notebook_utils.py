"""Fonctions communes aux notebooks d'évaluation (test*.ipynb) : chargement des
modèles, données de test, passages entre les espaces, lecture du domaine digit,
affichage et analyse des rotations enchaînées."""
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from numb_project.constants import BASE, DEVICE
from numb_project.gw_model import MyGlobalWorkspace
from numb_project.mnist_vae import VAE
from numb_project.utils import angle_to_cos_sin, clean_digit, cos_sin_to_angle, rotate_digit_angle, rotate_images


# --- Chargement des modèles ---

def load_global_workspace(checkpoint_path: str) -> MyGlobalWorkspace:
    """GW reconstruit avec la config enregistrée dans le checkpoint (`gw.config`),
    poids du checkpoint, en mode eval."""
    gw = MyGlobalWorkspace.load_from_checkpoint(checkpoint_path, map_location=DEVICE, weights_only=False)
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


# --- Rotations enchaînées : comparaison des espaces de rotation ---

# Voies de rotation, de la plus riche à la plus symbolique, et le module de
# rotation (gw.rotation_mods) de chacune ; seules les voies dont le module a
# été entraîné (cf `rotation_paths`) sont évaluées
ROTATION_MODULES = {"pixel": "pixel", "VAE": "vae", "GW": "gw", "GW via digit": "gw_digit", "GW fusionné": "gw_fused",
                    "digit": "digit", "VAE + digit": "vae_digit", "VAE + digit séparés": "vae_digit"}
ROTATION_PATHS = tuple(ROTATION_MODULES)
EXACT_DIGIT = "rotation exacte de l'angle lu (digit)"
REFERENCE = "référence : vraie rotation lue par le GW"
VAE_FLOOR = "plancher VAE (vraie rotation → VAE → image)"
GW_FLOOR = "plancher GW (vraie rotation → GW → image)"
SYMBOLIC_FLOOR = "plancher symbolique (vraie rotation → GW → digit propre → GW → image)"
# chaque plancher a la couleur de la voie qu'il borne ; le plancher symbolique
# borne à la fois « GW via digit » et « digit » ; la rotation exacte a la
# couleur de la voie digit, dont elle est la version idéale
CURVE_COLORS = {"pixel": "C0", "VAE": "C1", "GW": "C2", "GW via digit": "C4", "GW fusionné": "C5", "digit": "C3",
                "VAE + digit": "C6", "VAE + digit séparés": "C7",
                EXACT_DIGIT: "C3", REFERENCE: "black", VAE_FLOOR: "C1", GW_FLOOR: "C2", SYMBOLIC_FLOOR: "C4"}


def rotation_paths(gw: MyGlobalWorkspace) -> tuple[str, ...]:
    """Voies de rotation dont le module existe dans `gw` (rotations activées dans sa config)."""
    return tuple(path for path in ROTATION_PATHS if ROTATION_MODULES[path] in gw.rotation_mods)


def rotation_degrees(gw: MyGlobalWorkspace) -> float:
    return gw.config["global_workspace"]["rotation_operations"]["degrees"]


def format_fusion_weights(gw: MyGlobalWorkspace) -> str:
    """Poids de la fusion des voies « VAE + digit » (ceux du module vae_digit,
    utilisés par `rotate_state` et `path_outputs`), appris ou fixes."""
    if "vae_digit" not in gw.rotation_mods:
        return "pas de fusion VAE + digit dans ce modèle"
    fused = gw.rotation_mods["vae_digit"]
    image_weight = fused.image_weight.item()
    learned = isinstance(fused.image_weight_logit, torch.nn.Parameter)
    return (f"fusion VAE + digit : poids vae = {image_weight:.3f}, poids digit = {1 - image_weight:.3f} "
            f"({'appris' if learned else 'fixe'})")


# état d'une voie : un tenseur, ou (latent VAE, digit) pour « VAE + digit » et
# « VAE + digit séparés »
PathState = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


def start_states(gw: MyGlobalWorkspace, images: torch.Tensor) -> dict[str, PathState]:
    """Point de départ de chaque voie de `gw` (et de la rotation exacte) dans son espace.
    Les voies passant par digit partent du vecteur digit propre *perçu* (image ->
    GW -> digit), et non du vrai digit utilisé à l'entraînement : on évalue la
    perception et l'opération ensemble."""
    gw_state = image_to_gw(gw, images)
    digits = clean_digit(from_gw(gw, "digit", gw_state))
    gw_digit_state = to_gw(gw, "digit", digits)
    vae_latent = gw.gw_mod.domain_mods["image"].encode(images)
    states = {
        "pixel": images,
        "VAE": vae_latent,
        "GW": gw_state,
        "GW via digit": gw_digit_state,
        "GW fusionné": 0.5 * (gw_state + gw_digit_state),
        "digit": digits,
        "VAE + digit": (vae_latent, digits),
        "VAE + digit séparés": (vae_latent, digits),
    }
    return {path: states[path] for path in rotation_paths(gw)} | {EXACT_DIGIT: digits}


def rotate_state(gw: MyGlobalWorkspace, path: str, state: PathState) -> PathState:
    """Une rotation de plus sur une voie. « VAE + digit » repart des décodages du
    GW fusionné, sans nettoyer le digit, comme à l'entraînement. « VAE + digit
    séparés » enchaîne les rotations VAE et digit chacune dans son espace, la
    fusion ne servant qu'à lire le résultat (cf `path_outputs`)."""
    if path == EXACT_DIGIT:
        return rotate_digit_angle(state, rotation_degrees(gw))
    if path == "VAE + digit":
        return gw.rotation_mods["vae_digit"].step(gw.gw_mod, *state)
    if path == "VAE + digit séparés":
        vae_latent, digits = state
        fused = gw.rotation_mods["vae_digit"]
        return fused.vae_rotation(vae_latent), fused.digit_rotation(digits)
    return gw.rotation_mods[ROTATION_MODULES[path]](state)


def rotate_states(gw: MyGlobalWorkspace, states: dict[str, PathState]) -> dict[str, PathState]:
    """Une rotation de plus sur chaque voie."""
    return {path: rotate_state(gw, path, state) for path, state in states.items()}


def path_outputs(gw: MyGlobalWorkspace, path: str, state: PathState) -> tuple[torch.Tensor, torch.Tensor]:
    """(image produite, vecteur digit lu par le décodeur digit du GW) pour l'état d'une voie."""
    if path == "VAE + digit":  # déjà décodé du GW dans les deux domaines
        vae_latent, digits = state
        return gw.gw_mod.domain_mods["image"].decode(vae_latent), digits
    if path == "VAE + digit séparés":
        vae_latent, digits = gw.rotation_mods["vae_digit"].fuse_and_decode(gw.gw_mod, *state)
        return gw.gw_mod.domain_mods["image"].decode(vae_latent), digits
    if path == "pixel":
        return state, from_gw(gw, "digit", image_to_gw(gw, state))
    if path == "VAE":
        return gw.gw_mod.domain_mods["image"].decode(state), from_gw(gw, "digit", to_gw(gw, "image", state))
    if path in ("digit", EXACT_DIGIT):
        return gw_to_image(gw, to_gw(gw, "digit", state)), state
    return gw_to_image(gw, state), from_gw(gw, "digit", state)  # GW, GW via digit, GW fusionné


def chained_rotations(
    gw: MyGlobalWorkspace, images: torch.Tensor, labels: torch.Tensor, start_angles: torch.Tensor, max_n: int = 12
) -> dict[str, dict[str, torch.Tensor]]:
    """Applique 1 à `max_n` fois les rotations apprises. Pour chaque courbe, chaque n
    et chaque image, tenseurs (max_n, images) parmi : "mse" (image vs vraie
    rotation), "correct" (chiffre reconnu) et "angle_error" (°).
    Les voies de rotation entraînées dans `gw` (cf `path_outputs`), toutes lues
    par le même lecteur (le décodeur digit du GW) :
     - pixel : rotation sur l'image ;
     - VAE : rotation sur le latent du VAE ;
     - GW : rotation sur l'état GW ;
     - GW via digit : rotation sur l'état GW passé par digit (image -> GW -> digit propre -> GW) ;
     - GW fusionné : rotation sur 0.5 (GW + GW via digit) ;
     - digit : rotation sur le vecteur digit propre lu par le GW ;
     - VAE + digit : rotations VAE et digit fusionnées dans le GW, puis décodées ;
       chaque rotation repart de ces décodages ;
     - VAE + digit séparés : rotations VAE et digit enchaînées chacune dans son
       espace, fusionnées dans le GW et décodées seulement pour la lecture.
    Les courbes de comparaison :
     - EXACT_DIGIT : digit lu au départ, angle tourné exactement de k x degrees,
       le mieux que la voie digit puisse faire ;
     - REFERENCE (reconnaissance) : ce que le lecteur reconnaît de la vraie image tournée ;
     - VAE_FLOOR, GW_FLOOR, SYMBOLIC_FLOOR (MSE) : vraie image tournée passée par
       le VAE, le GW ou le GW puis digit propre, le mieux que les voies correspondantes
       puissent faire en pixels."""
    image_mod = gw.gw_mod.domain_mods["image"]
    degrees = rotation_degrees(gw)
    results = {path: {"mse": [], "correct": [], "angle_error": []} for path in (*rotation_paths(gw), EXACT_DIGIT)}
    results |= {REFERENCE: {"correct": [], "angle_error": []},
                VAE_FLOOR: {"mse": []}, GW_FLOOR: {"mse": []}, SYMBOLIC_FLOOR: {"mse": []}}

    def mse(produced: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((produced - target) ** 2).mean(dim=(1, 2, 3))

    states = start_states(gw, images)
    for n in range(1, max_n + 1):
        states = rotate_states(gw, states)
        target = rotate_by(images, degrees * n)
        target_angles = start_angles + degrees * n
        for path, state in states.items():
            produced, digits = path_outputs(gw, path, state)
            correct, err = recognition(digits, labels, target_angles)
            results[path]["mse"].append(mse(produced, target))
            results[path]["correct"].append(correct)
            results[path]["angle_error"].append(err)
        target_state = image_to_gw(gw, target)
        target_digits = from_gw(gw, "digit", target_state)
        correct, err = recognition(target_digits, labels, target_angles)
        results[REFERENCE]["correct"].append(correct)
        results[REFERENCE]["angle_error"].append(err)
        results[VAE_FLOOR]["mse"].append(mse(image_mod.decode(image_mod.encode(target)), target))
        results[GW_FLOOR]["mse"].append(mse(gw_to_image(gw, target_state), target))
        results[SYMBOLIC_FLOOR]["mse"].append(mse(gw_to_image(gw, to_gw(gw, "digit", clean_digit(target_digits))), target))
    return {path: {name: torch.stack(values).cpu() for name, values in r.items()} for path, r in results.items()}


def plot_chained_rotations(results: dict[str, dict[str, torch.Tensor]], title: str) -> None:
    """Moyennes sur les images, en fonction du nombre de rotations enchaînées.
    Traits pleins : voies de rotation ; pointillés : référence et planchers
    (un plancher a la couleur de la voie qu'il borne)."""
    max_n = results[REFERENCE]["correct"].shape[0]
    ns = range(1, max_n + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for ax, (metric, ylabel, chance) in zip(axes, [
        ("mse", "MSE en pixels (vs vraie rotation)", None),
        ("correct", "précision sur le chiffre", 0.1),
        ("angle_error", "erreur d'angle moyenne (°)", 90),
    ]):
        for path, values in results.items():
            if metric in values:
                ax.plot(ns, values[metric].mean(dim=1), "-o" if path in ROTATION_PATHS else "--",
                        color=CURVE_COLORS[path], label=path)
        if chance is not None:
            ax.axhline(chance, color="gray", linestyle=":", label="hasard")
        ax.set_xticks(ns)
        ax.set_xlabel("nombre de rotations enchaînées")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    # une seule légende sous les graphiques, avec toutes les courbes
    legend = {label: handle for ax in axes for handle, label in zip(*ax.get_legend_handles_labels())}
    fig.legend(legend.values(), legend.keys(), loc="lower center", ncol=4, fontsize=8)
    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0.17, 1, 1))
    plt.show()

    # tableau : une section par métrique, une ligne par courbe, une colonne par n
    name_width = max(len(path) for path in results)
    for metric, fmt in [("mse", "{:7.4f}"), ("correct", "{:7.1%}"), ("angle_error", "{:6.1f}°")]:
        print(f"\n{metric:<{name_width}} " + "".join(f"{f'n={n}':>8}" for n in ns))
        for path, values in results.items():
            if metric in values:
                print(f"{path:<{name_width}} " + "".join(f"{fmt.format(v):>8}" for v in values[metric].mean(dim=1).tolist()))


def plot_per_digit(results: dict[str, dict[str, torch.Tensor]], labels: torch.Tensor, title: str) -> None:
    """Précision et erreur d'angle par chiffre (lignes) et par nombre de rotations
    (colonnes), pour chaque voie, la rotation exacte et la référence."""
    labels = labels.cpu()
    paths = [*(path for path in ROTATION_PATHS if path in results), EXACT_DIGIT, REFERENCE]
    max_n = results[REFERENCE]["correct"].shape[0]
    fig, axes = plt.subplots(2, len(paths), figsize=(4.5 * len(paths), 9))
    for row, (metric, label, cmap, vmax, fmt) in enumerate([
        ("correct", "précision sur le chiffre", "RdYlGn", 1, "{:.0%}"),
        ("angle_error", "erreur d'angle moyenne (°)", "RdYlGn_r", 90, "{:.0f}"),
    ]):
        for col, path in enumerate(paths):
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
            ax.set_title(f"{path}\n{label}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


COLUMN_TITLES = {"GW via digit": "GW (d)", "GW fusionné": "GW (f)", "VAE + digit": "V+D",
                 "VAE + digit séparés": "V+D (s)"}  # titre court : les colonnes sont étroites


def show_chained_rotations(gw: MyGlobalWorkspace, images: torch.Tensor, max_n: int = 12,
                           n_images: int = 3, cell_size: float = 0.55) -> None:
    """Suivi pas à pas des rotations enchaînées : une ligne par étape (angle
    cumulé), et pour chaque image un bloc de colonnes : vraie rotation, puis
    chaque voie entraînée (pixel, VAE, GW, GW via digit, GW fusionné, digit,
    VAE + digit, VAE + digit séparés). À l'étape 0, chaque voie
    montre son point de départ (image d'origine, reconstruction par le VAE,
    demi-cycle par le GW, cycle par digit, moyenne des deux)."""
    degrees = rotation_degrees(gw)
    paths = rotation_paths(gw)
    columns = ("réel", *paths)
    images = images[:n_images]

    states = start_states(gw, images)
    steps = []
    for n in range(max_n + 1):
        if n > 0:
            states = rotate_states(gw, states)
        steps.append({"réel": rotate_by(images, degrees * n)}
                     | {path: path_outputs(gw, path, states[path])[0] for path in paths})
    # un bloc de colonnes par image, séparés par une colonne vide étroite
    n_cols = len(columns)
    width_ratios = ([1] * n_cols + [0.4]) * n_images
    width_ratios.pop()
    fig, axes = plt.subplots(
        max_n + 1, len(width_ratios), squeeze=False,
        figsize=(cell_size * sum(width_ratios) + 0.6, cell_size * (max_n + 1) + 0.4),
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.05, "hspace": 0.05},
    )
    for ax in axes.flat:
        ax.axis("off")
    for n, step in enumerate(steps):
        for i in range(n_images):
            for c, name in enumerate(columns):
                ax = axes[n, i * (n_cols + 1) + c]
                ax.imshow(step[name][i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
                if n == 0:
                    ax.set_title(COLUMN_TITLES.get(name, name), fontsize=7)
        axes[n, 0].text(-0.15, 0.5, f"{degrees * n:.0f}°", transform=axes[n, 0].transAxes,
                        ha="right", va="center", fontsize=7)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.01)
    plt.show()
