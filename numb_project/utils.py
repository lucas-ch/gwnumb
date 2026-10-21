from typing import Mapping
import torch
import torch.nn.functional as F


from numb_project.constants import *

def get_from_dict_or_val(
    val: int | Mapping[str, int], key: str, log: str
) -> int:
    """
    If val is int, return val, otherwise return val[key]
    """
    if isinstance(val, int):
        return val

    assert key in val, f"{key} should be defined in {log}."
    return val[key]

def rotate_images(images: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
    """Rotation exacte (non apprise) d'un batch d'images (B, C, H, W), d'un angle
    par image, sur le device des images (affine_grid + grid_sample bilinéaire).
    Seule fonction de rotation du projet : VAE, GW, cibles des opérations et
    notebooks tournent tous les images de la même façon.

    Un angle positif tourne l'image dans le sens des aiguilles d'une montre
    (sens inverse de torchvision.transforms.functional.rotate)."""
    theta = angles_deg * torch.pi / 180
    cos, sin = torch.cos(theta), torch.sin(theta)
    zeros = torch.zeros_like(theta)
    rotation_matrices = torch.stack(
        [torch.stack([cos, sin, zeros], dim=-1), torch.stack([-sin, cos, zeros], dim=-1)], dim=-2
    )
    grid = F.affine_grid(rotation_matrices, list(images.shape), align_corners=False)
    return F.grid_sample(images, grid, align_corners=False)


def angle_to_cos_sin(angle_deg: torch.Tensor) -> torch.Tensor:
    """Encode un angle en (cos, sin), sans discontinuité entre 359° et 0°."""
    theta = angle_deg * torch.pi / 180
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

def cos_sin_to_angle(cos_sin: torch.Tensor) -> torch.Tensor:
    """Inverse de angle_to_cos_sin : angle en degrés dans [0, 360[ (le vecteur
    (cos, sin) n'a pas besoin d'être de norme 1)."""
    return torch.rad2deg(torch.atan2(cos_sin[..., 1], cos_sin[..., 0])) % 360
