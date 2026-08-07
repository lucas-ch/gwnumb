import torch

BASE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_FOLDER = f"/home/lucas/gwnumb/config"
CONFIG_FILE = "config.yaml"
VAE_CHECKPOINT_FILE = "/home/lucas/gwnumb/checkpoints/mnist.ckpt"
GW_CHECKPOINT_FOLDER = "/home/lucas/gwnumb/checkpoints/numb"