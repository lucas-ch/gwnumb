import torch

BASE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_FOLDER = f"/home/lucasc/Projects/gwnumb/config"
CONFIG_FILE = "config.yaml"
VAE_CHECKPOINT_FILE = "/home/lucasc/Projects/gwnumb/checkpoints/mnist.ckpt"
GW_CHECKPOINT_FOLDER = "/home/lucasc/Projects/gwnumb/checkpoints/numb"