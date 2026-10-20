import torch

BASE = 10
DIGIT_DIM = BASE + 2  # one-hot du chiffre + (cos, sin) de l'angle de l'image
MNIST_IMAGE_SHAPE = (1, 28, 28)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = "/home/lucas/gwnumb"
CONFIG_FOLDER = f"{ROOT}/config"
CONFIG_FILE = "config.yaml"
VAE_CHECKPOINT_FILE = f"{ROOT}/checkpoints/mnist-20D.ckpt"
GW_CHECKPOINT_FOLDER = f"{ROOT}/checkpoints/numb"