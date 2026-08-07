
import torch

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from cfg_tools import load_config_files

from numb_project.data_module import MnistDataModule
from numb_project.gw_model import setup_global_workspace

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS = 5000
CONFIG_FILE = f"/home/lucas/gwnumb/config"

def main():
    config = load_config_files(
        CONFIG_FILE,
        use_cli=False,
        load_files=["config.yaml"])[0]

    data_module = MnistDataModule(config['training']['batch_size'])
    global_workspace = setup_global_workspace(config)

    checkpoint_callback = ModelCheckpoint(dirpath="/home/lucas/gwnumb/checkpoints/numb", filename="test")
    trainer = Trainer(
            max_epochs=MAX_EPOCHS,
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
            callbacks=[checkpoint_callback],
            accelerator="gpu" if DEVICE == "cuda" else "cpu", devices=[0] if DEVICE == "cuda" else "auto"
    )

    trainer.fit(global_workspace, data_module)

if __name__ == "__main__":
    main()