from typing import Any

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from cfg_tools import load_config_files

from numb_project.constants import *
from numb_project.data_module import MnistDataModule
from numb_project.domain_module import LoadedDomainConfig, get_domains_config
from numb_project.gw_model import setup_global_workspace, load_pretrained_global_workspace


def get_training_objects(
    config: dict[str, Any],
    domain_configs: list[LoadedDomainConfig],
    training_name: str,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    config = load_config_files(
        CONFIG_FOLDER,
        use_cli=False,
        load_files=[CONFIG_FILE])[0]

    data_module = MnistDataModule(config['training']['batch_size'])

    if checkpoint_path is not None:
        global_workspace = load_pretrained_global_workspace(
            config, domain_configs, checkpoint_path, device=DEVICE
        )
    else:
        global_workspace = setup_global_workspace(config, domain_configs)

    checkpoint_callback = ModelCheckpoint(dirpath=GW_CHECKPOINT_FOLDER, filename=training_name)
    trainer = Trainer(
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback],
        accelerator="gpu" if DEVICE == "cuda" else "cpu",
        devices=[0] if DEVICE == "cuda" else "auto"
    )

    return {
        "data_module": data_module,
        "global_workspace": global_workspace,
        "trainer": trainer
    }


def main() -> None:
    config = load_config_files(
        CONFIG_FOLDER,
        use_cli=False,
        load_files=[CONFIG_FILE])[0]
    domain_configs = get_domains_config(['image', 'digit'])

    checkpoint_path = "/home/lucas/gwnumb/checkpoints/numb/pretrain.ckpt"

    training_objects = get_training_objects(
        config, domain_configs, 'test', checkpoint_path=checkpoint_path
    )
    training_objects['trainer'].fit(training_objects['global_workspace'], training_objects['data_module'])


if __name__ == "__main__":
    main()