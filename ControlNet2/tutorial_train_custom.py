"""

다음과 같은 명령어로 실행시킬 수 있습니다. 
python tutorial_train_custom.py --config config/config1.yaml

"""

from share import *
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from custom_dataset import GeometryDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from cldm.cldm import ControlLDM, EpipolarWarpOperator
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn as nn
import wandb
from pytorch_lightning.loggers import WandbLogger
from utils import load_config
import argparse
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 

def main(config_path):
    config = load_config(config_path)
    # WandB 설정
    wandb.init(project=config['wandb_project'], name=config['wandb_name'])
    wandb_logger = WandbLogger(project=config['wandb_project'])

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(config['model_config_path']).cpu()

    # Config setting
    model.load_state_dict(load_state_dict(config['resume_path'], location='cpu'))
    model.learning_rate = config['learning_rate']
    model.sd_locked = config['sd_locked']
    model.only_mid_control = config['only_mid_control']
    checkpoint_dir = os.path.join(config['checkpoint_dir'], config['wandb_name'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    # ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='train/loss_simple',
        dirpath= checkpoint_dir,
        filename=config['checkpoint_filename'],
        save_top_k = 10,
        # save_top_k=config['checkpoint_save_top_k'],
        mode='min',
        every_n_epochs=config['checkpoint_every_n_epochs'],
        save_last=False,
        verbose=True,
    )

    # Misc
    dataset = GeometryDataset(config['annotation_file'], config['base_dir'])
    dataloader = DataLoader(dataset, num_workers=config['num_workers'], batch_size=config['batch_size'], shuffle=True)
    # Logger
    logger = ImageLogger(config, batch_frequency=config['logger_freq'],specific_image_indices=config['specific_image_indices'])

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator=config['accelerator'],
        devices=config['devices'],
        precision=config['precision'],
        callbacks=[logger, checkpoint_callback],
        logger=wandb_logger
    )

    print(f"Using {trainer.device_ids} GPU(s):")

    # Train!
    trainer.fit(model, dataloader)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config file을 입력해주세요.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    main(args.config)