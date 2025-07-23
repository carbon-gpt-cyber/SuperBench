import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import yaml
import os
from VQVAE_L import VQVAE_L
from VQVAE_H import VQVAE_H
from datamodule import ClimateDownscalingDataModule
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(config_path):
    config = load_config(config_path)
    set_seed(config['seed'])

    data_module = ClimateDownscalingDataModule(**config['data_module'])
    model = VQVAE_H(input_channels=3, output_channels=3, **config['model'])

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=3, mode='min')

    trainer = Trainer(
        max_epochs = config['trainer']['max_epochs'],
        strategy='ddp_find_unused_parameters_true',
        # gpus = config['trainer']['gpus'],
        callbacks=[checkpoint_callback],
        logger=CSVLogger(save_dir='./vae'),
        precision=16,
        accumulate_grad_batches=4
    )

    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    main(config_path='config.yaml')