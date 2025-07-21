import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import yaml
import os
from STtransformer import Step_t
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
    normalization = data_module.output_transforms
    lat, lon = data_module.get_lat_lon()
    mean_norm, std_norm = normalization.mean, normalization.std
    config['model'].update({'lat': lat,
                            'mean': mean_norm,
                            'std': std_norm})
    model = Step_t(**config['model'])

    checkpoint_callback = ModelCheckpoint(
                                          monitor='val_loss',
                                          dirpath='my_model/', 
                                          filename='best_model',
                                          save_top_k=1, 
                                          mode='min')

    trainer = Trainer(
        max_epochs = config['trainer']['max_epochs'],
        strategy='ddp_find_unused_parameters_true',
        callbacks=[checkpoint_callback],
        logger=CSVLogger(save_dir='./'),
        precision=16,
        accumulate_grad_batches=4
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module, ckpt_path='./my_model/best_model.ckpt')
    # model = DDPM(**config['model'])
    # model = model.load_from_checkpoint(checkpoint_path='my_model/best_model.ckpt')
    # trainer.test(model, datamodule=data_module)

if __name__ == '__main__':
    main(config_path='config.yaml')