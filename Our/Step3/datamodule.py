import os
from typing import Optional, List
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from pytorch_lightning import LightningDataModule

from src.data_loader import GetClimateDataset
from utils import get_data_info

class SequenceWrapper(Dataset):
    """Wrap single-frame dataset to provide a time dimension."""
    def __init__(self, dataset: Dataset, variables: List[str]):
        self.dataset = dataset
        self.variables = variables

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        # add time dimension
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        return x, y, self.variables, self.variables


def collate_fn(batch):
    inp = torch.stack([b[0] for b in batch])
    out = torch.stack([b[1] for b in batch])
    variables = batch[0][2]
    out_variables = batch[0][3]
    return inp, out, variables, out_variables


class ClimateDownscalingDataModule(LightningDataModule):
    def __init__(self,
                 root_dir: str,
                 variables: List[str],
                 batch_size: int = 4,
                 num_workers: int = 0,
                 upscale_factor: int = 4,
                 crop_size: int = 720,
                 method: str = "bicubic",
                 noise_ratio: float = 0.0):
        super().__init__()
        self.save_hyperparameters()

        _, _, _, mean, std = get_data_info('era5')
        self.transforms = transforms.Normalize(mean, std)
        self.output_transforms = transforms.Normalize(mean, std)
        self.std = std

        self.data_train: Optional[Dataset] = None
        self.data_val1: Optional[Dataset] = None
        self.data_val2: Optional[Dataset] = None
        self.data_test1: Optional[Dataset] = None
        self.data_test2: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if self.data_train is None:
            def build(subdir, train):
                ds = GetClimateDataset(os.path.join(self.hparams.root_dir, subdir),
                                        train=train,
                                        transform=torch.from_numpy,
                                        upscale_factor=self.hparams.upscale_factor,
                                        noise_ratio=self.hparams.noise_ratio,
                                        std=self.std,
                                        crop_size=self.hparams.crop_size,
                                        n_patches=1,
                                        method=self.hparams.method)
                return SequenceWrapper(ds, self.hparams.variables)

            self.data_train = build('train', True)
            self.data_val1 = build('valid_1', False)
            self.data_val2 = build('valid_2', False)
            self.data_test1 = build('test_1', False)
            self.data_test2 = build('test_2', False)

    def train_dataloader(self):
        return DataLoader(self.data_train,
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          collate_fn=collate_fn)

    def val_dataloader(self):
        loader1 = DataLoader(self.data_val1,
                             batch_size=self.hparams.batch_size,
                             shuffle=False,
                             num_workers=self.hparams.num_workers,
                             collate_fn=collate_fn)
        loader2 = DataLoader(self.data_val2,
                             batch_size=self.hparams.batch_size,
                             shuffle=False,
                             num_workers=self.hparams.num_workers,
                             collate_fn=collate_fn)
        return [loader1, loader2]

    def test_dataloader(self):
        loader1 = DataLoader(self.data_test1,
                             batch_size=self.hparams.batch_size,
                             shuffle=False,
                             num_workers=self.hparams.num_workers,
                             collate_fn=collate_fn)
        loader2 = DataLoader(self.data_test2,
                             batch_size=self.hparams.batch_size,
                             shuffle=False,
                             num_workers=self.hparams.num_workers,
                             collate_fn=collate_fn)
        return [loader1, loader2]

    def get_lat_lon(self):
        lat_path = os.path.join(self.hparams.root_dir, 'lat.npy')
        lon_path = os.path.join(self.hparams.root_dir, 'lon.npy')
        if os.path.exists(lat_path) and os.path.exists(lon_path):
            lat = np.load(lat_path)
            lon = np.load(lon_path)
            return lat, lon
        return None, None
