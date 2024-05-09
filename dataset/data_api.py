import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
from torch.utils.data import DataLoader

from dataset.temporal_dataset import TemporalDataset
from dataset.transforms import TrainTransform, ValTransform

def create_dataset(hparams, split):
    if split.startswith('train'):
        transform = TrainTransform(hparams)
    else:
        transform = ValTransform(hparams)
    if hparams.dataset_name == 'temporal':
        dataset = TemporalDataset(hparams.data_dir, transform=transform, split=split)
    else:
        raise ValueError(f'Unknown dataset name: {hparams.dataset_name}')
    return dataset

def collate_fn(batch):
    # transposed_batch = list(zip(*batch))
    # frms = torch.stack(transposed_batch[0], dim=0)
    # skls = torch.stack(transposed_batch[1], dim=0)
    # flow = torch.stack(transposed_batch[2], dim=0)
    # masks = torch.stack(transposed_batch[3], dim=0)
    # dmaps = torch.stack(transposed_batch[4], dim=0)
    # return frms, skls, flow, masks #, dmaps
    batch_data = {}
    for key in ['point_clouds', 'keypoints']:
        batch_data[key] = torch.stack([sample[key] for sample in batch], dim=0)
    return batch_data

class LitDataModule(pl.LightningDataModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            self.train_dataset = create_dataset(self.hparams, self.hparams.train_split)
            self.val_dataset = create_dataset(self.hparams, self.hparams.val_split)
        else:
            self.test_dataset = create_dataset(self.hparams, self.hparams.test_split)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.hparams.pin_memory,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size_eva,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.hparams.pin_memory,
            drop_last=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size_eva,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.hparams.pin_memory,
            drop_last=False
        )