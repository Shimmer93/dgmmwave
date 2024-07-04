import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
from torch.utils.data import DataLoader
import os

from dataset.temporal_dataset import TemporalDataset
from dataset.reference_dataset import ReferenceDataset
from dataset.transforms import TrainTransform, ValTransform

def create_dataset(hparams, split):
    if split.startswith('train'):
        transform = TrainTransform(hparams)
        data_pkl = hparams.train_data_pkl if hparams.train_data_pkl is not None else hparams.data_pkl
        dataset_name = hparams.train_dataset_name if hparams.train_dataset_name is not None else hparams.dataset_name
    else:
        transform = ValTransform(hparams)
        if split.startswith('val'):
            data_pkl = hparams.val_data_pkl if hparams.val_data_pkl is not None else hparams.data_pkl
            dataset_name = hparams.val_dataset_name if hparams.val_dataset_name is not None else hparams.dataset_name
        else:
            data_pkl = hparams.test_data_pkl if hparams.test_data_pkl is not None else hparams.data_pkl
            dataset_name = hparams.test_dataset_name if hparams.test_dataset_name is not None else hparams.dataset_name
    
    if dataset_name == 'temporal':
        dataset = TemporalDataset(os.path.join(hparams.data_dir, data_pkl), transform=transform, split=split)
        collate_fn = TemporalDataset.collate_fn
    elif dataset_name == 'reference':
        dataset = ReferenceDataset(os.path.join(hparams.data_dir, data_pkl), os.path.join(hparams.data_dir, hparams.ref_data_pkl), 
                                   transform=transform, split=split, ref_split=hparams.ref_split)
        collate_fn = ReferenceDataset.collate_fn
    else:
        raise ValueError(f'Unknown dataset name: {hparams.dataset_name}')
    return dataset, collate_fn

class LitDataModule(pl.LightningDataModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            self.train_dataset, self.train_collate_fn = create_dataset(self.hparams, self.hparams.train_split)
            self.val_dataset, self.val_collate_fn = create_dataset(self.hparams, self.hparams.val_split)
        else:
            self.test_dataset, self.test_collate_fn = create_dataset(self.hparams, self.hparams.test_split)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.train_collate_fn,
            pin_memory=self.hparams.pin_memory,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size_eva,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.val_collate_fn,
            pin_memory=self.hparams.pin_memory,
            drop_last=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size_eva,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.test_collate_fn,
            pin_memory=self.hparams.pin_memory,
            drop_last=False
        )