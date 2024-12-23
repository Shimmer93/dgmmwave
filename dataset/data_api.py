import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
from torch.utils.data import DataLoader
import os

from dataset.temporal_dataset import TemporalDataset
from dataset.biaug_dataset import BiAugDataset
from dataset.reference_dataset import ReferenceDataset
from dataset.posneg_dataset import PosNegDataset
from dataset.ti_inference_dataset import TiInferenceDataset
from dataset.transforms import TrainTransform, RefTransform, ValTransform, PosNegTransform

def create_dataset(hparams, split=None):
    if isinstance(split, list):
        split_id = split[0]
    elif isinstance(split, str):
        split_id = split
    else:
        split_id = 'predict'

    transform = TrainTransform(hparams) if split_id.startswith('train') else ValTransform(hparams)
    
    if split_id.startswith('train'):
        data_pkl = hparams.train_data_pkl if hasattr(hparams, 'train_data_pkl') else hparams.data_pkl
        dataset_name = hparams.train_dataset_name if hasattr(hparams, 'train_dataset_name') else hparams.dataset_name
    elif split_id.startswith('val'):
        data_pkl = hparams.val_data_pkl if hasattr(hparams, 'val_data_pkl') else hparams.data_pkl
        dataset_name = hparams.val_dataset_name if hasattr(hparams, 'val_dataset_name') else hparams.dataset_name
    elif split_id.startswith('test'):
        data_pkl = hparams.test_data_pkl if hasattr(hparams, 'test_data_pkl') else hparams.data_pkl
        dataset_name = hparams.test_dataset_name if hasattr(hparams, 'test_dataset_name') else hparams.dataset_name
    elif split_id.startswith('predict'):
        data_dir = hparams.predict_data_dir
        dataset_name = hparams.predict_dataset_name if hasattr(hparams, 'predict_dataset_name') else hparams.dataset_name

    if dataset_name == 'temporal':
        dataset = TemporalDataset(os.path.join(hparams.data_dir, data_pkl), transform=transform, split=split)
        collate_fn = TemporalDataset.collate_fn
    elif dataset_name == 'biaug':
        dataset = BiAugDataset(os.path.join(hparams.data_dir, data_pkl), transform=transform, split=split)
        collate_fn = BiAugDataset.collate_fn
    elif dataset_name == 'reference':
        ref_transform = RefTransform(hparams)
        dataset = ReferenceDataset(os.path.join(hparams.data_dir, data_pkl), os.path.join(hparams.data_dir, hparams.ref_data_pkl), 
                                   transform=transform, ref_transform=ref_transform, split=split, ref_split=hparams.ref_split)
        collate_fn = ReferenceDataset.collate_fn
    elif dataset_name == 'posneg':
        posneg_transform = PosNegTransform(transform, hparams)
        dataset = PosNegDataset(os.path.join(hparams.data_dir, data_pkl), transform=posneg_transform, split=split)
        collate_fn = PosNegDataset.collate_fn
    elif dataset_name == 'ti_inference':
        dataset = TiInferenceDataset(data_dir, transform=transform)
        collate_fn = TiInferenceDataset.collate_fn
    else:
        raise ValueError(f'Unknown dataset name: {dataset_name}')
    return dataset, collate_fn

class LitDataModule(pl.LightningDataModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            self.train_dataset, self.train_collate_fn = create_dataset(self.hparams, self.hparams.train_split)
            self.val_dataset, self.val_collate_fn = create_dataset(self.hparams, self.hparams.val_split)
        elif stage == 'test':
            self.test_dataset, self.test_collate_fn = create_dataset(self.hparams, self.hparams.test_split)
        elif stage == 'predict':
            self.predict_dataset, self.predict_collate_fn = create_dataset(self.hparams)
        else:
            raise ValueError(f'Unknown stage: {stage}')

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
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.batch_size_eva,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.predict_collate_fn,
            pin_memory=self.hparams.pin_memory,
            drop_last=False
        )