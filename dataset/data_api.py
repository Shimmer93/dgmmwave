import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset.transforms import RandomApply, ComposeTransform, MultipleKeyAggregate
from misc.utils import import_with_str

class PipelineTransform(ComposeTransform):

    def __init__(self, pipeline):
        tsfms = []
        for tsfm in pipeline:
            tsfm_class = import_with_str('dataset.transforms', tsfm['name'])
            tsfms_params = tsfm['params']
            if tsfms_params is None:
                tsfms_params = {}
            tsfm_ = tsfm_class(**tsfms_params)
            if 'prob' in tsfm:
                tsfm_ = RandomApply([tsfm_], prob=tsfm['prob'])
            if 'ori_key' in tsfm:
                tsfm_ = MultipleKeyAggregate([tsfm_], tsfm['ori_key'], tsfm['more_keys'])
            tsfms.append(tsfm_)

        super().__init__(tsfms)

def create_dataset(dataset_name, dataset_params, pipeline):
    if dataset_params is None:
        dataset_params = {}
    transform = PipelineTransform(pipeline)
    dataset_class = import_with_str('dataset', dataset_name)
    dataset = dataset_class(transform=transform, **dataset_params)
    collate_fn = dataset_class.collate_fn
    return dataset, collate_fn

def create_ref_dataset(dataset_name, dataset_params, pipeline, ref_pipeline):
    if dataset_params is None:
        dataset_params = {}
    transform = PipelineTransform(pipeline)
    ref_transform = PipelineTransform(ref_pipeline)
    dataset_class = import_with_str('dataset', dataset_name)
    dataset = dataset_class(transform=transform, ref_transform=ref_transform, **dataset_params)
    collate_fn = dataset_class.collate_fn
    return dataset, collate_fn

class LitDataModule(pl.LightningDataModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            if self.hparams.train_dataset['name'] in ['ReferenceDataset', 'ReferenceOneToOneDataset']:
                self.train_dataset, self.train_collate_fn = create_ref_dataset(self.hparams.train_dataset['name'], self.hparams.train_dataset['params'], self.hparams.train_pipeline, self.hparams.ref_pipeline)
            else:
                self.train_dataset, self.train_collate_fn = create_dataset(self.hparams.train_dataset['name'], self.hparams.train_dataset['params'], self.hparams.train_pipeline)
            self.val_dataset, self.val_collate_fn = create_dataset(self.hparams.val_dataset['name'], self.hparams.val_dataset['params'], self.hparams.val_pipeline)
        elif stage == 'test':
            self.test_dataset, self.test_collate_fn = create_dataset(self.hparams.test_dataset['name'], self.hparams.test_dataset['params'], self.hparams.test_pipeline)
        elif stage == 'predict':
            self.predict_dataset, self.predict_collate_fn = create_dataset(self.hparams.predict_dataset['name'], self.hparams.predict_dataset['params'], self.hparams.predict_pipeline)
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