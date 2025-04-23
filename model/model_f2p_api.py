import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import matplotlib.pyplot as plt
import os
import pickle

from model.metrics import calulate_error
from misc.utils import torch2numpy, import_with_str

def create_model(model_name, model_params):
    if model_params is None:
        model_params = {}
    model_class = import_with_str('model', model_name)
    model = model_class(**model_params)
    return model

def create_loss(loss_name, loss_params):
    if loss_params is None:
        loss_params = {}
    loss_class = import_with_str('torch.nn', loss_name)
    loss = loss_class(**loss_params)
    return loss

def create_optimizer(optim_name, optim_params, mparams):
    if optim_params is None:
        optim_params = {}
    optim_class = import_with_str('torch.optim', optim_name)
    optimizer = optim_class(mparams, **optim_params)
    return optimizer
    
def create_scheduler(sched_name, sched_params, optimizer):
    if sched_params is None:
        sched_params = {}
    if sched_name == 'LinearWarmupCosineAnnealingLR':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, **sched_params)
    else:
        sched_class = import_with_str('torch.optim.lr_scheduler', sched_name)
        scheduler = sched_class(optimizer, **sched_params)
    return scheduler

class F2PLitModel(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = create_model(hparams.model_name, hparams.model_params)
        if hparams.checkpoint_path is not None:
            self.load_state_dict(torch.load(hparams.checkpoint_path, map_location=self.device)['state_dict'], strict=False)
        self.loss = create_loss(hparams.loss_name, hparams.loss_params)
        if hparams.save_when_test:
            self.results = []

    def _calculate_loss(self, batch):
        y = batch['keypoints']
        y = y - y[:, :, 8:9, :]
        # if 'flow' in batch.keys():
        #     x = batch['flow']
            # if torch.rand(1) < 0.5:
            #     x = x + torch.randn_like(x) * 0.05
        # elif 'pred_keypoints' in batch.keys():
        #     y_ = batch['pred_keypoints']
        #     y_ = y_ - y_[:, :, 8:9, :]
        #     x = y_[:, 1:, ...] - y_[:, :-1, ...]
        #     y = y[:, :-1, ...]
        if torch.rand(1) < 0.5:
            y_ = y# + torch.randn_like(y) * 0.02
            x = y_[:, 1:, ...] - y_[:, :-1, ...]
            x = x + torch.randn_like(x) * 0.05
            y = y[:, :-1, ...]
        else:
            x = y[:, 1:, ...] - y[:, :-1, ...]
            y = y[:, :-1, ...]
        
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)

        return {'loss': loss}, torch2numpy(x), torch2numpy(y), torch2numpy(y_hat)

    def _evaluate(self, batch):
        y = batch['keypoints']
        y = y - y[:, :, 8:9, :]
        if 'flow' in batch.keys():
            x = batch['flow']
        # elif 'pred_keypoints' in batch.keys():
        #     y_ = batch['pred_keypoints']
        #     y_ = y_ - y_[:, :, 8:9, :]
        #     x = y_[:, 1:, ...] - y_[:, :-1, ...]
        #     y = y[:, :-1, ...]
        else:
            x = y[:, 1:, ...] - y[:, :-1, ...]
            y = y[:, :-1, ...]

        y_hat = self.model(x)
        # left_indices = [3, 5, 7, 10, 12, 14]
        # right_indices = [2, 4, 6, 9, 11, 13]
        # y_hat[:, :, left_indices+right_indices, :] = y_hat[:, :, right_indices+left_indices, :]

        return torch2numpy(x), torch2numpy(y), torch2numpy(y_hat)

    def training_step(self, batch, batch_idx):
        losses, x, y, y_hat = self._calculate_loss(batch)
        mpjpe, pampjpe = calulate_error(y_hat, y)

        log_dict = {'train_mpjpe': mpjpe, 'train_pampjpe': pampjpe}
        for loss_name, loss in losses.items():
            log_dict[f'train_{loss_name}'] = loss

        self.log_dict(log_dict, sync_dist=True)

        return losses['loss']

    def validation_step(self, batch, batch_idx):
        x, y, y_hat = self._evaluate(batch)
        mpjpe, pampjpe = calulate_error(y_hat, y)

        self.log_dict({'val_mpjpe': mpjpe, 'val_pampjpe': pampjpe}, sync_dist=True)

    def test_step(self, batch, batch_idx):
        si = batch['sequence_index']
        x, y, y_hat = self._evaluate(batch)
        mpjpe, pampjpe = calulate_error(y_hat, y)

        self.log_dict({'test_mpjpe': mpjpe, 'test_pampjpe': pampjpe}, sync_dist=True)

        if self.hparams.save_when_test:
            result = {'pred': y_hat, 'gt': y, 'flow': x, 'seq_idx': torch2numpy(si)}
            self.results.append(result)

    def on_test_end(self):
        if self.hparams.save_when_test:
            mpjpes = []
            pampjpes = []
            split_seqs = []
            split_idxs = []
            last_seq_idx = -1

            for result in self.results:
                for i in range(len(result['pred'])):
                    gt = result['gt'][i]
                    pred = result['pred'][i]
                    flow = result['flow'][i]
                    seq_idx = int(result['seq_idx'][i].item())
                    if seq_idx != last_seq_idx:
                        last_seq_idx = seq_idx
                        split_seqs.append({'keypoints': [], 'keypoints_pred': [], 'flow': []})
                        split_idxs.append(seq_idx)
                    split_seqs[-1]['keypoints'].append(gt)
                    split_seqs[-1]['keypoints_pred'].append(pred)
                    split_seqs[-1]['flow'].append(flow)

            for i in range(len(split_seqs)):
                split_seqs[i]['keypoints'] = np.array(split_seqs[i]['keypoints'])[:, 0, ...]
                split_seqs[i]['flow'] = np.array(split_seqs[i]['flow'])[:, 0, ...]
                
                kps_pred = np.array(split_seqs[i]['keypoints_pred'])
                N, T, J, D = kps_pred.shape
                kps_pred_ = np.zeros((N + T - 1, J, D))
                for j in range(T):
                    kps_pred_[j:j+N] += kps_pred[:, j]
                for j in range(T):
                    kps_pred_[j] *= (T / (j + 1))
                kps_pred = kps_pred_[:N]
                split_seqs[i]['keypoints_pred'] = kps_pred / T

                if kps_pred.shape[0] > 0:
                    mpjpe, pampjpe = calulate_error(split_seqs[i]['keypoints_pred'][:, np.newaxis], split_seqs[i]['keypoints'][:, np.newaxis], reduce=False)
                    mpjpes.append(mpjpe)
                    pampjpes.append(pampjpe)

            mpjpes = np.concatenate(mpjpes, axis=0)
            pampjpes = np.concatenate(pampjpes, axis=0)
            mpjpe = np.mean(mpjpes)
            pampjpe = np.mean(pampjpes)
            print(f'Final MPJPE: {mpjpe}, PAMJPE: {pampjpe}')

            with open(os.path.join('logs', self.hparams.exp_name, self.hparams.version, 'results.pkl'), 'wb') as f:
                pickle.dump(split_seqs, f)

        return super().on_test_end()

    def configure_optimizers(self):
        optimizer = create_optimizer(self.hparams.optim_name, self.hparams.optim_params, self.model.parameters())
        scheduler = create_scheduler(self.hparams.sched_name, self.hparams.sched_params, optimizer)
        return [optimizer], [scheduler]