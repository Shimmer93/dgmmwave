from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import matplotlib.pyplot as plt
import pickle

import os
from collections import OrderedDict
from copy import deepcopy

from model.metrics import calulate_error
from model.chamfer_distance import ChamferDistance
from loss.mpjpe import mpjpe as mpjpe_mmwave
from misc.utils import torch2numpy, import_with_str, delete_prefix_from_state_dict
from misc.skeleton import ITOPSkeleton, JOINT_COLOR_MAP
from misc.vis import visualize_sample

def create_model(model_name, model_params):
    if model_params is None:
        model_params = {}
    model_class = import_with_str('model', model_name)
    model = model_class(**model_params)
    return model

def chamfer_mask(x, y_hat0, y_hat1, thres_static=0.2, thres_dist=0.1):
    # x: B T N C
    # y_hat0: B 1 J 3
    # y_hat1: B 1 J 3

    x_t01 = x[:, -2:, ...]
    x_t0 = x_t01[:, 0:1, :, :3]  # B 1 N 3
    x_t1 = x_t01[:, 1:2, :, :3]  # B 1 N 3

    chamfer_dist = ChamferDistance()
    dist1, dist2 = chamfer_dist(x_t0[:, 0].to(torch.float), y_hat0[:, 0].to(torch.float))

    mask_dist_pos = (dist2 < thres_dist).unsqueeze(1).unsqueeze(-1).detach()  # B 1 J 1
    mask_dist_neg = (dist2 > thres_static).unsqueeze(1).unsqueeze(-1).detach()  # B 1 J 1
    return mask_dist_pos, mask_dist_neg

class UnsupLoss(torch.nn.Module):
    def __init__(self, thres_static=0.2, thres_dist=0.1):
        super().__init__()
        self.thres_static = thres_static
        self.thres_dist = thres_dist
        self.chamfer_dist = ChamferDistance()

    def forward(self, x, y_hat0, y_hat1):
        # x: B T N C
        # y_hat0: B 1 J 3
        # y_hat1: B 1 J 3

        mask_dist_pos, mask_dist_neg = chamfer_mask(x, y_hat0, y_hat1, self.thres_static, self.thres_dist)
        
        my_hat_dynamic = (y_hat1 - y_hat0) * mask_dist_pos
        my_norm_hat_dynamic = torch.norm(my_hat_dynamic, p=2, dim=-1)
        my_norm_hat_dynamic = my_norm_hat_dynamic[my_norm_hat_dynamic > 0]
        loss_dynamic = torch.relu(0.05 - my_norm_hat_dynamic).mean()
        
        my_hat_static = (y_hat1 - y_hat0) * mask_dist_neg
        my_hat_static = my_hat_static[my_hat_static > 0]
        loss_static = F.mse_loss(my_hat_static, torch.zeros_like(my_hat_static))

        return loss_dynamic, loss_static

def create_loss(loss_name, loss_params):
    if loss_params is None:
        loss_params = {}
    if loss_name == 'UnsupLoss':
        loss = UnsupLoss(**loss_params)
    else:
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

class EMA(nn.Module):
    """ Model Exponential Moving Average V2 from timm"""
    def __init__(self, model, decay=0.9999):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

class MeanTeacherLitModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = create_model(hparams.model_name, hparams.model_params)
        self.model_ema = EMA(self.model, decay=hparams.ema_decay)
        if hparams.checkpoint_path is not None:
            self.load_state_dict(torch.load(hparams.checkpoint_path, map_location=self.device)['state_dict'])
        self.loss = create_loss(hparams.loss_name, hparams.loss_params)
        if hparams.save_when_test or hparams.predict:
            self.results = []

    def _recover_point_cloud(self, x, center, radius):
        # print(radius)
        x[..., :3] = x[..., :3] + center.unsqueeze(-2).unsqueeze(-2)
        x = torch2numpy(x)
        return x
    
    def _recover_skeleton(self, y, center, radius):
        y = y * radius.unsqueeze(-2).unsqueeze(-2) + center.unsqueeze(-2).unsqueeze(-2)
        y = torch2numpy(y)
        return y

    def _recover_data(self, x, y, y_hat, center, radius):
        x = self._recover_point_cloud(x, center, radius)
        y = self._recover_point_cloud(y, center, radius)
        y_hat = self._recover_point_cloud(y_hat, center, radius)
        return x, y, y_hat

    def _vis_pred_gt_keypoints(self, x, y, y_hat):
        sample = x[0][0][:, [0, 2, 1]], y[0][0][:, [0, 2, 1]], y_hat[0][0][:, [0, 2, 1]]
        fig = visualize_sample(sample, edges=ITOPSkeleton.bones, point_size=2, joint_size=25, linewidth=2, padding=0.1)

        # wandb.log({'keypoints': wandb.Image(fig)})
        tensorboard = self.logger.experiment
        tensorboard.add_figure('sample', fig, global_step=self.global_step)
        plt.close(fig)
        plt.clf()

    def _calculate_loss(self, batch):
        if 'sup' in batch:
            x = batch['sup']['point_clouds']
            y = batch['sup']['keypoints']
        else:
            x = batch['point_clouds']
            y = batch['keypoints']
        if self.hparams.model_name in ['P4Transformer', 'P4TransformerAnchor', 'SPiKE']:
            if self.hparams.train_dataset['name'] in ['ReferenceDataset'] and self.hparams.ours:
                batch_sup = batch['sup']
                batch_unsup = batch['unsup']

                if 'both' in self.hparams.train_dataset['params'] and self.hparams.train_dataset['params']['both']:
                    x_sup_li = batch_sup['point_clouds'][..., :3]
                    x_sup_mm = torch.clone(x_sup_li)
                    y_sup = batch_sup['keypoints']
                    xr_sup_mm = batch_sup['ref_point_clouds'][..., :3]
                    yr_sup = batch_sup['ref_keypoints']
                    x_unsup = batch_unsup['point_clouds'][..., :3]

                    B = x_sup_li.shape[0]

                    y_sup_mm_hat = self.model_teacher(x_sup_mm)
                    yr_sup_mm_hat = self.model_teacher(xr_sup_mm)
                    y_sup_li_hat = self.model(x_sup_li)
                    yr_sup_mm_hat2 = self.model(xr_sup_mm)

                    y_hat = y_sup_mm_hat[:B]

                    loss_sup = F.mse_loss(y_sup_mm_hat, y_sup) + F.mse_loss(y_sup_li_hat, y_sup) + \
                                F.mse_loss(yr_sup_mm_hat, yr_sup) + F.mse_loss(yr_sup_mm_hat2, yr_sup)
                else:
                    x_sup = batch_sup['point_clouds']
                    y_sup = batch_sup['keypoints']
                    xr_sup = batch_sup['ref_point_clouds']
                    yr_sup = batch_sup['ref_keypoints']
                    x_unsup = batch_unsup['point_clouds']

                    y_sup_hat = self.model(x_sup)
                    yr_sup_hat = self.model(xr_sup)
                    y_hat = y_sup_hat

                    loss_sup = F.mse_loss(y_sup_hat, y_sup) + F.mse_loss(yr_sup_hat, yr_sup)
                
                x_unsup_t0 = x_unsup[:, :-1, ...]
                x_unsup_t1 = x_unsup[:, 1:, ...]

                self.model_ema.module.eval()
                with torch.no_grad():
                    y_unsup_t0_pseudo = self.model_ema.module(x_unsup_t0)
                    y_unsup_t1_pseudo = self.model_ema.module(x_unsup_t1)

                y_unsup_t0_hat = self.model(x_unsup_t0)
                y_unsup_t1_hat = self.model(x_unsup_t1)

                loss_pseudo = F.mse_loss(y_unsup_t0_hat, y_unsup_t0_pseudo.detach()) + \
                              F.mse_loss(y_unsup_t1_hat, y_unsup_t1_pseudo.detach())

                x_unsup_t0 = x_unsup[:, :-1, ...]
                x_unsup_t1 = x_unsup[:, 1:, ...]
                y_unsup_t0_hat = self.model(x_unsup_t0)
                y_unsup_t1_hat = self.model(x_unsup_t1)

                unsup_loss = self.loss.to(self.device)
                loss_unsup_dynamic, loss_unsup_static = unsup_loss(x_unsup, y_unsup_t0_hat, y_unsup_t1_hat)

                loss = loss_sup + self.hparams.w_dynamic * loss_unsup_dynamic + self.hparams.w_static * loss_unsup_static + self.hparams.w_pseudo * loss_pseudo
                losses = {'loss': loss, 'loss_sup': loss_sup, 'loss_unsup_dynamic': loss_unsup_dynamic, 'loss_unsup_static': loss_unsup_static, 'loss_pseudo': loss_pseudo}

            elif self.hparams.train_dataset['name'] in ['ReferenceDataset']:
                x_ref = batch['ref_point_clouds']
                y_ref = batch['ref_keypoints']

                y_hat = self.model(x)
                y_ref_hat = self.model(x_ref)
                loss = self.loss(y_hat, y) + self.loss(y_ref_hat, y_ref)
                losses = {'loss': loss}
            else:
                y_hat = self.model(x)
                loss = self.loss(y_hat, y)
                losses = {'loss': loss}
        else:
            raise NotImplementedError
        
        return losses, x, y, y_hat

    def _evaluate(self, batch):
        x = batch['point_clouds']
        y = batch['keypoints']
        
        y_hat = self.model(x)
        return x, y, y_hat

    def training_step(self, batch, batch_idx):
        if 'sup' in batch:
            c = batch['sup']['centroid']
            r = batch['sup']['radius']
        else:
            c = batch['centroid']
            r = batch['radius']

        losses, x, y, y_hat = self._calculate_loss(batch)

        if y_hat is None:
            log_dict = {}
        else:
            # print(f"y_hat shape: {y_hat.shape}, y shape: {y.shape}, c shape: {c.shape}, r shape: {r.shape}")
            y = self._recover_skeleton(y, c, r)
            y_hat = self._recover_skeleton(y_hat, c, r)
            mpjpe, pampjpe = calulate_error(y_hat, y)
            log_dict = {'train_mpjpe': mpjpe, 'train_pampjpe': pampjpe}

        for loss_name, loss in losses.items():
            log_dict[f'train_{loss_name}'] = loss

        self.log_dict(log_dict, sync_dist=True)

        return losses['loss']
    
    def validation_step(self, batch, batch_idx):
        c = batch['centroid']
        r = batch['radius']

        x, y, y_hat = self._evaluate(batch)
        x, y, y_hat = self._recover_data(x, y, y_hat, c, r)
        y = y[:, -1:, ...]
        y_hat = y_hat[:, -1:, ...]
        mpjpe, pampjpe = calulate_error(y_hat, y)

        log_dict = {'val_mpjpe': mpjpe, 'val_pampjpe': pampjpe}
        self.log_dict(log_dict, sync_dist=True)

        if batch_idx == 10:
            self._vis_pred_gt_keypoints(x, y, y_hat)
    
    def test_step(self, batch, batch_idx):
        c = batch['centroid']
        r = batch['radius']
        si = batch['sequence_index']

        x, y, y_hat = self._evaluate(batch)
        x, y, y_hat = self._recover_data(x, y, y_hat, c, r)
        y = y[:, -1:, ...]
        y_hat = y_hat[:, -1:, ...]
        mpjpe, pampjpe = calulate_error(y_hat, y)

        log_dict = {'test_mpjpe': mpjpe, 'test_pampjpe': pampjpe}
        self.log_dict(log_dict, sync_dist=True)
        
        if batch_idx == 10:
            self._vis_pred_gt_keypoints(x, y, y_hat)

        if self.hparams.save_when_test:
            result = {'pred': y_hat, 'gt': y, 'input': x, 'seq_idx': torch2numpy(si)}
            self.results.append(result)

    def on_test_end(self):
        if self.hparams.save_when_test:
            split_seqs = []
            split_idxs = []
            last_seq_idx = -1

            for result in self.results:
                for i in range(len(result['pred'])):
                    gt = result['gt'][i]
                    pred = result['pred'][i]
                    input = result['input'][i]
                    seq_idx = int(result['seq_idx'][i].item())
                    if seq_idx != last_seq_idx:
                        last_seq_idx = seq_idx
                        split_seqs.append({'keypoints': [], 'keypoints_pred': [], 'input': []})
                        split_idxs.append(seq_idx)
                    split_seqs[-1]['keypoints'].append(gt)
                    split_seqs[-1]['keypoints_pred'].append(pred)
                    split_seqs[-1]['input'].append(input)

            for i in range(len(split_seqs)):
                split_seqs[i]['keypoints'] = np.array(split_seqs[i]['keypoints'])[:, 0, ...]
                split_seqs[i]['input'] = np.array(split_seqs[i]['input'])[:, 0, ...]
                split_seqs[i]['keypoints_pred'] = np.array(split_seqs[i]['keypoints_pred'])[:, 0, ...]

            with open(os.path.join('logs', self.hparams.exp_name, self.hparams.version, 'results.pkl'), 'wb') as f:
                pickle.dump(split_seqs, f)

        return super().on_test_end()

    def predict_step(self, batch, batch_idx):
        x = batch['point_clouds']
        c = batch['centroid']
        r = batch['radius']
        si = batch['sequence_index']

        y_hat = self.model(x)
        x = self._recover_point_cloud(x, c, r)
        y_hat = self._recover_skeleton(y_hat, c, r)
        
        result = {'pred': y_hat, 'input': x, 'seq_idx': torch2numpy(si)}
        self.results.append(result)

    def on_predict_end(self):
        if self.hparams.save_when_test:
            split_seqs = []
            split_idxs = []
            last_seq_idx = -1

            for result in self.results:
                for i in range(len(result['pred'])):
                    pred = result['pred'][i]
                    input = result['input'][i]
                    seq_idx = int(result['seq_idx'][i].item())
                    if seq_idx != last_seq_idx:
                        last_seq_idx = seq_idx
                        split_seqs.append({'keypoints_pred': [], 'input': []})
                        split_idxs.append(seq_idx)
                    split_seqs[-1]['keypoints_pred'].append(pred)
                    split_seqs[-1]['input'].append(input)

            for i in range(len(split_seqs)):
                split_seqs[i]['input'] = np.array(split_seqs[i]['input'])[:, 0, ...]
                split_seqs[i]['keypoints_pred'] = np.array(split_seqs[i]['keypoints_pred'])[:, 0, ...]

            with open(os.path.join('logs', self.hparams.exp_name, self.hparams.version, 'results.pkl'), 'wb') as f:
                pickle.dump(split_seqs, f)

        return super().on_predict_end()

    def configure_optimizers(self):
        optimizer = create_optimizer(self.hparams.optim_name, self.hparams.optim_params, self.model.parameters())
        scheduler = create_scheduler(self.hparams.sched_name, self.hparams.sched_params, optimizer)
        return [optimizer], [scheduler]

    def on_before_backward(self, loss: torch.Tensor) -> None:
        if self.model_ema:
            self.model_ema.update(self.model)