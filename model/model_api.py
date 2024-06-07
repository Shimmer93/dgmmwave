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
import os
from collections import OrderedDict
import wandb

from model.P4Transformer.model import P4Transformer
from model.P4Transformer.model_da import P4TransformerDA
from model.debug_model import DebugModel
from model.metrics import calulate_error
from loss.pose import GeodesicLoss
from misc.utils import torch2numpy

def create_model(hparams):
    if hparams.model_name.lower() == 'p4t':
        model = P4Transformer(radius=hparams.radius, nsamples=hparams.nsamples, spatial_stride=hparams.spatial_stride,
                              temporal_kernel_size=hparams.temporal_kernel_size, temporal_stride=hparams.temporal_stride,
                              emb_relu=hparams.emb_relu,
                              dim=hparams.dim, depth=hparams.depth, heads=hparams.heads, dim_head=hparams.dim_head,
                              mlp_dim=hparams.mlp_dim, output_dim=hparams.output_dim, features=hparams.features)
    elif hparams.model_name.lower() == 'p4tda':
        model = P4TransformerDA(radius=hparams.radius, nsamples=hparams.nsamples, spatial_stride=hparams.spatial_stride,
                              temporal_kernel_size=hparams.temporal_kernel_size, temporal_stride=hparams.temporal_stride,
                              emb_relu=hparams.emb_relu,
                              dim=hparams.dim, depth=hparams.depth, heads=hparams.heads, dim_head=hparams.dim_head,
                              mlp_dim=hparams.mlp_dim, output_dim=hparams.output_dim, mem_size=hparams.mem_size, features=hparams.features)
    elif hparams.model_name.lower() == 'debug':
        model = DebugModel(in_dim=hparams.in_dim, out_dim=hparams.out_dim)
    else:
        raise ValueError(f'Unknown model name: {hparams.model_name}')
    
    return model

def create_loss(hparams):
    if hparams.loss_name == 'mse':
        return nn.MSELoss()
    elif hparams.loss_name == 'geodesic':
        return GeodesicLoss()
    else:
        raise NotImplementedError

def create_optimizer(hparams, mparams):
    if hparams.optim_name == 'adam':
        return optim.Adam(mparams, lr=hparams.lr, weight_decay=hparams.weight_decay)
    elif hparams.optim_name == 'adamw':
        return optim.AdamW(mparams, lr=hparams.lr, weight_decay=hparams.weight_decay)
    elif hparams.optim_name == 'sgd':
        return optim.SGD(mparams, lr=hparams.lr, momentum=hparams.momentum)
    else:
        raise NotImplementedError
    
def create_scheduler(hparams, optimizer):
    if hparams.sched_name == 'cosine':
        return LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=hparams.warmup_epochs, 
                max_epochs=hparams.epochs, warmup_start_lr=hparams.warmup_lr, eta_min=hparams.min_lr)
    elif hparams.sched_name == 'step':
        return sched.MultiStepLR(optimizer, milestones=hparams.milestones, gamma=hparams.gamma)
    elif hparams.sched_name == 'plateau':
        return sched.ReduceLROnPlateau(optimizer, patience=hparams.patience, factor=hparams.factor, 
                min_lr=hparams.min_lr)
    else:
        raise NotImplementedError

class LitModel(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = create_model(hparams)
        self.loss = create_loss(hparams)

    def _vis_pred_gt_keypoints(self, y_hat, y, x):
        fig = plt.figure()
        ax_pred = fig.add_subplot(231)
        ax_gt = fig.add_subplot(232)
        ax_2d = fig.add_subplot(233)
        ax_3d = fig.add_subplot(234, projection='3d')
        ax_pc = fig.add_subplot(235)

        ax_pred.set_aspect('equal')
        ax_gt.set_aspect('equal')
        ax_2d.set_aspect('equal')
        ax_3d.set_aspect('equal')
        ax_pc.set_aspect('equal')

        ax_pred.set_title('Predicted')
        ax_gt.set_title('Ground Truth')
        ax_2d.set_title('2D')
        ax_3d.set_title('3D')
        ax_pc.set_title('Point Cloud')

        for p_hat, p in zip(y_hat[0, 0], y[0, 0]):
            random_color = np.random.rand(3).tolist()
            ax_pred.plot(p_hat[0], p_hat[1], color=random_color, marker='o')
            ax_gt.plot(p[0], p[1], color=random_color, marker='o')
        ax_2d.plot(y_hat[0, 0, :, 0], y_hat[0, 0, :, 1], 'bo')
        ax_2d.plot(y[0, 0, :, 0], y[0, 0, :, 1], 'ro')
        ax_3d.scatter(y_hat[0, 0, :, 0], y_hat[0, 0, :, 1], y_hat[0, 0, :, 2], 'b')
        ax_3d.scatter(y[0, 0, :, 0], y[0, 0, :, 1], y[0, 0, :, 2], 'r')
        ax_pc.scatter(x[0, 0, :, 0], x[0, 0, :, 1], x[0, 0, :, 2], 'g')

        wandb.log({'keypoints': wandb.Image(fig)})
        plt.close(fig)

    def _calculate_loss(self, batch):
        x = batch['point_clouds']
        y = batch['keypoints']
        if self.hparams.model_name.lower() == 'p4t':
            y_hat = self.model(x)
            loss = self.loss(y_hat, y)
        elif self.hparams.model_name.lower() == 'p4tda':
            if self.hparams.mode == 'train':
                x, s = torch.split(x, [5, 1], dim=-1)
                y_hat, s_hat, l_rec = self.model(x)
                l_pc = self.loss(y_hat, y)
                # print(s_hat.squeeze().shape, s.squeeze().shape)
                l_seg = F.cross_entropy(s_hat.permute(0, 2, 1, 3), s.squeeze(-1).long())
                # print(l_pc, l_seg, l_rec)
                loss = l_pc + self.hparams.w_seg * l_seg + self.hparams.w_rec * l_rec
            elif self.hparams.mode == 'adapt':
                y_hat, _, l_rec = self.model(x)
                loss = l_rec
            else:
                raise ValueError('mode must be train or adapt!')
        else:
            raise NotImplementedError
        
        return loss, x, y, y_hat


    def training_step(self, batch, batch_idx):
        loss, x, y, y_hat = self._calculate_loss(batch)

        y_hat = torch2numpy(y_hat)
        y = torch2numpy(y)
        mpjpe, pampjpe = calulate_error(y_hat, y)

        self.log_dict({'train_loss': loss, 'train_mpjpe': mpjpe, 'train_pampjpe': pampjpe}, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        c = batch['centroid']
        r = batch['radius']

        loss, x, y, y_hat = self._calculate_loss(batch)

        y_hat = torch2numpy(y_hat)
        y = torch2numpy(y)
        c = torch2numpy(c)
        r = torch2numpy(r)
        y_hat = y_hat * r[..., np.newaxis, np.newaxis, np.newaxis] + c[:, np.newaxis, np.newaxis, ...]
        y = y * r[..., np.newaxis, np.newaxis, np.newaxis] + c[:, np.newaxis, np.newaxis, ...]
        mpjpe, pampjpe = calulate_error(y_hat, y)

        self.log_dict({'val_loss': loss, 'val_mpjpe': mpjpe, 'val_pampjpe': pampjpe}, sync_dist=True)
        if batch_idx == 0:
            self._vis_pred_gt_keypoints(y_hat, y, torch2numpy(x))
        return loss
    
    def test_step(self, batch, batch_idx):
        c = batch['centroid']
        r = batch['radius']

        loss, x, y, y_hat = self._calculate_loss(batch)

        y_hat = torch2numpy(y_hat)
        y = torch2numpy(y)
        c = torch2numpy(c)
        r = torch2numpy(r)
        y_hat = y_hat * r[..., np.newaxis, np.newaxis, np.newaxis] + c[:, np.newaxis, np.newaxis, ...]
        y = y * r[..., np.newaxis, np.newaxis, np.newaxis] + c[:, np.newaxis, np.newaxis, ...]
        mpjpe, pampjpe = calulate_error(y_hat, y)

        self.log_dict({'test_loss': loss, 'test_mpjpe': mpjpe, 'test_pampjpe': pampjpe}, sync_dist=True)
        if batch_idx == 0:
            self._vis_pred_gt_keypoints(y_hat, y, torch2numpy(x))
        return loss

    def configure_optimizers(self):
        optimizer = create_optimizer(self.hparams, self.model.parameters())
        scheduler = create_scheduler(self.hparams, optimizer)
        return [optimizer], [scheduler]