from typing import Any
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

# from model.P4Transformer.model import P4Transformer
from model.debug_model import DebugModel

def create_model(hparams):
    # if hparams.model_name.lower() == 'p4t':
    #     model = P4Transformer(radius=hparams.radius, nsamples=hparams.nsamples, spatial_stride=hparams.spatial_stride,
    #                           temporal_kernel_size=hparams.temporal_kernel_size, temporal_stride=hparams.temporal_stride,
    #                           emb_relu=hparams.emb_relu,
    #                           dim=hparams.dim, depth=hparams.depth, heads=hparams.heads, dim_head=hparams.dim_head,
    #                           mlp_dim=hparams.mlp_dim, output_dim=hparams.output_dim, features=hparams.features)
    if hparams.model_name.lower() == 'debug':
        model = DebugModel(in_dim=hparams.in_dim, out_dim=hparams.out_dim)
    else:
        raise ValueError(f'Unknown model name: {hparams.model_name}')
    
    return model

def create_loss(hparams):
    if hparams.loss_name == 'mse':
        return nn.MSELoss()
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

    def _vis_pred_gt_keypoints(self, y, y_hat):
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(y[0, 0, :, 0].detach().cpu().numpy(), y[0, 0, :, 1].detach().cpu().numpy(), 'ro')
        ax[0].set_title('Predicted')
        ax[1].plot(y_hat[0, 0, :, 0].detach().cpu().numpy(), y_hat[0, 0, :, 1].detach().cpu().numpy(), 'bo')
        ax[1].set_title('Ground Truth')
        wandb.log({'keypoints': wandb.Image(fig)})
        plt.close(fig)

    def training_step(self, batch, batch_idx):
        x = batch['point_clouds']
        y = batch['keypoints']
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['point_clouds']
        y = batch['keypoints']
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        self._vis_pred_gt_keypoints(y, y_hat)
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch['point_clouds']
        y = batch['keypoints']
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)
        self._vis_pred_gt_keypoints(y, y_hat)
        return loss

    def configure_optimizers(self):
        optimizer = create_optimizer(self.hparams, self.model.parameters())
        scheduler = create_scheduler(self.hparams, optimizer)
        return [optimizer], [scheduler]