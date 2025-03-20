import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import matplotlib.pyplot as plt
import os
import pickle

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

class AuxLitModel(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = create_model(hparams.model_name, hparams.model_params)
        if hparams.checkpoint_path is not None:
            self.load_state_dict(torch.load(hparams.checkpoint_path, map_location=self.device)['state_dict'], strict=False)
        self.loss = create_loss(hparams.loss_name, hparams.loss_params)
        if hparams.save_when_test:
            self.results = []

    def _add_random_direction_noise_spherical(self, dx, dy, dz, angle_noise_level=0.1):
        # dx: B
        # dy: B
        # dz: B
        # angle_noise_level: scalar

        # convert to spherical coordinates
        azimuth = torch.atan2(dy, dx)
        elevation = torch.asin(dz)

        # add noise
        d_azimuth = torch.randn_like(azimuth) * angle_noise_level
        d_elevation = torch.randn_like(elevation) * angle_noise_level
        magnitude = torch.sqrt(d_azimuth ** 2 + d_elevation ** 2)
        azimuth += d_azimuth
        elevation += d_elevation

        # print(dx.shape, dy.shape, dz.shape, azimuth.shape, elevation.shape, magnitude.shape)

        # convert back to cartesian coordinates
        dx = torch.cos(elevation) * torch.cos(azimuth)
        dy = torch.cos(elevation) * torch.sin(azimuth)
        dz = torch.sin(elevation)

        return dx, dy, dz, magnitude

    def _add_random_motion_noise(self, dx, dy, dz, noise_level=0.1):
        # dx: B
        # dy: B
        # dz: B
        # noise_level: scalar

        noise_x = torch.randn_like(dx) * noise_level
        noise_y = torch.randn_like(dy) * noise_level
        noise_z = torch.randn_like(dz) * noise_level
        magnitude = torch.sqrt(noise_x ** 2 + noise_y ** 2 + noise_z ** 2)
        dx += noise_x
        dy += noise_y
        dz += noise_z

        return dx, dy, dz, magnitude

    def _calculate_loss(self, batch):
        if self.hparams.model_name == 'PlausibilityRegressor':
            if self.hparams.model_variant == 'direction':
                x_pos = batch['bone_dirs'].squeeze()
                x_neg = x_pos.clone()
                x_neg[..., 0], x_neg[..., 1], x_neg[..., 2], magnitude = \
                    self._add_random_direction_noise_spherical(x_neg[..., 0], x_neg[..., 1], x_neg[..., 2], angle_noise_level = 0.5)
            else:
                x_pos = batch['bone_motions'].squeeze()
                x_neg = x_pos.clone()
                x_neg[..., 0], x_neg[..., 1], x_neg[..., 2], magnitude = \
                    self._add_random_motion_noise(x_neg[..., 0], x_neg[..., 1], x_neg[..., 2], noise_level = 0.5)
            x = torch.cat([x_pos, x_neg], dim=0)
            y = torch.cat([torch.ones_like(magnitude) * magnitude, torch.zeros_like(magnitude)], dim=0).to(x.device)
            y_hat = self.model(x)
            loss = self.loss(y_hat, y)

            return {'loss': loss}
        else:
            raise NotImplementedError   

    def training_step(self, batch, batch_idx):
        losses = self._calculate_loss(batch)

        log_dict = {}
        for loss_name, loss in losses.items():
            log_dict[f'train_{loss_name}'] = loss

        self.log_dict(log_dict, sync_dist=True)

        return losses['loss']

    def validation_step(self, batch, batch_idx):
        losses = self._calculate_loss(batch)

        log_dict = {}
        for loss_name, loss in losses.items():
            log_dict[f'val_{loss_name}'] = loss

        self.log_dict(log_dict, sync_dist=True)

        return losses['loss']

    def test_step(self, batch, batch_idx):
        losses = self._calculate_loss(batch)

        log_dict = {}
        for loss_name, loss in losses.items():
            log_dict[f'test_{loss_name}'] = loss

        self.log_dict(log_dict, sync_dist=True)

        if self.hparams.save_when_test:
            self.results.append(log_dict)

        return losses['loss']

    def configure_optimizers(self):
        optimizer = create_optimizer(self.hparams.optim_name, self.hparams.optim_params, self.model.parameters())
        scheduler = create_scheduler(self.hparams.sched_name, self.hparams.sched_params, optimizer)
        return [optimizer], [scheduler]