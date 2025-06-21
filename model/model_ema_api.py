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
from copy import deepcopy

from model.metrics import calulate_error
from model.chamfer_distance import ChamferDistance
from loss.mpjpe import mpjpe as mpjpe_mmwave
from misc.utils import torch2numpy, import_with_str, delete_prefix_from_state_dict
from misc.skeleton import ITOPSkeleton, JOINT_COLOR_MAP

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
        self.model = create_model(hparams)
        if hparams.checkpoint_path is not None:
            self.load_state_dict(torch.load(hparams.checkpoint_path, map_location=self.device)['state_dict'])
        self.model_ema = EMA(self.model, decay=hparams.ema_decay)
        self.loss = create_loss(hparams.loss_name, hparams.loss_params)
        if hparams.save_when_test:
            self.results = []

    def _recover_point_cloud(self, x, center, radius):
        x[..., :3] = x[..., :3] * radius.unsqueeze(-2).unsqueeze(-2) + center.unsqueeze(-2).unsqueeze(-2)
        x = torch2numpy(x)
        return x
    
    def _recover_skeleton(self, y, center, radius):
        y = y * radius.unsqueeze(-2).unsqueeze(-2) + center.unsqueeze(-2).unsqueeze(-2)
        y = torch2numpy(y)
        return y

    def _recover_data(self, x, y, y_hat, center, radius):
        x = self._recover_point_cloud(x, center, radius)
        y = self._recover_skeleton(y, center, radius)
        y_hat = self._recover_skeleton(y_hat, center, radius)
        return x, y, y_hat

    def _get_bounds(self, data):
        all_ps = data[..., :3].reshape(-1, 3)
        min_x, max_x = np.min(all_ps[:, 0]), np.max(all_ps[:, 0])
        min_y, max_y = np.min(all_ps[:, 1]), np.max(all_ps[:, 1])
        min_z, max_z = np.min(all_ps[:, 2]), np.max(all_ps[:, 2])
        return min_x, max_x, min_y, max_y, min_z, max_z
    
    def _set_3d_ax_limits(self, ax, bounds):
        min_x, max_x, min_y, max_y, min_z, max_z = bounds
        range_x = max_x - min_x
        range_y = max_y - min_y
        range_z = max_z - min_z

        ax.set_box_aspect([range_x, range_y, range_z])
        ax.set_xlim(min_x - range_x * 0.1, max_x + range_x * 0.1)
        ax.set_zlim(min_y - range_y * 0.1, max_y + range_y * 0.1)
        ax.set_ylim(min_z - range_z * 0.1, max_z + range_z * 0.1)

    def _vis_pred_gt_keypoints(self, x, y, y_hat):
        fig = plt.figure()
        ax_pred_xy = fig.add_subplot(331)
        ax_pred_xz = fig.add_subplot(332)
        ax_pred_yz = fig.add_subplot(333)
        ax_gt_xy = fig.add_subplot(334)
        ax_gt_xz = fig.add_subplot(335)
        ax_gt_yz = fig.add_subplot(336)
        ax_pred_3d = fig.add_subplot(337, projection='3d')
        ax_gt_3d = fig.add_subplot(338, projection='3d')
        ax_pc = fig.add_subplot(339, projection='3d')

        ax_pred_xy.set_aspect('equal')
        ax_pred_xz.set_aspect('equal')
        ax_pred_yz.set_aspect('equal')
        ax_gt_xy.set_aspect('equal')
        ax_gt_xz.set_aspect('equal')
        ax_gt_yz.set_aspect('equal')
        
        y_hat_bounds = self._get_bounds(y_hat[0, 0])
        y_bounds = self._get_bounds(y[0, 0])
        x_bounds = self._get_bounds(x[0, 0])

        self._set_3d_ax_limits(ax_pred_3d, y_hat_bounds)
        self._set_3d_ax_limits(ax_gt_3d, y_bounds)
        self._set_3d_ax_limits(ax_pc, x_bounds)

        ax_pred_xy.set_title('Predicted XY')
        ax_pred_xz.set_title('Predicted XZ')
        ax_pred_yz.set_title('Predicted YZ')
        ax_gt_xy.set_title('GT XY')
        ax_gt_xz.set_title('GT XZ')
        ax_gt_yz.set_title('GT YZ')
        ax_pred_3d.set_title('Predicted 3D')
        ax_gt_3d.set_title('GT 3D')
        ax_pc.set_title('Point Cloud')

        for i, (p_hat, p) in enumerate(zip(y_hat[0, 0], y[0, 0])):
            color = JOINT_COLOR_MAP[i]
            ax_pred_xy.plot(p_hat[0], p_hat[1], color=color, marker='o')
            ax_pred_xz.plot(p_hat[0], p_hat[2], color=color, marker='o')
            ax_pred_yz.plot(p_hat[1], p_hat[2], color=color, marker='o')
            ax_gt_xy.plot(p[0], p[1], color=color, marker='o')
            ax_gt_xz.plot(p[0], p[2], color=color, marker='o')
            ax_gt_yz.plot(p[1], p[2], color=color, marker='o')
            ax_pred_3d.scatter(p_hat[0], p_hat[2], p_hat[1], color=color, marker='o')
            ax_gt_3d.scatter(p[0], p[2], p[1], color=color, marker='o')
        ax_pc.scatter(x[0, 0, :, 0], x[0, 0, :, 2], x[0, 0, :, 1], 'g', marker='o')

        fig.tight_layout()

        # wandb.log({'keypoints': wandb.Image(fig)})
        tensorboard = self.logger.experiment
        tensorboard.add_figure('keypoints', fig, global_step=self.global_step)
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
                    x_sup0 = batch_sup['point_clouds'][..., :3]
                    x_sup1 = batch_sup['point_clouds_trans'][..., :-1, :, :3]
                    y_sup = batch_sup['keypoints']
                    xr_sup0 = batch_sup['ref_point_clouds'][..., :3]
                    yr_sup = batch_sup['ref_keypoints']
                    x_unsup = batch_unsup['point_clouds'][..., :3]

                    B = x_sup0.shape[0]

                    x_sup0_ = torch.cat((x_sup0, xr_sup0), dim=0)
                    y_sup_ = torch.cat((y_sup, yr_sup), dim=0)

                    if torch.rand(1).item() < 0.5:
                        perm = torch.randperm(x_sup0_.shape[-2])
                        num2exchange = torch.randint(0, x_sup0_.shape[-2], (1,)).item()
                        x_sup0__ = torch.cat((x_sup0_[:B, ..., perm[:num2exchange], :3], x_sup1[:B, ..., perm[num2exchange:], :3]), dim=-2)
                        x_sup1[:B] = torch.cat((x_sup1[:B, ..., perm[:num2exchange], :3], x_sup0_[:B, ..., perm[num2exchange:], :3]), dim=-2)
                        x_sup0_[:B] = x_sup0__

                    y_sup0_hat, f0 = self.model(x_sup0_)
                    y_sup1_hat, f1 = self.model(x_sup1)

                    y_hat = y_sup0_hat[:B]

                    loss_sup = F.mse_loss(y_sup0_hat, y_sup_) + F.mse_loss(y_sup1_hat, y_sup)
                    loss_con = F.mse_loss(f0[:B], f1[:B])
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
                    loss_con = 0.0
                
                x_unsup_t0 = x_unsup[:, :-1, ...]
                x_unsup_t1 = x_unsup[:, 1:, ...]

                self.model_ema.module.eval()
                with torch.no_grad():
                    y_unsup_t0_pseudo, _ = self.model_ema.module(x_unsup_t0)
                    y_unsup_t1_pseudo, _ = self.model_ema.module(x_unsup_t1)

                y_unsup_t0_hat, _ = self.model(x_unsup_t0)
                y_unsup_t1_hat, _ = self.model(x_unsup_t1)

                mask_dist_pos, mask_dist_neg = chamfer_mask(x_unsup, y_unsup_t0_pseudo, y_unsup_t1_pseudo)
                y_unsup_flow = torch.norm(y_unsup_t1_pseudo - y_unsup_t0_pseudo, p=2, dim=-1).unsqueeze(-1)
                mask_flow_pos = (y_unsup_flow > 0.1)
                mask_flow_neg = (y_unsup_flow < 0.05)
                mask_pos = mask_dist_pos & mask_flow_pos
                mask_neg = mask_dist_neg & mask_flow_neg
                mask = mask_pos | mask_neg

                loss_pseudo = F.mse_loss(y_unsup_t0_pseudo * mask, y_unsup_t0_hat * mask) + F.mse_loss(y_unsup_t1_pseudo * mask, y_unsup_t1_hat * mask)

                unsup_loss = self.loss.to(self.device)
                loss_unsup_dynamic, loss_unsup_static = unsup_loss(x_unsup, y_unsup_t0_hat, y_unsup_t1_hat)

                loss = loss_sup + self.hparams.w_dynamic * loss_unsup_dynamic + self.hparams.w_static * loss_unsup_static + self.hparams.w_pseudo * loss_pseudo
                losses = {'loss': loss, 'loss_sup': loss_sup, 'loss_unsup_dynamic': loss_unsup_dynamic, 'loss_unsup_static': loss_unsup_static, 'loss_pseudo': loss_pseudo}

            elif self.hparams.train_dataset['name'] in ['ReferenceDataset']:
                x_ref = batch['ref_point_clouds']
                y_ref = batch['ref_keypoints']

                if 'both' in self.hparams.train_dataset['params'] and self.hparams.train_dataset['params']['both']:
                    x_ = batch['point_clouds_trans'][..., :-1, :, :3]
                    if torch.rand(1).item() < 0.5:
                        perm = torch.randperm(x_.shape[-2])
                        num2exchange = torch.randint(0, x_.shape[-2], (1,)).item()
                        x__ = torch.cat((x[..., perm[:num2exchange], :3], x_[..., perm[num2exchange:], :3]), dim=-2)
                        x_ = torch.cat((x_[..., perm[:num2exchange], :3], x[..., perm[num2exchange:], :3]), dim=-2)
                        x = x__
                    y_hat = self.model(x)
                    y_hat_ = self.model(x_)
                    y_ref_hat = self.model(x_ref)
                    loss = self.loss(y_hat, y) + self.loss(y_hat_, y) + self.loss(y_ref_hat, y_ref)
                else:
                    y_hat = self.model(x)
                    y_ref_hat = self.model(x_ref)
                    loss = self.loss(y_hat, y) + self.loss(y_ref_hat, y_ref)
                losses = {'loss': loss}
            elif self.hparams.train_dataset['name'] in ['ReferenceOneToOneDataset']:
                x_ref = batch['ref_point_clouds']
                y_ref = batch['ref_keypoints']
                if torch.rand(1).item() < 0.5:
                    perm = torch.randperm(x_ref.shape[-2])
                    num2exchange = torch.randint(0, x_ref.shape[-2], (1,)).item()
                    x_ = torch.cat((x[..., perm[:num2exchange], :3], x_ref[..., perm[num2exchange:], :3]), dim=-2)
                    x_ref_ = torch.cat((x_ref[..., perm[:num2exchange], :3], x[..., perm[num2exchange:], :3]), dim=-2)
                    x = x_
                    x_ref = x_ref_
                y_hat = self.model(x)
                y_ref_hat = self.model(x_ref)
                loss = self.loss(y_hat, y) + self.loss(y_ref_hat, y_ref)
                losses = {'loss': loss}
            else:
                if 'both' in self.hparams.train_dataset['params'] and self.hparams.train_dataset['params']['both']:
                    x_ = batch['point_clouds_trans']
                    if torch.rand(1).item() < 0.5:
                        perm = torch.randperm(x_.shape[-2])
                        num2exchange = torch.randint(0, x_.shape[-2], (1,)).item()
                        x__ = torch.cat((x[..., perm[:num2exchange], :3], x_[..., perm[num2exchange:], :3]), dim=-2)
                        x_ = torch.cat((x_[..., perm[:num2exchange], :3], x[..., perm[num2exchange:], :3]), dim=-2)
                        x = x__
                    y_hat = self.model(x)
                    y_hat_ = self.model(x_)
                    loss = self.loss(y_hat, y) + self.loss(y_hat_, y)
                else:
                    y_hat = self.model(x)
                    loss = self.loss(y_hat, y)
                losses = {'loss': loss}
        else:
            raise NotImplementedError
        
        return losses, x, y, y_hat

    def _calculate_loss_eval(self, batch):
        x = batch['point_clouds']
        y = batch['keypoints']
        if self.hparams.model_name.lower() == 'p4tda5':
            x, _ = torch.chunk(x, 2, dim=1)
            y, _ = torch.chunk(y, 2, dim=1)
            y_hat, _ = self.model.forward_train(x, y)
        else:
            raise NotImplementedError
        
        return x, y, y_hat

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
        c, _ = torch.chunk(c, 2, dim=1)
        r, _ = torch.chunk(r, 2, dim=1)

        x, y, y_hat = self._calculate_loss_eval(batch)

        y_hat = y_hat * r.unsqueeze(-2).unsqueeze(-2) + c.unsqueeze(-2).unsqueeze(-2) # unnormalization
        y = y * r.unsqueeze(-2).unsqueeze(-2) + c.unsqueeze(-2).unsqueeze(-2) # unnormalization

        y_hat = torch2numpy(y_hat)
        y = torch2numpy(y)
        mpjpe, pampjpe = calulate_error(y_hat, y)

        self.log_dict({'val_mpjpe': mpjpe, 'val_pampjpe': pampjpe}, sync_dist=True)
        if batch_idx == 0:
            self._vis_pred_gt_keypoints(y_hat, y, torch2numpy(x))
    
    def test_step(self, batch, batch_idx):
        c = batch['centroid']
        r = batch['radius']
        c, _ = torch.chunk(c, 2, dim=1)
        r, _ = torch.chunk(r, 2, dim=1)

        x, y, y_hat = self._calculate_loss_eval(batch)

        y_hat = y_hat * r.unsqueeze(-2).unsqueeze(-2) + c.unsqueeze(-2).unsqueeze(-2) # unnormalization
        y = y * r.unsqueeze(-2).unsqueeze(-2) + c.unsqueeze(-2).unsqueeze(-2) # unnormalization

        y_hat = torch2numpy(y_hat)
        y = torch2numpy(y)
        mpjpe, pampjpe = calulate_error(y_hat, y)

        self.log_dict({'test_mpjpe': mpjpe, 'test_pampjpe': pampjpe}, sync_dist=True)
        if batch_idx == 0:
            self._vis_pred_gt_keypoints(y_hat, y, torch2numpy(x))
    
    # def validation_step(self, batch, batch_idx):
    #     c = batch['centroid']
    #     r = batch['radius']

    #     loss, x, y, y_hat = self._calculate_loss(batch)

    #     y_hat = torch2numpy(y_hat)
    #     y1, _ = torch.chunk(y, 2, dim=1)
    #     # c1, _ = torch.chunk(c, 2, dim=1)
    #     # r1, _ = torch.chunk(r, 2, dim=1)

    #     y1 = torch2numpy(y1)
    #     # c1 = torch2numpy(c1)
    #     # r1 = torch2numpy(r1)
    #     # print(y_hat.shape, y1.shape)
    #     # y_hat = y_hat * r1[:, np.newaxis, np.newaxis, ...] + c1[:, np.newaxis, np.newaxis, ...]
    #     # y1 = y1 * r1[:, np.newaxis, np.newaxis, ...] + c1[:, np.newaxis, np.newaxis, ...]
    #     mpjpe, pampjpe = calulate_error(y_hat, y1)

    #     self.log_dict({'val_loss': loss, 'val_mpjpe': mpjpe, 'val_pampjpe': pampjpe}, sync_dist=True)
    #     if batch_idx == 0:
    #         self._vis_pred_gt_keypoints(y_hat, y1, torch2numpy(x))
    #     return loss
    
    # def test_step(self, batch, batch_idx):
    #     c = batch['centroid']
    #     r = batch['radius']

    #     loss, x, y, y_hat = self._calculate_loss(batch)

    #     y_hat = torch2numpy(y_hat)
    #     y1, _ = torch.chunk(y, 2, dim=1)
    #     # c1, _ = torch.chunk(c, 2, dim=1)
    #     # r1, _ = torch.chunk(r, 2, dim=1)

    #     y1 = torch2numpy(y1)
    #     # c1 = torch2numpy(c1)
    #     # r1 = torch2numpy(r1)
    #     # y_hat = y_hat * r1[:, np.newaxis, np.newaxis, ...] + c1[:, np.newaxis, np.newaxis, ...]
    #     # y1 = y1 * r1[:, np.newaxis, np.newaxis, ...] + c1[:, np.newaxis, np.newaxis, ...]
    #     mpjpe, pampjpe = calulate_error(y_hat, y1)

    #     self.log_dict({'test_loss': loss, 'test_mpjpe': mpjpe, 'test_pampjpe': pampjpe}, sync_dist=True)
    #     if batch_idx == 0:
    #         self._vis_pred_gt_keypoints(y_hat, y1, torch2numpy(x))
    #     return loss

    def configure_optimizers(self):
        optimizer = create_optimizer(self.hparams, self.model.parameters())
        scheduler = create_scheduler(self.hparams, optimizer)
        return [optimizer], [scheduler]

    def shared_step(self, x, y, metric):
        y_hat = self.model(x) if self.training or self.model_ema is None else self.model_ema.module(x)
        loss = self.criterion(y_hat, y)
        self.log_dict(metric(y_hat, y), prog_bar=True)
        return loss

    def on_before_backward(self, loss: torch.Tensor) -> None:
        if self.model_ema:
            self.model_ema.update(self.model)