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

from model.P4Transformer.model import P4Transformer
from model.P4Transformer.model_da import P4TransformerDA
from model.P4Transformer.model_da2 import P4TransformerDA2
from model.P4Transformer.model_da3 import P4TransformerDA3
from model.P4Transformer.model_da4 import P4TransformerDA4
from model.P4Transformer.model_da5 import P4TransformerDA5
from model.debug_model import DebugModel
from model.dg_model import DGModel
from model.dg_model2 import DGModel2
from model.metrics import calulate_error
from loss.pose import GeodesicLoss, SymmetryLoss, ReferenceBoneLoss
from loss.adapt import EntropyLoss, ClassLogitContrastiveLoss
from misc.utils import torch2numpy
from misc.skeleton import SimpleCOCOSkeleton

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
    elif hparams.model_name.lower() == 'p4tda2':
        model = P4TransformerDA2(radius=hparams.radius, nsamples=hparams.nsamples, spatial_stride=hparams.spatial_stride,
                              temporal_kernel_size=hparams.temporal_kernel_size, temporal_stride=hparams.temporal_stride,
                              emb_relu=hparams.emb_relu,
                              dim=hparams.dim, depth=hparams.depth, heads=hparams.heads, dim_head=hparams.dim_head,
                              mlp_dim=hparams.mlp_dim, output_dim=hparams.output_dim, mem_size=hparams.mem_size, features=hparams.features)
    elif hparams.model_name.lower() == 'p4tda3':
        model = P4TransformerDA3(radius=hparams.radius, nsamples=hparams.nsamples, spatial_stride=hparams.spatial_stride,
                              temporal_kernel_size=hparams.temporal_kernel_size, temporal_stride=hparams.temporal_stride,
                              emb_relu=hparams.emb_relu,
                              dim=hparams.dim, depth=hparams.depth, heads=hparams.heads, dim_head=hparams.dim_head,
                              mlp_dim=hparams.mlp_dim, output_dim=hparams.output_dim, features=hparams.features)
    elif hparams.model_name.lower() == 'p4tda4':
        model = P4TransformerDA4(radius=hparams.radius, nsamples=hparams.nsamples, spatial_stride=hparams.spatial_stride,
                              temporal_kernel_size=hparams.temporal_kernel_size, temporal_stride=hparams.temporal_stride,
                              emb_relu=hparams.emb_relu,
                              dim=hparams.dim, depth=hparams.depth, heads=hparams.heads, dim_head=hparams.dim_head,
                              mlp_dim=hparams.mlp_dim, output_dim=hparams.output_dim, features=hparams.features)
    elif hparams.model_name.lower() == 'p4tda5':
        model = P4TransformerDA5(radius=hparams.radius, nsamples=hparams.nsamples, spatial_stride=hparams.spatial_stride,
                              temporal_kernel_size=hparams.temporal_kernel_size, temporal_stride=hparams.temporal_stride,
                              emb_relu=hparams.emb_relu,
                              dim=hparams.dim, depth=hparams.depth, heads=hparams.heads, dim_head=hparams.dim_head,
                              dim_proposal=hparams.dim_proposal, heads_proposal=hparams.heads_proposal, dim_head_proposal=hparams.dim_head_proposal,
                              mlp_dim=hparams.mlp_dim, num_joints=hparams.num_joints, features=hparams.features, num_proposal=hparams.num_proposal)
    elif hparams.model_name.lower() == 'debug':
        model = DebugModel(in_dim=hparams.in_dim, out_dim=hparams.out_dim)
    elif hparams.model_name.lower() == 'dg':
        model = DGModel(graph_layout=hparams.graph_layout, graph_mode=hparams.graph_mode, num_features=hparams.num_features, num_joints=hparams.num_joints,
                        num_layers_point=hparams.num_layers_point, num_layers_joint=hparams.num_layers_joint, dim=hparams.dim, num_heads=hparams.num_heads,
                        dim_feedforward=hparams.dim_feedforward, dropout=hparams.dropout)
    elif hparams.model_name.lower() == 'dg2':
        model = DGModel2(graph_layout=hparams.graph_layout, graph_mode=hparams.graph_mode, num_features=hparams.num_features, num_joints=hparams.num_joints,
                        num_layers_point=hparams.num_layers_point, num_layers_joint=hparams.num_layers_joint, dim=hparams.dim, num_heads=hparams.num_heads,
                        dim_feedforward=hparams.dim_feedforward, dropout=hparams.dropout)
    else:
        raise ValueError(f'Unknown model name: {hparams.model_name}')
    
    return model

def create_losses(hparams):
    losses = {}
    for loss_name in hparams.loss_names:
        if loss_name == 'mse':
            losses['pc'] = nn.MSELoss()
        elif loss_name == 'segment':
            losses['seg'] = nn.CrossEntropyLoss()
        elif loss_name == 'geodesic':
            losses['geo'] = GeodesicLoss()
        elif loss_name == 'symmetry':
            losses['sym'] = SymmetryLoss(left_bones=SimpleCOCOSkeleton.left_bones, right_bones=SimpleCOCOSkeleton.right_bones)
        elif loss_name == 'reference_bone':
            losses['ref'] = ReferenceBoneLoss(bones=SimpleCOCOSkeleton.bones, threshold=hparams.ref_bone_threshold)
        elif loss_name == 'entropy':
            losses['ent'] = EntropyLoss()
        elif loss_name == 'class_logit_contrastive':
            losses['clc'] = ClassLogitContrastiveLoss()
        else:
            raise NotImplementedError
    return losses

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
        self.losses = create_losses(hparams)

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

        # wandb.log({'keypoints': wandb.Image(fig)})
        tensorboard = self.logger.experiment
        tensorboard.add_figure('keypoints', fig, global_step=self.global_step)
        plt.close(fig)
        plt.clf()

    def _calculate_loss(self, batch):
        x = batch['point_clouds']
        y = batch['keypoints']
        if self.hparams.model_name.lower() == 'p4t' or self.hparams.model_name.lower() == 'p4tda3':
            y_hat = self.model(x)
            loss = self.losses['pc'](y_hat, y)
        elif self.hparams.model_name.lower() == 'p4tda':
            if self.hparams.mode == 'train':
                x, s = torch.split(x, [5, 1], dim=-1)
                y_hat, s_hat, l_rec = self.model(x)
                l_pc = self.losses['pc'](y_hat, y)
                # print(s_hat.squeeze().shape, s.squeeze().shape)
                l_seg = self.losses['seg'](s_hat.permute(0, 2, 1, 3), s.squeeze(-1).long())
                # print(l_pc, l_seg, l_rec)
                loss = l_pc + self.hparams.w_seg * l_seg + self.hparams.w_rec * l_rec
            elif self.hparams.mode == 'adapt':
                y_ref = batch['ref_keypoints']
                y_hat, y, l_rec = self.model(x)
                l_ref = self.losses['ref'](y, y_ref)
                l_sym = self.losses['sym'](y)
                loss = self.hparams.w_rec * l_rec + self.hparams.w_ref * l_ref + self.hparams.w_sym * l_sym
            else:
                raise ValueError('mode must be train or adapt!')
        elif self.hparams.model_name.lower() == 'p4tda2':
            if self.hparams.mode == 'train':
                x, s = torch.split(x, [5, 1], dim=-1)
                y_hat, s_hat, l_rec = self.model(x)
                l_pc = self.losses['pc'](y_hat, y)
                l_seg = self.losses['seg'](s_hat.permute(0, 2, 1, 3), s.squeeze(-1).long())
                loss = l_pc + self.hparams.w_seg * l_seg + self.hparams.w_rec * l_rec
            elif self.hparams.mode == 'adapt':
                y_ref = batch['ref_keypoints']
                y_hat, s_hat, l_rec = self.model(x, update_memory=False)
                # l_ref = self.losses['ref'](y, y_ref)
                # l_sym = self.losses['sym'](y)
                # l_ent = self.losses['ent'](s_hat)
                l_clc = self.losses['clc'](s_hat, x[..., :3])
                # print(f'l_rec: {torch2numpy(l_rec)}')
                # print(f'l_rec: {torch2numpy(l_rec)}, l_ent: {torch2numpy(l_ent)}, l_clc: {torch2numpy(l_clc)}')
                # print(f'l_rec: {torch2numpy(l_rec)}, l_ref: {torch2numpy(l_ref)}, l_sym: {torch2numpy(l_sym)}, l_ent: {torch2numpy(l_ent)}, l_clc: {torch2numpy(l_clc)}')
                # loss = self.hparams.w_rec * l_rec + self.hparams.w_ent * l_ent + self.hparams.w_clc * l_clc + self.hparams.w_ref * l_ref + self.hparams.w_sym * l_sym
                loss = self.hparams.w_rec * l_rec + self.hparams.w_clc * l_clc # + self.hparams.w_ref * l_ref
            else:
                raise ValueError('mode must be train or adapt!')
        elif self.hparams.model_name.lower() == 'p4tda4':
            if self.hparams.mode == 'train':
                y_hat, l_cls = self.model.forward_train(x)
                l_pc = self.losses['pc'](y_hat, y)
                loss = l_pc + self.hparams.w_cls * l_cls
            elif self.hparams.mode == 'adapt':
                y_hat, l_cls = self.model.forward_train(x)
                loss = l_cls
            else:
                raise ValueError('mode must be train or adapt!')
        elif self.hparams.model_name.lower() == 'p4tda5':
            if self.hparams.mode == 'train':
                y_hat, l_cls = self.model.forward_train(x, y)
                l_pc = self.losses['pc'](y_hat, y)
                loss = l_pc + self.hparams.w_cls * l_cls
            elif self.hparams.mode == 'adapt':
                c = batch['centroid']
                r = batch['radius']
                s = batch['scale']
                t = batch['translate']
                rot = batch['rotation_matrix']
                x1, x2 = torch.chunk(x, 2, dim=1)
                # y1, y2 = torch.chunk(y, 2, dim=1)
                c1, c2 = torch.chunk(c, 2, dim=1)
                r1, r2 = torch.chunk(r, 2, dim=1)
                s1, s2 = torch.chunk(s, 2, dim=1)
                t1, t2 = torch.chunk(t, 2, dim=1)
                rot1, rot2 = torch.chunk(rot, 2, dim=1)

                y_hat = self.model.forward_adapt(x1)
                y_pseudo = self.model_ema.module.forward_adapt(x2)
                # print(y_hat.shape, t1.shape, rot1.shape, s1.shape, r1.shape, c1.shape)

                y_hat = y_hat - t1.unsqueeze(-2).unsqueeze(-2) # centering
                y_pseudo = y_pseudo - t2.unsqueeze(-2).unsqueeze(-2) # centering

                y_hat = y_hat @ rot1.transpose(-1, -2).unsqueeze(1) # rotation
                y_pseudo = y_pseudo @ rot2.transpose(-1, -2).unsqueeze(1) # rotation

                y_hat = y_hat / s1.unsqueeze(-2).unsqueeze(-2) # scaling
                y_pseudo = y_pseudo / s2.unsqueeze(-2).unsqueeze(-2) # scaling

                y_hat = y_hat * r1.unsqueeze(-2).unsqueeze(-2) + c1.unsqueeze(-2).unsqueeze(-2) # unnormalization
                y_pseudo = y_pseudo * r2.unsqueeze(-2).unsqueeze(-2) + c2.unsqueeze(-2).unsqueeze(-2) # unnormalization
                
                l_cls = self.losses['pc'](y_hat, y_pseudo.detach())

                loss = l_cls
            else:
                raise ValueError('mode must be train or adapt!')
        elif self.hparams.model_name.lower() == 'dg':
            l_pos, y_hat = self.model.forward_train(x, y)
            # print(f'l_rec_pc: {torch2numpy(l_rec_pc)}, l_rec_skl: {torch2numpy(l_rec_skl)}, l_pos: {torch2numpy(l_pos)}')
            loss = l_pos
            # l_rec_pc, l_rec_skl, l_pos, y_hat = self.model.forward_train(x, y)
            # print(f'l_rec_pc: {torch2numpy(l_rec_pc)}, l_rec_skl: {torch2numpy(l_rec_skl)}, l_pos: {torch2numpy(l_pos)}')
            # loss = self.hparams.w_rec_pc * l_rec_pc + self.hparams.w_rec_skl * l_rec_skl + self.hparams.w_pos * l_pos
        elif self.hparams.model_name.lower() == 'dg2':
            l_pos, y_hat = self.model.forward_train(x, y)
            loss = l_pos
        else:
            raise NotImplementedError
        
        return loss, x, y, y_hat

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