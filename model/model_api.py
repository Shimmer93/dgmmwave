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
# import wandb
# import tensorboard

from model.P4Transformer.model import P4Transformer
from model.P4Transformer.model_da import P4TransformerDA
from model.P4Transformer.model_da2 import P4TransformerDA2
from model.P4Transformer.model_da3 import P4TransformerDA3
from model.P4Transformer.model_da4 import P4TransformerDA4
from model.P4Transformer.model_da5 import P4TransformerDA5
from model.P4Transformer.model_da6 import P4TransformerDA6
from model.debug_model import DebugModel
from model.model_poseformer import PoseTransformer
# from model.dg_model import DGModel
# from model.dg_model2 import DGModel2
from model.metrics import calulate_error
from loss.pose import GeodesicLoss, SymmetryLoss, ReferenceBoneLoss
from loss.adapt import EntropyLoss, ClassLogitContrastiveLoss
from loss.mpjpe import mpjpe as mpjpe_mmwave
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
    elif hparams.model_name.lower() == 'p4tda6':
        model = P4TransformerDA6(radius=hparams.radius, nsamples=hparams.nsamples, spatial_stride=hparams.spatial_stride,
                              temporal_kernel_size=hparams.temporal_kernel_size, temporal_stride=hparams.temporal_stride,
                              emb_relu=hparams.emb_relu,
                              dim=hparams.dim, depth=hparams.depth, heads=hparams.heads, dim_head=hparams.dim_head,
                              dim_proposal=hparams.dim_proposal, heads_proposal=hparams.heads_proposal, dim_head_proposal=hparams.dim_head_proposal,
                              mlp_dim=hparams.mlp_dim, num_joints=hparams.num_joints, features=hparams.features, num_proposal=hparams.num_proposal)
    elif hparams.model_name.lower() == 'ptr':
        model = PoseTransformer(num_frame=hparams.number_of_frames, num_joints=hparams.num_joints, num_input_dims = hparams.num_input_dims, in_chans=hparams.in_chans, embed_dim_ratio=hparams.embed_dim_ratio, depth=hparams.depth,
        num_heads=hparams.num_heads, mlp_ratio=hparams.mlp_ratio, qkv_bias=hparams.qkv_bias, qk_scale=None, drop_path_rate=hparams.drop_path_rate)
    elif hparams.model_name.lower() == 'debug':
        model = DebugModel(in_dim=hparams.in_dim, out_dim=hparams.out_dim)
    # elif hparams.model_name.lower() == 'dg':
    #     model = DGModel(graph_layout=hparams.graph_layout, graph_mode=hparams.graph_mode, num_features=hparams.num_features, num_joints=hparams.num_joints,
    #                     num_layers_point=hparams.num_layers_point, num_layers_joint=hparams.num_layers_joint, dim=hparams.dim, num_heads=hparams.num_heads,
    #                     dim_feedforward=hparams.dim_feedforward, dropout=hparams.dropout)
    # elif hparams.model_name.lower() == 'dg2':
    #     model = DGModel2(graph_layout=hparams.graph_layout, graph_mode=hparams.graph_mode, num_features=hparams.num_features, num_joints=hparams.num_joints,
    #                     num_layers_point=hparams.num_layers_point, num_layers_joint=hparams.num_layers_joint, dim=hparams.dim, num_heads=hparams.num_heads,
    #                     dim_feedforward=hparams.dim_feedforward, dropout=hparams.dropout)
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

class LitModel(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = create_model(hparams)
        if hparams.checkpoint_path is not None:
            self.load_state_dict(torch.load(hparams.checkpoint_path, map_location=self.device)['state_dict'])
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
                y_hat, l_cls = self.model.forward_train(x)
                loss = l_cls
            else:
                raise ValueError('mode must be train or adapt!')
        elif self.hparams.model_name.lower() == 'p4tda6':
            if self.hparams.mode == 'train':
                x_ref = batch['ref_point_clouds']
                y_ref = batch['ref_keypoints']
                y_hat, d, l_rec = self.model(x, mode='train')
                _, d_ref, _ = self.model(x_ref, mode='train')
                l_pc = self.losses['pc'](y_hat, y)
                d0 = torch.zeros_like(d, device=d.device)
                l_d = F.binary_cross_entropy_with_logits(d, d0)
                d1 = torch.ones_like(d_ref, device=d_ref.device)
                l_d_ref = F.binary_cross_entropy_with_logits(d_ref, d1)
                loss = l_pc + self.hparams.w_rec * l_rec + self.hparams.w_d * (l_d + l_d_ref)
                losses = {'loss': loss, 'l_pc': l_pc, 'l_rec': l_rec, 'l_d': l_d+l_d_ref}
            elif self.hparams.mode == 'adapt':
                _, d = self.model(x, mode='adapt')
                d0 = torch.zeros_like(d, device=d.device)
                l_d = F.binary_cross_entropy_with_logits(d, d0)
                loss = l_d
                losses = {'loss': loss, 'l_d': l_d}
            elif self.hparams.model_name.lower() == 'ptr':
            #TODO: Implement the loss function of posetransformer
                x = x[:, :, :, :3]
                y_hat = self.model(x)
                y_mod = torch.clone(y)
                y_mod[:, :, 0] = 0
                # loss = self.losses['pc'](y_hat, y)
                loss = mpjpe_mmwave(y_hat, y_mod)
                loss = y_mod.shape[0] * y_mod.shape[1] * loss
                print("The original loss is", loss)
                # The current problems:
                # 1. The input shape of x is [batch_size, receptive_frames = 5, joint_num = 1024, channels], however, if joint_num is set to 1024, it is too big for the model to initialize
                # 2. The output shape of y_hat is [batch_size, 1, joint_num, -1], which is different from y, whose shape is [batch_size, 1, 13, 3]
                # 3. In the validation step afterwards, we also calcualte mpjpe, why we need to calculate it twice?
                # # torch.cuda.empty_cache()
                torch.cuda.empty_cache()
            else:
                raise ValueError('mode must be train or adapt!')
        # elif self.hparams.model_name.lower() == 'dg':
        #     l_pos, y_hat = self.model.forward_train(x, y)
        #     # print(f'l_rec_pc: {torch2numpy(l_rec_pc)}, l_rec_skl: {torch2numpy(l_rec_skl)}, l_pos: {torch2numpy(l_pos)}')
        #     loss = l_pos
        #     # l_rec_pc, l_rec_skl, l_pos, y_hat = self.model.forward_train(x, y)
        #     # print(f'l_rec_pc: {torch2numpy(l_rec_pc)}, l_rec_skl: {torch2numpy(l_rec_skl)}, l_pos: {torch2numpy(l_pos)}')
        #     # loss = self.hparams.w_rec_pc * l_rec_pc + self.hparams.w_rec_skl * l_rec_skl + self.hparams.w_pos * l_pos
        # elif self.hparams.model_name.lower() == 'dg2':
        #     l_pos, y_hat = self.model.forward_train(x, y)
        #     loss = l_pos
        else:
            raise NotImplementedError
        
        return losses, x, y, y_hat


    def training_step(self, batch, batch_idx):
        losses, x, y, y_hat = self._calculate_loss(batch)

        y_hat = torch2numpy(y_hat)
        y = torch2numpy(y)
        mpjpe, pampjpe = calulate_error(y_hat, y)

        log_dict = {'train_mpjpe': mpjpe, 'train_pampjpe': pampjpe}
        for loss_name, loss in losses.items():
            log_dict[f'train_{loss_name}'] = loss

        self.log_dict(log_dict, sync_dist=True)
        # self.log_dict({'train_loss': loss, 'train_mpjpe': mpjpe, 'train_pampjpe': pampjpe}, sync_dist=True)
        return losses['loss']
    
    def validation_step(self, batch, batch_idx):
        c = batch['centroid']
        r = batch['radius']

        _, x, y, y_hat = self._calculate_loss(batch)

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

        _, x, y, y_hat = self._calculate_loss(batch)

        y_hat = y_hat * r.unsqueeze(-2).unsqueeze(-2) + c.unsqueeze(-2).unsqueeze(-2) # unnormalization
        y = y * r.unsqueeze(-2).unsqueeze(-2) + c.unsqueeze(-2).unsqueeze(-2) # unnormalization

        y_hat = torch2numpy(y_hat)
        y = torch2numpy(y)
        mpjpe, pampjpe = calulate_error(y_hat, y)

        self.log_dict({'test_mpjpe': mpjpe, 'test_pampjpe': pampjpe}, sync_dist=True)
        if batch_idx == 0:
            self._vis_pred_gt_keypoints(y_hat, y, torch2numpy(x))

    def configure_optimizers(self):
        optimizer = create_optimizer(self.hparams, self.model.parameters())
        scheduler = create_scheduler(self.hparams, optimizer)
        return [optimizer], [scheduler]