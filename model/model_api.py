import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import matplotlib.pyplot as plt
import os
import pickle
# from pytorch3d.loss import chamfer_distance

# import wandb
# import tensorboard

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

class UnsupLoss(torch.nn.Module):
    def __init__(self, loss_params):
        super().__init__()
        self.model = create_model('P4TransformerMotion', loss_params['model_params'])
        self.model.load_state_dict(torch.load(loss_params['model_checkpoint'], map_location='cpu')['state_dict'], strict=False)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.chamfer_dist = ChamferDistance()

    def forward(self, x, y_hat0, y_hat1):
        x_t01 = x[:, -2:, ...]
        m_hat = self.model(x_t01)
        x_t0 = x_t01[:, 0:1, :, :3]  # B 1 N 3
        x_t1 = x_t01[:, 1:2, :, :3]  # B 1 N 3
        x_t1_hat = x_t0 + m_hat  # B 1 N 3
        dist1, dist2 = self.chamfer_dist(x_t1_hat[:, 0, :, :3], x_t1[:, 0, :, :3])
        loss_chamfer = dist1.mean(dim=-1) + dist2.mean(dim=-1)
        mask_chamfer = (loss_chamfer < 0.1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).detach()

        dists = torch.norm(y_hat0[:, 0].unsqueeze(2) - x_t0[:, 0].
                           unsqueeze(1), dim=-1) # B J N
        dists2 = torch.norm(y_hat0[:, 0].unsqueeze(2) - torch.cat([x_t0[:, 0], x_t1[:, 0]], dim=1).
                           unsqueeze(1), dim=-1) # B J N
        min_dists, min_indices = torch.min(dists, dim=-1) # B J, B J
        min_dists2, min_indices2 = torch.min(dists2, dim=-1) # B J, B J
        mask_dist = (min_dists < 0.05).unsqueeze(1).unsqueeze(-1).detach() # B 1 J 1
        mask_dist_neg = (min_dists2 > 0.2).unsqueeze(1).unsqueeze(-1).detach() # B 1 J 1
        min_indices_expanded = min_indices.unsqueeze(-1).expand(-1, -1, y_hat0.shape[-1]) # B J 3
        nearest_m = m_hat[:, 0].gather(1, min_indices_expanded)

        dist1, dist2 = self.chamfer_dist(x_t0[:, 0].to(torch.float), y_hat0[:, 0].to(torch.float))
        loss_dist = dist1[dist1 < 0.1].mean()

        # print(f'mask_dist_neg: {mask_dist_neg.sum()}/{mask_dist_neg.numel()}')

        loss_dynamic = F.mse_loss((y_hat1 - y_hat0) * mask_chamfer * mask_dist, nearest_m * mask_chamfer * mask_dist)
        my_hat_static = (y_hat1 - y_hat0) * mask_dist_neg
        loss_static = F.mse_loss(my_hat_static, torch.zeros_like(my_hat_static))
        # loss_dynamic = F.mse_loss((y_hat1 - y_hat0) * mask_chamfer * mask_dist, nearest_m * mask_chamfer * mask_dist)
        # loss_static = F.mse_loss((y_hat1 - y_hat0) * mask_dist_neg, torch.zeros_like(y_hat0) * mask_dist_neg)

        return loss_dynamic, loss_static, loss_dist

def create_loss(loss_name, loss_params):
    if loss_params is None:
        loss_params = {}
    if loss_name == 'UnsupLoss':
        loss = UnsupLoss(loss_params)
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

class LitModel(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = create_model(hparams.model_name, hparams.model_params)
        if hparams.checkpoint_path is not None:
            self.load_state_dict(torch.load(hparams.checkpoint_path, map_location=self.device)['state_dict'], strict=False)
        self.loss = create_loss(hparams.loss_name, hparams.loss_params)
        if hparams.save_when_test:
            self.results = []

        if hasattr(hparams, 'use_aux') and hparams.use_aux:
            self.reg_dir = create_model('PlausibilityRegressor', hparams.reg_dir_params)
            self.reg_motion = create_model('PlausibilityRegressor', hparams.reg_motion_params)
            self.reg_dir.load_state_dict(delete_prefix_from_state_dict(torch.load(hparams.reg_dir_path, map_location=self.device)['state_dict'], 'model.'))
            self.reg_motion.load_state_dict(delete_prefix_from_state_dict(torch.load(hparams.reg_motion_path, map_location=self.device)['state_dict'], 'model.'))
            self.reg_dir.requires_grad_(False)
            self.reg_motion.requires_grad_(False)

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
                
                x_sup0 = batch_sup['point_clouds']
                x_sup1 = batch_sup['point_clouds_trans']
                y_sup = batch_sup['keypoints']
                xr_sup0 = batch_sup['ref_point_clouds']
                xr_sup1 = batch_sup['ref_point_clouds_trans']
                yr_sup = batch_sup['ref_keypoints']
                x_unsup = batch_unsup['point_clouds']

                # print(f'x_sup0: {x_sup0.shape}, xr_sup0: {xr_sup0.shape}')

                x_sup0_t0 = torch.cat((x_sup0[:, :-1, :, :3], xr_sup0[:, :-1, :, :3]), dim=0)
                x_sup1_t0 = torch.cat((x_sup1[:, :-1, :, :3], xr_sup1[:, :-1, :, :3]), dim=0)
                x_sup0_t1 = torch.cat((x_sup0[:, 1:, :, :3], xr_sup0[:, 1:, :, :3]), dim=0)
                x_sup1_t1 = torch.cat((x_sup1[:, 1:, :, :3], xr_sup1[:, 1:, :, :3]), dim=0)
                y_sup_t0 = torch.cat((y_sup[:, :-1, :, :3], yr_sup[:, :-1, :, :3]), dim=0)
                y_sup_t1 = torch.cat((y_sup[:, 1:, :, :3], yr_sup[:, 1:, :, :3]), dim=0)

                if torch.rand(1).item() < 0.5:
                    perm = torch.randperm(x_sup0_t0.shape[-2])
                    num2exchange = torch.randint(0, x_sup0_t0.shape[-2], (1,)).item()
                    x_sup0_t0_ = torch.cat((x_sup0_t0[..., perm[:num2exchange], :3], x_sup1_t0[..., perm[num2exchange:], :3]), dim=-2)
                    x_sup1_t0_ = torch.cat((x_sup1_t0[..., perm[:num2exchange], :3], x_sup0_t0[..., perm[num2exchange:], :3]), dim=-2)
                    x_sup0_t0 = x_sup0_t0_
                    x_sup1_t0 = x_sup1_t0_

                if torch.rand(1).item() < 0.5:
                    perm = torch.randperm(x_sup0_t0.shape[-2])
                    num2exchange = torch.randint(0, x_sup0_t0.shape[-2], (1,)).item()
                    x_sup0_t1_ = torch.cat((x_sup0_t1[..., perm[:num2exchange], :3], x_sup1_t1[..., perm[num2exchange:], :3]), dim=-2)
                    x_sup1_t1_ = torch.cat((x_sup1_t1[..., perm[:num2exchange], :3], x_sup0_t1[..., perm[num2exchange:], :3]), dim=-2)
                    x_sup0_t1 = x_sup0_t1_
                    x_sup1_t1 = x_sup1_t1_

                y_sup_t0_hat0 = self.model(x_sup0_t0)
                y_sup_t1_hat0 = self.model(x_sup0_t1)
                y_sup_t0_hat1 = self.model(x_sup1_t0)
                y_sup_t1_hat1 = self.model(x_sup1_t1)

                y_hat = y_sup_t1_hat0[:y_sup_t0.shape[0]//2]

                # with open('aaaaaaaaaaaa.txt', 'a') as f:
                #     f.write(f'{y_sup_t0}\n')

                loss_sup_t0 = F.mse_loss(y_sup_t0_hat0, y_sup_t0) + F.mse_loss(y_sup_t0_hat1, y_sup_t0.clone())
                loss_sup_t1 = F.mse_loss(y_sup_t1_hat0, y_sup_t1) + F.mse_loss(y_sup_t1_hat1, y_sup_t1.clone())
                
                x_unsup_t0 = x_unsup[:, :-1, ...]
                x_unsup_t1 = x_unsup[:, 1:, ...]

                y_unsup_t0_hat = self.model(x_unsup_t0)
                y_unsup_t1_hat = self.model(x_unsup_t1)

                unsup_loss = self.loss.to(self.device)
                loss_unsup_dynamic, loss_unsup_static, loss_unsup_dist = unsup_loss(x_unsup, y_unsup_t0_hat, y_unsup_t1_hat)

                # if self.current_epoch < 10:
                #     loss = loss_sup_t0 + loss_sup_t1
                # else:
                loss = loss_sup_t0 + loss_sup_t1 + self.hparams.w_dynamic * loss_unsup_dynamic + self.hparams.w_static * loss_unsup_static + self.hparams.w_dist * loss_unsup_dist
                losses = {'loss': loss, 'loss_sup': loss_sup_t0 + loss_sup_t1, 'loss_unsup_dynamic': loss_unsup_dynamic, 'loss_unsup_static': loss_unsup_static, 'loss_unsup_dist': loss_unsup_dist}

            elif self.hparams.train_dataset['name'] in ['ReferenceDataset']:
                x_ref = batch['ref_point_clouds']
                y_ref = batch['ref_keypoints']
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
                y_hat = self.model(x)
                loss = self.loss(y_hat, y)
                losses = {'loss': loss}
        elif self.hparams.model_name in ['P4TransformerSimCC']:
            c_hat, y_hat = self.model(x)

            n = self.model.cube_len
            bin_width = 3.0 / n
            x_bin_indices = ((y[..., 0] + 1.5) / bin_width).clamp(0, n-1).floor().to(torch.long)
            y_bin_indices = (y[..., 1] / bin_width).clamp(0, n-1).floor().to(torch.long)
            z_bin_indices = ((y[..., 2] + 1.5) / bin_width).clamp(0, n-1).floor().to(torch.long)
            c = torch.stack([x_bin_indices, y_bin_indices, z_bin_indices], dim=-1).to(torch.long) # B 1 J 3
            loss = self.loss(c_hat, c)
            losses = {'loss': loss}
        elif self.hparams.model_name in ['P4TransformerMotion']:
            batch_sup = batch['sup']
            batch_unsup = batch['unsup']

            x_sup = batch_sup['point_clouds']
            y_sup = batch_sup['keypoints']
            x_unsup = batch_unsup['point_clouds']

            m_sup_hat = self.model(x_sup) # B 1 N 3
            my_sup = y_sup[:, 1:2, :, :3] - y_sup[:, 0:1, :, :3] # B 1 J 3

            dists = torch.norm(x_sup[:, 0, :, :3].unsqueeze(2) - y_sup[:, 0, :, :3].unsqueeze(1), dim=-1) # B N J
            bottom_k_dists, bottom_k_indices = torch.topk(dists, k=3, dim=-1, largest=False) # B N k
            bottom_k_weights = torch.softmax(-bottom_k_dists, dim=-1) # B N k
            # print(my_sup.shape, bottom_k_indices.shape, bottom_k_weights.shape)
            B, N, K = bottom_k_indices.shape
            bottom_k_indices_ = bottom_k_indices.reshape(B, N * K, 1).expand(-1, -1, my_sup.shape[-1]) # B N*k 3
            nearest_m = my_sup[:, 0].gather(1, bottom_k_indices_).reshape(B, N, K, my_sup.shape[-1]) # B N k 3
            nearest_m = (nearest_m * bottom_k_weights.unsqueeze(-1)).sum(dim=-2) # B N 3

            min_dists, _ = torch.min(dists, dim=-1) # B N, B N
            mask_dist = (min_dists < 0.1).unsqueeze(1).unsqueeze(-1) # B 1 N 1
            # min_indices_expanded = min_indices.unsqueeze(-1).expand(-1, -1, m_sup_hat.shape[-1]) # B N 3
            # nearest_m = my_sup[:, 0].gather(1, min_indices_expanded).unsqueeze(1) # B N 3

            loss_sup = F.mse_loss(m_sup_hat * mask_dist, nearest_m * mask_dist)

            m_unsup_hat = self.model(x_unsup) # B 1 N 3
            x0_unsup = x_unsup[:, 0:1, :, :3] # B 1 N 3
            x1_unsup = x_unsup[:, 1:2, :, :3] # B 1 N 3
            x1_unsup_hat = x0_unsup + m_unsup_hat # B 1 N 3

            y_hat = y_sup

            chamfer_dist = ChamferDistance()
            dist1, dist2 = chamfer_dist(x1_unsup_hat[:, 0, :, :3], x1_unsup[:, 0, :, :3])
            loss_chamfer = dist1.mean() + dist2.mean()

            loss_reg = torch.norm(m_unsup_hat, dim=-1).mean()

            loss = loss_sup + self.hparams.w_chamfer * loss_chamfer + self.hparams.w_reg * loss_reg
            
            losses = {'loss': loss, 'loss_sup': loss_sup, 'loss_chamfer': loss_chamfer, 'loss_reg': loss_reg}

        elif self.hparams.model_name in ['P4TransformerFlow']:
            # B, T, J, C = y.shape
            y0 = y[:, 0:1, ...]
            loc0 = y0[:, :, 8:9, :]
            pose0 = y0 - loc0
            y1 = y[:, 1:2, ...]
            loc1 = y1[:, :, 8:9, :]
            pose1 = y1 - loc1
            flow = y[:, 1:, ...] - y[:, :-1, ...]

            pose_hat0, loc_hat0, flow_hat0 = self.model(x[:, :-1, ...])
            pose_hat1, loc_hat1, flow_hat1 = self.model(x[:, 1:, ...])

            l_pose = self.loss(pose_hat0, pose0) + self.loss(pose_hat1, pose1)
            l_loc = self.loss(loc_hat0, loc0) + self.loss(loc_hat1, loc1)
            l_flow = self.loss(flow_hat0, flow) + self.loss(flow_hat1[:, :-1, ...], flow[:, 1:, ...])
            l_con = self.loss(flow_hat0[:, 1:, ...], flow_hat1[:, :-1, ...])

            # y0_hat = pose_hat0 + loc_hat0
            # y1_hat = pose_hat1 + loc_hat1
            # y0_hat1 = y0_hat.clone() + flow_hat0[:, 0:1, ...]
            # l_con = self.loss(y0_hat1, y1_hat)

            y_hat = pose_hat0 + loc_hat0
            y = y0
            # flow_hat2 = y_hat[:, 1:, ...] - y_hat[:, :-1, ...]
            # l_flow2 = self.loss(flow_hat2, flow)
            # accum_flow = torch.cumsum(flow_hat0[:, :-1, ...], dim=1)
            # y_hat = torch.cat([y0_hat, y0_hat + accum_flow], dim=1)
            loss = self.hparams.w_loc * l_loc + self.hparams.w_flow * l_flow + self.hparams.w_con * l_con
            losses = {'loss': loss, 'l_pose': l_pose, 'l_loc': l_loc, 'l_flow': l_flow, 'l_con': l_con}
        elif self.hparams.model_name in ['P4TransformerFlowDA']:
            x_ref = batch['ref_point_clouds']
            y0 = y[:, 0:1, ...]
            loc0 = y0[:, :, 8:9, :]
            pose0 = y0 - loc0
            y1 = y[:, 1:2, ...]
            loc1 = y1[:, :, 8:9, :]
            pose1 = y1 - loc1
            flow = y[:, 1:, ...] - y[:, :-1, ...]

            flow_lidar0, loc_lidar0, flow_trans0, loc_trans0, l_rec_lidar0, l_rec_mmwave0, l_conf0 = self.model((x[:, :-1, ...], x_ref[:, :-1, ...]), mode='train')
            flow_lidar1, loc_lidar1, flow_trans1, loc_trans1, l_rec_lidar1, l_rec_mmwave1, l_conf1 = self.model((x[:, 1:, ...], x_ref[:, 1:, ...]), mode='train')
            l_flow = self.loss(flow_lidar0, flow) + self.loss(flow_trans0, flow) + self.loss(flow_lidar1[:, :-1, ...], flow[:, 1:, ...]) + self.loss(flow_trans1[:, :-1, ...], flow[:, 1:, ...])
            l_loc = self.loss(loc_lidar0, loc0) + self.loss(loc_trans0, loc0) + self.loss(loc_lidar1, loc1) + self.loss(loc_trans1, loc1)
            l_con = self.loss(flow_lidar0[:, 1:, ...], flow_lidar1[:, :-1, ...]) + self.loss(flow_trans0[:, 1:, ...], flow_trans1[:, :-1, ...])
            l_rec_lidar = l_rec_lidar0 + l_rec_lidar1
            l_rec_mmwave = l_rec_mmwave0 + l_rec_mmwave1
            l_conf = l_conf0 + l_conf1

            y_hat = pose0 + loc_lidar0
            y = y0

            loss = self.hparams.w_loc * l_loc + self.hparams.w_flow * l_flow + self.hparams.w_rec * (l_rec_lidar + l_rec_mmwave) + self.hparams.w_conf * l_conf
            losses = {'loss': loss, 'l_flow': l_flow, 'l_loc': l_loc, 'l_rec_lidar': l_rec_lidar, 'l_rec_mmwave': l_rec_mmwave, 'l_conf': l_conf}

        elif self.hparams.model_name in ['P4TransformerDG', 'P4TransformerDG2']:
            x_ref = batch['ref_point_clouds']
            if torch.rand(1).item() < 0.5:
                perm = torch.randperm(x_ref.shape[-2])
                num2exchange = torch.randint(0, x_ref.shape[-2], (1,)).item()
                x_ = torch.cat((x[..., perm[:num2exchange], :3], x_ref[..., perm[num2exchange:], :3]), dim=-2)
                x_ref_ = torch.cat((x_ref[..., perm[:num2exchange], :3], x[..., perm[num2exchange:], :3]), dim=-2)
                x = x_
                x_ref = x_ref_
                # print(x.shape, x_ref.shape)
            y_ref = batch['ref_keypoints']
            y_hat, y_hat_ref, l_rec = self.model((x, x_ref))
            l_pc = self.loss(y_hat, y)
            l_pc2 = self.loss(y_hat_ref, y_ref)
            loss = l_pc + l_pc2 + self.hparams.w_rec * l_rec #+ self.hparams.w_mem * l_mem # + w_con * l_con
            losses = {'loss': loss, 'l_pc': l_pc, 'l_pc2': l_pc2, 'l_rec': l_rec}
            
        elif self.hparams.model_name == 'PoseTransformer':
            #TODO: Implement the loss function of posetran
            # sformer
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
            raise NotImplementedError
        
        return losses, x, y, y_hat
    
    def _evaluate(self, batch):
        x = batch['point_clouds']
        y = batch['keypoints']
        if self.hparams.model_name in ['P4TransformerFlow']:
            y_hat, pose_hat, loc_hat, flow_hat = self.model(x[:, :-1, ...])
            accum_flow_hat = torch.cumsum(flow_hat, dim=1)
            y_hat = torch.cat((y_hat[:, 0:1, ...], accum_flow_hat), dim=1)
            return x, y, y_hat, pose_hat, loc_hat, flow_hat
        elif self.hparams.model_name in ['P4TransformerFlowDA']:
            flow_hat, loc_hat = self.model(x[:, :-1, ...])
            return x, y, y, loc_hat, flow_hat
        elif self.hparams.model_name in ['P4TransformerMotion']:
            m_hat = self.model(x)
            return x, y, y, m_hat
        else:
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

        if self.hparams.model_name in ['P4TransformerFlow']:
            x, y, y_hat_, pose_hat, loc_hat, flow_hat = self._evaluate(batch)
            y0 = y[:, 0:1, ...]
            loc = y0[:, :, 8:9, :]
            pose = y0 - loc
            flow = y[:, 1:, ...] - y[:, :-1, ...]
            y = y[:, 0:1, ...]
            y_hat_ = y_hat_[:, 0:1, ...]

            l_pose = F.mse_loss(pose_hat, pose)
            l_loc = F.mse_loss(loc_hat, loc)
            l_flow = F.mse_loss(flow_hat, flow)

        elif self.hparams.model_name in ['P4TransformerFlowDA']:
            x, y, y_hat_, loc_hat, flow_hat = self._evaluate(batch)
            y0 = y[:, 0:1, ...]
            loc = y0[:, :, 8:9, :]
            flow = y[:, 1:, ...] - y[:, :-1, ...]
            y = y[:, 0:1, ...]
            y_hat_ = y_hat_[:, 0:1, ...]

            l_loc = F.mse_loss(loc_hat, loc)
            l_flow = F.mse_loss(flow_hat, flow)

        elif self.hparams.model_name in ['P4TransformerMotion']:
            x, y, y_hat_, m_hat = self._evaluate(batch)

            x0 = x[:, 0:1, :, :3] # B 1 N 3
            x1 = x[:, 1:2, :, :3] # B 1 N 3
            x1_hat = x0 + m_hat # B 1 N 3
            y_hat = y
            
            chamfer_dist = ChamferDistance()
            dist1, dist2 = chamfer_dist(x1_hat[:, 0, :, :3], x1[:, 0, :, :3])
            loss_chamfer = dist1.mean() + dist2.mean()

            loss_reg = torch.norm(m_hat, dim=-1).mean()

            loss = loss_chamfer + self.hparams.w_reg * loss_reg

        else:
            x, y, y_hat_ = self._evaluate(batch)
            y = y[:, -1:, ...]
            y_hat_ = y_hat_[:, -1:, ...]
        
        x, y, y_hat = self._recover_data(x, y, y_hat_, c, r)
        mpjpe, pampjpe = calulate_error(y_hat, y)

        log_dict = {'val_mpjpe': mpjpe, 'val_pampjpe': pampjpe}
        if self.hparams.model_name in ['P4TransformerFlow']:
            log_dict['val_l_pose'] = l_pose
            log_dict['val_l_loc'] = l_loc
            log_dict['val_l_flow'] = l_flow
        elif self.hparams.model_name in ['P4TransformerFlowDA']:
            log_dict['val_l_loc'] = l_loc
            log_dict['val_l_flow'] = l_flow
        elif self.hparams.model_name in ['P4TransformerMotion']:
            log_dict['val_loss'] = loss
        self.log_dict(log_dict, sync_dist=True)

        if batch_idx == 0:
            self._vis_pred_gt_keypoints(x, y, y_hat)
    
    def test_step(self, batch, batch_idx):
        c = batch['centroid']
        r = batch['radius']
        si = batch['sequence_index']

        if self.hparams.model_name in ['P4TransformerFlow']:
            x, y, y_hat_, pose_hat, loc_hat, flow_hat = self._evaluate(batch)
            y0 = y[:, 0:1, ...]
            loc = y0[:, :, 8:9, :]
            pose = y0 - loc
            flow = y[:, 1:, ...] - y[:, :-1, ...]
            y = y[:, 0:1, ...]
            y_hat_ = y_hat_[:, 0:1, ...]

            l_pose = F.mse_loss(pose_hat, pose)
            l_loc = F.mse_loss(loc_hat, loc)
            l_flow = F.mse_loss(flow_hat, flow)

        elif self.hparams.model_name in ['P4TransformerFlowDA']:
            x, y, y_hat_, loc_hat, flow_hat = self._evaluate(batch)
            y0 = y[:, 0:1, ...]
            loc = y0[:, :, 8:9, :]
            flow = y[:, 1:, ...] - y[:, :-1, ...]
            y = y[:, 0:1, ...]
            y_hat_ = y_hat_[:, 0:1, ...]

            l_loc = F.mse_loss(loc_hat, loc)
            l_flow = F.mse_loss(flow_hat, flow)

        elif self.hparams.model_name in ['P4TransformerMotion']:
            x, y, y_hat_, m_hat = self._evaluate(batch)

            x0 = x[:, 0:1, :, :3] # B 1 N 3
            x1 = x[:, 1:2, :, :3] # B 1 N 3
            x1_hat = x0 + m_hat # B 1 N 3
            y_hat = y

            chamfer_dist = ChamferDistance()
            dist1, dist2 = chamfer_dist(x1_hat[:, 0, :, :3], x1[:, 0, :, :3])
            loss_chamfer = dist1.mean() + dist2.mean()

            loss_reg = torch.norm(m_hat, dim=-1).mean()

            loss = loss_chamfer + self.hparams.w_reg * loss_reg

        else:
            x, y, y_hat_ = self._evaluate(batch)
            y = y[:, -1:, ...]
            y_hat_ = y_hat_[:, -1:, ...]

        # start_idx = (batch_idx * 32) % (1024 - 32)
        # end_idx = start_idx + 32

        # y_mem = self.model.forward_debug(start_idx, end_idx)
        # y_mem = y_mem.squeeze().detach().cpu().numpy()
        # np.save(f'logs/{self.hparams.exp_name}/{self.hparams.version}/y_mem_{start_idx}_{end_idx}.npy', y_mem)

        if hasattr(self.hparams, 'use_aux') and self.hparams.use_aux:
            y_hat = y_hat_.squeeze(1)
            bds = []
            for b in ITOPSkeleton.bones:
                bd = y_hat[:, b[1]].clone() - y_hat[:, b[0]].clone()
                bd = bd / torch.linalg.norm(bd, axis=-1, keepdims=True)
                bds.append(bd)
            bds = torch.stack(bds, axis=1)
            pl_dir = self.reg_dir(bds)
            l_pl_dir = F.mse_loss(pl_dir, torch.zeros_like(pl_dir))

            y_hat0, y_hat1 = torch.chunk(y_hat.clone(), 2, dim=0)
            bms = []
            for b in ITOPSkeleton.bones:
                bm = y_hat1[:, b[1]].clone() - y_hat0[:, b[0]].clone()
                bms.append(bm)
            bms = torch.stack(bms, axis=1)
            pl_motion = self.reg_motion(bms)
            l_pl_motion = F.mse_loss(pl_motion, torch.zeros_like(pl_motion))

        x, y, y_hat = self._recover_data(x, y, y_hat_, c, r)
        mpjpe, pampjpe = calulate_error(y_hat, y)

        log_dict = {'test_mpjpe': mpjpe, 'test_pampjpe': pampjpe}
        if self.hparams.model_name in ['P4TransformerFlow']:
            log_dict['test_l_pose'] = l_pose
            log_dict['test_l_loc'] = l_loc
            log_dict['test_l_flow'] = l_flow
        if self.hparams.model_name in ['P4TransformerFlowDA']:
            log_dict['test_l_loc'] = l_loc
            log_dict['test_l_flow'] = l_flow
        if hasattr(self.hparams, 'use_aux') and self.hparams.use_aux:
            log_dict['test_pl_dir'] = l_pl_dir
            log_dict['test_pl_motion'] = l_pl_motion
        if self.hparams.model_name in ['P4TransformerMotion']:
            log_dict['val_loss'] = loss
        self.log_dict(log_dict, sync_dist=True)
        if batch_idx == 0:
            self._vis_pred_gt_keypoints(x, y, y_hat)

        if self.hparams.save_when_test:
            result = {'pred': y_hat, 'gt': y, 'input': x, 'seq_idx': torch2numpy(si)}
            if self.hparams.model_name in ['P4TransformerFlow', 'P4TransformerFlowDA']:
                result['flow'] = torch2numpy(flow_hat)
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
                    input = result['input'][i]
                    seq_idx = int(result['seq_idx'][i].item())
                    if self.hparams.model_name in ['P4TransformerFlow', 'P4TransformerFlowDA']:
                        flow = result['flow'][i]
                    if seq_idx != last_seq_idx:
                        last_seq_idx = seq_idx
                        split_seqs.append({'keypoints': [], 'keypoints_pred': [], 'input': []})
                        if self.hparams.model_name in ['P4TransformerFlow', 'P4TransformerFlowDA']:
                            split_seqs[-1]['flow'] = []
                        split_idxs.append(seq_idx)
                    split_seqs[-1]['keypoints'].append(gt)
                    split_seqs[-1]['keypoints_pred'].append(pred)
                    split_seqs[-1]['input'].append(input)
                    if self.hparams.model_name in ['P4TransformerFlow', 'P4TransformerFlowDA']:
                        split_seqs[-1]['flow'].append(flow)

            for i in range(len(split_seqs)):
                split_seqs[i]['keypoints'] = np.array(split_seqs[i]['keypoints'])[:, 0, ...]
                split_seqs[i]['input'] = np.array(split_seqs[i]['input'])[:, 0, ...]
                
                kps_pred = np.array(split_seqs[i]['keypoints_pred'])
                N, T, J, D = kps_pred.shape
                kps_pred_ = np.zeros((N + T - 1, J, D))
                for j in range(T):
                    kps_pred_[j:j+N] += kps_pred[:, j]
                for j in range(T):
                    kps_pred_[j] *= (T / (j + 1))
                kps_pred = kps_pred_[:N]
                split_seqs[i]['keypoints_pred'] = kps_pred / T
                # split_seqs[i]['keypoints_pred'] = kps_pred[:, 0, ...]

                if self.hparams.model_name in ['P4TransformerFlow', 'P4TransformerFlowDA']:
                    flow = np.array(split_seqs[i]['flow'])
                    N, T, J, D = flow.shape
                    flow_ = np.zeros((N + T - 1, J, D))
                    for j in range(T):
                        flow_[j:j+N] += flow[:, j]
                    for j in range(T):
                        flow_[j] *= (T / (j + 1))
                    flow = flow_[:N]
                    split_seqs[i]['flow'] = flow / T
                    # split_seqs[i]['flow'] = flow[:, 0, ...]

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
                    if self.hparams.model_name in ['P4TransformerFlow']:
                        flow = result['flow'][i]
                    if seq_idx != last_seq_idx:
                        last_seq_idx = seq_idx
                        split_seqs.append({'keypoints_pred': [], 'input': []})
                        if self.hparams.model_name in ['P4TransformerFlow']:
                            split_seqs[-1]['flow'] = []
                        split_idxs.append(seq_idx)
                    split_seqs[-1]['keypoints_pred'].append(pred)
                    split_seqs[-1]['input'].append(input)
                    if self.hparams.model_name in ['P4TransformerFlow']:
                        split_seqs[-1]['flow'].append(flow)

            for i in range(len(split_seqs)):
                split_seqs[i]['input'] = np.array(split_seqs[i]['input'])[:, 0, ...]
                
                kps_pred = np.array(split_seqs[i]['keypoints_pred'])
                N, T, J, D = kps_pred.shape
                kps_pred_ = np.zeros((N + T - 1, J, D))
                for j in range(T):
                    kps_pred_[j:j+N] += kps_pred[:, j]
                for j in range(T):
                    kps_pred_[j] *= (T / (j + 1))
                kps_pred = kps_pred_[:N]
                split_seqs[i]['keypoints_pred'] = kps_pred / T

                if self.hparams.model_name in ['P4TransformerFlow']:
                    flow = np.array(split_seqs[i]['flow'])
                    N, T, J, D = flow.shape
                    flow_ = np.zeros((N + T - 1, J, D))
                    for j in range(T):
                        flow_[j:j+N] += flow[:, j]
                    for j in range(T):
                        flow_[j] *= (T / (j + 1))
                    flow = flow_[:N]
                    split_seqs[i]['flow'] = flow / T

            with open(os.path.join('logs', self.hparams.exp_name, self.hparams.version, 'results.pkl'), 'wb') as f:
                pickle.dump(split_seqs, f)

        return super().on_predict_end()

    def configure_optimizers(self):
        optimizer = create_optimizer(self.hparams.optim_name, self.hparams.optim_params, self.model.parameters())
        scheduler = create_scheduler(self.hparams.sched_name, self.hparams.sched_params, optimizer)
        return [optimizer], [scheduler]