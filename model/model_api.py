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
from loss.mpjpe import mpjpe as mpjpe_mmwave
from misc.utils import torch2numpy, import_with_str, delete_prefix_from_state_dict
from misc.skeleton import ITOPSkeleton, JOINT_COLOR_MAP

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
        x = batch['point_clouds']
        y = batch['keypoints']
        if self.hparams.model_name in ['P4Transformer', 'SPiKE']:
            y_hat = self.model(x)
            loss = self.loss(y_hat, y)
            losses = {'loss': loss}
        if self.hparams.model_name in ['P4TransformerFlow']:
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
            l_flow = self.loss(flow_hat0[:, :-1, ...], flow[:, :-1, ...]) + self.loss(flow_hat1[:, :-1, ...], flow[:, 1:, ...])
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
            flow = y[:, 1:, ...] - y[:, :-1, ...]

            flow_lidar, loc_lidar, flow_trans, loc_trans, l_rec_lidar, l_rec_mmwave, l_conf = self.model((x, x_ref), mode='train')
            l_flow = self.loss(flow_lidar[:, :-1, ...], flow) + self.loss(flow_trans[:, :-1, ...], flow)
            l_loc = self.loss(loc_lidar, loc0) + self.loss(loc_trans, loc0)

            y_hat = y0
            y = y0

            loss = self.hparams.w_loc * l_loc + self.hparams.w_flow * l_flow + self.hparams.w_rec * (l_rec_lidar + l_rec_mmwave) + self.hparams.w_conf * l_conf
            losses = {'loss': loss, 'l_flow': l_flow, 'l_loc': l_loc, 'l_rec_lidar': l_rec_lidar, 'l_rec_mmwave': l_rec_mmwave, 'l_conf': l_conf}

        elif self.hparams.model_name in ['P4TransformerDA8', 'P4TransformerDA9']:
            x_ref = batch['ref_point_clouds']
            y_hat, y_hat_ref, y_hat2, y_hat_ref2, l_rec, l_rec_ref, l_mem = self.model((x, x_ref), mode='train')
            l_pc = self.loss(y_hat, y)
            l_pc2 = self.loss(y_hat2, y.clone())
            # l_con = self.losses['pc'](y_hat_ref, y_hat_ref2)
            # w_con = self.hparams.w_con if self.current_epoch > 40 else 0
            loss = l_pc + l_pc2 + self.hparams.w_rec * (l_rec + l_rec_ref) + self.hparams.w_mem * l_mem # + w_con * l_con
            losses = {'loss': loss, 'l_pc': l_pc, 'l_pc2': l_pc2, 'l_rec': l_rec, 'l_rec_ref': l_rec_ref, 'l_mem': l_mem}
        elif self.hparams.model_name in ['P4TransformerDA10']:
            x_ref = batch['ref_point_clouds']
            y_hat, y_hat2, l_prec, l_prec_ref, l_rec, l_rec_ref, l_mem = self.model((x, x_ref), mode='train')
            l_pc = self.loss(y_hat, y)
            l_pc2 = self.loss(y_hat2, y.clone())
            # l_con = self.losses['pc'](y_hat_ref, y_hat_ref2)
            # w_con = self.hparams.w_con if self.current_epoch > 40 else 0
            loss = l_pc + l_pc2 + self.hparams.w_rec * (l_rec + l_rec_ref) + self.hparams.w_mem * l_mem # + w_con * l_con
            losses = {'loss': loss, 'l_pc': l_pc, 'l_pc2': l_pc2, 'l_prec': l_prec, 'l_prec_ref': l_prec_ref, 'l_rec': l_rec, 'l_rec_ref': l_rec_ref, 'l_mem': l_mem}
        elif self.hparams.model_name in ['P4TransformerDA11']:
            x_ref = batch['ref_point_clouds']
            y_hat, y_hat2, l_conf, l_rec, l_rec_ref, l_mem, l_prec, l_prec_ref = self.model((x, x_ref), mode='train')
            l_pc = self.loss(y_hat, y)
            l_pc2 = self.loss(y_hat2, y.clone())
            loss = l_pc + l_pc2 + self.hparams.w_conf * l_conf + self.hparams.w_rec * (l_rec_ref) + \
                self.hparams.w_prec * (l_prec + l_prec_ref) # + self.hparams.w_dist * l_dist # + self.hparams.w_prec * l_prec  # + w_con * l_con self.hparams.w_mem * l_mem + 
            losses = {'loss': loss, 'l_pc': l_pc, 'l_pc2': l_pc2, 'l_conf': l_conf, 'l_rec': l_rec, 'l_rec_ref': l_rec_ref, 
                        'l_prec': l_prec, 'l_prec_ref': l_prec_ref}
        elif self.hparams.model_name in ['LMA_P4T']:
            x_ref = batch['ref_point_clouds']
            # vis_lidar, vis_transferred, \
            y_hat, y_hat2, y_ref_hat, l_rec_lidar, l_rec_mmwave, l_conf, vis_lidar, vis_transferred = self.model((x, x_ref), mode='train')
            l_pc = self.loss(y_hat, y) + self.loss(y_hat2, y)

            # l_tcon = F.mse_loss(y_mmwave0, y_mmwave1, reduction='none').mean(dim=-1)
            # l_tcon = torch.min(l_tcon, dim=-1)[0]
            # l_tcon[l_tcon < 0.01] = 0
            # l_tcon = l_tcon.mean()

            vis_gt = torch.zeros_like(vis_lidar, dtype=torch.float32)
            for i in range(len(vis_gt)):
                for j in range(len(vis_gt[i])):
                    vis_gt[i][j][0] = torch.any(x[i, -1, :, -1] == j+1)
            l_vis = F.binary_cross_entropy_with_logits(vis_lidar, vis_gt) + \
                    F.binary_cross_entropy_with_logits(vis_transferred, vis_gt)
            
            loss = l_pc + self.hparams.w_rec * (l_rec_lidar + l_rec_mmwave) + self.hparams.w_conf * l_conf + self.hparams.w_vis * l_vis

            losses = {'loss': loss, 'l_pc': l_pc, 'l_rec_mmwave': l_rec_mmwave, 'l_rec_lidar': l_rec_lidar, 'l_conf': l_conf, 'l_vis': l_vis}

            # if self.hparams.use_aux:
            #     y_ref_hat = y_ref_hat.squeeze(1)
            #     bds = []
            #     for b in ITOPSkeleton.bones:
            #         bd = y_ref_hat[:, b[1]].clone() - y_ref_hat[:, b[0]].clone()
            #         bd = bd / torch.linalg.norm(bd, axis=-1, keepdims=True)
            #         bds.append(bd)
            #     bds = torch.stack(bds, axis=1)
            #     pl_dir = self.reg_dir(bds)
            #     l_pl_dir = F.mse_loss(pl_dir, torch.zeros_like(pl_dir))

            #     # y_ref_hat0, y_ref_hat1 = torch.chunk(y_ref_hat.clone(), 2, dim=0)
            #     # bms = []
            #     # for b in ITOPSkeleton.bones:
            #     #     bm = y_ref_hat1[:, b[1]].clone() - y_ref_hat0[:, b[0]].clone()
            #     #     bms.append(bm)
            #     # bms = torch.stack(bms, axis=1)
            #     # pl_motion = self.reg_motion(bms)
            #     # l_pl_motion = F.mse_loss(pl_motion, torch.zeros_like(pl_motion))

            #     loss += self.hparams.w_pl_dir * l_pl_dir # + self.hparams.w_pl_motion * l_pl_motion
            #     # loss += self.hparams.w_pl_motion * l_pl_motion
            #     losses['l_pl_dir'] = l_pl_dir
            #     # losses['l_pl_motion'] = l_pl_motion
            
        elif self.hparams.model_name in ['LMA2_P4T']:
            x_ref = batch['ref_point_clouds']
            y_hat, y_hat2, l_rec = self.model((x, x_ref), mode='train')
            l_pc = self.loss(y_hat, y) + self.loss(y_hat2, y)
            loss = l_pc + self.hparams.w_rec * l_rec
            losses = {'loss': loss, 'l_pc': l_pc, 'l_rec': l_rec}

        elif self.hparams.model_name in ['LMA3_P4T']:
            x_ref = batch['ref_point_clouds']
            y_loc = y[:, :, 8:9]
            y_pose = y - y_loc
            y_pose_hat, y_loc_hat, vis_lidar, l_rec, l_rec_ref, l_ortho = self.model((x, x_ref), mode='train')

            vis_gt = torch.zeros_like(vis_lidar, dtype=torch.float32)
            for i in range(len(vis_gt)):
                for j in range(len(vis_gt[i])):
                    vis_gt[i][j][0] = torch.any(x[i, -1, :, -1] == j+1)
            l_vis = F.binary_cross_entropy_with_logits(vis_lidar, vis_gt)

            y_hat = y_pose_hat + y_loc_hat
            l_pose = self.loss(y_pose_hat, y_pose)
            l_loc = self.loss(y_loc_hat, y_loc)
            loss = l_pose + self.hparams.w_loc * l_loc + self.hparams.w_ortho * l_ortho + \
                    self.hparams.w_vis * l_vis
                #    self.hparams.w_rec * (l_rec + l_rec_ref) + 
            losses = {'loss': loss, 'l_pose': l_pose, 'l_loc': l_loc, 'l_rec': l_rec, 'l_rec_ref': l_rec_ref, 'l_ortho': l_ortho, 'l_vis': l_vis}
            
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
            y_hat, pose_hat, loc_hat, flow_hat = self.model(x)
            return x, y, y_hat, pose_hat, loc_hat, flow_hat
        elif self.hparams.model_name in ['P4TransformerFlowDA']:
            flow_hat, loc_hat = self.model(x)
            return x, y, y, loc_hat, flow_hat
        else:
            y_hat = self.model(x)
            return x, y, y_hat

    def training_step(self, batch, batch_idx):
        c = batch['centroid']
        r = batch['radius']

        losses, x, y, y_hat = self._calculate_loss(batch)

        if y_hat is None:
            log_dict = {}
        else:
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
            y = y[:, -2:-1, ...]
            y_hat_ = y_hat_[:, -2:-1, ...]

            l_pose = F.mse_loss(pose_hat, pose)
            l_loc = F.mse_loss(loc_hat, loc)
            l_flow = F.mse_loss(flow_hat[:, :-1, ...], flow)

        elif self.hparams.model_name in ['P4TransformerFlowDA']:
            x, y, y_hat_, loc_hat, flow_hat = self._evaluate(batch)
            y0 = y[:, 0:1, ...]
            loc = y0[:, :, 8:9, :]
            flow = y[:, 1:, ...] - y[:, :-1, ...]
            y = y[:, -2:-1, ...]
            y_hat_ = y_hat_[:, -2:-1, ...]

            l_loc = F.mse_loss(loc_hat, loc)
            l_flow = F.mse_loss(flow_hat[:, :-1, ...], flow)

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
            y = y[:, -2:-1, ...]
            y_hat_ = y_hat_[:, -2:-1, ...]

            l_pose = F.mse_loss(pose_hat, pose)
            l_loc = F.mse_loss(loc_hat, loc)
            l_flow = F.mse_loss(flow_hat[:, :-1, ...], flow)

        elif self.hparams.model_name in ['P4TransformerFlowDA']:
            x, y, y_hat_, loc_hat, flow_hat = self._evaluate(batch)
            y0 = y[:, 0:1, ...]
            loc = y0[:, :, 8:9, :]
            flow = y[:, 1:, ...] - y[:, :-1, ...]
            y = y[:, -2:-1, ...]
            y_hat_ = y_hat_[:, -2:-1, ...]

            l_loc = F.mse_loss(loc_hat, loc)
            l_flow = F.mse_loss(flow_hat[:, :-1, ...], flow)

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
        self.log_dict(log_dict, sync_dist=True)
        if batch_idx == 0:
            self._vis_pred_gt_keypoints(x, y, y_hat)

        if self.hparams.save_when_test:
            result = {'pred': y_hat, 'gt': y, 'input': x, 'seq_idx': torch2numpy(si)}
            if self.hparams.model_name in ['P4TransformerFlow']:
                result['flow'] = torch2numpy(flow_hat)
            self.results.append(result)

    def on_test_end(self):
        if self.hparams.save_when_test:
            results_to_save = {'pred': [], 'gt': [], 'input': [], 'seq_idx': []}
            if self.hparams.model_name in ['P4TransformerFlow']:
                results_to_save['flow'] = []

            for result in self.results:
                results_to_save['pred'].append(result['pred'])
                results_to_save['gt'].append(result['gt'])
                results_to_save['input'].append(result['input'])
                results_to_save['seq_idx'].append(result['seq_idx'])
                if self.hparams.model_name in ['P4TransformerFlow']:
                    results_to_save['flow'].append(result['flow'])

            results_to_save['pred'] = np.concatenate(results_to_save['pred'], axis=0)
            results_to_save['gt'] = np.concatenate(results_to_save['gt'], axis=0)
            results_to_save['input'] = np.concatenate(results_to_save['input'], axis=0)
            results_to_save['seq_idx'] = np.concatenate(results_to_save['seq_idx'], axis=0)
            if self.hparams.model_name in ['P4TransformerFlow']:
                results_to_save['flow'] = np.concatenate(results_to_save['flow'], axis=0)

            with open(os.path.join('logs', self.hparams.exp_name, self.hparams.version, 'results.pkl'), 'wb') as f:
                pickle.dump(results_to_save, f)

        return super().on_test_end()

    def predict_step(self, batch, batch_idx):
        x = batch['point_clouds']
        c = batch['centroid']
        r = batch['radius']

        y_hat = self.model(x)
        x = self._recover_point_cloud(x, c, r)
        y_hat = self._recover_skeleton(y_hat, c, r)
        
        pred = {'name': batch['name'], 'index': batch['index'], 'keypoints': y_hat, 'point_clouds': x}
        
        return pred

    def configure_optimizers(self):
        optimizer = create_optimizer(self.hparams.optim_name, self.hparams.optim_params, self.model.parameters())
        scheduler = create_scheduler(self.hparams.sched_name, self.hparams.sched_params, optimizer)
        return [optimizer], [scheduler]