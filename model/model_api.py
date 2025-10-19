import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import matplotlib
matplotlib.use('Agg')
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

class LitModel(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = create_model(hparams.model_name, hparams.model_params)
        if hparams.checkpoint_path is not None:
            self.load_state_dict(torch.load(hparams.checkpoint_path, map_location=self.device)['state_dict'], strict=False)
        self.loss = create_loss(hparams.loss_name, hparams.loss_params)
        if hparams.save_when_test or hparams.predict:
            self.results = []

        if hasattr(hparams, 'has_teacher') and hparams.has_teacher:
            self.model_teacher = create_model(hparams.model_name, hparams.model_params)
            if hasattr(hparams, 'teacher_checkpoint_path') and hparams.teacher_checkpoint_path is not None:
                from collections import OrderedDict
                state_dict = torch.load(hparams.teacher_checkpoint_path, map_location=self.device)['state_dict']
                teacher_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith('model_teacher.'):
                        teacher_state_dict[k[len('model_teacher.'):]] = v
                self.model_teacher.load_state_dict(teacher_state_dict, strict=False)

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
        if self.hparams.model_name in ['P4Transformer', 'P4TransformerAnchor', 'SPiKE', 'PoseTransformer', 'PointTransformer']:
            if hasattr(self.hparams, 'has_teacher') and self.hparams.has_teacher:
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
                    if hasattr(self.hparams, 'teacher_checkpoint_path') and self.hparams.teacher_checkpoint_path is not None:
                        loss_sup = F.mse_loss(y_sup_li_hat, y_sup) + F.mse_loss(yr_sup_mm_hat2, yr_sup)
                    else:
                        loss_sup = F.mse_loss(y_sup_mm_hat, y_sup) + F.mse_loss(y_sup_li_hat, y_sup) + \
                                F.mse_loss(yr_sup_mm_hat, yr_sup) + F.mse_loss(yr_sup_mm_hat2, yr_sup)

                else:
                    x_sup = batch_sup['point_clouds']
                    y_sup = batch_sup['keypoints']
                    xr_sup = batch_sup['ref_point_clouds']
                    yr_sup = batch_sup['ref_keypoints']
                    x_unsup = batch_unsup['point_clouds']

                    y_sup_hat = self.model_teacher(x_sup)
                    yr_sup_hat = self.model(xr_sup)
                    y_hat = y_sup_hat

                    if hasattr(self.hparams, 'teacher_checkpoint_path') and self.hparams.teacher_checkpoint_path is not None:
                        loss_sup = F.mse_loss(yr_sup_hat, yr_sup)
                    else:
                        loss_sup = F.mse_loss(y_sup_hat, y_sup) + F.mse_loss(yr_sup_hat, yr_sup)
                
                x_unsup_t0 = x_unsup[:, :-1, ...]
                x_unsup_t1 = x_unsup[:, 1:, ...]

                with torch.no_grad():
                    y_unsup_t0_pseudo = self.model_teacher(x_unsup_t0)
                    y_unsup_t1_pseudo = self.model_teacher(x_unsup_t1)

                y_unsup_t0_hat = self.model(x_unsup_t0)
                y_unsup_t1_hat = self.model(x_unsup_t1)

                loss_pseudo = F.mse_loss(y_unsup_t0_hat, y_unsup_t0_pseudo.detach()) + \
                              F.mse_loss(y_unsup_t1_hat, y_unsup_t1_pseudo.detach())

                unsup_loss = self.loss.to(self.device)
                loss_unsup_dynamic, loss_unsup_static = unsup_loss(x_unsup, y_unsup_t0_pseudo, y_unsup_t1_pseudo)

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
        
        y_hat = self.model(x[..., :3])
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
        
        result = {'pred': y_hat, 'input': x, 'seq_idx': torch2numpy(si), 'name': batch['name']}
        self.results.append(result)

    def on_predict_end(self):
        split_seqs = []
        split_idxs = []
        last_seq_idx = -1

        for result in self.results:
            # print(result['name'], int(result['seq_idx'][0].item()))


            # if result['name'] == '12_19_2024_15_37_08':
            #     print('debug', int(result['seq_idx'][0].item()))
            for i in range(len(result['pred'])):
                pred = result['pred'][i]
                input = result['input'][i]
                name = result['name'][i]
                seq_idx = int(result['seq_idx'][i].item())
                # print(f'predict: {name}, idx {seq_idx}')

                if seq_idx != last_seq_idx:
                    last_seq_idx = seq_idx
                    # print(f'???predict: {name}, idx {seq_idx}')
                    split_seqs.append({'keypoints_pred': [], 'input': [], 'name': name})
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
        if hasattr(self.hparams, 'has_teacher') and self.hparams.has_teacher:
            optimizer = create_optimizer(self.hparams.optim_name, self.hparams.optim_params, list(self.model.parameters()) + list(self.model_teacher.parameters()))
            scheduler = create_scheduler(self.hparams.sched_name, self.hparams.sched_params, optimizer)
        else:
            optimizer = create_optimizer(self.hparams.optim_name, self.hparams.optim_params, self.model.parameters())
            scheduler = create_scheduler(self.hparams.sched_name, self.hparams.sched_params, optimizer)
        return [optimizer], [scheduler]