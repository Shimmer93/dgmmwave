import numpy as np
import torch
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
from misc.utils import torch2numpy, import_with_str
from loss.pose import GeodesicLoss, SymmetryLoss, ReferenceBoneLoss
from loss.adapt import EntropyLoss, ClassLogitContrastiveLoss
from loss.mpjpe import mpjpe as mpjpe_poseformer
from misc.utils import torch2numpy
from misc.skeleton import SimpleCOCOSkeleton

# def create_model(hparams):
#     if hparams.model_name.lower() == 'p4t':
#         model = P4Transformer(radius=hparams.radius, nsamples=hparams.nsamples, spatial_stride=hparams.spatial_stride,
#                               temporal_kernel_size=hparams.temporal_kernel_size, temporal_stride=hparams.temporal_stride,
#                               emb_relu=hparams.emb_relu,
#                               dim=hparams.dim, depth=hparams.depth, heads=hparams.heads, dim_head=hparams.dim_head,
#                               mlp_dim=hparams.mlp_dim, output_dim=hparams.output_dim, features=hparams.features)
#     elif hparams.model_name.lower() == 'p4tda':
#         model = P4TransformerDA(radius=hparams.radius, nsamples=hparams.nsamples, spatial_stride=hparams.spatial_stride,
#                               temporal_kernel_size=hparams.temporal_kernel_size, temporal_stride=hparams.temporal_stride,
#                               emb_relu=hparams.emb_relu,
#                               dim=hparams.dim, depth=hparams.depth, heads=hparams.heads, dim_head=hparams.dim_head,
#                               mlp_dim=hparams.mlp_dim, output_dim=hparams.output_dim, mem_size=hparams.mem_size, features=hparams.features)
#     elif hparams.model_name.lower() == 'p4tda2':
#         model = P4TransformerDA2(radius=hparams.radius, nsamples=hparams.nsamples, spatial_stride=hparams.spatial_stride,
#                               temporal_kernel_size=hparams.temporal_kernel_size, temporal_stride=hparams.temporal_stride,
#                               emb_relu=hparams.emb_relu,
#                               dim=hparams.dim, depth=hparams.depth, heads=hparams.heads, dim_head=hparams.dim_head,
#                               mlp_dim=hparams.mlp_dim, output_dim=hparams.output_dim, mem_size=hparams.mem_size, features=hparams.features)
#     elif hparams.model_name.lower() == 'p4tda3':
#         model = P4TransformerDA3(radius=hparams.radius, nsamples=hparams.nsamples, spatial_stride=hparams.spatial_stride,
#                               temporal_kernel_size=hparams.temporal_kernel_size, temporal_stride=hparams.temporal_stride,
#                               emb_relu=hparams.emb_relu,
#                               dim=hparams.dim, depth=hparams.depth, heads=hparams.heads, dim_head=hparams.dim_head,
#                               mlp_dim=hparams.mlp_dim, output_dim=hparams.output_dim, features=hparams.features)
#     elif hparams.model_name.lower() == 'p4tda4':
#         model = P4TransformerDA4(radius=hparams.radius, nsamples=hparams.nsamples, spatial_stride=hparams.spatial_stride,
#                               temporal_kernel_size=hparams.temporal_kernel_size, temporal_stride=hparams.temporal_stride,
#                               emb_relu=hparams.emb_relu,
#                               dim=hparams.dim, depth=hparams.depth, heads=hparams.heads, dim_head=hparams.dim_head,
#                               mlp_dim=hparams.mlp_dim, output_dim=hparams.output_dim, features=hparams.features)
#     elif hparams.model_name.lower() == 'p4tda5':
#         model = P4TransformerDA5(radius=hparams.radius, nsamples=hparams.nsamples, spatial_stride=hparams.spatial_stride,
#                               temporal_kernel_size=hparams.temporal_kernel_size, temporal_stride=hparams.temporal_stride,
#                               emb_relu=hparams.emb_relu,
#                               dim=hparams.dim, depth=hparams.depth, heads=hparams.heads, dim_head=hparams.dim_head,
#                               dim_proposal=hparams.dim_proposal, heads_proposal=hparams.heads_proposal, dim_head_proposal=hparams.dim_head_proposal,
#                               mlp_dim=hparams.mlp_dim, num_joints=hparams.num_joints, features=hparams.features, num_proposal=hparams.num_proposal)
#     elif hparams.model_name.lower() == 'p4tda6':
#         model = P4TransformerDA6(radius=hparams.radius, nsamples=hparams.nsamples, spatial_stride=hparams.spatial_stride,
#                               temporal_kernel_size=hparams.temporal_kernel_size, temporal_stride=hparams.temporal_stride,
#                               emb_relu=hparams.emb_relu,
#                               dim=hparams.dim, depth=hparams.depth, heads=hparams.heads, dim_head=hparams.dim_head,
#                               dim_proposal=hparams.dim_proposal, heads_proposal=hparams.heads_proposal, dim_head_proposal=hparams.dim_head_proposal,
#                               mlp_dim=hparams.mlp_dim, num_joints=hparams.num_joints, features=hparams.features, num_proposal=hparams.num_proposal)
#     elif hparams.model_name.lower() == 'p4tda7':
#         model = P4TransformerDA7(radius=hparams.radius, nsamples=hparams.nsamples, spatial_stride=hparams.spatial_stride,
#                               temporal_kernel_size=hparams.temporal_kernel_size, temporal_stride=hparams.temporal_stride,
#                               emb_relu=hparams.emb_relu,
#                               dim=hparams.dim, depth=hparams.depth, heads=hparams.heads, dim_head=hparams.dim_head,
#                               mlp_dim=hparams.mlp_dim, num_joints=hparams.num_joints, features=hparams.features)
#     elif hparams.model_name.lower() == 'p4tda8':
#         model = P4TransformerDA8(radius=hparams.radius, nsamples=hparams.nsamples, spatial_stride=hparams.spatial_stride,
#                               temporal_kernel_size=hparams.temporal_kernel_size, temporal_stride=hparams.temporal_stride,
#                               emb_relu=hparams.emb_relu,
#                               dim=hparams.dim, depth=hparams.depth, heads=hparams.heads, dim_head=hparams.dim_head,
#                               dim_proposal=hparams.dim_proposal, heads_proposal=hparams.heads_proposal, dim_head_proposal=hparams.dim_head_proposal,
#                               mlp_dim=hparams.mlp_dim, num_joints=hparams.num_joints, features=hparams.features, num_proposal=hparams.num_proposal)
#     elif hparams.model_name.lower() == 'ptr' or hparams.model_name.lower() == 'ptr2':
#         model = PoseTransformer(num_frame=hparams.number_of_frames, num_joints=hparams.num_joints, num_input_dims = hparams.num_input_dims, in_chans=hparams.in_chans, embed_dim_ratio=hparams.embed_dim_ratio, depth=hparams.depth,
#         num_heads=hparams.num_heads, mlp_ratio=hparams.mlp_ratio, qkv_bias=hparams.qkv_bias, qk_scale=None, drop_path_rate=hparams.drop_path_rate)
#     elif hparams.model_name.lower() == 'debug':
#         model = DebugModel(in_dim=hparams.in_dim, out_dim=hparams.out_dim)
#     else:
#         raise ValueError(f'Unknown model name: {hparams.model_name}')
    
# =======
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

    def _vis_pred_gt_keypoints(self, x, y, y_hat):
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
        if self.hparams.model_name in ['P4Transformer', 'SPiKE']:
            y_hat = self.model(x)
            loss = self.loss(y_hat, y)
            losses = {'loss': loss}
        elif self.hparams.model_name in ['P4TransformerDA8', 'P4TransformerDA9']:
            if self.hparams.mode == 'train':
                x_ref = batch['ref_point_clouds']
                y_hat, y_hat_ref, y_hat2, y_hat_ref2, l_rec, l_rec_ref, l_mem = self.model((x, x_ref), mode='train')
                l_pc = self.loss(y_hat, y)
                l_pc2 = self.loss(y_hat2, y.clone())
                # l_con = self.losses['pc'](y_hat_ref, y_hat_ref2)
                # w_con = self.hparams.w_con if self.current_epoch > 40 else 0
                loss = l_pc + l_pc2 + self.hparams.w_rec * (l_rec + l_rec_ref) + self.hparams.w_mem * l_mem # + w_con * l_con
                losses = {'loss': loss, 'l_pc': l_pc, 'l_pc2': l_pc2, 'l_rec': l_rec, 'l_rec_ref': l_rec_ref, 'l_mem': l_mem}
            else:
                raise ValueError('mode must be train!')
        elif self.hparams.model_name in ['P4TransformerDA10']:
            if self.hparams.mode == 'train':
                x_ref = batch['ref_point_clouds']
                y_hat, y_hat2, l_prec, l_prec_ref, l_rec, l_rec_ref, l_mem = self.model((x, x_ref), mode='train')
                l_pc = self.loss(y_hat, y)
                l_pc2 = self.loss(y_hat2, y.clone())
                # l_con = self.losses['pc'](y_hat_ref, y_hat_ref2)
                # w_con = self.hparams.w_con if self.current_epoch > 40 else 0
                loss = l_pc + l_pc2 + self.hparams.w_rec * (l_rec + l_rec_ref) + self.hparams.w_mem * l_mem # + w_con * l_con
                losses = {'loss': loss, 'l_pc': l_pc, 'l_pc2': l_pc2, 'l_prec': l_prec, 'l_prec_ref': l_prec_ref, 'l_rec': l_rec, 'l_rec_ref': l_rec_ref, 'l_mem': l_mem}
        elif self.hparams.model_name in ['P4TransformerDA11']:
            if self.hparams.mode == 'train':
                x_ref = batch['ref_point_clouds']
                y_hat, y_hat2, l_conf, l_rec, l_rec_ref, l_mem = self.model((x, x_ref), mode='train')
                l_pc = self.loss(y_hat, y)
                l_pc2 = self.loss(y_hat2, y.clone())
                # B, T, L, C = x_new_hat.shape
                # l_dist = torch.cdist(x_new[:, 2], y[:, 0]).min(dim=-1)[0].mean()
                # l_dist = 0
                # for i in range(B):
                #     for j in range(T):
                #         x_new_ij = x_new[i, j]
                #         x_ij = x[i, j][..., :C]
                #         conf_ij = x[i, j][..., -1]
                #         l_dist += chamfer_distance(x_new_ij.reshape(1, -1, C).to(x.dtype), x_ij[conf_ij > 0].reshape(1, -1, C).detach())[0]
                # l_dist /= B * T
                # l_dist = chamfer_distance(x_new_hat.reshape(-1, L, C).to(x.dtype), x[..., :C].reshape(-1, L, C).detach())[0] + \
                #             chamfer_distance(x_new_ref_hat.reshape(-1, L, C).to(x.dtype), x_ref[..., :C].reshape(-1, L, C).detach())[0]
                # l_dist_ref, _ = chamfer_distance(x_new_ref.reshape(-1, L, C).to(x.dtype), x_ref[..., :C].reshape(-1, L, C).detach())
                # l_con = self.losses['pc'](y_hat_ref, y_hat_ref2)
                # w_con = self.hparams.w_con if self.current_epoch > 40 else 0
                loss = l_pc + l_pc2 + self.hparams.w_conf * l_conf + self.hparams.w_rec * (l_rec + l_rec_ref) + \
                    self.hparams.w_mem * l_mem # + self.hparams.w_dist * l_dist # + self.hparams.w_prec * l_prec  # + w_con * l_con
                losses = {'loss': loss, 'l_pc': l_pc, 'l_pc2': l_pc2, 'l_conf': l_conf, 'l_rec': l_rec, 'l_rec_ref': l_rec_ref, 'l_mem': l_mem} #, 'l_dist': l_dist}
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
            torch.cuda.empty_cache()
        elif self.hparams.model_name.lower() == 'ptr2':
            # print(x.shape)
            y_hat = self.model(x)
            y_mod = torch.clone(y)
            # loss = self.losses['pc'](y_hat, y)
            loss = mpjpe_poseformer(y_hat, y_mod)
            loss = y_mod.shape[0] * y_mod.shape[1] * loss
            losses = {'loss': loss}
            # print("The original loss is", loss)
            torch.cuda.empty_cache()
        else:
            raise NotImplementedError
        
        return losses, x, y, y_hat
    
    def _evaluate(self, batch):
        x = batch['point_clouds']
        y = batch['keypoints']
        if self.hparams.model_name in ['P4TransformerDA11']:
            y_hat = self.model(x, mode='inference')
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

        x, y, y_hat = self._evaluate(batch)
        x, y, y_hat = self._recover_data(x, y, y_hat, c, r)
        mpjpe, pampjpe = calulate_error(y_hat, y)

        self.log_dict({'val_mpjpe': mpjpe, 'val_pampjpe': pampjpe}, sync_dist=True)
        if batch_idx == 0:
            self._vis_pred_gt_keypoints(x, y, y_hat)
    
    def test_step(self, batch, batch_idx):
        c = batch['centroid']
        r = batch['radius']

        x, y, y_hat = self._evaluate(batch)
        x, y, y_hat = self._recover_data(x, y, y_hat, c, r)
        mpjpe, pampjpe = calulate_error(y_hat, y)

        self.log_dict({'test_mpjpe': mpjpe, 'test_pampjpe': pampjpe}, sync_dist=True)
        if batch_idx == 0:
            self._vis_pred_gt_keypoints(x, y, y_hat)

        if self.hparams.save_when_test:
            result = {'pred': y_hat, 'gt': y, 'input': x}
            self.results.append(result)

    def on_test_end(self):
        if self.hparams.save_when_test:
            results_to_save = {'pred': [], 'gt': [], 'input': []}

            for result in self.results:
                results_to_save['pred'].append(result['pred'])
                results_to_save['gt'].append(result['gt'])
                results_to_save['input'].append(result['input'])

            results_to_save['pred'] = np.concatenate(results_to_save['pred'], axis=0)
            results_to_save['gt'] = np.concatenate(results_to_save['gt'], axis=0)
            results_to_save['input'] = np.concatenate(results_to_save['input'], axis=0)

            with open(os.path.join('logs', self.hparams.exp_name, self.hparams.version, 'results.pkl'), 'wb') as f:
                pickle.dump(results_to_save, f)

        return super().on_test_end()

    def predict_step(self, batch, batch_idx):
        x = batch['point_clouds']
        c = batch['centroid']
        r = batch['radius']
        print(batch)

        y_hat = self.model(x)
        x = self._recover_point_cloud(x, c, r)
        y_hat = self._recover_skeleton(y_hat, c, r)
        
        pred = {'name': batch['name'], 'index': batch['index'], 'keypoints': y_hat, 'point_clouds': x}
        # pred = {
        #    'name': batch.get('name', 'unknown'),
        #    'index': batch.get('index', batch_idx),
        #    'keypoints': y_hat,
        #    'point_clouds': x
        # }
        return pred

    def configure_optimizers(self):
        optimizer = create_optimizer(self.hparams.optim_name, self.hparams.optim_params, self.model.parameters())
        scheduler = create_scheduler(self.hparams.sched_name, self.hparams.sched_params, optimizer)
        return [optimizer], [scheduler]