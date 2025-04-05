import warnings
warnings.filterwarnings("ignore")

from argparse import ArgumentParser
# import wandb
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger #WandbLogger, 
from pytorch_lightning.strategies.ddp import DDPStrategy

from dataset.data_api import LitDataModule
from model.model_api import LitModel
from model.model_aux_api import AuxLitModel
from model.model_f2p_api import F2PLitModel
# from model.model_ema_api import MeanTeacherLitModel
from misc.utils import load_cfg, merge_args_cfg

def main(args):
    dm = LitDataModule(hparams=args)
    # if args.mean_teacher:
    #     model = MeanTeacherLitModel(hparams=args)
    # else:
    if args.aux:
        model = AuxLitModel(hparams=args)
        monitor = 'val_loss'
        filename = args.model_name+'-{epoch}-{val_loss:.4f}'
    elif args.f2p:
        model = F2PLitModel(hparams=args)
        monitor = 'val_mpjpe'
        filename = args.model_name+'-{epoch}-{val_mpjpe:.4f}'
    else:
        model = LitModel(hparams=args)
        if args.model_name in ['P4TransformerFlow']:
            monitor = 'val_l_flow'
            filename = args.model_name+'-{epoch}-{val_l_flow:.4f}'
        else:
            monitor = 'val_mpjpe'
            filename = args.model_name+'-{epoch}-{val_mpjpe:.4f}'
    # model.load_from_checkpoint(args.checkpoint_path)

    callbacks = [
        ModelCheckpoint(
            monitor=monitor,
            dirpath=os.path.join('logs', args.exp_name, args.version),
            filename=filename,
            save_top_k=1,
            save_last=True,
            mode='min'),
        RichProgressBar(refresh_rate=20)
    ]

    # wandb_on = True #if args.dev+args.test==0 else False
    # if wandb_on:
    #     wandb_logger = WandbLogger(
    #         project=args.wandb_project_name,
    #         save_dir=args.wandb_save_dir,
    #         offline=args.wandb_offline,
    #         log_model=False,
    #         job_type='train')
    #     wandb_logger.log_hyperparams(args)
    logger = TensorBoardLogger(save_dir='logs', 
                               name=args.exp_name,
                               version=args.version)
    logger.log_hyperparams(args)

    trainer = pl.Trainer(
        fast_dev_run=args.dev,
        logger=logger, # wandb_logger if wandb_on else None,
        max_epochs=args.epochs,
        devices=args.gpus,
        accelerator="gpu",
        sync_batchnorm=args.sync_batchnorm,
        num_nodes=args.num_nodes,
        gradient_clip_val=args.clip_grad,
        strategy=DDPStrategy(find_unused_parameters=True) if args.strategy == 'ddp' else args.strategy,
        callbacks=callbacks,
        precision=args.precision,
        benchmark=args.benchmark
    )

    if bool(args.predict):
        import numpy as np
        import pickle
        predictions = trainer.predict(model, datamodule=dm, ckpt_path=None if args.only_load_model else args.checkpoint_path, return_predictions=True)
        data_out = {}
        for pred in predictions:
            if pred['name'] not in data_out:
                data_out[pred['name']] = {'raw': [], 'keypoints': [], 'point_clouds': []}
            data_out[pred['name']]['raw'].append(pred)

        for name in data_out:
            data_out[name]['raw'] = sorted(data_out[name]['raw'], key=lambda x: x['index'][0])
            data_out[name]['keypoints'] = np.concatenate([pred['keypoints'] for pred in data_out[name]['raw']], axis=0)
            data_out[name]['point_clouds'] = np.concatenate([pred['point_clouds'] for pred in data_out[name]['raw']], axis=0)
            del data_out[name]['raw']

        with open(os.path.join('logs', args.exp_name, args.version, 'predictions.pkl'), 'wb') as f:
            pickle.dump(data_out, f)
    elif bool(args.test):
        trainer.test(model, datamodule=dm, ckpt_path=None if args.only_load_model else args.checkpoint_path)
    else:
        trainer.fit(model, datamodule=dm, ckpt_path=None if args.only_load_model else args.checkpoint_path)
        if args.dev==0:
            trainer.test(ckpt_path="best", datamodule=dm)

    # if wandb_on:
    #     wandb.finish()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='cfg/test.yaml')
    parser.add_argument('-g', "--gpus", type=str, default=None,
                        help="Number of GPUs to train on (int) or which GPUs to train on (list or str) applied per node.")
    parser.add_argument('-d', "--dev", type=int, default=0, help='fast_dev_run for debug')
    parser.add_argument('-n', "--num_nodes", type=int, default=1)
    parser.add_argument('-w', "--num_workers", type=int, default=4)
    parser.add_argument('-b', "--batch_size", type=int, default=2048)
    parser.add_argument('-e', "--batch_size_eva", type=int, default=1000, help='batch_size for evaluation')
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument("--model_ckpt_dir", type=str, default="./model_ckpt/")
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--save_when_test', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--exp_name', type=str, default='fasternet')
    parser.add_argument("--version", type=str, default="0")
    parser.add_argument('--only_load_model', action='store_true')
    parser.add_argument('--mean_teacher', action='store_true')
    parser.add_argument('--aux', action='store_true')
    parser.add_argument('--f2p', action='store_true')

    args = parser.parse_args()
    cfg = load_cfg(args.cfg)
    args = merge_args_cfg(args, cfg)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    # please change {WANDB_API_KEY} to your personal api_key before using wandb
    # os.environ["WANDB_API_KEY"] = "60b29f8aae47df8755cbd430f0179c0cd8797bf6"

    main(args)