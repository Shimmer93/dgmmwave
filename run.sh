#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=project
##SBATCH --nodelist=10-0-8-18

export CUDA_VISIBLE_DEVICES=0
# /mnt/home/zpengac/.Miniconda3/envs/dnn/bin/pip install --force-reinstall natten
srun python main.py -g 1 -n 1 -w 10 -b 16 -e 1 --data_dir data/milipoint.pkl --pin_memory --wandb_project_name test --cfg cfg/sample_p4t.yml --model_ckpt_dir ./model_ckpt/$(date +'%Y%m%d_%H%M%S')
# srun python main.py -g 4 -n 1 -w 10 -b 32 -e 20 --data_dir /mnt/home/zpengac/USERDIR/HAR/datasets/jhmdb --pin_memory --wandb_project_name test --cfg cfg/sample.yaml --model_ckpt_dir ./model_ckpt/$(date +'%Y%m%d_%H%M%S') --checkpoint_path /mnt/home/zpengac/USERDIR/HAR/SkeletonFlow/model_ckpt/20231202_181426/deeplabv3-raft-epoch=82-val_loss=-20.8980.ckpt --test