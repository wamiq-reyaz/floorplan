#!/bin/bash --login
#SBATCH -N 1
#SBATCH --array=0-15
#SBATCH --export=ALL,NCCL_SOCKET_IFNAME=eth0
#SBATCH --partition=batch
#SBATCH -J rplan5walls
#SBATCH --cores-per-socket=5
#SBATCH -o slurm/%A_%a.out
#SBATCH -e slurm/%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=24G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:v100
#SBATCH -A conf-gpu-2020.11.23

conda activate faclab

#wandb agent wamreyaz/demo_conditional/ivu5j9qr
#wandb agent wamreyaz/demo_conditional/btzx2q3h # augmented
wandb agent wamreyaz/demo_conditional/srvpemjv # aug flip + rot