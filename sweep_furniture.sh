#!/bin/bash --login
#SBATCH -N 1
#SBATCH --array=0-20
#SBATCH --export=ALL,NCCL_SOCKET_IFNAME=eth0
#SBATCH --partition=batch
#SBATCH -J furn_nodes
#SBATCH --cores-per-socket=5
#SBATCH -o slurm/%A_%a.out
#SBATCH -e slurm/%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --mem=24G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:v100

conda activate faclab

#wandb agent wamreyaz/furniture_nodes/wk7yovfp

# wandb agent wamreyaz/furniture_nodes_suppl/k6r6py5e # furniture 18

# wandb agent wamreyaz/furniture_adj/ik38smdm # furniture adj h
# wandb agent wamreyaz/furniture_adj/6w035v2e # furniture adj v

# wandb agent wamreyaz/furniture_adj/pq6hmmbb # adj adj

wandb agent wamreyaz/furniture_adj/n5q39ssq # adj object