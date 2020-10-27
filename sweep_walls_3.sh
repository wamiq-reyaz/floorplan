#!/bin/bash --login
#SBATCH -N 1
#SBATCH --array=0-7
#SBATCH --export=ALL,NCCL_SOCKET_IFNAME=eth0
#SBATCH --partition=batch
#SBATCH -J rplan3walls
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

#wandb agent wamreyaz/Walls/5wg0l8sm Old actually only retrains doors. Shit
#wandb agent wamreyaz/FixedWalls/2h9tg04o # trains rplan 5 tuple model
wandb agent wamreyaz/FixedWalls/vu8kbe87 # rplan 3 tuple model
