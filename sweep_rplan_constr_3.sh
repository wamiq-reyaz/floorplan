#!/bin/bash --login 
#SBATCH -N 1
#SBATCH --array=0-5
#SBATCH --export=ALL,NCCL_SOCKET_IFNAME=eth0
#SBATCH --partition=batch
#SBATCH -J r3const
#SBATCH --cores-per-socket=5
#SBATCH -o slurm/%A-%a.out
#SBATCH -e slurm/%A-%a.err
#SBATCH --time=06:30:00
#SBATCH --mem=24G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:v100  
#SBATCH -A conf-gpu-2020.11.23


#module load anaconda3

conda activate faclab
which conda
which python

wandb agent wamreyaz/constrained/oc042fal

		      
