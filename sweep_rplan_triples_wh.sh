#!/bin/bash --login 
#SBATCH -N 1
#SBATCH --array=0-15
#SBATCH --export=ALL,NCCL_SOCKET_IFNAME=eth0
#SBATCH --partition=batch
#SBATCH -J doors
#SBATCH --cores-per-socket=5
#SBATCH -o slurm/%A-%a.out
#SBATCH -e slurm/%A-%a.err
#SBATCH --time=03:59:00
#SBATCH --mem=24G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:v100  
#SBATCH -A conf-gpu-2020.11.23


#module load anaconda3

conda activate faclab
#source activate faclab
which conda
which python

#wandb agent wamreyaz/Triplesxy/32q6vpqn
#wandb agent wamreyaz/Triples/bsx5zkgd
wandb agent wamreyaz/Triples_hw/aqhkplbr # hopefully correct

		      
