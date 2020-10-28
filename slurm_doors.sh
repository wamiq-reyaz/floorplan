#!/bin/bash --login 
#SBATCH -N 1
#SBATCH --export=ALL,NCCL_SOCKET_IFNAME=eth0
#SBATCH --partition=batch
#SBATCH -J doors
#SBATCH --cores-per-socket=5
#SBATCH -o slurm/%J.out
#SBATCH -e slurm/%J.err
#SBATCH --time=06:00:00  
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
python train_walls.py --tuples 3\
		      --notes 'demo' \
		      --lr 0.00001
		      
