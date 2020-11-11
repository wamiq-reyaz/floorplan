#!/bin/bash --login
#SBATCH -N 1
#SBATCH --array=0
#SBATCH --export=ALL,NCCL_SOCKET_IFNAME=eth0
#SBATCH --partition=batch
#SBATCH -J l3cdoors
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

python train_conditional_doors.py --tuples 3 --dec_layer=12 --dim=264 --enc_layer=16 --lr 0.00015 --lifull True