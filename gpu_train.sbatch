#!/bin/bash 
#SBATCH --partition=orchid 
#SBATCH --account=orchid
#SBATCH --gres=gpu:1
#SBATCH --job-name=infiltrain
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=8000

source /home/users/doran/software/envs/pytorch/bin/activate
python ./train.py
