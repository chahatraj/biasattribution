#!/bin/bash

# #SBATCH --account=cs678sp23
#SBATCH --job-name=success
##SBATCH --partition=normal
##SBATCH --nodes 1
##SBATCH --ntasks 1
##SBATCH --time 2-00:00:00
##SBATCH --cpus-per-task 1
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=90GB
#SBATCH --time=0-03:00:00
#SBATCH --array=1-1
#SBATCH --output=/scratch/craj/logs/batt/out/success_n.%A_%a.out.txt
#SBATCH --error=/scratch/craj/logs/batt/err/success_n.%A_%a.err.txt

# nvidia-smi

# Read arguments from the config file

echo $ARGUMENTS

# Execute the script with the arguments
python llama3_8bi.py
