#!/bin/bash
#SBATCH --ntasks=16          		        # 16 cores (CPU)
#SBATCH --nodes=1            		        # Use 1 node
#SBATCH --job-name=Tetris        	      	# Name of job
#SBATCH --partition=gpu      		        # Use GPU partition
#SBATCH --gres=gpu:1         		        # Use one GPUs
#SBATCH --mem=64G            		        # Default memory per CPU is 3GB
#SBATCH --output=./output/print.out 		# Stdout file

## Script commands
module load singularity

SIFFILE="/mnt/users/hallvlav/reinforcement-learning/singularity/singularity.sif"

## Executing the script.

singularity exec --nv $SIFFILE python train.py

## Execute with: sbatch train.sh
