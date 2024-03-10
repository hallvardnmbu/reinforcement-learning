#!/bin/bash
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --job-name=                      ##   !!! ENTER NAME HERE !!!
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=./output/print.out

## Script commands
module load singularity

SIFFILE="/path/to/singularity.sif"       ##  !!! ENTER PATH HERE !!!

## Executing the script.

singularity exec --nv $SIFFILE python train.py

## Execute with: sbatch train.sh
