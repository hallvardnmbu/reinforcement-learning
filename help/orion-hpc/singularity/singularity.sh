#!/bin/bash
#SBATCH --ntasks=1			                # 1 core (CPU)
#SBATCH --nodes=1			                # Use 1 node
#SBATCH --job-name=build_singularity        		# Name of job
#SBATCH --mem=3G 			                # Default memory per CPU is 3GB
#SBATCH --partition=gpu                     		# Use GPU partition
#SBATCH --gres=gpu:1                        		# Use one GPU
#SBATCH --output=./output.out 				# Stdout and stderr file


## Script commands
# This is a script to generate new singularity container .sif file taking a .def file as an argument. 
# The script will use the same name for the .sif file it returns as the .def file has (hello.def --> hello.sif)
# The script works by submitting singularity creation as a SLURM job. 
# This has to be done because creating a GPU capable singularity requires the use of a GPU node, and 
# using interactive login commands in a shell script does not work. 
#
# To run script: 
# >> sbatch singularity.sh PATH/TO/DEFINITION/FILE.def
# returns: 
#   * SINGULARITY_DEFINITION_FILE.sif => Singularity container.
#   * output.out => Output from container creation. Check this for errors.

if [ $# = 0 ]
then
    echo "Error. No definition file given"
elif [ $# -gt 1 ]
then
    echo "Too many arguments given. Do one file at the time"
else
    DIR_NAME=$(dirname "$1")
    BASENAME=$(basename "$1")
    SIF_FILENAME="${BASENAME:0:end-4}.sif"
    echo "Making container $SIF_FILENAME from file $BASENAME, and putting it in $DIR_NAME"
    SIF_PATH="${DIR_NAME}/${SIF_FILENAME}"
    module load singularity
    singularity build --fakeroot $SIF_PATH $1
fi 
