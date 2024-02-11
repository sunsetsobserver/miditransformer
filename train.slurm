#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="a.lukawski@kunsten.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=100M
#SBATCH --time=00:05:00
#SBATCH --partition=testing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# load modules (assuming you start from the default environment)
# we explicitly call the modules to improve reproducibility
# in case the default settings change
module load Python

# Source the Python virtual environment
source $HOME/data1/miditransformer/bin/activate

echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"


# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
python3 train.py
echo "[$SHELL] ## Script finished"

echo "[$SHELL] #### Finished Python test. Have a nice day"