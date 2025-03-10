#!/bin/bash
# SLURM Job Dispatcher for training
# basic_nn1
#
# by Alex João Peterson Santos
# March 7, 2025
# for CS 453 Project
#
# Runs an instance of the model trainign script for every permutation of hyperparameters defined below.
# Run this from the directory of the model, otherwise relative paths will not work.

###### CONFIGURATION #####

ACCOUNT= # fill this in with PIRG name
USERNAME= # fill this in with your username
MLPROJDIR= # fill this in with the path to the root of the project
MODELNAME="basic_nn1"

##### HYPERPARAMETER VALUES #####

VALS0= # dataset directory (will not iterate over this)
VALS1=("2" "4" "6" "8" "16") # depth
VALS2=("8" "16" "32" "64") # width
VALS3=("4" "8" "16" "32") # batch size
VALS4=("0.0001" "0.0005" "0.001" "0.01") # learning rate
VALS5=("-1" "0.9" "0.99") # SGD momentum (-1 for AlanW)
VALS6=("0" "1") # use leaky ReLU (uses standard ReLU if 0)

##### DO NOT MODIFY BELOW THIS LINE #####

create_batch_file() {
  cat << EOF > ./slurm/${1}.batch
#!/bin/bash

#SBATCH --account=${ACCOUNT}
#SBATCH --partition=gpu
#SBATCH --job-name=${1}
#SBATCH --mem=2000M
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:45:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --output=${MLPROJDIR}/models/${MODELNAME}/out/${1}_out.txt
#SBATCH --error=${MLPROJDIR}/models/${MODELNAME}/out/${1}_err.txt

date
module load miniconda3
conda activate cs453_proj
python3.11 -u ${MLPROJDIR}/job_scripts/train.py ${MODELNAME} ${2} ${3} ${4} ${5} ${6} ${7} ${8}
date
EOF
}

# create directories for output if they do not exist
mkdir -p ./out > /dev/null
mkdir -p ./curves > /dev/null
mkdir -p ./saves > /dev/null

# warn about cleaning directories
read -p "Clean dirs 'out', 'curves', 'saves', and records CSV file before running? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf ./out/*
    rm -rf ./curves/*
    rm -rf ./saves/*
    rm -f ./*records.csv
    echo "Cleaned."
else
    echo "Proceeding without cleaning."
fi

# fixed values (we did not iterate over these)
V0=$VALS0

# calculate number of jobs to submit
NUMJOBS=0
JOBLIST=()
mkdir ./slurm
for V1 in "${VALS1[@]}"; do
    for V2 in "${VALS2[@]}"; do
        for V3 in "${VALS3[@]}"; do
            for V4 in "${VALS4[@]}"; do
                for V5 in "${VALS5[@]}"; do
                    for V6 in "${VALS6[@]}"; do
                        NUMJOBS=$((NUMJOBS+1))
                        JOBNAME=run_train_${MODELNAME}_${V1}x${V2}_bs${V3}_lr${V4}_sg${V5}_lk${V6}
                        create_batch_file "$JOBNAME" "$V0" "$V1" "$V2" "$V3" "$V4" "$V5" "$V6"
                        JOBLIST+=("$JOBNAME")
                    done
                done
            done
        done
    done
done

# check with user before submitting jobs
read -p "Batch files created. You are about to submit ${NUMJOBS} jobs to slurm. Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deleting batch files and exiting..."
    rm -rf ./slurm
    exit 1
fi

# submit the jobs
for JOB in "${JOBLIST[@]}"; do
    sbatch ./slurm/${JOB}.batch
    sleep 0.1
done

# delete batch files after submission
sleep 1
rm -rf ./slurm

echo "Submitted."
squeue -u ${USERNAME}
echo "Use the following command to undo this:"
echo "scancel -u <username>"
