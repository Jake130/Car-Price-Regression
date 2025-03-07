# Conda Environment Creation
#
# by Alex João Peterson Santos
# March 7, 2025
# for CS 453 Project
#
# This script creates a conda environment with the necessary packages for the project.
# The conda environment is named mlproj and should be activated before working with models.

module load miniconda3
conda create -y -n cs453_proj python=3.11
conda activate cs453_proj
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 numpy pandas -c pytorch -c nvidia