#!/bin/bash
#SBATCH --job-name=btrabucco_morphology
#SBATCH --account=co_rail
#SBATCH --time=72:00:00
#SBATCH --partition=savio3_2080ti
#SBATCH --qos=rail_2080ti3_normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --gres=gpu:8

singularity exec --nv -B /usr/lib64 -B /var/lib/dcv-gl -w \
    /global/scratch/btrabucco/morphology.img \
    /global/scratch/btrabucco/generate_dkitty_dataset.sh 

