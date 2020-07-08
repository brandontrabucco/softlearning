#!/bin/bash

export PACKAGES=/packages
. $PACKAGES/anaconda3/etc/profile.d/conda.sh

conda activate softlearning

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200_linux/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro150/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro131/bin
cp -r $PACKAGES/.mujoco $HOME/.mujoco

ln -s $PACKAGES/.mujoco/mujoco200_linux $PACKAGES/.mujoco/mujoco200

python $PACKAGES/softlearning/morphology/dkitty/evaluate_designs.py \
    --local-dir /global/scratch/btrabucco/data \
    --designs /global/scratch/btrabucco/designs/dkitty/forward_model.pkl \
    --num-samples 3 \
    --num-parallel 32 
