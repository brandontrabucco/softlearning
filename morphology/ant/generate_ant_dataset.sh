#!/bin/bash

. /packages/anaconda3/etc/profile.d/conda.sh
conda activate morphing-datasets

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200_linux/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro150/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro131/bin

python /packages/morphing-datasets/make_data.py \
  --local-dir /global/shared/btrabucco/data \
  --num-legs 4 \
  --dataset-size 2000 \
  --num-parallel 32 \
  --num-gpus 8 \
  --n-envs 4 \
  --max-episode-steps 1000 \
  --total-timesteps 1000000 \
  --method centered \
  --domain $1
