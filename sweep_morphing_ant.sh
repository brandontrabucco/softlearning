#!/bin/bash

for THIGH_SIZE1 in 0.2 0.4
do
for ANKLE_SIZE1 in 0.4 0.8
do
for THIGH_SIZE2 in 0.2 0.4
do
for ANKLE_SIZE2 in 0.4 0.8
do
for THIGH_SIZE3 in 0.2 0.4
do
for ANKLE_SIZE3 in 0.4 0.8
do
for THIGH_SIZE4 in 0.2 0.4
do
for ANKLE_SIZE4 in 0.4 0.8
do
  softlearning run_example_local examples.development \
    --algorithm SAC \
    --universe gym \
    --domain "MorphingAnt_${THIGH_SIZE1}_${ANKLE_SIZE1}_${THIGH_SIZE2}_${ANKLE_SIZE2}_${THIGH_SIZE3}_${ANKLE_SIZE3}_${THIGH_SIZE4}_${ANKLE_SIZE4}" \
    --task v0 \
    --exp-name MorphingAnt \
    --checkpoint-frequency 10000
done
done
done
done
done
done
done
done