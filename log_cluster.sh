#!/bin/bash

# quick script to take a cluster training and log it to weights and biases

LOADFROM=~/mymujoco/rl/models/dqn/
MACHINE=luke-PC
STAGGER=8
TIMESTAMP=21-12-22-17:07
JOBS="1:32"

cd ~/mymujoco

./pc_job.sh --log-wandb --savedir $LOADFROM --machine $MACHINE \
  --stagger $STAGGER --timestamp $TIMESTAMP --device cpu \
  -j "$JOBS"