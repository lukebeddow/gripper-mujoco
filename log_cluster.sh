#!/bin/bash

# quick script to take a cluster training and log it to weights and biases

LOADFROM=~/cluster/rl/models/dqn/
MACHINE=cluster
STAGGER=8
TIMESTAMP=20-10-22-17:08
JOBS="1:15"

cd ~/mymujoco

./pc_job.sh --log-wandb --savedir $LOADFROM --machine $MACHINE \
  --stagger $STAGGER --timestamp $TIMESTAMP --device cpu \
  -j "$JOBS"