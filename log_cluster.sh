#!/bin/bash

# quick script to take a cluster training and log it to weights and biases

LOADFROM=~/cluster/rl/models/dqn/
MACHINE=cluster
STAGGER=4
TIMESTAMP=16-08-22-17:00
JOBS="1:4"

cd ~/mymujoco

./pc_job.sh --log-wandb --savedir $LOADFROM --machine $MACHINE \
  --stagger $STAGGER --timestamp $TIMESTAMP --device cpu \
  -j "$JOBS"