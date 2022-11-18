#!/bin/bash

# quick script to take a cluster training and log it to weights and biases

LOADFROM=~/cluster/rl/models/dqn/
MACHINE=cluster
STAGGER=8
TIMESTAMP=07-11-22-cluster_17:43
JOBS="1:27"

cd ~/mymujoco

./pc_job.sh --log-wandb --savedir $LOADFROM --machine $MACHINE \
  --stagger $STAGGER --timestamp $TIMESTAMP --device cpu \
  -j "$JOBS"