#!/bin/bash

# quick script to take a cluster training and log it to weights and biases

LOADFROM=~/cluster/rl/models/dqn/
MACHINE=cluster
<<<<<<< HEAD
STAGGER=8
TIMESTAMP=16-08-22-16:42
JOBS="1:30"
=======
STAGGER=6
TIMESTAMP=22-08-22-14:44
JOBS="1:18"
>>>>>>> 4afdcecb76346d1fc2b0f8ee6ab66bda779e81fe

cd ~/mymujoco

./pc_job.sh --log-wandb --savedir $LOADFROM --machine $MACHINE \
  --stagger $STAGGER --timestamp $TIMESTAMP --device cpu \
  -j "$JOBS"