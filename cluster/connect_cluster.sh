#!/bin/bash

# set variables
USERNAME=lbeddow
PASSFILE=key.txt
PORT=3333
LOGIN_NODE=tails.cs.ucl.ac.uk
JOB_NODE=vic.cs.ucl.ac.uk

# ssh -J lbeddow@tails.cs.ucl.ac.uk lbeddow@vic.cs.ucl.ac.uk
# ssh -L 3333:vic.cs.ucl.ac.uk:22 lbeddow@tails.cs.ucl.ac.uk

# make a new terminal tab for the tunnel
terminator --new-tab -x sshpass -f $PASSFILE ssh -L $PORT:$JOB_NODE:22 $USERNAME@$LOGIN_NODE

# make a new terminal tab for transfers
terminator --new-tab --working-directory=~/mymujoco

# jump into the job submit node (sshpass doesn't work with double jump -J)
ssh -J $USERNAME@$LOGIN_NODE $USERNAME@$JOB_NODE
