# These are flags you must include - Two memory and one runtime.
# Runtime is either seconds or hours:min:sec (1day=24, 2days=48, 3days=72, 4days=96)

#$ -l tmem=2G
#$ -l h_vmem=2G
#$ -l h_rt=71:59:0 

# Some important notes
#dollar -t X-Y   -> submit array job inclusive of both X and Y
#dollar -t X-Y:Z -> submit with stride Z, eg 2-10:2 means { 2, 4, 6, 8, 10 }
#dollar -tc Z    -> Z is max number of concurrent jobs at once

#$ -S /bin/bash
#$ -j y
#$ -N test_without_wandb
#$ -t 1-6

# The code you want to run now goes here.

hostname
date

# safety measure to stagger processes starting
sleep ${SGE_TASK_ID}

# source python and export the library location
source mypython/python3/bin/activate
export LD_LIBRARY_PATH=/share/apps/python-3.6.9/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/clusterlibs/mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/clusterlibs/mujoco/mujoco-2.1.5/lib

cd ~/mymujoco/rl

# run the script and pass in flags
python3 array_training_DQN.py \
  -j ${SGE_TASK_ID} \
  -m cluster \
  "$@"

date
