# These are flags you must include - Two memory and one runtime.
# Runtime is either seconds or hours:min:sec

#$ -l tmem=8G
#$ -l h_vmem=8G
#$ -l h_rt=95:59:0 

# Some important notes
#dollar -t X-Y   -> submit array job inclusive of both X and Y
#dollar -tc Z    -> Z is max number of concurrent jobs at once

#$ -S /bin/bash
#$ -j y
#$ -N ArrayTrainDQN_13
#$ -t 1-20

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
python3 array_training_DQN.py ${SGE_TASK_ID} $LUKE_JOB_SUBMIT_TIME

date
