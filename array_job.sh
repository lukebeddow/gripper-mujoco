# These are flags you must include - Two memory and one runtime.
# Runtime is either seconds or hours:min:sec

#$ -l tmem=20G
#$ -l h_vmem=20G
#$ -l h_rt=73:59:0 

# These are optional flags but you probably want them in all jobs
# t X-Y is inclusive of both X and Y
# tc is total number of concurrent jobs at once

#$ -S /bin/bash
#$ -j y
#$ -N ArrayTrainDQN_8
#$ -t 1-30
#$ -tc 15

#The code you want to run now goes here.

hostname
date

source mypython/python3/bin/activate
export LD_LIBRARY_PATH=/share/apps/python-3.6.9/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

cd ~/mymujoco/rl
python3 array_training_DQN.py ${SGE_TASK_ID}

date
