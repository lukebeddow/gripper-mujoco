# These are flags you must include - Two memory and one runtime.
# Runtime is either seconds or hours:min:sec

#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l h_rt=23:59:0 

#These are optional flags but you probably want them in all jobs

#$ -S /bin/bash
#$ -j y
#$ -N MyTrainDQN_4

#The code you want to run now goes here.

hostname
date

source mypython/python3/bin/activate
export LD_LIBRARY_PATH=/share/apps/python-3.6.9/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

cd ~/mymujoco/rl
python3 TrainDQN.py

date
