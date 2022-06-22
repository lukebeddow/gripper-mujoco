# These are flags you must include - Two memory and one runtime.
# Runtime is either seconds or hours:min:sec

#$ -l tmem=2G
#$ -l h_vmem=2G
#$ -l h_rt=72:0:0

#These are optional flags but you probably want them in all jobs

#$ -S /bin/bash
#$ -j y
#$ -N cluster_job

#The code you want to run now goes here.

hostname
date

# source python and export the library location
source mypython/python3/bin/activate
export LD_LIBRARY_PATH=/share/apps/python-3.6.9/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/clusterlibs/mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/clusterlibs/mujoco/mujoco-2.1.5/lib

cd ~/mymujoco/rl

# run the script and pass in flags
python3 array_training_DQN.py "$@"

date