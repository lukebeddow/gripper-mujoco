# These are flags you must include - Two memory and one runtime.
# Runtime is either seconds or hours:min:sec

#$ -l tmem=4G
#$ -l h_vmem=4G
#$ -l h_rt=72:0:0

#These are optional flags but you probably want them in all jobs

#$ -S /bin/bash
#$ -j y
#$ -N cluster_job_4GB

#The code you want to run now goes here.

hostname
date

# echo commands
echo Command line given arguments are: "$@"

export OPENBLAS_NUM_THREADS=1 # fix for weird cluster bug

# default options
TIMESTAMP="--timestamp $(date +%d-%m-%y-%H:%M)"
WANDB="--no-wandb"
JOB="-j 1"

# loop through input args, look for script specific arguments
for (( i = 1; i <= "$#"; i++ ));
do
  case ${!i} in
    # with arguments, incrememnt i
    -t | --timestamp ) (( i++ )); TIMESTAMP="-t ${!i}"; echo Timestamp set to ${!i} ;;
    -j | --jobs ) (( i++ )); JOB="-j ${!i}"; echo Job input set to: $JOB ;;
    # without arguments
    --yes-wandb ) WANDB=; echo use_wandb set to TRUE ;;
    # everything else passed directly to python (note quoted inputs will UNQUOTE eg -j "1 2 3 4" > -j 1 2 3 4)
    * ) ARGS+=( ${!i} ) ;;
  esac
done

# now prepare to submit the job

cd

# source python and export the library location
source mypython/python3/bin/activate
export LD_LIBRARY_PATH=/share/apps/python-3.6.9/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/clusterlibs/mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/clusterlibs/mujoco/mujoco-2.1.5/lib

cd ~/mymujoco/rl

# run the script and pass in flags
python3 array_training_DQN.py \
  $JOB \
  $TIMESTAMP \
  $WANDB \
  ${ARGS[@]}

date