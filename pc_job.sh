# bash script to run several trainings on the lab PC

# where to save terminal output to
LOG_FOLDER=/home/luke/training_logs

# take input arguments into an array, if specified
if [ "$#" -eq 0 ]; then
    ARRAY_INDEXES=(1 2 3)
    echo No arguments specified, using default which is: $ARRAY_INDEXES
else
    ARRAY_INDEXES=( "$@" )
    echo Arguments specified were: $ARRAY_INDEXES
fi

# current time for naming training files
LUKE_JOB_SUBMIT_TIME=$(date +%d-%m-%y-%H:%M)

# add mujoco to the shared library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mujoco-2.1.5/lib

# navigate to correct directory
cd ~/mymujoco/rl

# wrapper to catch ctrl+c and kill all background processes
trap 'trap - SIGINT && kill 0' SIGINT

echo Submitting jobs now
echo Saving logs to $LOG_FOLDER/

for I in ${ARRAY_INDEXES[@]}
do
    python3 array_training_DQN.py $I $LUKE_JOB_SUBMIT_TIME \
    >> $LOG_FOLDER/train_luke-PC_${LUKE_JOB_SUBMIT_TIME}_${I}.txt &
    echo Submitted job: train_luke-PC_${LUKE_JOB_SUBMIT_TIME}_${I}
done

echo All jobs submitted

echo Waiting for submitted jobs to complete...
wait 

echo ...finished all jobs