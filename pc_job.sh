# bash script to run several trainings on the lab PC

# to run a new training from array_training.py and train args 1,2,3 use:
#   $ ./pc_job.sh 1 2 3

# to continue a training from array_training.py and train args 1,2,3 use:
#   $ ./pc_job.sh continue 07-06-22-16:34 1 2 3
#   

# from: https://stackoverflow.com/questions/1401002/how-to-trick-an-application-into-thinking-its-stdout-is-a-terminal-not-a-pipe
faketty() {
    script -qfc "$(printf "%q " "$@")" /dev/null
}

# where to save terminal output to
LOG_FOLDER=/home/luke/training_logs

# add mujoco to the shared library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mujoco-2.1.5/lib

# current time for naming training files
LUKE_JOB_SUBMIT_TIME=$(date +%d-%m-%y-%H:%M)

# navigate to correct directory
cd ~/mymujoco/rl

# take input arguments into an array, if specified
if [ "$#" -eq 0 ]; then
    echo No arguments specified
    exit
else
    if [ "$1" == "continue" ]; then

        # parse arguments
        CONTINUE="$1"
        LUKE_JOB_SUBMIT_TIME="$2"
        ARRAY_INDEXES=( "${*:3}" )

        # echo information to the terminal
        echo PC job instructed to continue training
        echo The following jobs will be continued:
        for I in ${ARRAY_INDEXES[@]}
        do
            echo luke-PC_${LUKE_JOB_SUBMIT_TIME}_A${I}
        done
        
    else
        ARRAY_INDEXES=( "$*" )
    fi
    echo PC job arguments specified were: $ARRAY_INDEXES
fi

# wrapper to catch ctrl+c and kill all background processes
trap 'trap - SIGINT && kill 0' SIGINT

echo Submitting jobs now
echo Saving logs to $LOG_FOLDER/

for I in ${ARRAY_INDEXES[@]}
do
    JOB_NAME=luke-PC_A${I}_${LUKE_JOB_SUBMIT_TIME}
    faketty python3 array_training_DQN.py $I $LUKE_JOB_SUBMIT_TIME $CONTINUE \
    > $LOG_FOLDER/$JOB_NAME.txt &
    echo Submitted job: $JOB_NAME
done

echo All jobs submitted

echo Waiting for submitted jobs to complete...
wait 

echo ...finished all jobs