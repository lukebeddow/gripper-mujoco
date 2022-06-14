# bash script to run several trainings on the lab PC

# to run a new training from array_training.py and train args 1,2,3 use:
#   $ ./pc_job.sh 1 2 3

# to continue a training from array_training.py and train args 1,2,3 use:
#   $ ./pc_job.sh continue 07-06-22-16:34 1 2 3
#   

helpFunction()
{
   echo ""
   echo "Usage: $0 -d direction -c command"
   echo -e "\t-j jobs to run, eg 1,2,3,4"
   echo -e "\t-c continue, include if training is continuing"
   echo -e "\t-m machine, if continuing, on what machine eg luke-PC"
   echo -e "\t-t timestamp, what is the timestamp of the training eg 07-12-22-15:34"
   exit 1 # exit script after printing help
}

# defaults
continue=false
default_machine=luke-PC

while getopts "j:c:m:t:" opt
do
   case "$opt" in
      j ) jobs="$OPTARG" ;;
      c ) continue=true ;;
      m ) machine="$OPTARG" ;;
      t ) timestamp="$OPTARG" ;;
      ? ) helpFunction ;; # print helpFunction in case parameter is non-existent
   esac
done

# print helpFunction in case parameters are empty
if [ -z "$jobs" ]
then
   echo "All or some of the parameters are empty";
   helpFunction
fi

# echo parameter selection to the user
echo "Echoing user inputs:"
echo "jobs are: $jobs"
echo "continue is: $continue"
echo "machine is: $machine"
echo "timestamp is:" $timestamp

# from: https://stackoverflow.com/questions/1401002/how-to-trick-an-application-into-thinking-its-stdout-is-a-terminal-not-a-pipe
faketty() {
    script -qfc "$(printf "%q " "$@")" /dev/null
}

# where to save terminal output to
LOG_FOLDER=/home/luke/training_logs

# add mujoco to the shared library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mujoco-2.1.5/lib

# navigate to correct directory
cd ~/mymujoco/rl

# if a timestamp is not specified use current time
if [ -z "$timestamp" ]
then
    LUKE_JOB_SUBMIT_TIME=$(date +%d-%m-%y-%H:%M)
else
    LUKE_JOB_SUBMIT_TIME="$timestamp"
fi

# if we are continuing training
if [ $continue = true ]
then

    # on what machine are we continuing
    if [ -z "$machine" ]
    then
        CONTINUE="continue"
    else
        CONTINUE="continue_${machine}"
    fi
fi

# if a machine is specified, override
if [ -z "$machine" ]
then
   machine="$default_machine"
fi

#     # parse arguments
#     CONTINUE="$1"
#     LUKE_JOB_SUBMIT_TIME="$2"
#     ARRAY_INDEXES=( "${*:3}" )

#     # echo information to the terminal
#     echo PC job instructed to continue training
#     echo The following jobs will be continued:
#     for I in ${ARRAY_INDEXES[@]}
#     do
#         echo luke-PC_${LUKE_JOB_SUBMIT_TIME}_A${I}
#     done
    
# else
#     ARRAY_INDEXES=( "$*" )
# fi

ARRAY_INDEXES=("$jobs")

# echo information to the terminal
echo The following jobs will be done:
for I in ${ARRAY_INDEXES[@]}
do
    echo ${machine}_${LUKE_JOB_SUBMIT_TIME}_A${I}
done

echo CONTINUE is $CONTINUE

# exit

# # take input arguments into an array, if specified
# if [ "$#" -eq 0 ]; then
#     echo No arguments specified
#     exit
# else
#     if [ "$1" == "continue" ]; then

#         # parse arguments
#         CONTINUE="$1"
#         LUKE_JOB_SUBMIT_TIME="$2"
#         ARRAY_INDEXES=( "${*:3}" )

#         # echo information to the terminal
#         echo PC job instructed to continue training
#         echo The following jobs will be continued:
#         for I in ${ARRAY_INDEXES[@]}
#         do
#             echo luke-PC_${LUKE_JOB_SUBMIT_TIME}_A${I}
#         done
        
#     else
#         ARRAY_INDEXES=( "$*" )
#     fi
#     echo PC job arguments specified were: $ARRAY_INDEXES
# fi

# wrapper to catch ctrl+c and kill all background processes
trap 'trap - SIGINT && kill 0' SIGINT

echo Submitting jobs now
echo Saving logs to $LOG_FOLDER/

for I in ${ARRAY_INDEXES[@]}
do
    JOB_NAME=${machine}_${LUKE_JOB_SUBMIT_TIME}_A${I}
    faketty python3 array_training_DQN.py $I $LUKE_JOB_SUBMIT_TIME $CONTINUE \
    > $LOG_FOLDER/$JOB_NAME.txt &
    echo Submitted job: $JOB_NAME
done

echo All jobs submitted

echo Waiting for submitted jobs to complete...
wait 

echo ...finished all jobs