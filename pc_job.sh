# bash script to run several trainings on the lab PC

# to run a new training from array_training.py and train args 1,2,3 use:
#   $ ./pc_job.sh -j "1 2 3"

# to continue a training from array_training.py and train args 1,2,3 use:
#   $ ./pc_job.sh -j "1 2 3" -c y -t 07-06-22-16:34 -m luke-PC
#   

helpFunction()
{
   echo -e "\nUsage:"
   echo -e "Run jobs 1-4:\t $0 -j '1 2 3 4'"
   echo -e "Continue 1-4:\t $0 -j '1 2 3 4' -c -t old_timestamp"
   echo -e "\nOptions:"
   echo -e "\t -j ['ARGS'] jobs to run (need to be in quotes), eg -j '1 2 3 4'"
   echo -e "\t -c continue previous training"
   echo -e "\t -m [ARG] machine, if continuing, on what machine eg -m luke-PC"
   echo -e "\t -t [ARG] timestamp in form dd-mm-yy-hr:mn eg -t 07-12-22-15:34"
   echo -e "\t -h print help information"
   exit 1 # exit script after printing help
}

# defaults
continue=false
machine=luke-PC
timestamp=$(date +%d-%m-%y-%H:%M)

# where to save terminal output to
LOG_FOLDER=/home/luke/training_logs

# a colon after a flag character indicates it expects an argument
while getopts "j:cm:t:h" opt
do
   case "$opt" in
      j ) jobs="$OPTARG" ; echo Jobs input are $jobs ;;
      c ) continue=true ; echo Continue has been set to true ;;
      m ) machine="$OPTARG" ; echo Machine has been specified as $machine ;;
      t ) timestamp="$OPTARG" ; echo Timestamp has been specified as $timestamp ;;
      h ) helpFunction ;; # help flag
    #   ? ) helpFunction ;; # print helpFunction in case parameter is non-existent
   esac
done

# if jobs are not specified
if [ -z "$jobs" ]
then
    echo Incorrect inputs, jobs to do must be specified with -j flag
    helpFunction
fi

# from: https://stackoverflow.com/questions/1401002/how-to-trick-an-application-into-thinking-its-stdout-is-a-terminal-not-a-pipe
faketty() {
    script -qfc "$(printf "%q " "$@")" /dev/null
}

# add mujoco to the shared library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mujoco-2.1.5/lib

# navigate to correct directory
cd ~/mymujoco/rl

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

# extract the job indicies
ARRAY_INDEXES=("$jobs")

# echo information to the terminal
echo -e "\nThe following jobs will be done:"
for I in ${ARRAY_INDEXES[@]}
do
    echo ${machine}_${timestamp}_A${I}
done

# wrapper to catch ctrl+c and kill all background processes
trap 'trap - SIGINT && kill 0' SIGINT

echo Submitting jobs now
echo Saving logs to $LOG_FOLDER/

for I in ${ARRAY_INDEXES[@]}
do
    JOB_NAME=${machine}_${timestamp}_A${I}
    faketty python3 array_training_DQN.py $I $timestamp $CONTINUE \
    > $LOG_FOLDER/$JOB_NAME.txt &
    echo Submitted job: $JOB_NAME
done

echo All jobs submitted

echo Waiting for submitted jobs to complete...
wait 

echo ...finished all jobs