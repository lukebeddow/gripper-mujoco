# bash script to run several trainings on the lab PC

# to run a new training from array_training.py and train args 1,2,3 use:
#   $ ./pc_job.sh -j "1 2 3"

# to continue a training from array_training.py and train args 1,2,3 use:
#   $ ./pc_job.sh -j "1 2 3" -c -t 07-06-22-16:34 -m luke-PC
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
   echo -e "\t -n do not use weights and biases for logging live"
   echo -e "\t -l logging job, log to weights and biases"
   echo -e "\t -p logging job, plot graphs on screen"
   echo -e "\t -h print help information"
   exit 1 # exit script after printing help
}

# defaults
machine=luke-PC
timestamp="$(date +%d-%m-%y-%H:%M)"
FAKETTY=faketty

# a colon after a flag character indicates it expects an argument
while getopts "j:t:m:clpnfh" opt
do
   case "$opt" in
      j ) jobs="$OPTARG" ; echo Jobs input are "$jobs" ;;
      t ) timestamp="$OPTARG" ; echo Timestamp has been specified as "$timestamp" ;;
      m ) machine="$OPTARG" ; MACHINE="-m $machine" ; echo Machine has been specified as "$machine" ;;
      c ) CONTINUE="-c" ; echo Continue has been set to true ;;
      l ) LOG_WANDB="-l" ; echo log_wandb has been set to true ;;
      p ) LOG_PLOT="-p" ; echo log_plot has been set to true ;;
      n ) NO_WANDB="-n" ; echo no_wandb has been set to true ;;
      f ) FAKETTY=; echo faketty has been disabled ;;
      h ) helpFunction ;; # help flag
      * ) echo Invalid flag received ; helpFunction ;;
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

# where to save terminal output to
LOG_FOLDER=~/training_logs

# navigate to correct directory
cd ~/mymujoco/rl || exit

# extract the job indicies
ARRAY_INDEXES=("$jobs")

# echo information to the terminal
echo -e "\nThe following jobs have been selected:"
for I in ${ARRAY_INDEXES[@]}
do
    echo -e "${machine}_${timestamp}_A${I}"
done

# wrapper to catch ctrl+c and kill all background processes
trap 'trap - SIGINT && kill 0' SIGINT

echo -e "\nSubmitting jobs now"
echo Saving logs to $LOG_FOLDER/

for I in ${ARRAY_INDEXES[@]}
do
    JOB_NAME="${machine}_${timestamp}_A${I}"
    $FAKETTY python3 array_training_DQN.py \
        -j $I \
        -t $timestamp \
        $MACHINE \
        $CONTINUE \
        $NO_WANDB \
        $LOG_WANDB \
        $LOG_PLOT \
        > "$LOG_FOLDER/$JOB_NAME.txt" &
    echo Submitted job: "$JOB_NAME"
done

echo All jobs submitted

echo Waiting for submitted jobs to complete...
wait 

echo ...finished all jobs