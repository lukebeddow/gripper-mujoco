# bash script to run several trainings on the lab PC

# to run a new training from array_training.py and train args 1,2,3 use:
#   $ ./pc_job.sh -j "1 2 3"

# to continue a training from array_training.py and train args 1,2,3 use:
#   $ ./pc_job.sh -j "1 2 3" -c -t 07-06-22-16:34 -m luke-PC
#   

helpFunction()
{
   echo -e "\nUsage:"
   echo -e "Run jobs 1-4:      $0 -j '1 2 3 4'"
   echo -e "Run 2 at a time:   $0 -j '1 2 3 4' --stagger 2"
   echo -e "Pass python args:  $0 -j '1 2 3 4' -c --stagger 2 --device cpu"
   echo -e "\nOptions:"
   echo -e "\t -j, --jobs ['ARGS'] jobs to run (need to be in quotes), eg -j '1 2 3 4'"
   echo -e "\t -s, --stagger [ARG] staggered job, submit jobs in groups of ARG eg 3 at a time"
   echo -e "\t -f, --no-faketty disable faketty output file logging "
   echo -e "\t -h print help information"
   exit 1 # exit script after printing help
}

# defaults
machine=luke-PC
timestamp="$(date +%d-%m-%y-%H:%M)"
FAKETTY=faketty

PY_ARGS=() # arguments passed directly into python without parsing

# loop through input args, look for script specific arguments
for (( i = 1; i <= "$#"; i++ ));
do
  case ${!i} in
    -j | --jobs ) (( i++ )); jobs=${!i}; echo jobs are $jobs ;;
    -t | --timestamp ) (( i++ )); timestamp=${!i}; echo Timestamp set to $timestamp ;;
    -m | --machine ) (( i++ )); machine=${!i}; MACHINE="-m $machine"; echo Machine set to $machine ;;
    -f | --no-faketty ) FAKETTY=; echo faketty disabled ;;
    -s | --stagger ) (( i++ )); STAGGER=${!i}; echo stagger is $STAGGER ;;
    * ) PY_ARGS+=( ${!i} ) ;;
  esac
done

echo Python only arguments are: ${PY_ARGS[@]}

# if jobs are not specified, throw an error
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

IND=0

for I in ${ARRAY_INDEXES[@]}
do
    JOB_NAME="${machine}_${timestamp}_A${I}"
    $FAKETTY python3 array_training_DQN.py \
        -j $I \
        -t $timestamp \
        $MACHINE \
        ${PY_ARGS[@]} \
        > "$LOG_FOLDER/$JOB_NAME.txt" &
    echo Submitted job: "$JOB_NAME"

    # for submitting staggered jobs
    IND=$((IND + 1))
    if [ ! -z "$STAGGER" ]
    then
        if [ $(expr $IND % $STAGGER) == "0" ];
        then
            echo -e "Staggering now, waiting for all jobs to finish..."
            wait
            echo -e " ...finished\n"
        fi
    fi

done

echo All jobs submitted

echo Waiting for submitted jobs to complete...
wait 

echo ...finished all jobs