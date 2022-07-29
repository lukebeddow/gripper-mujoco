# bash script to run several trainings on the lab PC

# to run a new training from array_training.py and train args 1,2,3 use:
#   $ ./pc_job.sh -j "1 2 3"

# to continue a training from array_training.py and train args 1,2,3 use:
#   $ ./pc_job.sh -j "1 2 3" -c -t 07-06-22-16:34 -m luke-PC
#   

# define key directory structure
path_to_mymujoco=~/mymujoco
python_folder=rl
mujoco_lib_path=~/mujoco-2.1.5/lib
LOG_FOLDER=~/training_logs

helpFunction()
{
   echo -e "\nUsage:"
   echo -e "Run jobs 1-4:      $0 -j '1 2 3 4'"
   echo -e "Run 2 at a time:   $0 -j '1 2 3 4' --stagger 2"
   echo -e "Pass python args:  $0 -j '1 2 3 4' -c --stagger 2 --device cpu"
   echo -e "\nOptions:"
   echo -e "\t -j, --jobs ['ARGS'] jobs to run (need to be in quotes), eg -j '1 2 3 4'"
   echo -e "\t -s, --stagger [ARG] staggered job, submit jobs in groups of ARG eg 3 at a time"
   echo -e "\t -f, --no-faketty disable faketty output file logging"
   echo -e "\t -d, --debug print output in the terminal, not a log file"
   echo -e "\t --print special mode for array_training_dqn.py where index options are printed"
   echo -e "\t -h print help information"
   exit 1 # exit script after printing help
}

# from: https://stackoverflow.com/questions/1401002/how-to-trick-an-application-into-thinking-its-stdout-is-a-terminal-not-a-pipe
faketty() {
    script -qfc "$(printf "%q " "$@")" /dev/null
}

# ----- handle inputs ----- #

# defaults
machine=luke-PC
timestamp="$(date +%d-%m-%y-%H:%M)"
FAKETTY=faketty
LOGGING='Y'

PY_ARGS=() # arguments passed directly into python without parsing

# loop through input args, look for script specific arguments
for (( i = 1; i <= "$#"; i++ ));
do
  case ${!i} in
    # with arguments, incrememnt i
    -j | --jobs ) (( i++ )); jobs=${!i}; echo jobs are $jobs ;;
    -t | --timestamp ) (( i++ )); timestamp=${!i}; echo Timestamp set to $timestamp ;;
    -m | --machine ) (( i++ )); machine=${!i}; MACHINE="-m $machine"; echo Machine set to $machine ;;
    -s | --stagger ) (( i++ )); STAGGER=${!i}; echo stagger is $STAGGER ;;
    # without arguments
    -f | --no-faketty ) FAKETTY=; echo faketty disabled ;;
    -d | --debug ) LOGGING='N'; DEBUG="--no-wandb"; echo Debug mode on, terminal logging, no wandb ;;
    --print ) LOGGING='N'; PRINT="--print --no-wandb"; echo Printing mode on, no training ;;
    # everything else passed directly to python
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

# ----- begin main shell scripting ----- #

# add mujoco to the shared library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mujoco_lib_path

# navigate to correct directory
cd $path_to_mymujoco/$python_folder || exit

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

    # if we are logging terminal output to a seperate log file
    if [ $LOGGING = 'Y' ]
    then
        # first create the needed file (: is null operator), then direct output to it
        : > $LOG_FOLDER/$JOB_NAME
        exec > $LOG_FOLDER/$JOB_NAME
    fi

    # execute the command in the background
    $FAKETTY python3 array_training_DQN.py \
        -j $I \
        -t $timestamp \
        $MACHINE \
        ${PY_ARGS[@]} \
        $PRINT \
        $DEBUG \
        &

    # return output to terminal
    exec > /dev/tty

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