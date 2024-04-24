# bash script to run several trainings at the same time

# define key directory structure
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
path_to_mymujoco=$SCRIPT_DIR
python_folder=rl
LOG_FOLDER=~/training_logs
RUN_PREFIX=run

# ----- helpful functions ----- #

parseJobs()
{
    # if jobs are colon seperated, eg '1:4', expand this to '1 2 3 4'
    if [[ "$@" == *":"* ]]
    then
        tosearch="$@"
        colonarray=(${tosearch//:/ })
        firstnum=${colonarray[0]}
        endnum=${colonarray[-1]}
        newjobarray=( )
        for ((i=$firstnum;i<=$endnum;i++)); do
            newjobarray+="$i " 
        done
        echo $newjobarray
    else
        # if not colon seperated, return what we were given
        echo $@
    fi
}

# from: https://stackoverflow.com/questions/1401002/how-to-trick-an-application-into-thinking-its-stdout-is-a-terminal-not-a-pipe
faketty() {
    script -qfc "$(printf "%q " "$@")" /dev/null
}

# ----- handle inputs ----- #

# default inputs
timestamp="$(date +%d-%m-%y_%H-%M)"
KEEP_TIMESTAMP='N'
FAKETTY=faketty
LOGGING='Y'
DEBUG=
PRINT_RESULTS='N'
PRINT_RESULTS_AFTER='N'
RANDOMISE='N'

PY_ARGS=() # arguments passed directly into python without parsing

# loop through input args, look for script specific arguments
for (( i = 1; i <= "$#"; i++ ));
do
  case ${!i} in
    # with arguments, increment i
    -j | --jobs ) (( i++ )); jobs=$(parseJobs ${!i}); echo jobs are $jobs ;;
    -t | --timestamp ) (( i++ )); timestamp=${!i}; echo Timestamp set to $timestamp ;;
    -s | --stagger ) (( i++ )); STAGGER=${!i}; echo stagger is $STAGGER ;;
    # without arguments
    -k | --keep-time ) KEEP_TIMESTAMP='Y'; echo Keeping current timestamp ;;
    -f | --no-faketty ) FAKETTY=; echo faketty disabled ;;
    -d | --debug ) LOGGING='N'; PRINT_RESULTS_AFTER='N'; echo Debug mode on, terminal logging or printing results after ;;
    --debug-2 ) LOGGING='N'; PRINT_RESULTS_AFTER='N'; DEBUG='--log-level 2'; echo FULL DEBUG MODE ON, nothing will be saved ;;
    --print ) LOGGING='N'; PRINT="--print"; echo Printing mode on, no training ;;
    --print-results ) PRINT_RESULTS='Y' ;;
    --randomise ) RANDOMISE='Y'; echo Randomising job order submission ;;
    # everything else passed directly to python
    --program ) PRINT_RESULTS_AFTER='Y' ; PY_ARGS+=( ${!i} ) ;;
    * ) PY_ARGS+=( ${!i} ) ;;
  esac
done

echo Python only arguments are: ${PY_ARGS[@]}

# ----- main job submission ----- #

# create the log folder if needed
if [ $LOGGING = 'Y' ]
then
    mkdir -p $LOG_FOLDER
fi

# navigate to correct directory
cd $path_to_mymujoco/$python_folder || exit

# are we printing a results table
if [ $PRINT_RESULTS = 'Y' ]
then
    echo -e "\nPrinting results table\n"

    python3 launch_training.py \
        -t $timestamp \
        ${PY_ARGS[@]} \
        --print-results \
        --no-delay \
        --no-saving \
        --log-level 0

    echo -e "\nResults table complete"
    exit
fi

# extract the job indicies
ARRAY_INDEXES=("$jobs")

# echo information to the terminal
echo -e "\nThe following jobs have been selected:"
for I in ${ARRAY_INDEXES[@]}
do
    echo -e "${RUN_PREFIX}_${timestamp}_A${I}"
done

# wrapper to catch ctrl+c and kill all background processes
trap 'trap - SIGINT && kill 0' SIGINT

echo -e "\nSubmitting jobs now"
echo Saving logs to $LOG_FOLDER/

IND=0

# extracts the first job number (so an input of "2 3 1" gives 2 incorrectly)
MIN_JOB_NUM=${ARRAY_INDEXES%% *} # https://stackoverflow.com/questions/15685736/how-to-extract-a-particular-element-from-an-array-in-bash

# randomly shuffle the order (only makes sense if stagger is set)
if [ $RANDOMISE = 'Y' ]
then
    RAND_INDEXES=( $(shuf -e "$jobs"))
    echo ${RAND_INDEXES[@]}
    echo Shuffle should have happened
    echo $(shuf -e "$jobs")
fi

# loop through the jobs we have been assigned
for I in ${ARRAY_INDEXES[@]}
do
    if [ $KEEP_TIMESTAMP = 'Y' ]
    then
        keep_timestamp="$(date +%d-%m-%y_%H-%M)"
        JOB_NAME="${RUN_PREFIX}_${keep_timestamp}_A${I}"
    else
        JOB_NAME="${RUN_PREFIX}_${timestamp}_A${I}"
    fi

    # if we are logging terminal output to a seperate log file
    if [ $LOGGING = 'Y' ]
    then
        # first create the needed file (: is null operator), then direct output to it
        : > $LOG_FOLDER/$JOB_NAME
        exec > $LOG_FOLDER/$JOB_NAME
    fi

    # execute the command in the background
    $FAKETTY python3 launch_training.py \
        -j $I \
        -t $timestamp \
        ${PY_ARGS[@]} \
        $PRINT \
        $DEBUG \
        --smallest-job-num $MIN_JOB_NUM \
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

# print a results table upon completion of the training
if [ $PRINT_RESULTS_AFTER = 'Y' ]
then
    echo -e "\nNow printing final results table\n"

    python3 launch_training.py \
        -t $timestamp \
        ${PY_ARGS[@]} \
        --print-results \
        --no-delay \
        --no-saving \
        --log-level 0

    echo -e "\nResults table complete"
fi