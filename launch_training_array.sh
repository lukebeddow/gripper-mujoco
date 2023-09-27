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
FAKETTY=faketty
LOGGING='Y'
PRINT_RESULTS_AFTER='Y'

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
    -f | --no-faketty ) FAKETTY=; echo faketty disabled ;;
    -d | --debug ) LOGGING='N'; DEBUG=" --no-delay"; echo Debug mode on, terminal logging ;;
    --print ) LOGGING='N'; PRINT="--print"; echo Printing mode on, no training ;;
    --print-results ) LOGGING='N'; PRINT="--print-results --no-delay" ;;
    # everything else passed directly to python
    * ) PY_ARGS+=( ${!i} ) ;;
  esac
done

echo Python only arguments are: ${PY_ARGS[@]}

# ----- main job submission ----- #

# navigate to correct directory
cd $path_to_mymujoco/$python_folder || exit

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

# loop through the jobs we have been assigned
for I in ${ARRAY_INDEXES[@]}
do

    JOB_NAME="${RUN_PREFIX}_${timestamp}_A${I}"

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
        --log-level 0

    echo -e "\nResults table complete"
fi