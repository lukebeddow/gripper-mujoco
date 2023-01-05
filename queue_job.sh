# bash script to run several trainings on the lab PC

# usage: ./queue_job.sh -j "1:45" -q 15 [pc_job.sh args eg --device cuda --debug]

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


# ----- handle inputs ----- #

QUEUE_ARGS=() # arguments passed to sub scripts only in a queue scenario

# loop through input args
for (( i = 1; i <= "$#"; i++ ));
do
case ${!i} in
    # look for the jobs, queues and stagger flags
    -j | --jobs ) (( i++ )); jobs=$(parseJobs ${!i}); echo Jobs are $jobs ;;
    -q | --queue ) (( i++ )); QUEUE_NUM=${!i}; echo Queue mode selected, there will be $QUEUE_NUM queues ;;
    -s | --stagger ) (( i++ )); echo Stagger flag removed before running pc_job.sh ;;
    # everything else leave it unchanged to pass into pc_job.sh
    * ) QUEUE_ARGS+=( ${!i} ) ;;
esac
done

# if jobs are not specified, throw an error
if [ -z "$jobs" ]
then
    echo Incorrect inputs, jobs to do must be specified with -j or --jobs flags
    exit
fi

# if queue number is not specified, throw an error
if [ -z "$QUEUE_NUM" ]
then
    echo Incorrect inputs, queue number must be specified with -q or --queue flags
    exit
fi

# extract the job indicies
ARRAY_INDEXES=("$jobs")
    
# submit the jobs into QUEUE_NUM number of queues
for (( q = 0; q < $QUEUE_NUM; q++ ));
do
    JOBS_IN_QUEUE=()
    for I in ${ARRAY_INDEXES[@]}
    do
        # echo $I gives $(( $I % $QUEUE_NUM ))
        if [ $(( $I % $QUEUE_NUM )) == "$q" ]
        then
            JOBS_IN_QUEUE+=( $I )
        fi
    done

    # randomly shuffle the queue order
    JOBS_IN_QUEUE=( $(shuf -e "${JOBS_IN_QUEUE[@]}"))

    # print and submit (sleep so all print statements complete before jobs start spamming teminal)
    echo Queue number $(( $q + 1 )) jobs are "${JOBS_IN_QUEUE[*]}"
    sleep 1 && ./pc_job.sh \
                    ${QUEUE_ARGS[@]} \
                    --jobs "${JOBS_IN_QUEUE[*]}" \
                    --stagger 1 \
                    --timestamp "$(date +%d-%m-%y-%H:%M)" \
                    &

done

echo All queues submitted

echo Waiting for submitted queues to complete...
wait 

echo ...finished all queues