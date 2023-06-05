# submit an array job, setting a global environment variable
# the flag -V means the qsub jobs will copy our current environment variables
# "$@" passes all arguments, no arguments are needed for a new job
# but there are optinal arguments:
#   -c continue a previous job
#   -t [ARG] timestamp to use with job
#   -m [ARG] machine name of job that is being continued
#
# eg ./submit_array_job.sh
# eg ./submit_array_job.sh -c -t 12-07-22-09:43 -m cluster

# echo commands
echo Command line given arguments are: "$@"

export OPENBLAS_NUM_THREADS=1 # fix for weird cluster bug

# default options
TIMESTAMP="--timestamp $(date +%d-%m-%y-%H:%M)"
WANDB="--no-wandb"

# loop through input args, look for script specific arguments
for (( i = 1; i <= "$#"; i++ ));
do
  case ${!i} in
    # with arguments, incrememnt i
    -t | --timestamp ) (( i++ )); TIMESTAMP="-t ${!i}"; echo Timestamp set to ${!i} ;;
    # without arguments
    --yes-wandb ) WANDB=; echo use_wandb set to TRUE ;;
    # everything else passed directly to python (note quoted inputs will UNQUOTE eg -j "1 2 3 4" > -j 1 2 3 4)
    * ) ARGS+=( ${!i} ) ;;
  esac
done

# submit the job and pass over any command line arguments (nb "$@" is required)
qsub -V ~/mymujoco/array_job.sh \
  $TIMESTAMP \
  $WANDB \
  ${ARGS[@]}
