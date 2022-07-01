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

export LUKE_JOB_SUBMIT_TIME=$(date +%d-%m-%y-%H:%M)
export OPENBLAS_NUM_THREADS=1 # fix for weird cluster bug

qsub -V ~/mymujoco/array_job.sh -t $LUKE_JOB_SUBMIT_TIME
