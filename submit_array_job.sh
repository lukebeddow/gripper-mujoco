# submit an array job, setting a global environment variable
# the flag -V means the qsub jobs will copy our current environment variables

export LUKE_JOB_SUBMIT_TIME=$(date +%d-%m-%y-%H:%M)

qsub -V ~/mymujoco/array_job.sh
