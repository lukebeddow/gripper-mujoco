# bash script to run several trainings on the lab PC

# it is not recommended to use more than 4 trainings
NUM_TRAININGS=4
ARRAY_INDEXES=(5 10 15 20)
LOG_FOLDER=/home/luke/training_logs

# current time for naming training files
LUKE_JOB_SUBMIT_TIME=$(date +%d-%m-%y-%H:%M)

# add mujoco to the shared library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mujoco-2.1.5/lib

# navigate to correct directory
cd ~/mymujoco/rl

echo Submitting jobs now
echo Saving logs to $LOG_FOLDER/

for I in ${ARRAY_INDEXES[@]}
do
    python3 array_training_DQN.py $I $LUKE_JOB_SUBMIT_TIME \
    > $LOG_FOLDER/train_luke-PC_${LUKE_JOB_SUBMIT_TIME}_${I}.txt &
    echo Submitted job: train_luke-PC_${LUKE_JOB_SUBMIT_TIME}_${I}
done

echo All jobs submitted