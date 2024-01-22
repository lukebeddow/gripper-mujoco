# bash script to run several trainings at the same time

# from: https://stackoverflow.com/questions/1401002/how-to-trick-an-application-into-thinking-its-stdout-is-a-terminal-not-a-pipe
faketty() {
    script -qfc "$(printf "%q " "$@")" /dev/null
}

# wrapper to catch ctrl+c and kill all background processes
trap 'trap - SIGINT && kill 0' SIGINT

MODULUS=193e9
BACKGROUND=&
SLEEP=1
LOGGING_TO=data_gather_logs

THICK_1=0.86e-3
THICK_2=0.96e-3

# direct all logging to /dev/null except last (to keep an eye on progress)
: > $LOGGING_TO/log_0p87_28_1.txt
exec > $LOGGING_TO/log_0p87_28_1.txt
python3 plot_finger_models.py -t $THICK_1 -w 28e-3 -E $MODULUS -f 0 &

sleep $SLEEP
: > $LOGGING_TO/log_0p97_24_1.txt
exec > $LOGGING_TO/log_0p97_24_1.txt
python3 plot_finger_models.py -t $THICK_2 -w 24e-3 -E $MODULUS -f 0 &

sleep $SLEEP
: > $LOGGING_TO/log_0p97_28_1.txt
exec > $LOGGING_TO/log_0p97_28_1.txt
python3 plot_finger_models.py -t $THICK_2 -w 28e-3 -E $MODULUS -f 0 &

sleep $SLEEP
: > $LOGGING_TO/log_0p87_28_2.txt
exec > $LOGGING_TO/log_0p87_28_2.txt
python3 plot_finger_models.py -t $THICK_1 -w 28e-3 -E $MODULUS -f 1 &

sleep $SLEEP
: > $LOGGING_TO/log_0p97_24_2.txt
exec > $LOGGING_TO/log_0p97_24_2.txt
python3 plot_finger_models.py -t $THICK_2 -w 24e-3 -E $MODULUS -f 1 &

sleep $SLEEP
: > $LOGGING_TO/log_0p97_28_2.txt
exec > $LOGGING_TO/log_0p97_28_2.txt
python3 plot_finger_models.py -t $THICK_2 -w 28e-3 -E $MODULUS -f 1 &

sleep $SLEEP
: > $LOGGING_TO/log_0p87_28_3.txt
exec > $LOGGING_TO/log_0p87_28_3.txt
python3 plot_finger_models.py -t $THICK_1 -w 28e-3 -E $MODULUS -f 2 &

sleep $SLEEP
: > $LOGGING_TO/log_0p97_24_3.txt
exec > $LOGGING_TO/log_0p97_24_3.txt
python3 plot_finger_models.py -t $THICK_2 -w 24e-3 -E $MODULUS -f 2 &

sleep $SLEEP
: > $LOGGING_TO/log_0p97_28_3.txt
exec > $LOGGING_TO/log_0p97_28_3.txt
python3 plot_finger_models.py -t $THICK_2 -w 28e-3 -E $MODULUS -f 2 &

echo All jobs submitted

echo Waiting for submitted jobs to complete...
wait 

echo ...finished all jobs

exit