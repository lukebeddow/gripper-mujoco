#!/bin/bash

# before running this script, you need to set up a tunnel and leave it running
# in another terminal. For example:
#     $ ssh -L 3333:vic.cs.ucl.ac.uk:22 lbeddow@tails.cs.ucl.ac.uk

# Most useful commands commands:
#     - upload all source code and job scripts with $ ./transfer.sh -c source -d up
#     - upload all object sets with $ ./transfer.sh -c mjcf -d up
#     - download all source code and trained models with $ ./transfer.sh -c all -d down

# set variables
PASSFILE=key.txt
PORT=3333

# folder path to the upload/download directories eg UPLOADFROM=myfolder -> ~/myfolder
UPLOADFROM=mymujoco
UPLOADTO=mymujoco
DOWNLOADFROM=mymujoco
DOWNLOADTO=cluster

helpFunction()
{
   echo ""
   echo "Usage: $0 -d direction -c command"
   echo -e "\t-d transfer direction, two options either 'up' or 'down'"
   echo -e "\t-c command eg 'py', 'cpp', 'models', 'job', 'source', 'output', 'all'"
   echo -e "\t-f filename, directly transfer a file"
   exit 1 # exit script after printing help
}

while getopts "d:f:c:" opt
do
   case "$opt" in
      d ) direction="$OPTARG" ;;
      c ) command="$OPTARG" ;;
      f ) filename="$OPTARG" ;;
      ? ) helpFunction ;; # print helpFunction in case parameter is non-existent
   esac
done

# print helpFunction in case parameters are empty
if [ -z "$direction" ] || [ -z "$command" ] && [ -z "$filename" ]
then
   echo "All or some of the parameters are empty";
   helpFunction
fi

# echo parameter selection to the user
echo "direction is: $direction"
echo "command is: $command"
echo "filename is: $filename"

if [ "$direction" != "up" ] && [ "$direction" != "down" ]
then
   echo "Direction can only be 'up' or 'down'";
   helpFunction
fi

# python - transfer the source code in rl and rl/env
if [ "$command" = "py" ] || [ "$command" = "source" ] || [ "$command" = "all" ]
then
   if [ "$direction" = "up" ]
   then
      echo uploading the rl folder to /$UPLOADTO/rl
      sshpass -f $PASSFILE scp -P $PORT \
         ~/$UPLOADFROM/rl/* \
         lbeddow@localhost:~/$UPLOADTO/rl/
      echo uploading the env folder to /$UPLOADTO/rl/env
      sshpass -f $PASSFILE scp -P $PORT \
         ~/$UPLOADFROM/rl/env/* \
         lbeddow@localhost:~/$UPLOADTO/rl/env/
   fi
   if [ "$direction" = "down" ] 
   then
      echo downloading the rl folder to /$DOWNLOADTO/rl
      sshpass -f $PASSFILE scp -P $PORT \
         lbeddow@localhost:~/$DOWNLOADFROM/rl/* \
         ~/$DOWNLOADTO/rl
      echo downloading env folder to /$DOWNLOADTO/rl/env
      sshpass -f $PASSFILE scp -P $PORT \
         lbeddow@localhost:~/$DOWNLOADFROM/rl/env/* \
         ~/$DOWNLOADTO/rl/env/
   fi
fi

# cpp - transfer the source code in src and the Makefile
if [ "$command" = "cpp" ] || [ "$command" = "source" ] || [ "$command" = "all" ]
then
   if [ "$direction" = "up" ]
   then
      echo uploading the src folder to /$UPLOADTO/src
      sshpass -f $PASSFILE scp -P $PORT \
         ~/$UPLOADFROM/src/* \
         lbeddow@localhost:~/$UPLOADTO/src/
      echo uploading the makefile to /$UPLOADFROM
      sshpass -f $PASSFILE scp -P $PORT \
         ~/$UPLOADFROM/Makefile \
         lbeddow@localhost:~/$UPLOADTO/Makefile
      echo uploading buildsettings.mk to /$UPLOADFROM
      sshpass -f $PASSFILE scp -P $PORT \
         ~/$UPLOADFROM/buildsettings.mk \
         lbeddow@localhost:~/$UPLOADTO/buildsettings.mk
      echo uploading the configure.sh script
      sshpass -f $PASSFILE scp -P $PORT \
         ~/$UPLOADFROM/configure.sh \
         lbeddow@localhost:~/$UPLOADTO/configure.sh
   fi
   if [ "$direction" = "down" ]
   then
      echo downloading the src folder to /$DOWNLOADTO/src
      sshpass -f $PASSFILE scp -P $PORT \
         lbeddow@localhost:~/$DOWNLOADFROM/src/* \
         ~/$DOWNLOADTO/src/
      echo downloading the makefile to /$DOWNLOADTO
      sshpass -f $PASSFILE scp -P $PORT \
         lbeddow@localhost:~/$DOWNLOADFROM/Makefile \
         ~/$DOWNLOADTO/
      echo downloading the configure.sh script to /$DOWNLOADTO
      sshpass -f $PASSFILE scp -P $PORT \
         lbeddow@localhost:~/$UPLOADFROM/configure.sh \
         ~/$DOWNLOADTO/
   fi
fi

# models - transfer all of the trained model files
if [ "$command" = "models" ] || [ "$command" = "output" ] || [ "$command" = "all" ]
then
   if [ "$direction" = "up" ]
   then
      echo uploading the entire models folder to /$UPLOADFROM/rl
      sshpass -f $PASSFILE scp -P $PORT -r \
         ~/$UPLOADTO/rl/models \
         lbeddow@localhost:~/$UPLOADFROM/rl/
   fi
   if [ "$direction" = "down" ]
   then
      echo downloading entire models folder to /$DOWNLOADTO/rl
      sshpass -f $PASSFILE scp -P $PORT -r \
         lbeddow@localhost:~/$DOWNLOADFROM/rl/models \
         ~/$DOWNLOADTO/rl/
   fi
fi

# job - cluster job files (ending in job.sh), cluster output files
if [ "$command" = "job" ] || [ "$command" = "output" ] || [ "$command" = "all" ] \
   || { [ "$command" = "source" ] && [ "$direction" = "up" ]; }
then
   if [ "$direction" = "up" ]
   then
      echo uploading *job.sh to /$UPLOADTO
      sshpass -f $PASSFILE scp -P $PORT \
         ~/$UPLOADFROM/*job.sh \
         lbeddow@localhost:~/$UPLOADTO/
   fi
   if [ "$direction" = "down" ]
   then 
      echo downloading the job output files from root ~/ to /$DOWNLOADTO/output
      sshpass -f $PASSFILE scp -P $PORT \
         lbeddow@localhost:~/*.o* \
         ~/$DOWNLOADTO/output/
      echo downloading *job.sh to /$DOWNLOADTO
      sshpass -f $PASSFILE scp -P $PORT \
         lbeddow@localhost:~/$DOWNLOADFROM/*job.sh \
         ~/$DOWNLOADTO/
   fi
fi

# mjcf - transfer xml files
if [ "$command" = "mjcf" ]
then
   if [ "$direction" = "up" ]
   then
      echo uploading the mjcf files to /mjcf
      sshpass -f $PASSFILE scp -P $PORT -r \
         ~/$UPLOADFROM/mjcf \
         lbeddow@localhost:~/$UPLOADTO/
   fi
   if [ "$direction" = "down" ]
   then 
      echo downloading the mjcf files to /$DOWNLOADTO/mjcf
      sshpass -f $PASSFILE scp -P $PORT -r \
         lbeddow@localhost:~/$UPLOADTO/mjcf \
         ~/$DOWNLOADTO/
   fi
fi

# mymujoco - upload the entire folder (note you will need to recompile!)
if [ "$command" = "mymujoco" ]
then
   if [ "$direction" = "up" ]
   then
      echo uploading the entire $UPLOADFROM folder to root
      sshpass -f $PASSFILE scp -P $PORT -r \
         ~/$UPLOADFROM \
         lbeddow@localhost:~/
   fi
   if [ "$direction" = "down" ]
   then 
      echo downloading the entire $DOWNLOADFROM folder to $DOWNLOADTO
      sshpass -f $PASSFILE scp -P $PORT -r \
         lbeddow@localhost:~/$DOWNLOADFROM/* \
         ~/$DOWNLOADTO/
   fi
fi

# upload or download the cluster libraries
if [ "$command" = "clusterlibs" ]
then
   if [ "$direction" = "up" ]
   then
      echo clusterlibs upload not yet implemented
   fi
   if [ "$direction" = "down" ]
   then 
      echo downloading ~/clusterlibs folder to $DOWNLOADTO/clusterlibs
      sshpass -f $PASSFILE scp -P $PORT -r \
         lbeddow@localhost:~/clusterlibs/* \
         ~/$DOWNLOADTO/clusterlibs/
   fi
fi

# transfer a specific file
if [ ! -z "$filename" ]
then
   if [ "$direction" = "up" ]
   then
      echo uploading ~/$UPLOADFROM/$filename
      sshpass -f $PASSFILE scp -P $PORT \
         ~/$UPLOADFROM/$filename \
         lbeddow@localhost:~/$UPLOADTO/$filename
   fi
   if [ "$direction" = "down" ]
   then 
      echo download ~/$DOWNLOADFROM/$filename
      sshpass -f $PASSFILE scp -P $PORT \
         lbeddow@localhost:~/$DOWNLOADFROM/$filename \
         ~/$DOWNLOADTO/$filename
   fi
fi