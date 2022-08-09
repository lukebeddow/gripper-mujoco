#!/bin/bash

cd

source mypython/python3/bin/activate
export LD_LIBRARY_PATH=/share/apps/python-3.6.9/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/clusterlibs/mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/clusterlibs/mujoco/mujoco-2.1.5/lib

scl enable devtoolset-8 bash
