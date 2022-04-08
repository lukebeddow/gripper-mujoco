# grippermujoco

Code for simulating my gripper with mujoco. The ```src``` folder contains c++ code which interfaces with mujoco, this is compiled into binaries in ```bin``` as well as a python module in ```rl/env/mjpy/bind.so``` for use with machine learning frameworks.

To build, run ```make``` in the root of the directory.

Before running this code, you need access to the ```gripper_task.xml``` files which are not in this repository.

## Installations

This code has three key library dependencies:

* MuJoCo - install here: https://github.com/deepmind/mujoco/releases
* Pybind11 - source code here: https://github.com/pybind/pybind11, ```sudo apt-get install pybind11-dev``` may also work
* Armadillo - install here: http://arma.sourceforge.net/download.html, or source: https://gitlab.com/conradsnicta/armadillo-code

Once the libraries are available, their file path location (or system library name) needs to be inserted into the Makefile in this project.

## Cluster

For building on the cluster, use:
```bash
./configure.sh
make cluster
```

Then to submit a job for example ```qsub array_job.sh```. It is important to use ```make cluster``` otherwise it will fail.


