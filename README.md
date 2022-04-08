# grippermujoco

Code for simulating my gripper with mujoco. The ```src``` folder contains c++ code which interfaces with mujoco, this is compiled into binaries in ```bin``` as well as a python module in ```rl/env/mjpy/bind.so``` for use with machine learning frameworks.

To build, run ```make``` in the root of the directory.

## Installations

This code has three key library dependencies:

* MuJoCo - install here: https://github.com/deepmind/mujoco/releases
* Pybind11 - source code here: https://github.com/pybind/pybind11, ```sudo apt-get install pybind11-dev``` may also work
* Armadillo - (if you have ROS you may already have this library, use ```ldconfig -p | grep armadillo``` to see if the library already exists), install here: http://arma.sourceforge.net/download.html, or source: https://gitlab.com/conradsnicta/armadillo-code

## Building

Building the project is done using a Makefile in the root of the directory. The Makefile has two types of target, it compiles an executable only from c++ code as well as compiling a python module which can be imported into python code.

Create the following folders:

* In the root:
  * bin
  * build
* In build:
  * py
  * cpp
  * depends
* In rl/env:
  * mjpy

Notice that all of these folders are ignored in the projects .gitignore file. The Makefile will output temporaries into ```build```, the executables into ```bin```, and the python module into ```rl/env/mjpy```. In the Makefile this is specified by the following lines:

```make
# define directory structure (these folders must exist)
SOURCEDIR := src
BUILDDIR := build
BUILDPY := $(BUILDDIR)/py
BUILDCPP := $(BUILDDIR)/cpp
BUILDDEP := $(BUILDDIR)/depends
OUTCPP := bin
OUTPY := rl/env/mjpy
```

Next, the locations of the library installations should be put into the Makefile. There are two variants, one for the cluster and one for the local machine. Edit the entries for the local machine to put in where you have the libraries:

```make
# local machine library locations
PYTHON_PATH = /usr/include/python3.6m
PYBIND_PATH = /home/luke/pybind11
ARMA_PATH = # none, use system library
MUJOCO_PATH = /home/luke/.mujoco/mujoco210
CORE_LIBS = -lmujoco210 -larmadillo
RENDER_LIBS = -lGL -lglew ~/.mujoco/mujoco210/bin/libglfw.so.3
DEFINE_VAR = # none
```

The choice of python path defines what version of python the module will run on (so in this case Python 3.6 only). In cases where you have installed the libraries on the system, you do not need a path, only the library name in the ```CORE_LIBS``` variable with a -l prefix. From source, if it is a header only library (like pybind11 and armadillo), you only need to put the path.

To build the project simply naviage to the root directory and run ```make```.

Other options:

```make
make py    # build only the python targets
make cpp   # build only the cpp targets
make clean # wipe all build files
```

## Run

Before running this code, you need access to the ```gripper_task.xml``` files which are not in this repository.

To run the simluation and play around, use run ```bin/mysimulate task 0```

To run the python training, have a look at ```rl/TrainDQN.py```.

## Cluster

For building on the cluster, use:
```bash
./configure.sh
make cluster
```

Then to submit a job for example ```qsub array_job.sh```. It is important to use ```make cluster``` otherwise it will fail.


