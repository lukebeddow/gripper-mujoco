# grippermujoco

Code for simulating my gripper with mujoco. The ```src``` folder contains c++ code which interfaces with mujoco, this is compiled into binaries in ```bin``` as well as a python module in ```rl/env/mjpy/bind.so``` for use with machine learning frameworks.

## Installations

This code has the following dependencies:

* MuJoCo - download here: https://github.com/deepmind/mujoco/releases, this repo has been tested with version ```2.1.5```. Save the folder in your files.
* Pybind11 - download the source code here: https://github.com/pybind/pybind11 and save it amoungst your files, ```sudo apt-get install pybind11-dev``` unfortunately does not work
* Armadillo - (if you have ROS you likely already have this library, use ```ldconfig -p | grep armadillo``` to see if the library already exists), otherwise install here: http://arma.sourceforge.net/download.html, or source: https://gitlab.com/conradsnicta/armadillo-code
* GLFW3 - install with ```sudo apt-get install libglfw3-dev```

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

Next, the locations of the library installations should be put into the Makefile. Every machine that compiles this project has slightly different paths and library locations. Open the ```buildsettings.mk``` makefile and you will see a variety of options for different library locations. Add a new one which corresponds to your machine, for example ```mybuild```. You will need to set:

```make
ifeq ($(filter mybuild, $(MAKECMDGOALS)), mybuild)

# path to the mjcf (mujoco model) files, most likely they are in the mjcf folder of this repo
MJCF_PATH = /home/luke/mymujoco/mjcf/object_set_1

# local machine library locations
PYTHON_PATH = /usr/include/python3.6m # path to python version you want to use for the python module
PYBIND_PATH = /home/luke/pybind11 # path to your pybind source folder
ARMA_PATH = # none, use system library # path to armadillo, you can leave this blank if you have it installed already
MUJOCO_PATH = /home/luke/mujoco-2.1.5 # path to your mujoco folder
CORE_LIBS = -larmadillo -$(MUJOCO_PATH)/lib/libmujoco.so # core libraries for armadillo and mujoco
RENDER_LIBS = -lglfw # rendering library
DEFINE_VAR = -DLUKE_MJCF_PATH='"$(MJCF_PATH)"' # define c++ macros, you don't need to edit this

endif
```

Currently, compilation is set up as well for the old version of mujoco (2.1.0) so for the time being you will also need to move some files. Go to the folder where you have saved your version of mujoco (eg ```mujoco-2.1.5```), then copy across the two uitools files into the include folder:

```
cd /your/path/to/mujoco-2.1.5
cp sample/uitools.c sample/uitools.h include/
```

To build the project simply naviage to the root directory and run ```make all mybuild``` - swap ```mybuild``` for the name you chose.

Make options:

```make
make all     # build all targets
make py      # build only the python targets
make cpp     # build only the cpp targets
make clean   # wipe all build files
make debug   # build targets in debug mode
```

## Run

To run the simluation and play around, use run ```bin/mysimulate task 0```.

To run the python training, have a look at ```rl/TrainDQN.py```.

## Cluster

For building on the cluster, use:
```bash
./configure.sh
make cluster
```

Then to submit a job for example ```qsub mymujoco/array_job.sh```. It is important to have built with ```make cluster``` otherwise it will fail.


