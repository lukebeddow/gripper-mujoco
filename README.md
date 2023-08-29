# grippermujoco

Code for simulating my gripper with mujoco. The ```src``` folder contains c++ code which interfaces with mujoco, this is compiled into binaries in ```bin``` as well as a python module in ```rl/env/mjpy/bind.so``` for use with machine learning frameworks.

## Installations

This code has the following dependencies:

* MuJoCo - download here: https://github.com/deepmind/mujoco/releases, this repo has been tested with version ```2.1.5```. Save the folder in your files.
* Pybind11 - download the source code here: https://github.com/pybind/pybind11 and save it amoungst your files, ```sudo apt-get install pybind11-dev``` unfortunately does not work
* Armadillo - (if you have ROS you likely already have this library, use ```ldconfig -p | grep armadillo``` to see if the library already exists). If not, try the terminal commands here: https://www.uio.no/studier/emner/matnat/fys/FYS4411/v13/guides/installing-armadillo/. If that fails install here: http://arma.sourceforge.net/download.html, or source: https://gitlab.com/conradsnicta/armadillo-code
* GLFW3 - install with ```sudo apt-get install libglfw3-dev```

## Building

Building the project is done using a Makefile in the root of the directory. The Makefile has two types of target, it compiles an executable only from c++ code as well as compiling a python module which can be imported into python code.

The c++ outputs will be put into a folder called ```bin```, the python module into ```rl/env/mjpy```, and build files into a folder called ```build```. These folders are automatically created when the Makefile is run. This folder structure is specified in the Makefile, and can be adjusted if desired, but this is not needed.

**Define library locations**

In order to build, the locations of the dependent libraries needs to be specified. This is specified in the ```buildsettings.mk``` file. **You will need to edit ```buildsettings.mk``` in order to build**. This file contains library locations for a variety of compile locations, you will need to add your compile location to this file. The file is structured as an ```if ... else if ... else if ... endif```. Copy the following code to the bottom of the file:

```
ifeq ($(filter mybuild, $(MAKECMDGOALS)), mybuild)

# set this command goal as a phony target (important)
.PHONY: mybuild

# what machine are we compiling for
MACHINE = machine-name

# mjcf files location (model files like gripper/objects)
MJCF_PATH = /path/to/repo/luke-gripper-mujoco/mjcf

# path to python executable (eg venv) and header files
# if using system python, simply 'PYTHON_EXE = python3'
PYTHON = /home/luke/pyenv/py3.8_mujoco
PYTHON_EXE = $(PYTHON)/bin/python
PYTHON_INCLUDE = $(PYTHON)/include/python3.8

# local machine library locations
# PYTHON_PATH = /home/luke/pyenv/py3.8_mujoco/bin/python
PYBIND_PATH = /home/luke/luke-gripper-mujoco/libs/pybind11
ARMA_PATH = # none, use system library
MUJOCO_PATH = /home/luke/luke-gripper-mujoco/libs/mujoco/mujoco-2.1.5
MUJOCO_LIB = $(MUJOCO_PATH)/lib
RENDER_PATH = # none, use system library
CORE_LIBS = -L$(MUJOCO_LIB) -lmujoco -larmadillo 
RENDER_LIBS = -lglfw
DEFINE_VAR = -DLUKE_MJCF_PATH='"$(MJCF_PATH)"' \
             -DLUKE_MACHINE='"$(MACHINE)"'

# extras
MAKEFLAGS += -j8 # jN => use N parallel cores

endif
```

In the above code, the keyword ```mybuild``` is used as a phony make command goal to select these path/library locations. So you would run ```make all mybuild``` to have these paths set. Change ```mybuild``` to the command of your choice:

<pre>
ifeq ($(filter <b>mybuild</b>, $(MAKECMDGOALS)), <b>mybuild</b>)

# add a phony target for this command goal (important)
.PHONY: <b>mybuild</b>
</pre>

Next, select the path to the mjcf files in ```$(MJCF_PATH)```. These are the models that are the robot models and object models that will be loaded into mujoco. They are contained in the ```mjcf``` folder of this repository. There are multiple object sets you can choose from. The object set can also be changed later in the code, here sets the default option.

Next, configure python:
* The ```$(PYTHON)``` variable is for convenience only (for the next two variables) and is not used. It does not need to be set.
* ```$(PYTHON_EXE)``` indicates which python executable should be run, this is for ```make sets```. For system python, simply use ```python3```, otherwise, use the path to the python executable. Above shows this for a python ```venv```.
* ```$(PYTHON_INCLUDE)``` should be the path to the python header files. Common examples:
  * For a ```venv```, do as above. However, note that sometimes this include folder is empty, in which case make a symbolic link to the system python header files with:
```ln -s /usr/include/python3.8 /path/to/venv/venvname/include/```
  * For system python it will likely be ```/usr/include/python3.6``` or similar. 

Finally, edit all of the library locations with the correct paths.
* If you have the source of a header only library (eg pybind11 above), simply add the path to this
* If you have a system library already (eg armadillo above), simply add the library with ```-l``` + ```libary-name```
* If you have a local library (eg mujoco above), either add the full path to the library location (not shown above) or use ```-L``` to specify the library folder and then ```-l``` to specify the library name in that folder (shown above)
* If using a python virtual environment, ensure the ```include``` folder for this virtual environment is not empty. If it is, you can create a symbolic link from the main python installation:

**Build commands**

To build the project simply naviage to the root directory and run ```make all mybuild``` -> swap ```mybuild``` for the name you chose.

Make options:

```make
make all     # build all targets
make py      # build only the python targets
make cpp     # build only the cpp targets
make clean   # wipe all build files
make debug   # build targets in debug mode
make sets    # build the mujoco model files and object sets
```

## Build sets

The repository requires a mujoco model file in order to work. The current build supports autogeneration of these model files, in which case you may be able to simply run. However, note that the ```luke-gripper-description``` submodule must be downloaded (check that the ```description``` folder is not empty). If clonining use:

```git clone --recurse-submodules https/.....```

If already cloned, you can pull the submodule with:

```git submodule update --init```

 Ensure the ```$(PYTHON_EXE)``` you have chosen has the following modules installed:
* ```pip install xacro```
* ```pip install lxml```
* ```pip install pyyaml```

Finally, if you wish to manually build sets:
* Choose your gripper settings by editing ```description/config/gripper.yaml```
* Next, choose or create your object set. Look in the ```description/mujoco/object_sets``` folder. A valid set must be inside this folder, being a yaml file whose name begins with 'set' and will be found by the ```set*.yaml``` wildcard.
* Run ```make sets SET=set_name```, replacing ```set_name``` with your desired set. You should see the set copied into the ```mjcf``` folder from the root of the repository.
* Many additional options exist, for further usage look into the ```luke-gripper-description``` submodule.

## Run

To run the c++ part of the code and visualise MuJoCo in a GUI, you can use the the executable ```bin/mysimulate```. There are optional command line arguments to specify additional parameters:

```
bin/mysimulate

Command line syntax:
  [-g, --gripper] gripper_file      choose a specific gripper file eg gripper_N8_28
  [-N, --segments] number           how many finger segments, default: 8. Note this is overriden by [-g, --gripper]
  [-w, --width] number              what finger width to use, default: 28. Note this is overriden by [-g, --gripper]
  [-o, --object-set] set_name       which object set to use, default: set6_fullset_800_50i
  [-t, --task] number               which task file number to use, default: 0
  [-p, --path] path                 path to object set, default: /home/luke/mymujoco/mjcf
```

To run the python training, have a look at ```rl/TrainDQN.py``` and ```rl/array_training_DQN.py```.

To run multiple trainings concurrently look at ```pc_job.sh``` or to run in queues look at ```queue_job.sh```.

If you get an error that the mujoco shared library ```libmujoco.so``` is not found, you will need to tell the computer where this library is:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/mujoco-2.1.5/lib
```

## Cluster

For building on the cluster, use:
```bash
./configure.sh
make cluster
```

Then to submit a job for example ```qsub mymujoco/array_job.sh```. It is important to have built with ```make cluster``` otherwise it will fail.


