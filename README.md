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

The c++ outputs will be put into a folder called ```bin```, the python module into ```rl/env/mjpy```, and build files into a folder called ```build```. These folders are automatically created when the Makefile is run. This folder structure is specified in the Makefile, and can be adjusted if desired, but this is not needed.

**Define library locations**

In order to build, the locations of the dependent libraries needs to be specified. This is specified in the ```buildsettings.mk``` file. **You will need to edit ```buildsettings.mk``` in order to build**. This file contains library locations for a variety of compile locations, you will need to add your compile location to this file. The file is structured as an ```if ... else if ... else if ... endif```. Copy the following code to the bottom of the file:


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
DEFINE_VAR = -DLUKE_MJCF_PATH='"$(MJCF_PATH)"' \
             -DLUKE_MACHINE='"$(MACHINE)"'      # define c++ macros, no need to edit

endif
```

In the above code, the keyword ```mybuild``` is used to select this path/library locations. So you would run ```make all mybuild``` to have these paths set. Change ```mybuild``` to the command of your choice:

<pre>
ifeq ($(filter <b>mybuild</b>, $(MAKECMDGOALS)), <b>mybuild</b>)
</pre>

Next, select the path to the mjcf files in ```$(MJCF_PATH)```. These are the models that are the robot models and object models that will be loaded into mujoco. They are contained in the ```mjcf``` folder of this repository. There are multiple object sets you can choose from. The object set can also be changed later in the code, here sets the default option.

Finally, edit all of the library locations with the correct paths.
* If you have the source of a header only library (eg pybind11 above), simply add the path to this
* If you have a system library already (eg armadillo above), simply add the library with ```-l``` + ```libary-name```
* If you have a local library (eg mujoco above), either add the full path to the library location (as above) or use ```-L``` to specify the library folder and then ```-l``` to specify the library name in that folder (not shown above)

**Temporary extra step for building**

Currently, compilation is set up as well for the old version of mujoco (2.1.0) so for the time being you will also need to move some files. Go to the folder where you have saved your version of mujoco (eg ```mujoco-2.1.5```), then copy across the two uitools files into the include folder:

```shell
cd /your/path/to/mujoco-2.1.5
cp sample/uitools.c sample/uitools.h include/
```
**Build commands**

To build the project simply naviage to the root directory and run ```make all mybuild``` -> swap ```mybuild``` for the name you chose.

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


