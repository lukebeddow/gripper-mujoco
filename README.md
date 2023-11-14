# grippermujoco

Code for simulating my gripper with mujoco. The ```src``` folder contains c++ code which interfaces with mujoco, this is compiled into binaries in ```bin``` as well as a python module in ```rl/env/mjpy/bind.so``` for use with machine learning frameworks.

## Download

Clone with submodules to include the ```description``` submodule:

```git clone --recurse-submodules /git/address/...```

Then update the submodule to track the main branch, so that it stays up to date and also you can push commits directly to the ```luke-gripper-description``` repo by editing within this repository.

```
git submodule update --remote
cd description/
git checkout main
cd ..
```

## Installations

This code has the following dependencies:

* MuJoCo - download here: https://github.com/deepmind/mujoco/releases, this repo has been tested with version ```2.1.5``` and ```2.2.0```. Save the folder in your files.
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
PYTHON = /home/luke/pyenv/py3.8_mujoco
PYTHON_EXE = $(PYTHON)/bin/python
PYTHON_INCLUDE = $(PYTHON)/include/python3.8

# local machine library locations
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

Select the path to the mjcf files in ```$(MJCF_PATH)```. These are the models that are the robot models and object models that will be loaded into mujoco. They are contained in the ```mjcf``` folder of this repository. There are multiple object sets you can choose from. The object set can also be changed later in the code, here sets the default option. Further information is given in the 'Build sets' section of this readme.

Next, configure python:
* The ```$(PYTHON)``` variable is for convenience only (for the next two variables) and is not used. It does not need to be set.
* ```$(PYTHON_EXE)``` indicates which python executable should be run, this is for ```make sets```. For system python, simply use ```python3```, otherwise, use the path to the python executable. Above shows this for a python ```venv```.
* ```$(PYTHON_INCLUDE)``` should be the path to the python header files. Common examples:
  * For a ```venv```, do as above. However, note that sometimes this include folder is empty, in which case make a symbolic link to the system python header files. See the 'Troubleshooting' section at the bottom.
  * For system python it will likely be ```/usr/include/python3.6``` or similar. 

Finally, edit all of the library locations with the correct paths.
* If you have the source of a header only library (eg pybind11 above), simply add the path to this
* If you have a system library already (eg armadillo above), simply add the library with ```-l``` + ```libary-name```
* If you have a local library (eg mujoco above), either add the full path to the library location (not shown above) or use ```-L``` to specify the library folder and then ```-l``` to specify the library name in that folder (shown above)
* The ```RENDER_PATH``` and ```RENDER_LIBS``` refer to GLFW, this should be a system install so do not need to be changed. They are seperate to enable compiling without rendering support.
* If using a python virtual environment, ensure the ```include``` folder for this virtual environment is not empty. See the 'Troubleshooting' section at the bottom.

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
* ```pip install pyyaml==5.4.1 # must be this version to maintain compatibility with xacro```

Finally, if you wish to manually build sets:
* Choose your gripper settings by editing ```description/config/gripper.yaml```
* Next, choose or create your object set. Look in the ```description/mujoco/object_sets``` folder. A valid set must be inside this folder, being a yaml file whose name begins with 'set' and will be found by the ```set*.yaml``` wildcard.
* Run ```make sets SET=set_name```, replacing ```set_name``` with your desired set. You should see the set copied into the ```mjcf``` folder from the root of the repository.
* Many additional options exist, for further usage look into the ```luke-gripper-description``` submodule.

## Run

Having followed the commands above, and specified your library locations to a particular build option (your version of ```mybuild```), now compile and create the default object set (swapping ```mybuild``` for your chosen command):

```
make all mybuild
make sets mybuild SET=set9_fullset
```

### Run c++ GUI

To run the c++ part of the code and visualise MuJoCo in a GUI, you can use the the executable ```bin/mysimulate```. There are optional command line arguments to specify additional parameters.

```
bin/mysimulate

Command line syntax:
  [-g, --gripper] gripper_file      choose a specific gripper file eg gripper_N8_28
  [-N, --segments] number           how many finger segments, default: 8. Note this is overriden by [-g, --gripper]
  [-w, --width] number              what finger width to use, default: 28. Note this is overriden by [-g, --gripper]
  [-o, --object-set] set_name       which object set to use, default: set9_fullset
  [-t, --task] number               which task file number to use, default: 0
  [-p, --path] path                 path to object set, default path set as $(MJCF_PATH) in 'buildsettings.mk'
```

### Run a python training

To run a python training, we can use ```rl/launch_training.py```. This file has a variety of command line arguments (with are specified under ```if __name__ == "__main__":```. The main ones are:

```
rl/launch_training.py

Command line syntax:
  [-p, --program] program_name      name of training program (you will define this) which must have NO whitespace
  [-j, --job] job_number            job number for this training, you will define how the job number affects the training, if at all
  [-t, --timestamp] %d-%m-%y_%H-%M  timestamp for training, mainly used to load old trainings, must follow format eg 15-01-23_12:34
  [-d, --device] device             select cpu or gpu in pytorch, should be either 'cpu' or 'cuda'
  [-r, --render]                    sets training to render to the screen
  [-c, --continue] number           continue a training specified with a timestamp and job number, can set [--new-endpoint] X, or [--extra-episodes] Y
  [-g, --plot]                      load and then plot graphs from an existing training specified with a timestamp and job number
  [--print-results]                 print table of results from previous training batch specified with a timestamp and for specific job numbers a [--job-string] "A:B" or "A B C D"
  [--demo] [num_trials]             render to the screen a demonstration test of a training specified with a timestamp and job number, default num_trials=30
  [--test] [object_set]             run a full test of a training specified with a timestamp and job number, optionally on a new object set, default=training set
  [--log-level] int                 set the log level, 0=none, 1=key info (default) 2=extra info, 3=per episode key info, 4=all info, actions/rewards etc
  [--no-delay]                      turn off default behaviour of sleeping for job_number seconds before running any job
  [--rngseed] seed                  set a specific rngseed (warning: trainings are NOT currently reproducible, seeding is pointless)
```

For the majority of cases, the workflow is to define a new 'program' in ```rl/launch_training.py```, and then run this with ```python3 launch_training.py -p program_name -j job_num```.

You define a new program by addding it to the bottom of ```rl/launch_training.py```, after ```if __name__ == "__main__":```. The very bottom part of this file is structured as an ```if args.program == "A"...elif args.program == "B"...elif args.program == "C" etc...```, switching between different training programs. You can add your own training program by copying the following template into the ```if..elif...``` and before the last ```else:```. Note that the python object ```tm``` is the training manager from ```rl/TrainingManager.py```.

```
elif args.program == "example_template":

    # define what to vary this training, dependent on job number
    vary_1 = None
    vary_2 = None
    vary_3 = None
    repeats = None
    tm.param_1_name = None
    tm.param_2_name = None
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()
    
    # # apply the varied settings (very important!)
    # tm.settings["A"]["B"] = tm.param_1
    # tm.settings["C"]["D"] = tm.param_2
    # tm.settings["E"]["F"] = tm.param_3

    # # choose any additional settings to change
    # tm.settings["A"]["B"] = X
    # tm.settings["C"]["D"] = Y
    # tm.settings["E"]["F"] = Z

    # create the environment
    env = tm.make_env()

    # make the agent, may depend on variable settings above
    layers = [env.n_obs, 64, 64, env.n_actions]
    network = networks.VariableNetwork(layers, device=args.device)
    agent = Agent_DQN(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()
```

Agents are defined in the folder ```rl/agents``` and the actual trainer which performs the training is defined in ```rl/Trainer.py```. The environment is defined in ```env/MjEnv.py``` and is dependent also on the compiled c++ code. These are the three essential parts, and each have their own settings, which are then managed by the ```TrainingMananger```. By writing these settings in the dictionary ```TrainingManager.settings``` when you call ```TrainingMananger.make_env()``` the environment settings are overwritten into the resultant environment called ```env```. Hence, changing the dictionary after this point will not change the actual ```env```, instead you would have to edit the env directly (advanced usage). During the call to ```TrainingManager.run_training()``` settings for the trainer and the agent are overwritten from the dictionary to the underlying objects, therefore the dictionary can be edited at any time before that call.

### Python training batch example

The program template given above expects that multiple different jobs will be run within this program, performing a grid search across a set of parameters. For example, if you wanted to check three different learning rates, ```1e-5, 1e-4, 1e-3``` and two different lengths of training ```10_000, 20_000``` that would take 6 trainings to cover all the possible combinations. Say we wanted to repeat each training 3 times, now we will have 18 total trainings. So our job numbers will vary from 1 to 18, and we can write that like this:

```
elif args.program == "vary_lr_and_num_ep":

    # define what to vary this training, dependent on job number
    vary_1 = [1e-5, 1e-4, 1e-3]
    vary_2 = [10_000, 20_000]
    vary_3 = None
    repeats = 3
    tm.param_1_name = "learning_rate"
    tm.param_2_name = "num_episodes"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()
    
    # apply the varied settings (very important!)
    tm.settings["Agent"]["learning_rate"] = tm.param_1
    tm.settings["Trainer"]["num_episodes"] = tm.param_2

    # create the environment
    env = tm.make_env()

    # make the agent, may depend on variable settings above
    layers = [env.n_obs, 64, 64, env.n_actions]
    network = networks.VariableNetwork(layers, device=args.device)
    agent = Agent_DQN(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()
```

Then, multiple terminals we can run our trainings with our job_numbers going 1 to 18.

However, rather than make 18 different terminals, a convienience script, ```launch_training_array.sh``` is provided. We can run all 18 trainings with only one command:

```
./launch_training_array.sh -p vary_lr_and_num_ep -j "1:18"
```

The job string ```1:18``` will be parsed into the numbers "1 2 3 ... 18". Then, 18 processes will start, each one given a different job number (1-18). Similarly, we can also use ```-j "1 4 7 10 13 16"``` to run only those job numbers. The bash script ```launch_training_array.sh``` passes your arguments to the processes running the python script ```rl/launch_training.py```, but has some optional arguments which are not passed. By default, terminal output from all of the processes will go into a folder called ```training_logs```. You must make this folder: ```mkdir ~/training_logs```.

```
./launch_training_array.sh

Command line syntax:
  [-d, --debug]                     debug mode, print training output in terminal and not in ~/training_logs, also add '--no-delay' to trainings
  [-s, --stagger] num               run only num trainings at a time, all must finish before next batch will start
```

By default, if you run a training program, this script prints a table of results for those trainings once they are all finished (using the ```--print-results``` input to ```rl/launch_training.py```).

## Cluster

For building on the cluster, use:
```bash
./configure.sh
make cluster
```

Then to submit a job for example ```qsub mymujoco/array_job.sh```. It is important to have built with ```make cluster``` otherwise it will fail.

## Troubleshooting

### Mujoco library not found at runtime

If you get an error that the mujoco shared library ```libmujoco.so``` is not found, you will need to tell the computer where this library is:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/mujoco-2.1.5/lib
```

### Python not compiling, no python.h

On ubuntu 20, when you create a virtual environment it does not seem to populate the ```include``` folder, hence the compilation fails to find python.h. What you should do is:

```
sudo apt install python3.8-dev
ln -s /usr/include/python3.8/ /home/path/to/venv/name/include/python3.8
```

First we install the headers, then create a symbolic link to them. Now python should compile correctly.

