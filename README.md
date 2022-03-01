# grippermujoco

Code for simulating my gripper with mujoco. The ```src``` folder contains c++ code which interfaces with mujoco, this is compiled into binaries in ```bin``` as well as a python module in ```rl/env/mjpy/bind.so``` for use with machine learning frameworks.

To build, run ```make``` in the root of the directory.

For building on the cluster, use:
```bash
./configure.sh
make cluster
```

Then to submit a job for example ```qsub array_job.sh```

Before running this code, you need access to the ```gripper_task.xml``` files which are not in this repository.
