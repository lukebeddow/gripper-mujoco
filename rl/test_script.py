#!/usr/bin/env python3

#!/usr/bin/env python3

from env.MjEnv import MjEnv
import numpy as np
from matplotlib import pyplot as plt

# create and prepare the mujoco instance
mj = MjEnv()
mj.testing_xmls = 0

# define the names of the object sets we want to investigate
object_sets = [
  "set_test_seg5",
  "set_test_seg6",
  "set_test_seg7",
  "set_test_seg8",
  # "set_test_seg9",
  # "set_test_seg10",
  # "set_test_seg15",
  # "set_test_seg20",
  # "set_test_seg30"
]

# list of stable timesteps for each object set
mujoco_timesteps = [
  0.00353,
  0.0024,
  0.00176,
  0.00133,
  # 0.00102,
  # 0.000841,
  # 0.000334,
  # 0.000198,
  # 8.42E-05
]

# initialise/wipe variables
data = []
num_sets = len(object_sets)

# loop through each object set and gather data
done_one = False
for i in range(0, num_sets):

  # if done_one: break

  mj._load_object_set(name=object_sets[i])
  mj._load_xml()
  mj.mj.set.mujoco_timestep = mujoco_timesteps[i] * 0.5
  mj.task_reload_chance = -1
  mj.mj.hard_reset()
  mj.reset()

  # run the validation regime
  print("Curve validation running for object set:", object_sets[i])
  print_out = False
  finger_data = mj.mj.curve_validation_regime(print_out)
  data.append(finger_data)

  done_one = True

# now process the data and plot it

fig, axs = plt.subplots(num_sets, 1)

for ind in range(num_sets):

  force = 2

  # print out
  data[ind].entries[force - 1].print()

  # lets start by plotting a basic example
  x =        np.array(data[ind].entries[force - 1].f1.x)
  pred_x =   np.array(data[ind].entries[force - 1].f1.pred_x)
  y =        np.array(data[ind].entries[force - 1].f1.y)
  pred_y =   np.array(data[ind].entries[force - 1].f1.pred_y)
  theory_y = np.array(data[ind].entries[force - 1].f1.theory_y)

  axs[ind].plot(x,      y,        "b-o",  label="mujoco")
  axs[ind].plot(pred_x, pred_y,   "r--*", label="mymodel")
  axs[ind].plot(x,      theory_y, "g",    label="theory")
  
  # axs[ind].set_title("No. segments N =", len(y) - 1)
  # axs[ind].axis("equal")
  # axs[ind].legend()

fig.set_size_inches(10, 10)