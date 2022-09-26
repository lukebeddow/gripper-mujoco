#!/usr/bin/env python3

from env.MjEnv import MjEnv
import numpy as np
from matplotlib import pyplot as plt

# create and prepare the mujoco instance
mj = MjEnv(noload=True)
mj.testing_xmls = 0

# define the names of the object sets we want to investigate
object_set_folder = "free_first_joint"
object_sets = [
  "set_test_seg5",
  # "set_test_seg6",
  # "set_test_seg7",
  # "set_test_seg8",
  # "set_test_seg9",
  # "set_test_seg10",
  # "set_test_seg15",
  # "set_test_seg20",
  # "set_test_seg25" # NO MJ TIMESTEP
  # "set_test_seg30"
]

mj.mj.set.auto_set_timestep = False
mj.mj.set.auto_calibrate_gauges = False
mj.mj.set.auto_sim_steps = False

# set finger stiffness algorithm
mj.mj.set.finger_stiffness = -100 # hardcoded real data convergence
mj.mj.set.finger_stiffness = -0.5 # original model based on linear bending

# list of stable timesteps for each object set
# mujoco_timesteps = [
#   0.00353,
#   0.0024,
#   0.00176,
#   0.00133,
#   0.00102,
#   0.000841,
#   0.000334,
#   0.000198,
#   #2.5e-4 # i made this later
#   # 10e-5 # 8.42E-05
# ]

mujoco_timesteps = [
  3.105e-3,
  2.43e-3,
  1.935e-3,
  1.71e-3,
  1.395e-3,
  1.215e-3,
  0.72e-3,
  0.405e-3,
  0.009e-3,
  0.045e-3
]

# initialise/wipe variables
data = []
num_sets = len(object_sets)

# ----- #

from real_data import real_data

# this data uses MASSES, it is [100g, 200g, 300g, 400g]
REAL_xy = [
  1e-3 * real_data[2],
  1e-3 * real_data[4],
  1e-3 * real_data[6],
  1e-3 * real_data[8]
]

from FEA_data import fea_data

# this data uses MASSES, it is [100g, 200g, 300g, 400g]
FEA_xy2 = [ 
  1e-3 * fea_data[1],
  1e-3 * fea_data[2],
  1e-3 * fea_data[3],
  1e-3 * fea_data[4]
]

# ----- #
# loop through each object set and gather data
done_one = False
for i in range(0, num_sets):

  # if done_one: break

  set_name = object_set_folder + "/" + object_sets[i]
  mj._load_object_set(name=set_name)
  mj._load_xml()
  mj.mj.set.mujoco_timestep = mujoco_timesteps[i] * 0.9
  mj.task_reload_chance = -1
  mj.mj.hard_reset()
  mj.reset()

  # testing
  mj.mj.numerical_stiffness_converge(REAL_xy[2][:,0], REAL_xy[2][:,1])

  # run the validation regime
  print("Curve validation running for object set:", object_sets[i])
  print_out = False
  finger_data = mj.mj.curve_validation_regime(print_out)
  data.append(finger_data)

  done_one = True

# ----- #

# now process the data and plot it

entries = list(range(num_sets))
# entries = [0, 5, 6]#, 9]
# entries = [0, -1]

max_force = 4

forces = list(range(1, max_force + 1))
# forces = [1, max_force]

if len(entries) == 1: len_entries = 2
else:
  len_entries = len(entries)

fig, axs = plt.subplots(len_entries, len(forces), sharex=True)

for i, force in enumerate(forces):
  for j, ind in enumerate(entries):

    # lets start by plotting a basic example
    x =        np.array(data[ind].entries[force - 1].f1.x)
    y =        np.array(data[ind].entries[force - 1].f1.y)
    pred_x =   np.array(data[ind].entries[force - 1].f1.pred_x)
    pred_y =   np.array(data[ind].entries[force - 1].f1.pred_y)
    theory_x = np.array(data[ind].entries[force - 1].f1.theory_x_curve)
    theory_y = np.array(data[ind].entries[force - 1].f1.theory_y_curve)

    axs[j][i].plot(x,        y,        "b-o",  label="mujoco")
    axs[j][i].plot(pred_x,   pred_y,   "r--*", label="my model")
    axs[j][i].plot(theory_x, theory_y, "g",    label="theory")
    # axs[j][i].plot(FEA_xy[force - 1][:,0], FEA_xy[force - 1][:,1], label="FEA") # this is NEWTONS
    axs[j][i].plot(REAL_xy[force - 1][:,0], REAL_xy[force - 1][:,1], label="Real") # this is GRAMS
    axs[j][i].plot(FEA_xy2[force - 1][:,0], FEA_xy2[force - 1][:,1], label="FEA") # this is GRAMS
    
    temp_N = len(data[ind].entries[force - 1].f1.y) - 1
    # axs[j][i].text(0.005, data[ind].entries[force - 1].f1.theory_y[-1] * 0.4,
    #   f"mujoco stddev wrt model= {1000 * data[ind].entries[force - 1].f1.error.std_y_wrt_pred_y:.2f}mm\n" +
    #   f"mujoco stddev wrt theory = {1000 * data[ind].entries[force - 1].f1.error.std_y_wrt_theory_y:.2f}mm\n" +
    #   f"model stddev wrt theory = {1000 * data[ind].entries[force - 1].f1.error.std_y_pred_wrt_theory_y:.2f}mm",
    #   fontsize=14)
    # axs[j][i].axis("equal")
    axs[j][i].legend()
    axs[j][i].set(ylim=(0, 70e-3))

    if j == 0:
      axs[j][i].set_title(f"Applied force = {force * 100} grams", fontsize=20)
    if i == 0:
      axs[j][i].set_ylabel(f"N = {temp_N}", fontsize=20, rotation=90)

# fig.set_size_inches(35, 25)
fig.set_size_inches(20, 15)
fig.tight_layout()

plt.show()

exit()

# now process the data and plot it

entries = list(range(num_sets))
# entries = [0, 3, 6, 8]

fig, axs = plt.subplots(len(entries), 2, sharex=True)

for i, force in enumerate([1, 5]):
  for j, ind in enumerate(entries):

    # lets start by plotting a basic example
    x =        np.array(data[ind].entries[force - 1].f1.x)
    pred_x =   np.array(data[ind].entries[force - 1].f1.pred_x)
    y =        np.array(data[ind].entries[force - 1].f1.y)
    pred_y =   np.array(data[ind].entries[force - 1].f1.pred_y)
    theory_x = np.array(data[ind].entries[force - 1].f1.theory_x_curve)
    theory_y = np.array(data[ind].entries[force - 1].f1.theory_y_curve)

    # add in (0,0)
    # x = np.insert(x, 0, 0)
    # pred_x = np.insert(pred_x, 0, 0)
    # y = np.insert(y, 0, 0)
    # pred_y = np.insert(pred_y, 0, 0)
    # theory_y = np.insert(theory_y, 0, 0)

    axs[j][i].plot(x,        y,        "b-o",  label="mujoco")
    axs[j][i].plot(pred_x,   pred_y,   "r--*", label="my model")
    axs[j][i].plot(theory_x, theory_y, "g",    label="theory")
    axs[j][i].plot(FEA_xy[i][:,0], FEA_xy[i][:,1], label="FEA")
    
    temp_N = len(data[ind].entries[force - 1].f1.y) - 1
    axs[j][i].text(0.005, data[ind].entries[force - 1].f1.theory_y[temp_N - 1] * 0.4,
      f"mujoco error = {100 * data[ind].entries[force - 1].f1.error.y_wrt_theory_y_tipratio:.1f}%" +
      f"\nmy model error = {100 * data[ind].entries[force - 1].f1.error.y_pred_wrt_theory_y_tipratio:.1f}%",
      fontsize=14)
    # axs[j][i].axis("equal")
    axs[j][i].legend()

    if j == 0:
      axs[j][i].set_title(f"Applied force = {force} N", fontsize=20)
    if i == 0:
      axs[j][i].set_ylabel(f"N = {temp_N}", fontsize=20, rotation=90)

fig.set_size_inches(15, 15)
fig.tight_layout()