#!/usr/bin/env python3

# add the path to the folder above, hardcoded
import sys
pathhere = "/home/luke/mujoco-devel/rl/"
sys.path.insert(0, pathhere)

from env.MjEnv import MjEnv
import numpy as np
from matplotlib import pyplot as plt
import pickle
import argparse

# define arguments and parse them
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--thickness", type=float, default=0.9e-3)
parser.add_argument("-w", "--width", type=float, default=28e-3)
parser.add_argument("-f", "--force-style", type=int, default=0)
parser.add_argument("-s", "--set", default="set9_testing")
parser.add_argument("-E", "--youngs-modulus", type=float, default=193e9)

args = parser.parse_args()

save_folder = "sim_data"

# create and prepare the mujoco instance
mj = MjEnv()
mj.params.test_objects = 1
mj.params.task_reload_chance = -1
mj.log_level = 4

# PREPARING THE OBJECT SET
"""
description/config/gripper.yaml should have:
  fixed_hook_segment: false
  fixed_motor_joints: true
  fingertip_clearance: 0.1
  segment_inertia_scaling: 50.0
  finger_length: 0.23

Then, generate the object sets before with:

make sets luke SET=set9_testing SEGMENTS="all" WIDTHS="24 28" USE_HASHES=yes

where 'luke' above is the machine name, swap out eg lab/lab-op etc

Now below, we have settings which match 
"""
mj.load_next.fixed_finger_hook = False # required as this overwrites gripper.yaml
mj.load_next.segment_inertia_scaling = 50.0
mj.load_next.fingertip_clearance = 0.1
mj.load_next.finger_length = 230e-3

# specify finger dimensions
mj.load_next.finger_thickness = args.thickness
mj.load_next.finger_width = args.width
mj.load_next.finger_modulus = args.youngs_modulus
mj.load_next.object_set_name = "set9_testing"

# 0=point end load, 1=UDL, 2=point end moment
force_style = args.force_style

if force_style == 0: force_style_str = "PL"
elif force_style == 1: force_style_str = "UDL"
elif force_style == 2: force_style_str = "EM"

# mj.load()

# global variables
max_force = 3
set_name = args.set

# # TESTING
# mj.mj.set_finger_modulus(193e9 * pow(1.0, 3)) # 190e9 gives <2.5% error for 0.9x28

# specify which segments to test
# segments = [5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
# segments = [5, 6, 7, 8, 9, 10]
segments = [3, 4]
segments = list(range(3, 31))
# segments = list(range(3, 6))

print(f"Finger thickness is {mj.load_next.finger_thickness * 1000:.2f} mm")
print(f"Finger width is {mj.load_next.finger_width * 1000:.1f} mm")
print(f"Youngs modulus is {mj.load_next.finger_modulus * 1e-9:.1f} GPa")
print(f"Force style is", force_style_str)
# print(f"Finger rigidity is {mj.mj.get_finger_rigidity():.2f}")

# will we recalculate data
get_data = True
use_converged = False

def run_curve_data(mjenv, segments, object_set, auto=True, force_style=0, get_timesteps_only=False):
  """
  This function returns a data structure containing curve validation data for
  a given mjenv across a given list of segments eg [5, 10, 15, 20]
  """

  data = []

  # turn on automatic finding of highest stable timestep
  mjenv.mj.set.auto_set_timestep = auto

  # turn off automatic finding of calibrations as they are not needed
  mjenv.mj.set.auto_calibrate_gauges = False

  if get_timesteps_only:
    timesteps = []

  # loop through each object set and gather data
  for N in segments:

    mjenv.params.test_objects = 1
    mjenv.params.task_reload_chance = -1
    mjenv.load(num_segments=N)
    mjenv.reset(hard=True, nospawn=True)

    if get_timesteps_only:

      this_timestep = mjenv.mj.set.mujoco_timestep
      timesteps.append(this_timestep)
      print(f"Timestep for N = {N} is {this_timestep * 1000:.1f} ms")
      continue

    else:

      print("Curve validation running for N =", N, "\t N in sim is", mjenv.mj.get_N(), flush=True)
      print_out = False
      finger_data = mjenv.mj.curve_validation_regime(print_out, force_style)
      data.append(finger_data)

  if get_timesteps_only: return timesteps

  print(f"force style was {force_style}")

  return data

# set finger stiffness algorithm
auto_timestep = True
converge = None
accuracy = None

if get_data:
  
  data = run_curve_data(mj, segments, set_name, auto=auto_timestep, force_style=force_style)

  # take care with overwrites - use the .py version in the terminal for overwrites
  overwrite = True

  name_style = "sim_bending_E{0:.2f}_{1}.pickle"
  rigidity = mj.mj.get_finger_rigidity()

  if overwrite:
    thickness_mm = mj.params.finger_thickness * 1000
    width_mm = mj.params.finger_width * 1000
    with open(save_folder + "/" + name_style.format(rigidity, force_style_str), "wb") as f:
      pickle.dump(data, f)