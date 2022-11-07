#!/usr/bin/env python3

from env.MjEnv import MjEnv
import numpy as np
from matplotlib import pyplot as plt
import argparse

# define arguments and parse them
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--thickness", type=float, default=0.9e-3)
parser.add_argument("-a", "--accuracy", type=float,  default=5e-4)
parser.add_argument("-m", "--method", type=int, default=1)

args = parser.parse_args()

# global variables
max_force = 4
set_name = "set4_fullset_795"

# create and prepare the mujoco instance
mj = MjEnv(noload=True)
mj.testing_xmls = 0
mj.task_reload_chance = -1

# specify the finger stiffness to use
if args.thickness is not None:
  mj.params.finger_thickness = args.thickness
else:
  mj.params.finger_thickness = args.thickness
print("Finger thickness set to ", mj.params.finger_thickness)

# specify which segments to test
segments = list(range(5, 31))
# segments = [5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
# segments = [5, 6, 7, 8, 9, 10]
# segments = [5]

# mujoco_timesteps = [
#   # 3.105e-3,
#   # 2.43e-3,
#   # 1.935e-3,
#   # 1.71e-3,
#   # 1.395e-3,
#   # 1.215e-3,
#   0.72e-3,
#   0.405e-3,
#   0.009e-3,
#   0.045e-3
# ]

def run_curve_data(mjenv, segments, converge_to=None, converge_target_accuracy=5e-4, auto=True, stiffness=-7.5):
  """
  This function returns a data structure containing curve validation data for
  a given mjenv across a given list of segments eg [5, 10, 15, 20]
  """

  data = []

  # turn on automatic finding of highest stable timestep
  mjenv.mj.set.auto_set_timestep = auto

  # set finger stiffness style (-7.5=final theory, -100=hardcoded real, -101=hardcoded theory)
  mjenv.mj.set.finger_stiffness = stiffness

  if converge_to is not None: 
    stiffness_code_string = ""
    loops_code_string = "std::vector<int> loops { "
    errors_code_string = "std::vector<float> errors { "

  # loop through each object set and gather data
  for N in segments:

    mjenv.load(object_set_name=set_name, num_segments=N)
    mjenv.reset(hard=True)

    # if we are converging stiffness before recording data
    if converge_to is not None:

      # # add safety cushion to timestep
      # mjenv.mj.set.mujoco_timestep *= 0.8
      # mjenv.reset()

      # TESTING converge to all forces
      if args.method == 2:
        print(f"Curve convergence to theory deflection at {converge_to:.2f} newtons, target_accuracy is {converge_target_accuracy * 100}%, N =", N, "\t N in sim is", mjenv.mj.get_N(), flush=True)
        info = mjenv.mj.numerical_stiffness_converge_2(converge_target_accuracy) # converge to 300g, basic theory curve

      # elif False:
      #   for f in [4 * 0.981, 3 * 0.981, 2 * 0.981, 1 * 0.981]:
      #     print(f"Curve convergence to theory deflection at {converge_to:.2f} newtons, target_accuracy is {converge_target_accuracy * 100}%, N =", N, "\t N in sim is", mjenv.mj.get_N(), flush=True)
      #     info = mjenv.mj.numerical_stiffness_converge(f, converge_target_accuracy)

      elif isinstance(converge_to, (int, float)):
        # if given an int/float value converge to the theory point load deflection curve for this value
        print(f"Curve convergence to theory deflection at {converge_to:.2f} newtons, target_accuracy is {converge_target_accuracy * 100}%, N =", N, "\t N in sim is", mjenv.mj.get_N(), flush=True)
        info = mjenv.mj.numerical_stiffness_converge(converge_to, converge_target_accuracy) # converge to 300g, basic theory curve
      else:
        # if given a numpy array of curve values, converge to this curve
        print(f"Curve convergence to given curve, N =", N, "\t N in sim is", mjenv.mj.get_N(), flush=True)
        info = mjenv.mj.numerical_stiffness_converge(converge_to[:,0], converge_to[:,1], converge_target_accuracy) # eg converge on real data

      vec_str = ""
      vec = mjenv.mj.get_finger_stiffnesses()
      for v in vec: vec_str += f"{v:.4f}, "
      code_stiffness = "std::vector<float> N{0} {{ {1} }};\n".format(N, vec_str[:-2]) # trim ', '

      split = info.replace(",", "=").split("=")
      loops = int(split[1])
      error = float(split[3])
      
      # save the code snippets
      stiffness_code_string += code_stiffness
      loops_code_string += str(loops) + ", "
      errors_code_string += str(error) + ", "

      print("Convergence finished with:", info, flush=True)
      print("Stiffness vector is:", mjenv.mj.get_finger_stiffnesses(), flush=True)

    else:
      # do not do any convergence
      print("Curve validation running for N =", N, "\t N in sim is", mjenv.mj.get_N(), flush=True)

    print_out = False
    finger_data = mjenv.mj.curve_validation_regime(print_out)
    data.append(finger_data)

  if converge_to is not None:
    print("\n" + stiffness_code_string)
    print(loops_code_string[:-2] + " };")
    print(errors_code_string[:-2] + " };")

  return data

# set finger stiffness algorithm
finger_stiffness = -7.5 # finalised theory result
# finger_stiffness = -101 # last convergence as initial guess
finger_stiffness = -102 # testing new convergence

auto_timestep = True
converge = None

# uncomment to recalculate stiffnesses and converge
converge = 3 * 0.981 # grams force, not newtons
accuracy = 2e-4 if args.accuracy is None else args.accuracy
print(f"Convergence accuracy set to {accuracy * 100}%")

mj.mj.tick()

data = run_curve_data(mj, segments, auto=auto_timestep, stiffness=finger_stiffness, converge_to=converge, converge_target_accuracy=accuracy)

print(f"\nThe finger thickness was: {mj.params.finger_thickness * 1000:.1f}")
print("The time taken was", mj.mj.tock())
