#!/usr/bin/env python3

import sys
from TrainDQN import TrainDQN
import networks

def wipe_all_rewards(model):
  """
  Wipe all the rewards to zero, and never triggering
  """

  model.env.mj.set.wipe_rewards()

  return model

  never = 10000
  
  # binary rewards                       reward   done    trigger
  model.env.mj.set.step_num.set          (0.0,    False,  never)
  model.env.mj.set.lifted.set            (0.0,    False,  never)
  model.env.mj.set.oob.set               (0.0,    False,  never)
  model.env.mj.set.dropped.set           (0.0,    False,  never)
  model.env.mj.set.target_height.set     (0.0,    False,  never)
  model.env.mj.set.exceed_limits.set     (0.0,    False,  never)
  model.env.mj.set.object_contact.set    (0.0,    False,  never)
  model.env.mj.set.object_stable.set     (0.0,    False,  never)

  # linear rewards                       reward   done    trigger   min   max  overshoot
  model.env.mj.set.finger_force.set      (0.0,    False,  never,    0.0,  0.0,  -1)
  model.env.mj.set.palm_force.set        (0.0,    False,  never,    0.0,  0.0,  -1)
  model.env.mj.set.exceed_axial.set      (0.0,    False,  never,    0.0,  0.0,  -1)
  model.env.mj.set.exceed_lateral.set    (0.0,    False,  never,    0.0,  0.0,  -1)
  model.env.mj.set.exceed_palm.set       (0.0,    False,  never,    0.0,  0.0,  -1)

  return model

def apply_to_all_models(model):
  """
  Settings we want to apply to every single running model
  """
  # ensure debug mode is off
  model.env.mj.set.debug = False

  # wipe all of the default settings
  model = wipe_all_rewards(model)

  return model

def terminate_early_on_exceed_limits(model):
  # is_done() = true for multiple exceeded limits, penalty also increased to prevent exploits
  model.env.mj.set.exceed_limits.set    (-0.5, 2, 1)
  model.env.mj.set.exceed_axial.set     (-0.2, 6, 1, 2.0, 6.0, -1)
  model.env.mj.set.exceed_lateral.set   (-0.2, 6, 1, 4.0, 6.0, -1)
  model.env.mj.set.exceed_palm.set      (-0.2, 6, 1, 6.0, 10.0, -1)

  return model

def make_rewards_negative(model):
  # shift the rewards to always be negative

  # binary rewards                       reward   done   trigger
  model.env.mj.set.step_num.set          (-0.01,  False,  1)
  model.env.mj.set.lifted.set            (0.002,  False,  1)
  model.env.mj.set.target_height.set     (0.002,  False,  1)
  model.env.mj.set.object_stable.set     (0.002,  False,  1)
  model.env.mj.set.stable_height.set     (0.0,    1,      1)
  
  # linear rewards                       reward   done   trigger min   max  overshoot
  model.env.mj.set.finger_force.set      (0.002,  False,  1,     0.2,  1.0,  -1)
  model.env.mj.set.palm_force.set        (0.002,  False,  1,     1.0,  6.0,  -1)
  model.env.mj.set.exceed_lateral.set    (-0.002, False,  1,     1.0,  6.0,  -1)
  model.env.mj.set.exceed_palm.set       (-0.002, False,  1,     1.0,  6.0,  -1)

  # terminate early for bad behaviour    reward   done   trigger min   max  overshoot
  model.env.mj.set.exceed_limits.set     (-0.5,   2,      1)
  model.env.mj.set.oob.set               (-1.0,   1,      1)
  model.env.mj.set.exceed_axial.set      (-0.2,   5,      1,     3.0,  6.0,  -1)
  model.env.mj.set.exceed_lateral.set    (-0.2,   5,      1,     4.0,  6.0,  -1)
  model.env.mj.set.exceed_palm.set       (-0.2,   5,      1,     6.0,  10.0, -1)

  return model

def no_palm_grasping(model):
  # remove the palm as an action

  # disable using the palm
  model.env.mj.set.use_palm_action = False

  # switch to negative rewards
  model = make_rewards_negative(model)

  # now overwrite
  model.env.mj.set.target_height.set     (0.0,    1,      1)
  model.env.mj.set.object_stable.set     (0.002,  False,  1)
  model.env.mj.set.stable_height.set     (0.0,    False,  1)


if __name__ == "__main__":

  cluster = True

  inputarg = int(sys.argv[1])
  print("Input argument: ", inputarg)

  # ----- use default network ----- #
  if inputarg <= 5:

    # create a model training instance
    model = TrainDQN(cluster=cluster, save_suffix=f"array_{inputarg}")
    model = apply_to_all_models(model)
    model = make_rewards_negative(model)

    # ----- adjust the eps_decay ----- #
    if inputarg == 1:
      model.params.eps_decay = 250
      model.train()

    elif inputarg == 2:
      model.params.eps_decay = 500
      model.train()

    elif inputarg == 3:
      model.params.eps_decay = 1000
      model.train()

    elif inputarg == 4:
      model.params.eps_decay = 2000
      model.train()

    elif inputarg == 5:
      model.params.eps_decay = 4000
      model.train()

  # ----- use deeper network ----- #
  else:

    # make a new model with a different network
    network = networks.DQN_3L60
    model = TrainDQN(network=network, cluster=cluster, save_suffix=f"array_{inputarg}")
    model = apply_to_all_models(model)
    model = make_rewards_negative(model)

    # ----- adjust the eps_decay ----- #
    if inputarg == 6:
      model.params.eps_decay = 250
      model.train()

    elif inputarg == 7:
      model.params.eps_decay = 500
      model.train()

    elif inputarg == 8:
      model.params.eps_decay = 1000
      model.train()

    elif inputarg == 9:
      model.params.eps_decay = 2000
      model.train()

    elif inputarg == 10:
      model.params.eps_decay = 4000
      model.train()