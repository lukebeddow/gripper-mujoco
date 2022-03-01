#!/usr/bin/env python3

import sys
from TrainDQN import TrainDQN
import networks

def apply_to_all_models(model):
  """
  Settings we want to apply to every single running model
  """
  # ensure debug mode is off
  model.env.mj.set.debug = False

  return model

def terminate_early_on_exceed_limits(model):
  # is_done() = true for multiple exceeded limits, penalty also increased to prevent exploits
  model.env.mj.set.exceed_limits.set(   -0.5, 2, 1)
  model.env.mj.set.exceed_axial.set(    -0.2, 6, 1, 2.0, 6.0, -1)
  model.env.mj.set.exceed_lateral.set(  -0.2, 6, 1, 4.0, 6.0, -1)
  model.env.mj.set.exceed_palm.set(     -0.2, 6, 1, 6.0, 10.0, -1)

  return model


if __name__ == "__main__":

  cluster = True

  inputarg = int(sys.argv[1])
  print("Input argument: ", inputarg)

  # ----- use default network ----- #
  if inputarg <= 10:

    # create a model training instance
    model = TrainDQN(cluster=cluster, save_suffix=f"array_{inputarg}")
    model = apply_to_all_models(model)

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

    # ----- adjust the eps_decay and terminate earlier on limits exceeded ----- #
    elif inputarg == 6:
      model.params.eps_decay = 250
      model = terminate_early_on_exceed_limits(model)
      model.train()

    elif inputarg == 7:
      model.params.eps_decay = 500
      model = terminate_early_on_exceed_limits(model)
      model.train()

    elif inputarg == 8:
      model.params.eps_decay = 1000
      model = terminate_early_on_exceed_limits(model)
      model.train()

    elif inputarg == 9:
      model.params.eps_decay = 2000
      model = terminate_early_on_exceed_limits(model)
      model.train()

    elif inputarg == 10:
      model.params.eps_decay = 4000
      model = terminate_early_on_exceed_limits(model)
      model.train()

  # ----- use deeper network ----- #
  else:

    # make a new model with a different network
    network = networks.DQN_3L60
    model = TrainDQN(network=network, cluster=cluster, save_suffix=f"array_{inputarg}")
    model = apply_to_all_models(model)

    # ----- adjust the eps_decay ----- #
    if inputarg == 11:
      model.params.eps_decay = 250
      model.train()

    elif inputarg == 12:
      model.params.eps_decay = 500
      model.train()

    elif inputarg == 13:
      model.params.eps_decay = 1000
      model.train()

    elif inputarg == 14:
      model.params.eps_decay = 2000
      model.train()

    elif inputarg == 15:
      model.params.eps_decay = 4000
      model.train()