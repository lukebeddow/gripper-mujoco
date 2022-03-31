#!/usr/bin/env python3

import sys
from TrainDQN import TrainDQN
import networks

def make_rewards_negative(model):
  # shift the rewards to always be negative

  # binary rewards                       reward   done   trigger
  model.env.mj.set.step_num.set          (-0.01,  False,   1)
  model.env.mj.set.lifted.set            (0.002,  False,   1)
  model.env.mj.set.target_height.set     (0.002,  False,   1)
  model.env.mj.set.object_stable.set     (0.002,  False,   1)
  
  # linear rewards                       reward   done   trigger min   max  overshoot
  model.env.mj.set.finger_force.set      (0.002,  False,   1,    0.2,  1.0,  -1)
  model.env.mj.set.palm_force.set        (0.002,  False,   1,    1.0,  6.0,  -1)

  # penalties                            reward   done   trigger min   max  overshoot
  model.env.mj.set.exceed_limits.set     (-0.002, False,   1)
  model.env.mj.set.exceed_axial.set      (-0.002, False,   1,    3.0,  6.0,  -1)
  model.env.mj.set.exceed_lateral.set    (-0.002, False,   1,    4.0,  6.0,  -1)
  model.env.mj.set.exceed_palm.set       (-0.002, False,   1,    6.0,  10.0, -1)

  # end criteria                         reward   done   trigger
  model.env.mj.set.stable_height.set     (0.0,    True,    1)
  model.env.mj.set.oob.set               (-1.0,   True,    1)

  # terminate episode when reward drops below -1.01, also cap at this value
  model.env.mj.set.quit_on_reward_below = -1.01
  model.env.mj.set.quit_reward_capped = True

  return model

def finger_only_lifting(model):
  # try only to lift objects with the fingers

  model.env.mj.set.use_palm_action = False

  # binary rewards                       reward   done   trigger
  model.env.mj.set.step_num.set          (-0.01,  False,   1)
  model.env.mj.set.lifted.set            (0.005,  False,   1)
  
  # linear rewards                       reward   done   trigger min   max  overshoot
  model.env.mj.set.finger_force.set      (0.005,  False,   1,    0.2,  1.0,  -1)

  # penalties                            reward   done   trigger min   max  overshoot
  model.env.mj.set.exceed_limits.set     (-0.005, False,   1)
  model.env.mj.set.exceed_axial.set      (-0.005, False,   1,    3.0,  6.0,  -1)
  model.env.mj.set.exceed_lateral.set    (-0.005, False,   1,    4.0,  6.0,  -1)

  # end criteria                         reward   done   trigger
  model.env.mj.set.height_target.set     (0.0,    True,    1)
  model.env.mj.set.oob.set               (-1.0,   True,    1)

  # terminate episode when reward drops below -1.01, also cap at this value
  model.env.mj.set.quit_on_reward_below = -1.01
  model.env.mj.set.quit_reward_capped = True

def add_palm_force_sensor(model):
  # add force sensor to the palm
  model.env.mj.set.use_palm_sensor = True
  model.env.mj.set.palm_force_normalise = 8.0
  return model

def add_palm_bumper_sensor(model):
  # add a palm bumper sensor
  model.env.mj.set.use_palm_sensor = True
  model.env.mj.set.palm_force_normalise = -1
  return model

def apply_to_all_models(model):
  """
  Settings we want to apply to every single running model
  """
  # ensure debug mode is off
  model.env.mj.set.debug = False

  # these settings should never be changed
  model.env.mj.set.gauge_read_rate_hz = 10
  model.env.mj.set.use_render_delay = False
  model.env.mj.set.render_on_step = False
  model.env.mj.set.use_settling = False

  # ensure key settings are set to defaults
  model.env.mj.set.normalising_force = 100
  model.env.mj.set.use_palm_sensor = False
  model.env.mj.set.oob_distance = 75e-3
  model.env.mj.set.height_target = 25e-3
  model.env.mj.set.stable_finger_force = 0.4
  model.env.mj.set.stable_palm_force = 1.0
  model.env.mj.set.obs_raw_data = False
  model.env.mj.set.paired_motor_X_step = True
  model.env.mj.set.use_palm_action = True
  model.env.mj.set.use_height_action = True

  # wipe all rewards so none trigger
  model.env.mj.set.wipe_rewards()

  return model

if __name__ == "__main__":

  cluster = True

  inputarg = int(sys.argv[1])
  print("Input argument: ", inputarg)

  # ----- 1 - 5, default network, default settings, vary rewards ----- #
  if inputarg <= 5:

    # create training instance and apply settings
    model = TrainDQN(cluster=cluster, save_suffix=f"array_{inputarg}")
    model = apply_to_all_models(model)
    model = make_rewards_negative(model)

    # now form the network
    network = networks.DQN_2L60
    model.init(network)

    # ----- adjust the rewards and step number ----- #
    if inputarg == 1:
      model.params.eps_decay = 1000
      model.env.max_episode_steps = 100
      model.env.mj.set.quit_on_reward_below = -1.01
      model.env.mj.set.quit_reward_capped = True
      model.train()

    elif inputarg == 2:
      model.params.eps_decay = 1000
      model.env.max_episode_steps = 150
      model.env.mj.set.quit_on_reward_below = -1.51
      model.env.mj.set.quit_reward_capped = True
      model.train()

    elif inputarg == 3:
      model.params.eps_decay = 1000
      model.env.max_episode_steps = 200
      model.env.mj.set.quit_on_reward_below = -2.01
      model.env.mj.set.quit_reward_capped = True
      model.train()

    elif inputarg == 4:
      model.params.eps_decay = 1000
      model.env.max_episode_steps = 100
      model.env.mj.set.quit_on_reward_below = -100
      model.env.mj.set.quit_reward_capped = False
      model.train()

    elif inputarg == 5:
      model.params.eps_decay = 1000
      model.env.max_episode_steps = 150
      model.env.mj.set.quit_on_reward_below = -100
      model.env.mj.set.quit_reward_capped = False
      model.train()

  # ----- 6 - 10, default network, no palm, vary rewards ----- #
  elif inputarg > 5 and inputarg <= 10:

    # create training instance and apply settings
    model = TrainDQN(cluster=cluster, save_suffix=f"array_{inputarg}")
    model = apply_to_all_models(model)
    model = finger_only_lifting(model)

    # now form the network
    network = networks.DQN_2L60
    model.init(network)

    # ----- adjust the rewards and step number ----- #
    if inputarg == 6:
      model.params.eps_decay = 1000
      model.env.max_episode_steps = 100
      model.env.mj.set.quit_on_reward_below = -1.01
      model.env.mj.set.quit_reward_capped = True
      model.train()

    elif inputarg == 7:
      model.params.eps_decay = 1000
      model.env.max_episode_steps = 150
      model.env.mj.set.quit_on_reward_below = -1.51
      model.env.mj.set.quit_reward_capped = True
      model.train()

    elif inputarg == 8:
      model.params.eps_decay = 1000
      model.env.max_episode_steps = 200
      model.env.mj.set.quit_on_reward_below = -2.01
      model.env.mj.set.quit_reward_capped = True
      model.train()

    elif inputarg == 9:
      model.params.eps_decay = 1000
      model.env.max_episode_steps = 100
      model.env.mj.set.quit_on_reward_below = -100
      model.env.mj.set.quit_reward_capped = False
      model.train()

    elif inputarg == 10:
      model.params.eps_decay = 1000
      model.env.max_episode_steps = 150
      model.env.mj.set.quit_on_reward_below = -100
      model.env.mj.set.quit_reward_capped = False
      model.train()

  # ----- 11 - 15, default network, palm force sensor, vary rewards ----- #
  elif inputarg > 10 and inputarg <= 15:

    # create training instance and apply settings
    model = TrainDQN(cluster=cluster, save_suffix=f"array_{inputarg}")
    model = apply_to_all_models(model)
    model = make_rewards_negative(model)

    model = add_palm_force_sensor(model)

    # now form the network
    network = networks.DQN_2L60
    model.init(network)

    # ----- adjust the rewards and step number ----- #
    if inputarg == 11:
      model.params.eps_decay = 1000
      model.env.max_episode_steps = 100
      model.env.mj.set.quit_on_reward_below = -1.01
      model.env.mj.set.quit_reward_capped = True
      model.train()

    elif inputarg == 12:
      model.params.eps_decay = 1000
      model.env.max_episode_steps = 150
      model.env.mj.set.quit_on_reward_below = -1.51
      model.env.mj.set.quit_reward_capped = True
      model.train()

    elif inputarg == 13:
      model.params.eps_decay = 1000
      model.env.max_episode_steps = 200
      model.env.mj.set.quit_on_reward_below = -2.01
      model.env.mj.set.quit_reward_capped = True
      model.train()

    elif inputarg == 14:
      model.params.eps_decay = 1000
      model.env.max_episode_steps = 100
      model.env.mj.set.quit_on_reward_below = -100
      model.env.mj.set.quit_reward_capped = False
      model.train()

    elif inputarg == 15:
      model.params.eps_decay = 1000
      model.env.max_episode_steps = 150
      model.env.mj.set.quit_on_reward_below = -100
      model.env.mj.set.quit_reward_capped = False
      model.train()

  # ----- BELOW NOT USED, array trained 1-15 ----- #

  # ----- 16 - 20, deeper network, palm force sensor, vary eps decay ----- #
  elif inputarg > 15 and inputarg <= 20:

    # create training instance and apply settings
    model = TrainDQN(cluster=cluster, save_suffix=f"array_{inputarg}")
    model = apply_to_all_models(model)
    model = make_rewards_negative(model)

    model = add_palm_force_sensor(model)

    # now form the network
    network = networks.DQN_3L60
    model.init(network)

    # ----- adjust the eps_decay ----- #
    if inputarg == 16:
      model.params.eps_decay = 250
      model.train()

    elif inputarg == 17:
      model.params.eps_decay = 500
      model.train()

    elif inputarg == 18:
      model.params.eps_decay = 1000
      model.train()

    elif inputarg == 19:
      model.params.eps_decay = 2000
      model.train()

    elif inputarg == 20:
      model.params.eps_decay = 4000
      model.train()

  # ----- 21 - 25, default network, default settings, vary target update ----- #
  elif inputarg > 20 and inputarg <= 25:

    # create training instance and apply settings
    model = TrainDQN(cluster=cluster, save_suffix=f"array_{inputarg}")
    model = apply_to_all_models(model)
    model = make_rewards_negative(model)

    model = add_palm_force_sensor(model)

    # now form the network
    network = networks.DQN_2L60
    model.init(network)

    # ----- adjust the eps_decay ----- #
    if inputarg == 21:
      model.params.target_update = 50
      model.train()

    elif inputarg == 22:
      model.params.target_update = 100
      model.train()

    elif inputarg == 23:
      model.params.target_update = 200
      model.train()

    elif inputarg == 24:
      model.params.target_update = 400
      model.train()

    elif inputarg == 25:
      model.params.target_update = 800
      model.train()

  # ----- 26 - 30, default network, palm force sensor, vary target update ----- #
  elif inputarg > 25 and inputarg <= 30:

    # create training instance and apply settings
    model = TrainDQN(cluster=cluster, save_suffix=f"array_{inputarg}")
    model = apply_to_all_models(model)
    model = make_rewards_negative(model)

    # now form the network
    network = networks.DQN_2L60
    model.init(network)

    # ----- adjust the eps_decay ----- #
    if inputarg == 26:
      model.params.target_update = 50
      model.train()

    elif inputarg == 27:
      model.params.target_update = 100
      model.train()

    elif inputarg == 28:
      model.params.target_update = 200
      model.train()

    elif inputarg == 29:
      model.params.target_update = 400
      model.train()

    elif inputarg == 30:
      model.params.target_update = 800
      model.train()