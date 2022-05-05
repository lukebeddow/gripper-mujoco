#!/usr/bin/env python3

import sys
from time import sleep
from datetime import datetime
from TrainDQN import TrainDQN
import networks

def mixed_rewards(model):
  # mixture of positive and negative rewards

  # binary rewards                       reward   done   trigger
  model.env.mj.set.step_num.set          (0.00,   False,   1)
  model.env.mj.set.lifted.set            (0.002,  False,   1)
  model.env.mj.set.target_height.set     (0.002,  False,   1)
  model.env.mj.set.object_stable.set     (0.002,  False,   1)

  # linear rewards                       reward   done   trigger min   max  overshoot
  model.env.mj.set.finger_force.set      (0.002,  False,   1,    0.2,  1.0,  -1)
  model.env.mj.set.palm_force.set        (0.002,  False,   1,    1.0,  6.0,  -1)

  # penalties                            reward   done   trigger min   max  overshoot
  model.env.mj.set.exceed_limits.set     (-0.005, 5,       1)
  model.env.mj.set.exceed_axial.set      (-0.005, 5,       1,    3.0,  6.0,  -1)
  model.env.mj.set.exceed_lateral.set    (-0.005, 5,       1,    4.0,  6.0,  -1)
  model.env.mj.set.exceed_palm.set       (-0.005, 5,       1,    6.0,  10.0, -1)

  # end criteria                         reward   done   trigger
  model.env.mj.set.stable_height.set     (0.0,    True,    1)
  model.env.mj.set.oob.set               (-1.0,   True,    1)

  # terminate episode when reward drops below -1.01, also cap at this value
  model.env.mj.set.quit_on_reward_below = -100
  model.env.mj.set.quit_reward_capped = True

  return model

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
  model.env.mj.set.exceed_limits.set     (-0.005, False,   1)
  model.env.mj.set.exceed_axial.set      (-0.005, False,   1,    3.0,  6.0,  -1)
  model.env.mj.set.exceed_lateral.set    (-0.005, False,   1,    4.0,  6.0,  -1)
  model.env.mj.set.exceed_palm.set       (-0.005, False,   1,    6.0,  10.0, -1)

  # end criteria                         reward   done   trigger
  model.env.mj.set.stable_height.set     (0.0,    True,    1)
  model.env.mj.set.oob.set               (-2.0,   True,    1)

  # terminate episode when reward drops below -1.01, also cap at this value
  model.env.mj.set.quit_on_reward_below = -2.01
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
  model.env.mj.set.target_height.set     (0.0,    True,    1)
  model.env.mj.set.oob.set               (-1.0,   True,    1)

  # terminate episode when reward drops below -1.01, also cap at this value
  model.env.mj.set.quit_on_reward_below = -2.01
  model.env.mj.set.quit_reward_capped = True

  return model

def apply_to_all_models(model):
  """
  Settings we want to apply to every single running model
  """

  # set up the object set
  model.env._load_object_set(name="set1_nocuboid_525")

  # number of steps in an episode
  model.env.max_episode_steps = 200

  # ensure we know what parameters we are using
  model.params.batch_size = 128
  model.params.learning_rate = 0.01
  model.params.gamma = 0.999
  model.params.eps_start = 0.9
  model.params.eps_end = 0.05
  model.params.eps_decay = 2000
  model.params.target_update = 100
  model.params.num_episodes = 20_000
  model.params.memory_replay = 20_000
  model.params.min_memory_replay = 5_000
  model.params.save_freq = 2_000
  model.params.test_freq = 2_000
  model.params.wandb_freq_s = 300

  # ensure debug mode is off
  model.env.log_level = 0
  model.env.mj.set.debug = False

  # disable all rendering
  model.env.disable_rendering = True
  model.env.mj.set.use_render_delay = False
  model.env.mj.set.render_on_step = False

  # define lengths and forces
  model.env.mj.set.oob_distance = 75e-3
  model.env.mj.set.done_height = 25e-3
  model.env.mj.set.stable_finger_force = 0.4
  model.env.mj.set.stable_palm_force = 1.0

  # what actions are we using
  model.env.mj.set.paired_motor_X_step = True
  model.env.mj.set.use_palm_action = True
  model.env.mj.set.use_height_action = True

  # remove all extra sensors
  model.env.mj.set.motor_state_sensor.in_use = True
  model.env.mj.set.motor_state_sensor.read_rate = -1 # -2 means 2 readings, current + prev
  model.env.mj.set.bending_gauge.in_use = True
  model.env.mj.set.axial_gauge.in_use = False
  model.env.mj.set.palm_sensor.in_use = False
  model.env.mj.set.wrist_sensor_XY.in_use = False
  model.env.mj.set.wrist_sensor_Z.in_use = False

  # what sensing mode (0=raw data, 1=change, 2=average)
  model.env.mj.set.sensor_sample_mode = 1
  model.env.mj.set.state_sample_mode = 0

  # wipe all rewards so none trigger
  model.env.mj.set.wipe_rewards()

  return model

if __name__ == "__main__":

  # will we publish results to weights and biases
  use_wandb = True

  # extract input arguments
  inputarg = int(sys.argv[1])

  if len(sys.argv) > 1:
    timestamp = sys.argv[2]
    notimestamp = True
  else:
    timestamp = ""
    notimestamp = None

  save_suffix = f"{timestamp}_array_{inputarg}"

  print("Input argument: ", inputarg)
  print("Timestamp is:", timestamp)

  # create and configure the model to default
  model = TrainDQN(notimestamp=notimestamp, save_suffix=save_suffix,
                   use_wandb=use_wandb)
  model = apply_to_all_models(model)
  model.no_plot = True

  # create the name of the run and configure for wandb
  run_name = f"train_{model.machine}_{save_suffix}"
  model.wandb_name = run_name

  print("This run will be saved as", run_name)

  # ----- 1 - 5, default network, negative rewards, vary number of sensors ----- #
  if inputarg <= 5:

    # apply settings
    model = make_rewards_negative(model)

    # now form the network
    network = networks.DQN_2L60

    # ----- adjust the rewards and step number ----- #
    if inputarg == 1:
      model.env.max_episode_steps = 200
      model.train(network)

    elif inputarg == 2:
      model.env.max_episode_steps = 200
      model.env.mj.set.motor_state_sensor.read_rate = -2
      model.train(network)

    elif inputarg == 3:
      model.env.max_episode_steps = 200
      model.env.mj.set.motor_state_sensor.read_rate = -2
      model.env.mj.set.axial_gauge.in_use = True
      model.train(network)

    elif inputarg == 4:
      model.env.max_episode_steps = 200
      model.env.mj.set.motor_state_sensor.read_rate = -2
      model.env.mj.set.axial_gauge.in_use = True
      model.env.mj.set.wrist_sensor_Z.in_use = True
      model.train(network)

    elif inputarg == 5:
      model.env.max_episode_steps = 200
      model.env.mj.set.motor_state_sensor.read_rate = -2
      model.env.mj.set.axial_gauge.in_use = True
      model.env.mj.set.wrist_sensor_Z.in_use = True
      model.env.mj.set.wrist_sensor_XY.in_use = True
      model.train(network)

  # ----- 6 - 10, deeper network, negative rewards, vary number of sensors ----- #
  elif inputarg > 5 and inputarg <= 10:

    # create training instance and apply settings
    model = make_rewards_negative(model)

    # now form the network
    network = networks.DQN_3L60

    # ----- adjust the rewards and step number ----- #
    if inputarg == 6:
      model.env.max_episode_steps = 200
      model.train(network)

    elif inputarg == 7:
      model.env.max_episode_steps = 200
      model.env.mj.set.motor_state_sensor.read_rate = -2
      model.train(network)

    elif inputarg == 8:
      model.env.max_episode_steps = 200
      model.env.mj.set.motor_state_sensor.read_rate = -2
      model.env.mj.set.axial_gauge.in_use = True
      model.train(network)

    elif inputarg == 9:
      model.env.max_episode_steps = 200
      model.env.mj.set.motor_state_sensor.read_rate = -2
      model.env.mj.set.axial_gauge.in_use = True
      model.env.mj.set.wrist_sensor_Z.in_use = True
      model.train(network)

    elif inputarg == 10:
      model.env.max_episode_steps = 200
      model.env.mj.set.motor_state_sensor.read_rate = -2
      model.env.mj.set.axial_gauge.in_use = True
      model.env.mj.set.wrist_sensor_Z.in_use = True
      model.env.mj.set.wrist_sensor_XY.in_use = True
      model.train(network)

  # ----- 11 - 15, default network, mixed rewards, vary sensors ----- #
  elif inputarg > 10 and inputarg <= 15:

    # create training instance and apply settings
    model = mixed_rewards(model)

    # now form the network
    network = networks.DQN_2L60

    # ----- adjust the rewards and step number ----- #
    if inputarg == 11:
      model.env.max_episode_steps = 200
      model.train(network)

    elif inputarg == 12:
      model.env.max_episode_steps = 200
      model.env.mj.set.motor_state_sensor.read_rate = -2
      model.train(network)

    elif inputarg == 13:
      model.env.max_episode_steps = 200
      model.env.mj.set.motor_state_sensor.read_rate = -2
      model.env.mj.set.axial_gauge.in_use = True
      model.train(network)

    elif inputarg == 14:
      model.env.max_episode_steps = 200
      model.env.mj.set.motor_state_sensor.read_rate = -2
      model.env.mj.set.axial_gauge.in_use = True
      model.env.mj.set.wrist_sensor_Z.in_use = True
      model.train(network)

    elif inputarg == 15:
      model.env.max_episode_steps = 200
      model.env.mj.set.motor_state_sensor.read_rate = -2
      model.env.mj.set.axial_gauge.in_use = True
      model.env.mj.set.wrist_sensor_Z.in_use = True
      model.env.mj.set.wrist_sensor_XY.in_use = True
      model.train(network)

  # ----- 16 - 20, deeper network, mixed rewards, vary sensors ----- #
  elif inputarg > 15 and inputarg <= 20:

    # create training instance and apply settings
    model = mixed_rewards(model)

    # now form the network
    network = networks.DQN_3L60

    # ----- adjust the rewards and step number ----- #
    if inputarg == 16:
      model.env.max_episode_steps = 200
      model.train(network)

    elif inputarg == 17:
      model.env.max_episode_steps = 200
      model.env.mj.set.motor_state_sensor.read_rate = -2
      model.train(network)

    elif inputarg == 18:
      model.env.max_episode_steps = 200
      model.env.mj.set.motor_state_sensor.read_rate = -2
      model.env.mj.set.axial_gauge.in_use = True
      model.train(network)

    elif inputarg == 19:
      model.env.max_episode_steps = 200
      model.env.mj.set.motor_state_sensor.read_rate = -2
      model.env.mj.set.axial_gauge.in_use = True
      model.env.mj.set.wrist_sensor_Z.in_use = True
      model.train(network)

    elif inputarg == 20:
      model.env.max_episode_steps = 200
      model.env.mj.set.motor_state_sensor.read_rate = -2
      model.env.mj.set.axial_gauge.in_use = True
      model.env.mj.set.wrist_sensor_Z.in_use = True
      model.env.mj.set.wrist_sensor_XY.in_use = True
      model.train(network)