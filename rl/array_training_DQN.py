#!/usr/bin/env python3

import sys
from time import sleep
from datetime import datetime
from TrainDQN import TrainDQN
import networks

def set_penalties(model, value, done=False, trigger=1):
  """
  Set penalty rewards with given value, alongside defaults
  """

  # penalties                            reward   done   trigger  min   max  overshoot
  model.env.mj.set.exceed_limits.set     (value,  done,  trigger)
  model.env.mj.set.exceed_axial.set      (value,  done,  trigger, 3.0,  6.0,  -1)
  model.env.mj.set.exceed_lateral.set    (value,  done,  trigger, 4.0,  6.0,  -1)
  model.env.mj.set.exceed_palm.set       (value,  done,  trigger, 6.0,  10.0, -1)

  return model

def set_bonuses(model, value):
  """
  Set bonus rewards with a given value
  """

  # binary rewards                       reward   done   trigger
  model.env.mj.set.lifted.set            (value,  False,   1)
  model.env.mj.set.target_height.set     (value,  False,   1)
  model.env.mj.set.object_stable.set     (value,  False,   1)
  
  # linear rewards                       reward   done   trigger min   max  overshoot
  model.env.mj.set.finger_force.set      (value,  False,   1,    0.2,  1.0,  -1)
  model.env.mj.set.palm_force.set        (value,  False,   1,    1.0,  6.0,  -1)

  return model

def setup_HER(model, use=True, style="basic", mode="final", k=4):
  """
  Set the goal for the simulation and enable HER
  """

  if use == False:
    model.env.mj.set.use_HER = False
    return model

  # enable HER and set the mode
  model.params.use_HER = True # python setting OVERRIDES cpp
  model.params.HER_mode = mode
  model.params.HER_k = k

  # set the HER goal reward style
  model.env.mj.set.goal_reward = 1.0
  model.env.mj.set.divide_goal_reward = True

  if style == "basic":
    # binary events
    model.env.mj.set.binary_goal_vector = True
    model.env.mj.goal.lifted.involved = True
    model.env.mj.goal.object_contact.involved = True
    model.env.mj.goal.object_stable.involved = True
    model.env.mj.goal.target_height.involved = True
    model.env.mj.goal.stable_height.involved = True
    # linear events
    model.env.mj.goal.finger_force.involved = True
    model.env.mj.goal.palm_force.involved = True
    # specify the thresholds               reward   done   trigger min   max  overshoot
    model.env.mj.set.finger_force.set      (0.0,    False,   1,    1.0,  2.0,  6.0)
    model.env.mj.set.palm_force.set        (0.0,    False,   1,    1.0,  3.0,  6.0)

  elif style == "forces":
    model.env.mj.set.binary_goal_vector = False
    # turn on the linear events for the gripper forces
    model.env.mj.goal.finger1_force.involved = True
    model.env.mj.goal.finger2_force.involved = True
    model.env.mj.goal.finger3_force.involved = True
    model.env.mj.goal.palm_force.involved = True
    # specify the thresholds               reward   done   trigger min   max  overshoot
    model.env.mj.set.finger1_force.set     (0.0,    False,   1,    0.0,  2.0,  6.0)
    model.env.mj.set.finger2_force.set     (0.0,    False,   1,    0.0,  2.0,  6.0)
    model.env.mj.set.finger3_force.set     (0.0,    False,   1,    0.0,  2.0,  6.0)
    model.env.mj.set.palm_force.set        (0.0,    False,   1,    0.0,  3.0,  6.0)

  else:
    raise RuntimeError("style was not set to a valid option in setup_HER()")

  model.wandb_note += f"HER goal style: '{style}', mode: '{mode}', k: {k}\n"

  return model

def create_reward_function(model, style="negative", options=[]):
  """
  Set the reward structure for the learning, with different style options
  """

  if style == "negative":
    # reward each step                     reward   done   trigger
    model.env.mj.set.step_num.set          (-0.01,  False,   1)
    # penalties and bonuses
    model = set_bonuses(model, 0.002)
    model = set_penalties(model, -0.002,
                          done=5 if "terminate_early" in options else False)
    # scale based on steps allowed per episode
    model.env.mj.set.scale_rewards(100 / model.env.max_episode_steps)
    # end criteria                         reward   done   trigger
    model.env.mj.set.stable_height.set     (0.0,    True,    1)
    model.env.mj.set.oob.set               (-1.0,   True,    1)
    # termination with poor reward
    model.env.mj.set.quit_on_reward_below = -1.0 if "cap" in options else -1e5
    model.env.mj.set.quit_reward_capped = True

  elif style == "mixed":
    # reward each step                     reward   done   trigger
    model.env.mj.set.step_num.set          (0.0,    False,   1)
    # penalties and bonuses
    model = set_bonuses(model, 0.002)
    model = set_penalties(model, -0.005,  
                          done=5 if "terminate_early" in options else False)
    # scale based on steps allowed per episode
    model.env.mj.set.scale_rewards(100 / model.env.max_episode_steps)
    # end criteria                         reward   done   trigger
    model.env.mj.set.stable_height.set     (1.0,    True,    1)
    model.env.mj.set.oob.set               (-1.0,   True,    1)
    # termination with poor reward
    model.env.mj.set.quit_on_reward_below = -1.0 if "cap" in options else -1e5
    model.env.mj.set.quit_reward_capped = True

  elif style == "mixed_v2":
    # reward each step                     reward   done   trigger
    model.env.mj.set.step_num.set          (-0.01,  False,   1)
    # penalties and bonuses
    model = set_bonuses(model, 0.005)
    model = set_penalties(model, -0.002,  
                          done=5 if "terminate_early" in options else False)
    # scale based on steps allowed per episode
    model.env.mj.set.scale_rewards(100 / model.env.max_episode_steps)
    # end criteria                         reward   done   trigger
    model.env.mj.set.stable_height.set     (1.0,    True,    1)
    model.env.mj.set.oob.set               (-1.0,   True,    1)
    # termination with poor reward
    model.env.mj.set.quit_on_reward_below = -1.0 if "cap" in options else -1e5
    model.env.mj.set.quit_reward_capped = True

  elif style == "sparse":
    # reward each step                     reward   done   trigger
    model.env.mj.set.step_num.set          (0.0,    False,   1)
    # penalties and bonuses
    model = set_bonuses(model, 0.0)
    model = set_penalties(model, 0.0,  
                          done=5 if "terminate_early" in options else False)
    # scale based on steps allowed per episode
    model.env.mj.set.scale_rewards(100 / model.env.max_episode_steps)
    # end criteria                         reward   done   trigger
    model.env.mj.set.object_stable.set     (1.0,    True,    1)
    model.env.mj.set.oob.set               (0.0,    True,    1)
    # termination with poor reward
    model.env.mj.set.quit_on_reward_below = -1.0 if "cap" in options else -1e5
    model.env.mj.set.quit_reward_capped = True

  elif style == "sparse_no_rewards":
    # reward each step                     reward   done   trigger
    model.env.mj.set.step_num.set          (0.0,    False,   1)
    # penalties and bonuses
    model = set_bonuses(model, 0.0)
    model = set_penalties(model, 0.0,  
                          done=5 if "terminate_early" in options else False)
    # scale based on steps allowed per episode
    model.env.mj.set.scale_rewards(100 / model.env.max_episode_steps)
    # end criteria                         reward   done   trigger
    model.env.mj.set.object_stable.set     (0.0,    True,    1)
    model.env.mj.set.oob.set               (0.0,    True,    1)
    # termination with poor reward
    model.env.mj.set.quit_on_reward_below = -1.0 if "cap" in options else -1e5
    model.env.mj.set.quit_reward_capped = True
  
  else:
    raise RuntimeError("style was not set to a valid option in create_reward_function()")

  model.wandb_note += f"Reward style: '{style}', options: [ "
  for extra in options: model.wandb_note += f"'{extra}' "
  model.wandb_note += "]\n"

  return model

def add_sensors(model, num=10, sensor_mode=1, state_mode=0):
  """
  Add a number of sensors
  """

  # what sensing mode (0=raw data, 1=change, 2=average)
  model.env.mj.set.sensor_sample_mode = sensor_mode
  model.env.mj.set.state_sample_mode = state_mode

  # default: state sensor and bending gauge sensor
  model.env.mj.set.motor_state_sensor.in_use = True
  model.env.mj.set.motor_state_sensor.read_rate = -1
  model.env.mj.set.bending_gauge.in_use = True

  # state sensor with two readings (current, prev)
  if num >= 1: model.env.mj.set.motor_state_sensor.read_rate = -2

  # palm force sensor
  if num >= 2: model.env.mj.set.palm_sensor.in_use = True

  # wrist z force sensor
  if num >= 3: model.env.mj.set.wrist_sensor_Z.in_use = True

  # wrist xy force sensor
  if num >= 4: model.env.mj.set.wrist_sensor_XY.in_use = True

  # finger axial gauges
  if num >= 5: model.env.mj.set.axial_gauge.in_use = True

  model.wandb_note += f"Num sensors: {num}, state mode: {state_mode}, sensor mode: {sensor_mode}\n"

  return model

def simplest_sphere_lifting(model):
  # try only to lift objects with the fingers

  # set up the object set
  model.env._load_object_set(name="set1_sphereonly_120")

  # data logging
  model.params.save_freq = 500
  model.params.test_freq = 2_000
  model.params.wandb_freq_s = 300

  # what actions are we using
  model.env.mj.set.paired_motor_X_step = True
  model.env.mj.set.use_palm_action = True
  model.env.mj.set.use_height_action = False

  # what sensing mode (0=raw data, 1=change, 2=average)
  model.env.mj.set.sensor_sample_mode = 1
  model.env.mj.set.state_sample_mode = 0

  # reward each step                     reward   done   trigger
  model.env.mj.set.step_num.set          (0,      False,   1)

  # penalties and bonuses
  pvalue = -0.025
  pdone = 5
  rvalue = 0.05

  # penalties                            reward   done   trigger  min   max  overshoot
  model.env.mj.set.exceed_limits.set     (pvalue, pdone,   1)
  model.env.mj.set.exceed_axial.set      (pvalue, pdone,   1,     3.0,  6.0,  -1)
  model.env.mj.set.exceed_lateral.set    (pvalue, pdone,   1,     4.0,  6.0,  -1)
  model.env.mj.set.exceed_palm.set       (pvalue, pdone,   1,     6.0,  10.0, -1)

  # binary rewards                       reward   done   trigger
  model.env.mj.set.lifted.set            (rvalue, False,   1)
  model.env.mj.set.target_height.set     (rvalue, False,   1)
  # model.env.mj.set.object_stable.set     (rvalue, False,   1)
  
  # linear rewards                       reward   done   trigger min   max  overshoot
  model.env.mj.set.finger_force.set      (rvalue, False,   1,    0.2,  1.0,  -1)
  model.env.mj.set.palm_force.set        (rvalue, False,   1,    1.0,  6.0,  -1)

  # scale based on steps allowed per episode
  model.env.mj.set.scale_rewards(100 / model.env.max_episode_steps)

  # end criteria                         reward   done   trigger
  # model.env.mj.set.stable_height.set     (0.0,    True,    1)
  model.env.mj.set.object_stable.set     (rvalue, True,    1)
  model.env.mj.set.oob.set               (-100.0, True,    1)

  # termination with poor reward
  model.env.mj.set.quit_on_reward_below = -1
  model.env.mj.set.quit_reward_capped = True

  model.wandb_note += "Simplest sphere lifting\n"

  return model

def apply_to_all_models(model):
  """
  Settings we want to apply to every single running model. This can also be used
  as a reference for which options are possible to change.
  """

  # set up the object set
  model.env._load_object_set(name="set1_nocuboid_525")

  # number of steps in an episode
  model.env.max_episode_steps = 200

  # key learning hyperparameters
  model.params.batch_size = 128
  model.params.learning_rate = 0.0001
  model.params.gamma = 0.999
  model.params.eps_start = 0.9
  model.params.eps_end = 0.05
  model.params.eps_decay = 2000
  model.params.target_update = 100
  model.params.num_episodes = 10_000
  model.params.optimiser = "adam"
  model.params.adam_beta1 = 0.9
  model.params.adam_beta2 = 0.999

  # memory replay and HER
  model.params.memory_replay = 20_000
  model.params.min_memory_replay = 5_000
  model.params.use_HER = False # python setting OVERRIDES cpp
  model.params.HER_mode = "final"
  model.params.HER_k = 4

  # data logging
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

  # what sensing mode (0=raw data, 1=change, 2=average)
  model.env.mj.set.sensor_sample_mode = 1
  model.env.mj.set.state_sample_mode = 0

  # turn off all HER by default
  # model.env.mj.set.use_HER = False
  model.env.mj.set.goal_reward = 1.0
  model.env.mj.set.divide_goal_reward = True
  model.env.mj.set.reward_on_end_only = True
  model.env.mj.set.binary_goal_vector = True

  # wipe all rewards so none trigger
  model.env.mj.set.wipe_rewards()

  # disable use of all sensors, then add back defaults
  model.env.mj.set.disable_sensors()
  model.env.mj.set.motor_state_sensor.in_use = True
  model.env.mj.set.motor_state_sensor.read_rate = -1
  model.env.mj.set.base_state_sensor.read_rate = -1
  model.env.mj.set.bending_gauge.in_use = True

  return model

def continue_training(model, run_name, group_name):
  """
  Continue the training of a model
  """

  print("Continuing training in group:", group_name)
  print("Continuing training of run:", run_name)

  new_endpoint = 20000
  model.wandb_note += f"Continuing training until new endpoint of {new_endpoint} episodes\n"
  model.continue_training(run_name, model.savedir + group_name + "/",
                          new_endpoint=new_endpoint)

  # we are finished when training has finished
  exit()

if __name__ == "__main__":

  """
  This script should be called as follows:

  Option A: start a series of trainings right now:
    $ ./array_training_DQN.py <number> $(date +%d-%m-%y-%H:%M)

  Option B: continue a series of trainings from a previous time:
    $ ./array_training_DQN.py <number> <timestamp> continue 

  where: <number> is used to specify what training regime the user wants. This
         script saves this number as 'inputarg' and it is used with if...elif...
         to determine what settings are used in the training.
         <timestamp> is the training time in the format '%d-%m-%y-%H:%M' as
         can be achieved with the linux 'date' command as seen above.

  eg: ./array_training_DQN.py 2 $(date +%d-%m-%y-%H:%M)
      ./array_training_DQN.py 5 22-06-2022-11:51
  """

  # key settings
  use_wandb = True
  no_plot = True
  log_level = 1

  # print all the inputs we have received
  print("Script inputs are:", sys.argv[1:])

  # extract input arguments
  inputarg = int(sys.argv[1])
  timestamp = sys.argv[2]

  # check if we are continuing a training
  if len(sys.argv) > 3 and sys.argv[3] == "continue":
    resume_training = True
  else:
    resume_training = False

  # save_suffix = f"A{inputarg}_{timestamp[-5:]}" # only include hr:min
  save_suffix = f"{timestamp[-5:]}_A{inputarg}" # only include hr:min

  print("Input argument:", inputarg)
  print("Timestamp is:", timestamp)
  print("Resume training is:", resume_training)

  # create and configure the model to default
  model = TrainDQN(use_wandb=use_wandb, no_plot=no_plot, log_level=log_level)
  model = apply_to_all_models(model)

  # cpu training only on cluster or PC
  if model.machine in ["cluster", "luke-PC"]: 
    model.device = "cpu"

  # override default run/group names
  model.run_name = f"{model.machine}_{save_suffix}"
  model.group_name = timestamp[:8] # include only day-month-year

  # if we are resuming training (currently can only resume on the SAME machine)
  if resume_training: continue_training(model, model.run_name, model.group_name)

  print("Run group name is:", model.group_name)
  print("This run will be saved as:", model.run_name)

  # TEMPORARY TESTING
  if inputarg == 1:

    network = networks.DQN_2L60
    model.wandb_note = f"Network: {network.name}\n"
    model.env_max_episode_steps = 100
    model = simplest_sphere_lifting(model)
    model.params.learning_rate = 0.0001
    model.wandb_note += "Learning rate 0.0001\n"
    model.train(network)

    exit()

  # ----- 3 layer network ----- #
  if inputarg <= 10:

    # now form the network
    network = networks.DQN_3L60
    model.wandb_note += f"Network: {network.name}\n"

    # set parameters
    model.env.max_episode_steps = 250
    
    # learning rate 0.00001
    if inputarg == 1:
      model = create_reward_function(model, style="sparse", options=[])
      model = add_sensors(model, num=5, sensor_mode=1, state_mode=0)
      model = setup_HER(model, use=False)
      model.params.learning_rate = 0.00001
      model.wandb_note += "Learning rate 0.00001\n"
      model.train(network)

    elif inputarg == 2:
      model = create_reward_function(model, style="negative", options=["terminate_early", "cap"])
      model = add_sensors(model, num=5, sensor_mode=1, state_mode=0)
      model = setup_HER(model, use=False)
      model.params.learning_rate = 0.00001
      model.wandb_note += "Learning rate 0.00001\n"
      model.train(network)

    elif inputarg == 3:
      model = create_reward_function(model, style="mixed_v2", options=["terminate_early", "cap"])
      model = add_sensors(model, num=5, sensor_mode=1, state_mode=0)
      model = setup_HER(model, use=False)
      model.params.learning_rate = 0.00001
      model.wandb_note += "Learning rate 0.00001\n"
      model.train(network)

    elif inputarg == 4:
      model = create_reward_function(model, style="sparse_no_rewards", options=[])
      model = add_sensors(model, num=5, sensor_mode=1, state_mode=0)
      model = setup_HER(model, use=True, style="basic", mode="final", k=4)
      model.params.learning_rate = 0.00001
      model.wandb_note += "Learning rate 0.00001\n"
      model.train(network)

    elif inputarg == 5:
      model = create_reward_function(model, style="sparse_no_rewards", options=[])
      model = add_sensors(model, num=5, sensor_mode=1, state_mode=0)
      model = setup_HER(model, use=True, style="forces", mode="final", k=4)
      model.params.learning_rate = 0.00001
      model.wandb_note += "Learning rate 0.00001\n"
      model.train(network)

    # learning rate 0.0001
    elif inputarg == 6:
      model = create_reward_function(model, style="sparse", options=[])
      model = add_sensors(model, num=5, sensor_mode=1, state_mode=0)
      model = setup_HER(model, use=False)
      model.params.learning_rate = 0.0001
      model.wandb_note += "Learning rate 0.0001\n"
      model.train(network)

    elif inputarg == 7:
      model = create_reward_function(model, style="negative", options=["terminate_early", "cap"])
      model = add_sensors(model, num=5, sensor_mode=1, state_mode=0)
      model = setup_HER(model, use=False)
      model.params.learning_rate = 0.0001
      model.wandb_note += "Learning rate 0.0001\n"
      model.train(network)

    elif inputarg == 8:
      model = create_reward_function(model, style="mixed_v2", options=["terminate_early", "cap"])
      model = add_sensors(model, num=5, sensor_mode=1, state_mode=0)
      model = setup_HER(model, use=False)
      model.params.learning_rate = 0.0001
      model.wandb_note += "Learning rate 0.0001\n"
      model.train(network)

    elif inputarg == 9:
      model = create_reward_function(model, style="sparse_no_rewards", options=[])
      model = add_sensors(model, num=5, sensor_mode=1, state_mode=0)
      model = setup_HER(model, use=True, style="basic", mode="final", k=4)
      model.params.learning_rate = 0.0001
      model.wandb_note += "Learning rate 0.0001\n"
      model.train(network)

    elif inputarg == 10:
      model = create_reward_function(model, style="sparse_no_rewards", options=[])
      model = add_sensors(model, num=5, sensor_mode=1, state_mode=0)
      model = setup_HER(model, use=True, style="forces", mode="final", k=4)
      model.params.learning_rate = 0.0001
      model.wandb_note += "Learning rate 0.0001\n"
      model.train(network)

  # ----- half the amount of movement of each action, double number episode steps ----- #
  elif inputarg >= 11 and inputarg <= 15:

    # now form the network
    network = networks.DQN_3L60
    model.wandb_note += f"Network: {network.name}\n"

    # set parameters
    model.env.max_episode_steps = 500
    model.env.mj.set.action_motor_steps = 100
    model.env.mj.set.action_base_translation = 1e-3
    model.env.mj.set.sim_steps_per_action = 100

    # learning rate 0.00001
    if inputarg == 11:
      model = create_reward_function(model, style="sparse", options=[])
      model = add_sensors(model, num=5, sensor_mode=1, state_mode=0)
      model = setup_HER(model, use=False)
      model.params.learning_rate = 0.00001
      model.wandb_note += "Half action effects\n"
      model.train(network)

    elif inputarg == 12:
      model = create_reward_function(model, style="negative", options=["terminate_early", "cap"])
      model = add_sensors(model, num=5, sensor_mode=1, state_mode=0)
      model = setup_HER(model, use=False)
      model.params.learning_rate = 0.00001
      model.wandb_note += "Half action effects\n"
      model.train(network)

    elif inputarg == 13:
      model = create_reward_function(model, style="mixed_v2", options=["terminate_early", "cap"])
      model = add_sensors(model, num=5, sensor_mode=1, state_mode=0)
      model = setup_HER(model, use=False)
      model.params.learning_rate = 0.00001
      model.wandb_note += "Half action effects\n"
      model.train(network)

    elif inputarg == 14:
      model = create_reward_function(model, style="sparse_no_rewards", options=[])
      model = add_sensors(model, num=5, sensor_mode=1, state_mode=0)
      model = setup_HER(model, use=True, style="basic", mode="final", k=4)
      model.params.learning_rate = 0.00001
      model.wandb_note += "Half action effects\n"
      model.train(network)

    elif inputarg == 15:
      model = create_reward_function(model, style="sparse_no_rewards", options=[])
      model = add_sensors(model, num=5, sensor_mode=1, state_mode=0)
      model = setup_HER(model, use=True, style="forces", mode="final", k=4)
      model.params.learning_rate = 0.00001
      model.wandb_note += "Half action effects\n"
      model.train(network)

  # ----- slow target net updates ----- #
  elif inputarg >= 16 and inputarg <= 20:

    # now form the network
    network = networks.DQN_3L60
    model.wandb_note += f"Network: {network.name}\n"

    # set parameters
    model.env.max_episode_steps = 250
    model.params.target_update = 500
    
    # learning rate 0.00001
    if inputarg == 16:
      model = create_reward_function(model, style="sparse", options=[])
      model = add_sensors(model, num=5, sensor_mode=1, state_mode=0)
      model = setup_HER(model, use=False)
      model.params.learning_rate = 0.00001
      model.wandb_note += "Learning rate 0.00001\n"
      model.train(network)

    elif inputarg == 17:
      model = create_reward_function(model, style="negative", options=["terminate_early", "cap"])
      model = add_sensors(model, num=5, sensor_mode=1, state_mode=0)
      model = setup_HER(model, use=False)
      model.params.learning_rate = 0.00001
      model.wandb_note += "Learning rate 0.00001\n"
      model.train(network)

    elif inputarg == 18:
      model = create_reward_function(model, style="mixed_v2", options=["terminate_early", "cap"])
      model = add_sensors(model, num=5, sensor_mode=1, state_mode=0)
      model = setup_HER(model, use=False)
      model.params.learning_rate = 0.00001
      model.wandb_note += "Learning rate 0.00001\n"
      model.train(network)

    elif inputarg == 19:
      model = create_reward_function(model, style="sparse_no_rewards", options=[])
      model = add_sensors(model, num=5, sensor_mode=1, state_mode=0)
      model = setup_HER(model, use=True, style="basic", mode="final", k=4)
      model.params.learning_rate = 0.00001
      model.wandb_note += "Learning rate 0.00001\n"
      model.train(network)

    elif inputarg == 20:
      model = create_reward_function(model, style="sparse_no_rewards", options=[])
      model = add_sensors(model, num=5, sensor_mode=1, state_mode=0)
      model = setup_HER(model, use=True, style="forces", mode="final", k=4)
      model.params.learning_rate = 0.00001
      model.wandb_note += "Learning rate 0.00001\n"
      model.train(network)

   