#!/usr/bin/env python3

# fix for cluster, numpy causes segfault
# import os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
from datetime import datetime
from TrainDQN import TrainDQN
from time import sleep
from random import random
import networks
import argparse

def set_penalties(model, value, done=False, trigger=1, make_binary=None):
  """
  Set penalty rewards with given value, alongside defaults
  """

  # penalties                            reward   done   trigger  min   max  overshoot
  model.env.mj.set.exceed_limits.set     (value,  done,  trigger)
  model.env.mj.set.exceed_axial.set      (value,  done,  trigger, 3.0,  6.0,  -1)
  model.env.mj.set.exceed_lateral.set    (value,  done,  trigger, 4.0,  6.0,  -1)
  model.env.mj.set.exceed_palm.set       (value,  done,  trigger, 6.0,  10.0, -1)

  # make rewards binary trigger by setting 'max' to 'min' for immediate saturation
  if make_binary == True:
    tol = 1e-5 # just in case add a tiny tolerance
    model.env.mj.set.exceed_axial.max = model.env.mj.set.exceed_axial.min + tol
    model.env.mj.set.exceed_lateral.max = model.env.mj.set.exceed_lateral.min + tol
    model.env.mj.set.exceed_palm.max = model.env.mj.set.exceed_palm.min + tol

  return model

def set_bonuses(model, value, make_binary=None):
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

  # make linear rewards binary by setting 'max' to 'min' for immediate saturation
  if make_binary == True:
    tol = 1e-5 # just in case add a tiny tolerance
    model.env.mj.set.finger_force.max = model.env.mj.set.finger_force.min + tol
    model.env.mj.set.palm_force.max = model.env.mj.set.palm_force.min + tol

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

def create_reward_function(model, style="negative", options=[], scale_rewards=1, scale_penalties=1,
                           penalty_termination=False):
  """
  Set the reward structure for the learning, with different style options
  """

  if style == "negative":

    # negative rewards do not work with early termination as the agent can exploit
    # ending the episode early in order to prevent paying a step cost

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

  elif style == "mixed_v2":
    # reward each step                     reward   done   trigger
    model.env.mj.set.step_num.set          (-0.01,  False,   1)
    # penalties and bonuses
    model = set_bonuses(model, 0.005,
                        make_binary=True if "make_binary" in options else None)
    model = set_penalties(model, -0.002,  
                          done=5 if "terminate_early" in options else False,
                          make_binary=True if "make_binary" in options else None)
    # scale based on steps allowed per episode
    model.env.mj.set.scale_rewards(100 / model.env.max_episode_steps)
    # end criteria                         reward   done   trigger
    model.env.mj.set.stable_height.set     (1.0,    True,    1)
    model.env.mj.set.oob.set               (-1.0,   True,    1)

  elif style == "mixed_v3":
    # reward each step                     reward   done   trigger
    model.env.mj.set.step_num.set          (-0.01,  False,   1)
    # penalties and bonuses
    model = set_bonuses(model, 0.002 * scale_rewards,
                        make_binary=True if "make_binary" in options else None)
    model = set_penalties(model, -0.002 * scale_penalties,  
                          done=penalty_termination,
                          make_binary=True if "make_binary" in options else None)
    # scale based on steps allowed per episode
    model.env.mj.set.scale_rewards(100 / model.env.max_episode_steps)
    # end criteria                         reward   done   trigger
    model.env.mj.set.stable_height.set     (1.0,    True,    1)
    model.env.mj.set.oob.set               (-1.0,   True,    1)

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
  
  else:
    raise RuntimeError("style was not set to a valid option in create_reward_function()")

  # termination on specific reward
  model.env.mj.set.quit_on_reward_below = -1.0 if "neg_cap" in options else -1e6
  model.env.mj.set.quit_on_reward_above = +1.0 if "pos_cap" in options else 1e6
  model.env.mj.set.quit_reward_capped = True

  model.wandb_note += f"Reward style: '{style}', options: [ "
  for extra in options: model.wandb_note += f"'{extra}' "
  model.wandb_note += "]\n"

  return model

def add_sensors(model, num=None, sensor_mode=None, state_mode=None, sensor_steps=None,
  state_steps=None, noise_std=None, z_state=None):
  """
  Add a number of sensors
  """

  if num is None: num = 10 # default, include all sensors

  # define defaults that can be overriden by function inputs
  if sensor_mode is None: sensor_mode = 1
  if state_mode is None: state_mode = 0
  if noise_std is None: noise_std = 0.05
  if sensor_steps is None: sensor_steps = 1
  if state_steps is None: state_steps = 2

  # default: state sensor and bending gauge sensor
  model.env.mj.set.motor_state_sensor.in_use = True
  model.env.mj.set.bending_gauge.in_use = True

  # what sensing mode (0=raw data, 1=change, 2=average, 3=median)
  model.env.mj.set.sensor_sample_mode = sensor_mode
  model.env.mj.set.state_sample_mode = state_mode

  # add minor gaussian sensing noise with mean 0
  model.env.mj.set.sensor_noise_std = noise_std
  model.env.mj.set.state_noise_std = noise_std

  # set the number of steps in the past we use for observations
  model.env.mj.set.sensor_n_prev_steps = sensor_steps
  model.env.mj.set.state_n_prev_steps = state_steps

  # palm force sensor
  if num >= 1: model.env.mj.set.palm_sensor.in_use = True

  # wrist z force sensor
  if num >= 2: model.env.mj.set.wrist_sensor_Z.in_use = True

  # finger axial gauges
  if num >= 3: model.env.mj.set.axial_gauge.in_use = True

  # know where z is in physical space (base z state sensor)
  if num >= 4 or z_state is True: model.env.mj.set.base_state_sensor.in_use = True

  model.wandb_note += (
    f"Num sensors: {num}, state mode: {state_mode}, sensor mode: {sensor_mode}" +
    f", state steps: {state_steps}, sensor steps: {sensor_steps}, sensor noise std: {noise_std}\n"
  )

  return model

def apply_to_all_models(model):
  """
  Settings we want to apply to every single running model. This can also be used
  as a reference for which options are possible to change.
  """

  # number of steps in an episode
  model.env.max_episode_steps = 250

  # key learning hyperparameters
  model.params.object_set = "set2_nocuboid_525"
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
  model.params.memory_replay = 50_000
  model.params.min_memory_replay = 5_000
  model.params.use_HER = False # python setting OVERRIDES cpp
  model.params.HER_mode = "final"
  model.params.HER_k = 4

  # curriculum learning
  model.params.use_curriculum = False
  model.params.curriculum_ep_num = 8000
  model.params.curriculum_object_set = "set2_fullset_795"

  # data loggings
  model.params.save_freq = 1_000
  model.params.test_freq = 1_000
  model.params.plot_freq_s = 300
  model.params.wandb_freq_s = 300

  # ensure debug mode is off
  model.env.log_level = 0
  model.env.mj.set.debug = False

  # disable all rendering
  model.env.disable_rendering = True
  model.env.mj.set.use_render_delay = False
  model.env.mj.set.render_on_step = False

  # define lengths and forces
  model.env.mj.set.finger_stiffness = 5
  model.env.mj.set.oob_distance = 75e-3
  model.env.mj.set.done_height = 35e-3
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
  model.env.mj.set.quit_on_reward_below = -1e6
  model.env.mj.set.quit_on_reward_above = 1e6
  model.env.mj.set.quit_reward_capped = False

  # disable use of all sensors
  model.env.mj.set.disable_sensors()
  model.env.mj.set.sensor_n_prev_steps = 1 # lookback only 1 step
  model.env.mj.set.state_n_prev_steps = 1 # lookback only 1 step

  # add back default sensors
  model.env.mj.set.motor_state_sensor.in_use = True
  model.env.mj.set.bending_gauge.in_use = True

  # ensure state sensors only give one reading per step (read_rate < 0)
  model.env.mj.set.motor_state_sensor.read_rate = -1
  model.env.mj.set.base_state_sensor.read_rate = -1

  # sensor noise options
  model.env.mj.set.sensor_noise_mag = 0
  model.env.mj.set.sensor_noise_mu = 0
  model.env.mj.set.sensor_noise_std = 0
  model.env.mj.set.state_noise_mag = 0
  model.env.mj.set.state_noise_mu = 0
  model.env.mj.set.state_noise_std = 0

  # logging/plotting options
  model.track.moving_avg_num = 100
  model.track.static_avg_num = model.track.moving_avg_num
  model.track.plot_raw = False
  model.track.plot_moving_avg = False
  model.track.plot_static_avg = True
  model.track.plot_test_raw = True
  model.track.plot_test_metrics = True
  model.track.plot_success_rate = True
  model.track.success_rate_metric = "stable height"
  model.track.plot_time_taken = True

  return model

def continue_training(model, run_name, group_name, object_set=None):
  """
  Continue the training of a model
  """

  print("Continuing training in group:", group_name)
  print("Continuing training of run:", run_name)

  # set up the object set
  model.env.mj.model_folder_path = "/home/luke/mymujoco/mjcf"

  # new_endpoint = 20_000
  # model.wandb_note += f"Continuing training until new endpoint of {new_endpoint} episodes\n"

  extra_episodes = 10_000
  model.wandb_note += f"Continuing training with an extra {extra_episodes} episodes\n"
  
  model.continue_training(run_name, model.savedir + group_name + "/",
                          extra_episodes=extra_episodes, object_set=object_set)

def logging_job(model, run_name, group_name):
  """
  Log training data, either to wandb or plot it to screen, or both
  """

  print("Logging training in group:", group_name)
  print("Logging training of run:", run_name)

  model.load(folderpath=model.savedir + group_name + "/", foldername=run_name)
  
  # logging/plotting options
  model.track.plot_raw = False
  model.track.plot_moving_avg = False
  model.track.plot_static_avg = True
  model.track.plot_test_raw = True
  model.track.plot_test_metrics = True
  model.track.plot_success_rate = True
  model.track.success_rate_metric = "stable height"
  model.track.plot_time_taken = True
  
  model.log_wandb(force=True, end=True)
  model.plot(force=True, end=True, hang=True)

def baseline_training(model, lr=5e-5, eps_decay=2000, sensors=2, network=networks.DQN_4L100, 
                      memory=50_000, state_steps=1, sensor_steps=1, z_state=True, sensor_mode=2,
                      state_mode=1, reward_style="mixed_v3", reward_options=[], scale_rewards=2.5,
                      scale_penalties=1.0, penalty_termination=False, finger_stiffness=8):
  """
  Runs a baseline training on the model
  """

  # set parameters
  model.env.max_episode_steps = 250
  model.params.learning_rate = lr
  model.params.eps_decay = eps_decay
  model.params.memory_replay = memory
  model.env.mj.set.finger_stiffness = finger_stiffness

  # wandb notes
  model.wandb_note += f"Network: {network.name}\n"
  model.wandb_note += f"Learning rate {lr}\n"
  model.wandb_note += f"eps_decay = {eps_decay}\n"
  
  # configure rewards and sensors
  model = create_reward_function(model, style=reward_style, options=reward_options,
                                 scale_rewards=scale_rewards, scale_penalties=scale_penalties,
                                 penalty_termination=penalty_termination)
  model = add_sensors(model, num=sensors, sensor_mode=sensor_mode, state_mode=state_mode,
                      state_steps=state_steps, sensor_steps=sensor_steps,
                      z_state=z_state)
  model = setup_HER(model, use=False)

  # train and finish
  model.train(network)
  exit()

if __name__ == "__main__":

  """
  This script should be called using the flags defined below:

  Required:
    -j [ARG] job input number, one integer only

  Optional:
    -t, --timestamp [ARG]   timestamp in 'DD-MM-YY-HR:MN' format
    -m, --machine [ARG]     machine name for run eg 'cluster', 'luke-PC'
    -o, --object-set [ARG]  object set name to use, eg 'set1_fullset_795'
    -c, --continue          continue a previous training
    -l, --log-wandb         logging job, log to weights and biases
    -p, --plot              logging job, plot graphs of training to screen
    -n, --no-wandb          no weights and biases, disable live logging
    --device                what device to use, 'cpu' or 'cuda'
    --savedir               directory to save/load into eg '/home/luke/models/'
    --print                 print info on current comparison parameters

  Examples:
    ./array_training_DQN.py -j 1
    ./array_training_DQN.py -j 3 -t 12-05-22-09:42 -c -n
  """

  # key default settings
  use_wandb = True
  no_plot = True
  log_level = 1
  datestr = "%d-%m-%Y-%H:%M" # all date inputs must follow this format

  # # print all the inputs we have received
  # print("array_training_DQN.py inputs are:", sys.argv[1:])

  # define arguments and parse them
  parser = argparse.ArgumentParser()
  parser.add_argument("-j", "--job",          type=int)            # job input number
  parser.add_argument("-t", "--timestamp",    default=None)        # timestamp
  parser.add_argument("-m", "--machine",      default=None)        # machine
  parser.add_argument("-o", "--object-set",   default=None)        # object set name
  parser.add_argument("-c", "--continue",     action="store_true", dest="resume") # continue training
  parser.add_argument("-l", "--log-wandb",    action="store_true") # log to wandb job
  parser.add_argument("-p", "--plot",         action="store_true") # plot to wandb job
  parser.add_argument("-n", "--no-wandb",     action="store_true") # no wandb logging
  parser.add_argument("--device",             default=None)        # override device
  parser.add_argument("--savedir",            default=None)        # override save/load directory
  parser.add_argument("--print",              action="store_true") # don't train, print help

  args = parser.parse_args()

  # # parse arguments but allow unknown arguments
  # args, unknown = parser.parse_known_args()

  # extract primary inputs
  inputarg = args.job
  timestamp = args.timestamp if args.timestamp else datetime.now().strftime(datestr)
  if args.no_wandb: use_wandb = False

  if args.print: 
    args.log_wandb = False
    log_level = 0

  # echo these inputs
  if log_level > 0:
    print("\narray_training_DQN is preparing to train:")
    print(" -> Input arg:", inputarg)
    print(" -> Timestamp is:", timestamp)
    print(" -> Use wandb is:", use_wandb)

  # seperate process for safety
  sleep(inputarg)
  sleep(random())

  save_suffix = f"{timestamp[-5:]}_A{inputarg}" # only include hr:min

  # create and configure the model to default
  model = TrainDQN(use_wandb=use_wandb, no_plot=no_plot, log_level=log_level,
                   object_set = args.object_set)
  model = apply_to_all_models(model)

  # cpu training only on cluster or PC
  if model.machine in ["cluster", "luke-PC"] and args.device is None: 
    model.set_device("cpu")
    if log_level > 0: print(" -> Setting to default 'cpu' device, to override use '--device cuda'")
  elif args.device is not None:
    if log_level > 0: print(f" -> Device override of '{args.device}'")
    model.set_device(args.device)

  # override default run/group names
  model.run_name = f"{model.machine}_{save_suffix}"
  model.group_name = timestamp[:8] # include only day-month-year

  if args.machine is not None:
    if log_level > 0: print(f" -> Machine override of '{args.machine}'")
    model.run_name = f"{args.machine}_{save_suffix}"

  # override save location
  if args.savedir is not None:
    if log_level > 0: print(f" -> Savedir override of '{args.savedir}'")
    model.savedir = args.savedir

  if log_level > 0:
    print(" -> Run group is:", model.group_name)
    print(" -> Run name is:", model.run_name, "\n")

  # ----- SPECIAL JOB OPTIONS ----- #

  # if we are resuming training (currently can only resume on the SAME machine)
  if args.resume:
    if log_level > 0: print("Resuming training")
    # we need to pass the object set to override the loaded default
    continue_training(model, model.run_name, model.group_name,
                      object_set=args.object_set)
    exit()

  # if we are doing a logging job
  if args.log_wandb or args.plot: 
    if log_level > 0: print(f"Logging job, plot is {args.plot} and wandb is {args.log_wandb}")
    model.no_plot = not args.plot
    model.use_wandb = args.log_wandb
    logging_job(model, model.run_name, model.group_name)
    if args.plot:
      input("Press enter to quit plotting windows and terminate program")
    exit()

  # ----- BEGIN TRAININGS ----- #

  """ 
  # How to perform a baseline training example:

  # varying two parameters 6x5 = 30 possible trainings 1-30
  stiffness_list = [5, 6, 7, 8, 9, 10]
  sensors_list = [1, 2, 3, 4, 5]

  # lists are zero indexed so adjust inputarg
  inputarg -= 1

  # we vary wrt the second list every inputarg increment
  x = len(sensors_list)

  # choose the values of each parameter for this training (based on inputarg)
  this_stiffness = stiffness_list[inputarg // x]       # vary every x steps
  this_sensors = sensors_list[inputarg % x]            # vary every +1 & loop

  # The pattern goes (with list_1=A,B,C... and list_2=1,2,3...)
  #   A1, A2, A3, ...
  #   B1, B2, B3, ...
  #   C1, C2, C3, ...

  # make note of the parameters chosen
  param_1 = f"Finger stiffness used: {this_stiffness}\n"
  param_2 = f"Sensors used: {this_sensors}\n"
  model.wandb_note += param_1 + param_2

  # if we are just printing help information
  if args.print:
    print("Input arg", inputarg + 1)
    print("\t" + param_1, end="")
    print("\t" + param_2, end="")
    exit()

  # parameter changes can be applied here or in the following function args
  model.env.mj.set.finger_stiffness = this_stiffness

  # perform the training with standard baseline settings unless specified by args
  baseline_training(model, sensors=this_sensors)
  """

  # sensors_list = [
  #   0, # bending and z state
  #   1, # + palm
  #   2  # + wrist
  # ]

  # # lists are zero indexed
  # inputarg -= 1

  # model.params.num_episodes = 40000
  # baseline_training(model, sensors=sensors_list[inputarg]) 

  # varying 3x3 = possible trainings 1-9
  sensors_list = [
    0, # bending and z state
    1, # + palm
    2  # + wrist
  ]

  num_segments_list = [
    "set3_fullset_795_N/5_free_seg",
    "set3_fullset_795_N/7_free_seg",
    "set3_fullset_795_N/10_free_seg"
  ]

  # lists are zero indexed so adjust inputarg
  inputarg -= 1

  # allow looping input indexes
  while inputarg >= 9: inputarg -= 9

  # we vary wrt memory_list every inputarg increment
  x = len(num_segments_list)

  # get the sensors and memory size for this training
  this_sensor = sensors_list[inputarg // x]                 # vary every x steps
  this_set = num_segments_list[inputarg % x]                # vary every +1 & loop

  # The pattern goes (with list_1=A,B,C... and list_2=1,2,3...)
  #   A1, A2, A3, ...
  #   B1, B2, B3, ...
  #   C1, C2, C3, ...

  # make note
  param_1 = f"Sensors is {this_sensor}\n"
  param_2 = f"Object set is {this_set}\n"
  model.wandb_note += param_1 + param_2

  # if we are just printing help information
  if args.print:
    print("Input arg", inputarg + 1)
    print("\t" + param_1, end="")
    print("\t" + param_2, end="")
    exit()

  # # if we use curriculum learning
  # if this_lr[1]:
  #   model.params.object_set = "set3_nocuboid_525"
  #   model.params.use_curriculum = True
  #   model.params.curriculum_object_set = "set3_fullset_795"
  # else:
  #   model.params.object_set = "set3_fullset_795"

  # set number of training episodes
  model.params.num_episodes = 40000

  # automatically find the highest stable timestep
  model.env.mj.set.mujoco_timestep = -1

  # apply the number of segments
  model.params.object_set = this_set

  # reduce the amount of saving/testing/logging
  model.params.save_freq = 2000
  model.params.test_freq = 2000
  model.params.wandb_freq_s = 3600

  # perform the training with other parameters standard
  baseline_training(model, sensors=this_sensor) 

  # ----- END ----- #
   