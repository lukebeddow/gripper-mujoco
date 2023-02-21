#!/usr/bin/env python3

# fix for cluster, numpy causes segfault
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
from datetime import datetime
from TrainDQN import TrainDQN
from time import sleep
from random import random
import networks
import argparse

def set_penalties(model, value, done=False, trigger=1, make_binary=None,
                  exceed_lims_multiplier=1.0):
  """
  Set penalty rewards with given value, alongside defaults
  """

  # TESTING: extra punishment for exceeding limits (touching table in particular we want to target)
  print("TESTING: exceed_lims_multiplier is", exceed_lims_multiplier, "\n")
  modval = value * exceed_lims_multiplier

  # penalties                            reward   done   trigger  min   max  overshoot
  model.env.mj.set.exceed_limits.set     (modval, done,  trigger)
  model.env.mj.set.exceed_axial.set      (value,  done,  trigger, 3.0,  6.0,  -1)
  model.env.mj.set.exceed_lateral.set    (value,  done,  trigger, 4.0,  6.0,  -1) # min and max currently overwritten with (1.0 and 1.5)*yield_load()
  model.env.mj.set.exceed_palm.set       (value,  done,  trigger, 6.0,  15.0, -1)

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
  
  # # OLD: linear rewards                       reward   done   trigger min   max  overshoot
  # model.env.mj.set.finger_force.set      (value,  False,   1,    0.2,  1.0,  -1)
  # model.env.mj.set.palm_force.set        (value,  False,   1,    1.0,  6.0,  -1)

  # NEW IDEA: choose reward ranges based on the desired stable force
  min = 0.2
  sbf = model.env.mj.set.stable_finger_force
  sbp = model.env.mj.set.stable_palm_force

  # linear rewards                       reward   done   trigger min   max  overshoot
  model.env.mj.set.finger_force.set      (value,  False,   1,    min,  sbf,  -1)
  model.env.mj.set.palm_force.set        (value,  False,   1,    min,  sbp,  -1)

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
                           penalty_termination=False, exceed_lims_multiplier=1.0):
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
                          done=5 if "terminate_early" in options else False,
                          exceed_lims_multiplier=exceed_lims_multiplier)
    # scale based on steps allowed per episode
    model.env.mj.set.scale_rewards(100 / model.env.params.max_episode_steps)
    # end criteria                         reward   done   trigger
    model.env.mj.set.stable_height.set     (0.0,    True,    1)
    model.env.mj.set.oob.set               (-1.0,   True,    1)

  elif style == "mixed":

    # reward each step                     reward   done   trigger
    model.env.mj.set.step_num.set          (0.0,    False,   1)
    # penalties and bonuses
    model = set_bonuses(model, 0.002)
    model = set_penalties(model, -0.005,  
                          done=5 if "terminate_early" in options else False,
                          exceed_lims_multiplier=exceed_lims_multiplier)
    # scale based on steps allowed per episode
    model.env.mj.set.scale_rewards(100 / model.env.params.max_episode_steps)
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
                          make_binary=True if "make_binary" in options else None,
                          exceed_lims_multiplier=exceed_lims_multiplier)
    # scale based on steps allowed per episode
    model.env.mj.set.scale_rewards(100 / model.env.params.max_episode_steps)
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
                          make_binary=True if "make_binary" in options else None,
                          exceed_lims_multiplier=exceed_lims_multiplier)
    # scale based on steps allowed per episode
    model.env.mj.set.scale_rewards(100 / model.env.params.max_episode_steps)
    # end criteria                         reward   done   trigger
    model.env.mj.set.stable_height.set     (1.0,    True,    1)
    model.env.mj.set.oob.set               (-1.0,   True,    1)

  elif style == "sparse":
    # reward each step                     reward   done   trigger
    model.env.mj.set.step_num.set          (0.0,    False,   1)
    # penalties and bonuses
    model = set_bonuses(model, 0.0)
    model = set_penalties(model, 0.0,  
                          done=5 if "terminate_early" in options else False,
                          exceed_lims_multiplier=exceed_lims_multiplier)
    # scale based on steps allowed per episode
    model.env.mj.set.scale_rewards(100 / model.env.params.max_episode_steps)
    # end criteria                         reward   done   trigger
    model.env.mj.set.object_stable.set     (1.0,    True,    1)
    model.env.mj.set.oob.set               (0.0,    True,    1)

  elif style == "sparse_no_rewards":
    # reward each step                     reward   done   trigger
    model.env.mj.set.step_num.set          (0.0,    False,   1)
    # penalties and bonuses
    model = set_bonuses(model, 0.0)
    model = set_penalties(model, 0.0,  
                          done=5 if "terminate_early" in options else False,
                          exceed_lims_multiplier=exceed_lims_multiplier)
    # scale based on steps allowed per episode
    model.env.mj.set.scale_rewards(100 / model.env.params.max_episode_steps)
    # end criteria                         reward   done   trigger
    model.env.mj.set.object_stable.set     (0.0,    True,    1)
    model.env.mj.set.oob.set               (0.0,    True,    1)
  
  else:
    raise RuntimeError("style was not set to a valid option in create_reward_function()")

  # specific options
  if "terminate_on_exceed_limits" in options:
    # reward each step                     reward   done   trigger
    model.env.mj.set.exceed_limits.set     (-1.0,   True,    3)

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

  # enable noise and normalisation for every sensor (should be enabled by default anyway)
  model.env.mj.set_sensor_noise_and_normalisation_to(True)

  # what sensing mode (0=raw data, 1=change, 2=average, 3=median)
  model.env.mj.set.sensor_sample_mode = sensor_mode
  model.env.mj.set.state_sample_mode = state_mode

  # set the same noise to regular sensors and state sensors
  model.env.mj.set.sensor_noise_std = noise_std
  model.env.mj.set.state_noise_std = noise_std

  # set the number of steps in the past we use for observations
  model.env.mj.set.sensor_n_prev_steps = sensor_steps
  model.env.mj.set.state_n_prev_steps = state_steps

  # --- start adding sensors, all should initially be disabled --- #

  # state sensor (default)
  if num >= 0: model.env.mj.set.motor_state_sensor.in_use = True

  # bending sensor
  if num >= 1: model.env.mj.set.bending_gauge.in_use = True

  # palm force sensor
  if num >= 2: model.env.mj.set.palm_sensor.in_use = True

  # wrist z force sensor
  if num >= 3: model.env.mj.set.wrist_sensor_Z.in_use = True

  # finger axial gauges
  if num >= 4: model.env.mj.set.axial_gauge.in_use = True

  # know where z is in physical space (base z state sensor)
  if num >= 5 or z_state is True: model.env.mj.set.base_state_sensor.in_use = True

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
  model.env.params.max_episode_steps = 250

  # key learning hyperparameters
  model.params.object_set = "set4_fullset_795"
  model.params.batch_size = 128
  model.params.learning_rate = 5e-5
  model.params.gamma = 0.999
  model.params.eps_start = 0.9
  model.params.eps_end = 0.05
  model.params.eps_decay = 4000
  model.params.target_update = 100
  model.params.num_episodes = 60_000
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
  model.params.save_freq = 2_000
  model.params.test_freq = 2_000
  model.params.plot_freq_s = 300
  model.params.wandb_freq_s = 900

  # ensure debug mode is off
  model.env.log_level = 0
  model.env.mj.set.debug = False

  # disable all rendering
  model.env.disable_rendering = True
  model.env.mj.set.use_render_delay = False
  model.env.mj.set.render_on_step = False

  # automatically calibrate
  model.env.mj.set.auto_set_timestep = True
  model.env.mj.set.auto_calibrate_gauges = True
  model.env.mj.set.auto_sim_steps = True
  model.env.mj.set.auto_exceed_lateral_lim = True # THIS OVERRIDES LATERAL PUNISHMENT ONLY
  # model.env.mj.set.bend_gauge_normalise = 5.0 # calibrate saturation to 5.0N # setting deleted
  model.env.mj.set.time_for_action = 0.2
  model.env.mj.set.saturation_yield_factor = 1.0
  model.env.mj.set.exceed_lat_min_factor = 0.75
  model.env.mj.set.exceed_lat_max_factor = 1.5

  # define lengths and forces
  model.env.mj.set.finger_stiffness = -7.5 # finalised theory (101? 102?)
  model.env.mj.set.oob_distance = 75e-3
  model.env.mj.set.done_height = 15e-3
  model.env.mj.set.stable_finger_force = 1.0
  model.env.mj.set.stable_palm_force = 1.0

  # what actions are we using
  model.env.mj.set.paired_motor_X_step = True
  model.env.mj.set.use_palm_action = True
  model.env.mj.set.use_height_action = True
  model.env.mj.set.XYZ_action_mm_rad = True
  model.env.mj.set.X_action_mm = 1.0
  model.env.mj.set.Y_action_rad = 0.01
  model.env.mj.set.Z_action_mm = 2.0
  model.env.mj.set.base_action_mm = 2.0 # not used currently
  model.env.mj.set.fingertip_min_mm = -12.5 # MOVEMENT BELOW THIS SETS within_limits=false;

  # what sensing mode (0=raw data, 1=change, 2=average)
  model.env.mj.set.sensor_sample_mode = 1
  model.env.mj.set.state_sample_mode = 0

  # turn off all HER by default
  # model.env.mj.set.use_HER = False # this setting is OVERRIDEN by model.params.use_HER
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

  # DONT ADD BACK DEFAULT SENSORS, leave to add_sensors(...) function
  # # add back default sensors
  # model.env.mj.set.motor_state_sensor.in_use = True
  # model.env.mj.set.bending_gauge.in_use = True

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
  model.track.moving_avg_num = 250
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

def continue_training(model, run_name, group_name, object_set=None, new_endpoint=None,
                      extra_episodes=None):
  """
  Continue the training of a model
  """

  print("Continuing training in group:", group_name)
  print("Continuing training of run:", run_name)

  # set up the object set
  model.env.mj.model_folder_path = "/home/luke/mymujoco/mjcf"

  # new_endpoint = 100_010
  # model.wandb_note += f"Continuing training until new endpoint of {new_endpoint} episodes\n"

  extra_episodes = 40_000
  model.wandb_note += f"Continuing training with an extra {extra_episodes} episodes\n"
  
  model.continue_training(run_name, model.savedir + group_name + "/",
                          new_endpoint=new_endpoint, extra_episodes=extra_episodes,
                          object_set=object_set, overridelib=args.override_lib)
  test(model)

def logging_job(model, run_name, group_name):
  """
  Log training data, either to wandb or plot it to screen, or both
  """

  print("Logging training in group:", group_name)
  print("Logging training of run:", run_name)

  # turn off any and all auto-calibrations
  model.env.mj.set.auto_set_timestep = False
  model.env.mj.set.auto_calibrate_gauges = False
  model.env.mj.set.auto_sim_steps = False

  model.load(folderpath=model.savedir + group_name + "/", foldername=run_name,
             overridelib=args.override_lib)
  
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

def baseline_settings(model, lr=5e-5, eps_decay=4000, sensors=3, network=[150, 100, 50], 
                      memory=50_000, state_steps=1, sensor_steps=1, z_state=True, sensor_mode=2,
                      state_mode=1, sensor_noise=0.025, reward_style="mixed_v3", reward_options=[], 
                      scale_rewards=2.5, scale_penalties=1.0, penalty_termination=False,
                      finger_stiffness=-7.5, num_segments=8, finger_thickness=0.9e-3, finger_width=28e-3,
                      max_episode_steps=250, XYZ_mm_rad=[1.0, 0.01, 2.0],
                      exceed_lims_multiplier=2.0):

  """
  Runs a baseline training on the model
  """

  # set parameters
  model.env.params.max_episode_steps = max_episode_steps
  model.params.learning_rate = lr
  model.params.eps_decay = eps_decay
  model.params.memory_replay = memory
  model.env.mj.set.finger_stiffness = finger_stiffness # -7.5 is final derivation
  model.num_segments = num_segments                    # 6 gives fast training primarily
  model.env.params.finger_thickness = finger_thickness # options are 0.8e-3, 0.9e-3, 1.0e-3
  model.env.load_finger_width = finger_width           # options are 24e-3, 28e-3

  # options for setting step size, see current defaults in this function
  if XYZ_mm_rad is False: 
    model.env.mj.set.XYZ_action_mm_rad = False
  elif isinstance(XYZ_mm_rad, list) and len(XYZ_mm_rad) == 3:
    model.env.mj.set.XYZ_action_mm_rad = True
    model.env.mj.set.X_action_mm = XYZ_mm_rad[0]
    model.env.mj.set.Y_action_rad = XYZ_mm_rad[1]
    model.env.mj.set.Z_action_mm = XYZ_mm_rad[2]
  elif XYZ_mm_rad is True:
    # apply defaults
    model.env.mj.set.X_action_mm = 1.0
    model.env.mj.set.Y_action_rad = 0.01
    model.env.mj.set.Z_action_mm = 2.0
  else:
    raise RuntimeError(f"array_training_DQN.baseline_settings(...) error with XYZ_mm_rad being: {XYZ_mm_rad}")

  # wandb notes
  model.wandb_note += f"Learning rate {lr}\n"
  model.wandb_note += f"eps_decay = {eps_decay}\n"
  model.wandb_note += f"finger_stiffness = {finger_stiffness}\n"
  model.wandb_note += f"num_segments = {num_segments}\n"
  
  # configure rewards and sensors
  model = create_reward_function(model, style=reward_style, options=reward_options,
                                 scale_rewards=scale_rewards, scale_penalties=scale_penalties,
                                 penalty_termination=penalty_termination,
                                 exceed_lims_multiplier=exceed_lims_multiplier)
  model = add_sensors(model, num=sensors, sensor_mode=sensor_mode, state_mode=state_mode,
                      state_steps=state_steps, sensor_steps=sensor_steps,
                      z_state=z_state, noise_std=sensor_noise)
  model = setup_HER(model, use=False)

  # finish initialisation of model
  if network is not None:
    model.init(network)

  return model

def heuristic_test(model, inputarg=None, render=False):
  """
  Do a heuristic test with baseline settings. Most of these settings are irrelevant
  (RL hyperparameters) but matter, like sensor setup. Best to be safe
  """

  # # temporary override!
  # model.env.params.test_objects = 5

  vary_1 = [0, 1, 2, 3]
  vary_2 = [0.9e-3]
  vary_3 = [8]
  repeats = 5
  param_1_name = "Num sensors"
  param_2_name = "Finger thickness"
  param_3_name = "Num segments"
  param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                              param_3=vary_3, repeats=repeats)
  baseline_args = {
    "sensors" : param_1,
    "finger_thickness" : param_2,
    "num_segments" : param_3
  }

  # note and printing information
  param_1_string = f"{param_1_name} is {param_1}\n" if param_1 is not None else ""
  param_2_string = f"{param_2_name} is {param_2}\n" if param_2 is not None else ""
  param_3_string = f"{param_3_name} is {param_3}\n" if param_3 is not None else ""
  model.wandb_note += param_1_string + param_2_string + param_3_string

  # if we are just printing help information
  if args.print:
    print("Input arg", args.job)
    print("\t" + param_1_string, end="")
    print("\t" + param_2_string, end="")
    print("\t" + param_3_string, end="\n")
    exit()

  # apply baseline settings
  model = baseline_settings(model, **baseline_args)

  # perform the test
  if render: model.env.disable_rendering = False
  model.test_heuristic_baseline()

  print(f"Finished heurisitc test with sensors = {param_1} and thickness = {param_2} and num segments = {param_3}")

def vary_all_inputs(raw_inputarg=None, param_1=None, param_2=None, param_3=None, repeats=None):
  """
  Helper function for adjusting parameters. With param_1 set to list_1 and param_2 set to list_2:

  The pattern goes (with param_1=[A,B,C...] and param_2=[1,2,3...])
    A1, A2, A3, ...
    B1, B2, B3, ...
    C1, C2, C3, ...

  With param_3=[X,Y,Z,...] we repeat the above grid first for X, then Y etc

  Set repeats to get sequential repeats, eg repeats=3 gives
    A1, A1, A1, A2, A2, A2, A3, A3, A3, ...
  """

  # convert input arg from 1...Max to 0...Max-1
  inputarg = raw_inputarg - 1

  # understand inputs
  if param_1 is not None:
    if isinstance(param_1, list):
      list_1 = param_1
    else:
      list_1 = [param_1]
    len_list_1 = len(list_1)
  else: return None, None, None

  if param_2 is not None:
    if isinstance(param_2, list):
      list_2 = param_2
    else:
      list_2 = [param_2]
    len_list_2 = len(list_2)
  else:
    len_list_2 = 1

  if param_3 is not None:
    if param_2 is None: raise RuntimeError("param_2 must be specified before param_3 in vary_all_inputs()")
    if isinstance(param_3, list):
      list_3 = param_3
    else:
      list_3 = [param_3]
    len_list_3 = len(list_3)
  else:
    len_list_3 = 1

  if repeats is None: repeats = 1

  # how fast do we move through lists
  list_1_changes = repeats
  list_2_changes = repeats * len_list_1
  list_3_changes = repeats * len_list_1 * len_list_2

  # don't allow overflow
  num_trainings = len_list_1 * len_list_2 * len_list_3 * repeats
  if raw_inputarg > num_trainings:
    raise RuntimeError(f"vary_all_inputs() got raw_inputarg={raw_inputarg} too high, num_trainings={num_trainings}")

  var_1 = list_1[(inputarg // list_1_changes) % len_list_1]
  if param_2 is not None:
    var_2 = list_2[(inputarg // list_2_changes) % len_list_2]
  else: var_2 = None
  if param_3 is not None:
    var_3 = list_3[(inputarg // list_3_changes) % len_list_3]
  else: var_3 = None

  return var_1, var_2, var_3

def get_best_performance(model):
  """
  Return the best performance of the model
  """

  folderpath = model.savedir + model.group_name + "/" + model.run_name + "/"

  # get all files with pickle extension in the target directory
  pkl_files = [x for x in os.listdir(folderpath) if x.endswith("training_finished.txt")]

  # if there are no candidate files
  if len(pkl_files) == 0: return "training finished file not found"
  if len(pkl_files) > 1: return "multiple finished files found"

  with open(folderpath + pkl_files[0], 'r') as f:
    output = f.read()
  
  output = output.splitlines()[-1]

  return output

def test(model, heuristic=False, trials_per_obj=5, render=False, pause=False):
  """
  Perform a thorough test on the model, including loading the best performing network
  """

  print("\nPreparing to perform a model test, heuristic =", heuristic)

  # load the best performing network
  if not heuristic: model.load(best_id=True)

  # adjust settings
  model.env.params.test_trials_per_object = trials_per_obj
  if render: model.env.disable_rendering = False

  # perform the test
  test_data = model.test(heuristic=heuristic, pause_each_episode=pause)
  test_report = model.create_test_report(test_data)

  # save data to a text file
  savetxt = f"array_training_DQN.test(...) final success rate = {model.last_test_success_rate}\n"
  savetxt += "\n" + test_report
  if heuristic: savename = "heuristic_test_"
  else: savename = "full_test_"
  currenttime = datetime.now().strftime(datestr)
  model.modelsaver.save(savename + currenttime, txtonly=True, txtstr=savetxt)

  return model

def print_results(model, filename="results.txt", savefile="table.txt"):
  """
  Create a results table, presumes a file called 'results.txt' which is the terminal
  output from running './pc_job -j "X:Y" -t DD-MM-YY-HR:MN --program xxxxx --print
  """

  filepath = model.savedir + model.group_name + "/" + filename

  table = []

  with open(filepath, 'r') as f:
    text = f.readlines()

  # print("the text is:\n\n", text)

  print_str = """"""

  first_elem = True
  done_first_elem = False
  temp_headings = []
  headings = []
  new_elem = []

  for line in text:

    if line.startswith("Input arg"):

      if done_first_elem:
        table.append(new_elem)
        if len(temp_headings) > len(headings): headings = temp_headings[:]
        temp_headings = []
        new_elem = []

      splits = line.split(" ")
      # int(...) drops the newline character
      new_elem.append(int(splits[-1]))

      temp_headings.append("Input arg")
      done_first_elem = True
        

    elif line.startswith("\t"):
      if line.startswith("\tTraining time best"):
        splits = line.split(" = ")
        new_elem.append(float(splits[1].split(" at ")[0]))
        new_elem.append(int(splits[-1]))
        temp_headings.append("Train best SR")
        temp_headings.append("Train best episode")
    
      elif line.startswith("\tFinal full test"):
        splits = line.split(" = ")
        new_elem.append(float(splits[-1]))
        temp_headings.append("Final test SR")

      else:
        splits = line.split(" is ")
        item = splits[-1]
        if item.endswith("\n"): item = item.strip("\n")
        new_elem.append(item)
        item = splits[0]
        if item.startswith("\t"): item = item.strip("\t")
        temp_headings.append(splits[0][1:])

  table.append(new_elem)

  # now prepare to print the table
  heading_str = ""
  for x in range(len(headings) - 1): heading_str += "{" + str(x) + "} | "
  heading_str += "{" + str(len(headings) - 1) + "}"
  row_str = heading_str[:]
  # formatters = ["{" + f"{x}:<{len(headings[x]) + 2}" + "}" for x in range(len(headings))]
  heading_formatters = []
  row_formatters = []
  for x in range(len(headings)):
    if False and isinstance(new_elem[x], float):
      row_formatters.append("{" + f"{x}:<{len(headings[x]) + 2}.4f" + "}")
    else:
      row_formatters.append("{" + f"{x}:<{len(headings[x]) + 2}" + "}")
    heading_formatters.append("{" + f"{x}:<{len(headings[x]) + 2}" + "}")
  heading_str = heading_str.format(*heading_formatters)
  row_str = row_str.format(*row_formatters)

  # assemble the table text
  print_str += heading_str.format(*headings) + "\n"
  # print(heading_str.format(*headings))
  for i in range(len(table)):
    # check if entry is incomplete
    while len(table[i]) < len(headings): table[i] += ["N/F"]
    for j, elem in enumerate(table[i]):
      if isinstance(elem, float):
        table[i][j] = "{:.4f}".format(elem)
    # print(row_str.format(*table[i]))
    print_str += row_str.format(*table[i]) + "\n"

  # print and save the table
  print()
  print(print_str)
  savepath = model.savedir + model.group_name + "/" + savefile
  with open(savepath, 'w') as f:
    f.write(print_str)

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

  # starting time
  starting_time = datetime.now()

  # key default settings
  use_wandb = True
  no_plot = True
  datestr = "%d-%m-%y-%H:%M" # all date inputs must follow this format

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
  parser.add_argument("-H", "--heuristic",    action="store_true") # run a test using heuristic actions
  parser.add_argument("--program",            default=None)        # program name to select from if..else if
  parser.add_argument("--device",             default=None)        # override device
  parser.add_argument("--savedir",            default=None)        # override save/load directory
  parser.add_argument("--print",              action="store_true") # don't train, print help
  parser.add_argument("--log-level",          type=int, default=1) # set script log level
  parser.add_argument("--best-performance",   action="store_true") # print best run performance
  parser.add_argument("--override-lib",       action="store_true") # override bind.so library with loaded data
  parser.add_argument("--no-delay",           action="store_true") # prevent a sleep(...) to seperate processes
  parser.add_argument("--test",               action="store_true") # run a thorough test on existing model
  parser.add_argument("--results",            action="store_true") # print a table of results.txt
  parser.add_argument("--print-results",      action="store_true") # prepare and print all results
  parser.add_argument("--delete-results",     action="store_true") # delete any results.txt data

  args = parser.parse_args()

  # # parse arguments but allow unknown arguments
  # args, unknown = parser.parse_known_args()

  # extract primary inputs
  inputarg = args.job
  timestamp = args.timestamp if args.timestamp else datetime.now().strftime(datestr)
  if args.no_wandb: use_wandb = False

  log_level = args.log_level

  if args.best_performance: 
    args.print = True

  if args.print: 
    args.log_wandb = False
    log_level = 0

  # echo these inputs
  if log_level > 0:
    print("\narray_training_DQN is preparing to train:")
    print(" -> Input arg:", inputarg)
    print(" -> Timestamp is:", timestamp)
    print(" -> Use wandb is:", use_wandb)
    print(" -> Program name override is:", args.program)

  # seperate process for safety
  if not args.no_delay:
    sleep(inputarg)
    sleep(0.25 * random())

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

  # if we are printing a results table
  if args.results:
    if log_level > 0: print("Printing a results table")
    print_results(model)
    exit()

  if args.delete_results:
    try:
      if log_level > 0: print("Deleting a results table")
      filepath = model.savedir + model.group_name + "/results.txt"
      with open(filepath, 'w') as f:
        f.write("")
    except FileNotFoundError as e:
      print("No results table found")
    exit()



  # ----- BEGIN TRAININGS ----- #




  # CONFIGURE KEY SETTINGS (take care that baseline_settings(...) does not overwrite)
  model.params.use_curriculum = False
  model.params.num_episodes = 50_000
  model.params.object_set = "set6_fullset_800_50i"

  if args.program is None:
    # cannot have whitespace
    raise RuntimeError("must specify the program to run with '--program xxxxxxx'")
    training_type = "vary_xmas"
  else:
    training_type = args.program

  # OLD, avoid: defaults to pass into baseline_settings(...)
  this_segments = 8 # was 8, change to 6 for speed
  this_noise = 0.025 # was 0.05, change to 0.025 for stability

  extra_info_string = ""

  if training_type == "vary_sensors_and_thickness":

    sensors_list = [
      0, # no sensors, state only
      1, # bending and z state
      2, # + palm
      3  # + wrist
    ]
    thickness_list = [
      # 0.8e-3,
      0.9e-3,
      1.0e-3
    ]
    repeats = 4
    param_1_name = "Sensors"
    param_2_name = "Thickness"
    param_3_name = None

    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=sensors_list,
                                                param_2=thickness_list, repeats=repeats)

    this_sensor = param_1
    this_thickness = param_2

    baseline_args = {
      "sensors" : this_sensor,
      "finger_thickness" : this_thickness,
      "sensor_noise" : this_noise,
      "num_segments" : this_segments
    }

  elif training_type == "vary_sensors_only":

    vary_1 = [
      0, # no sensors, state only
      1, # bending and z state
      2, # + palm
      3  # + wrist
    ]
    vary_2 = None
    vary_3 = None
    repeats = 5
    param_1_name = "Sensors"
    param_2_name = None
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)
    this_sensor = param_1
    this_thickness = 0.9e-3

    baseline_args = {
      "sensors" : this_sensor,
      "finger_thickness" : this_thickness,
      "sensor_noise" : this_noise,
      "num_segments" : this_segments
    }

  elif training_type == "vary_sensors_and_noise":

    vary_1 = [
      0, # no sensors, state only
      1, # bending and z state
      2, # + palm
      3  # + wrist
    ]
    vary_2 = [0, 0.025, 0.05]
    vary_3 = None
    repeats = 3
    param_1_name = "Sensors"
    param_2_name = "Noise std"
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)
    this_sensor = param_1
    this_thickness = 0.9e-3
    this_noise = param_2

    baseline_args = {
      "sensors" : this_sensor,
      "finger_thickness" : this_thickness,
      "sensor_noise" : this_noise,
      "num_segments" : this_segments
    }

  elif training_type == "vary_lr_and_eps":

    vary_1 = [5e-6, 1e-5, 5e-5, 1e-4, 5e-4]
    vary_2 = [1000, 2000, 4000, 6000]
    vary_3 = None
    repeats = None
    param_1_name = "learning rate"
    param_2_name = "eps decay"
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)
    this_lr = param_1
    this_eps_decay = param_2
    this_three = param_3

    baseline_args = {
      "lr" : this_lr,
      "eps_decay" : this_eps_decay,
      "sensor_noise" : this_noise,
      "num_segments" : this_segments
    }

  elif training_type == "vary_reward_and_network":

    vary_1 = [
      (1.5, 1.0),
      (2.5, 1.0),
      (2.5, 2.5),
      (3.5, 1.0),
      (3.5, 2.5)
    ]
    vary_2 = [networks.DQN_3L100, networks.DQN_5L100, networks.DQN_7L100]
    vary_3 = None
    repeats = 2
    param_1_name = "reward/penalty scaling"
    param_2_name = "network size"
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    baseline_args = {
      "scale_rewards" : param_1[0],
      "scale_penalties" : param_1[1],
      "network" : param_2,
      "sensor_noise" : this_noise,
      "num_segments" : this_segments
    }

  elif training_type == "vary_episode_length":

    vary_1 = [200, 250, 300, 350]
    vary_2 = None
    vary_3 = None
    repeats = 3
    param_1_name = "max episode steps"
    param_2_name = None
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)
    baseline_args = {
      "max_episode_steps" : param_1,
      "sensor_noise" : this_noise,
      "num_segments" : this_segments
    }

  elif training_type == "network_architecture":

    vary_1 = [
      [100, 100, 100],
      [82,  82,  82,  82,  82],
      [74,  74,  74,  74,  74,  74,  74],
      [150, 100, 50],
      [150, 100, 50,  50,  50],
      [150, 100, 50,  50,  50,  50,  50],
      [128, 96,  64],
      [128, 96,  64,  64,  64],
      [128, 96,  64,  64,  64,  64,  64]
    ]
    vary_2 = None
    vary_3 = None
    repeats = 4
    param_1_name = "hidden layer weights"
    param_2_name = None
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)
    baseline_args = {
      "network" : param_1,
      "sensor_noise" : this_noise,
      "num_segments" : this_segments
    }

  elif training_type == "vary_action_size":

    vary_1 = [0.7, 1.0]
    vary_2 = [0.01, 0.02]
    vary_3 = [1.0, 2.0]
    repeats = 4
    param_1_name = "X_action_mm"
    param_2_name = "Y_action_rad"
    param_3_name = "Z_action_mm"
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)
    baseline_args = {
      "XYZ_mm_rad" : True,
      "max_episode_steps" : 250, # default: scaled below
      "sensor_noise" : this_noise,
      "num_segments" : 8
    }
    model.env.mj.set.X_action_mm = param_1
    model.env.mj.set.Y_action_rad = param_2
    model.env.mj.set.Z_action_mm = param_3

    # what is the scale change in our action workspace
    scale = (0.67 * 0.02 * 1.22) / (param_1 * param_2 * param_3)

    # adjust replay memory to have size based on actions
    new_mem = int(((scale * model.params.memory_replay // 10000) + 1) * 10000)
    if new_mem < model.params.memory_replay: new_mem = model.params.memory_replay # only increase size
    model.params.memory_replay = new_mem

    # adjust steps per episode based on actions size
    new_steps = int(((scale * baseline_args['max_episode_steps'] // 50) + 1) * 50)
    if new_steps < baseline_args['max_episode_steps']: new_steps = baseline_args['max_episode_steps']
    baseline_args['max_episode_steps'] = new_steps

    extra_info_string = f"\tNew replay memory size is: {model.params.memory_replay}"
    extra_info_string += f"\n\tNew max episode steps is: {baseline_args['max_episode_steps']}"

  elif training_type == "vary_xmas":
 
    vary_1 = [1, 1.5, 2.0]    # exceed_lims multiplier
    vary_2 = [0, 1, 2, 3]     # sensors
    vary_3 = [0.9e-3, 1.0e-3] # thickness
    repeats = 5               # 120 trainings total
    param_1_name = "exceed limits multiplier"
    param_2_name = "sensors"
    param_3_name = "thickness"
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)
    baseline_args = {
      "XYZ_mm_rad" : [1.0, 0.01, 2.0],
      "max_episode_steps" : 250, # default: scaled below
      "sensor_noise" : this_noise,
      "num_segments" : 8,
      "exceed_lims_multiplier" : param_1,
      "sensors" : param_2,
      "finger_thickness" : param_3
    }

  elif training_type == "heuristic":

    # special case: block terminal logging
    model.log_level = 0

    vary_1 = [0, 1, 2, 3]
    vary_2 = [0.9e-3]
    vary_3 = [8]
    repeats = 1
    param_1_name = "Num sensors"
    param_2_name = "Finger thickness"
    param_3_name = "Num segments"
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)
    baseline_args = {
      "sensors" : param_1,
      "finger_thickness" : param_2,
      "num_segments" : param_3
    }

  elif training_type == "vary_sensor_steps":

    vary_1 = [1, 2, 3]
    vary_2 = [1, 2, 3]
    vary_3 = None
    repeats = 5
    param_1_name = "sensor steps"
    param_2_name = "state steps"
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)
    baseline_args = {
      "sensor_steps" : param_1,
      "state_steps" : param_2
    }

  elif training_type == "ski_training":

    vary_1 = [0.9e-3, 1.0e-3] # thickness
    vary_2 = [(1, 1), (2, 2), (3, 3)]
    vary_3 = None
    repeats = 10
    param_1_name = "finger thickness"
    param_2_name = "sensor/state steps"
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)
    baseline_args = {
      "finger_thickness" : param_1,
      "sensor_steps" : param_2[0],
      "state_steps" : param_2[1]
    }

    # run long trainings
    model.params.num_episodes = 100_000

  elif training_type == "paper_baseline_1":

    vary_1 = [0.9e-3, 1.0e-3] # thickness
    vary_2 = [0, 1, 2, 3]
    vary_3 = None
    repeats = 15
    param_1_name = "finger thickness"
    param_2_name = "num sensors"
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)
    baseline_args = {
      "finger_thickness" : param_1,
      "sensors" : param_2,
      "sensor_steps" : 3,
      "state_steps" : 3,
    }

    # run long trainings
    model.params.num_episodes = 100_000

    # run longer tests
    model.env.params.test_trials_per_object = 5

  elif training_type == "paper_baseline_1.5":

    # vary_1 = [0.9e-3, 1.0e-3] # thickness
    vary_1 = [24e-3] # width
    vary_2 = [0, 1, 2, 3]
    vary_3 = None
    repeats = 10
    param_1_name = "finger width"
    param_2_name = "num sensors"
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)
    baseline_args = {
      "finger_thickness" : 1.0e-3,
      "finger_width" : param_1,
      "sensors" : param_2,
      "sensor_steps" : 3,
      "state_steps" : 3,
    }

    # run long trainings
    model.params.num_episodes = 100_000

    # run longer tests
    model.env.params.test_trials_per_object = 5

  elif training_type == "paper_baseline_2":

    vary_1 = [
      (0.9e-3, 28e-3),
      (1.0e-3, 24e-3),
      (1.0e-3, 28e-3),
    ]
    vary_2 = [0, 1, 2, 3]
    vary_3 = None
    repeats = 10
    param_1_name = "thickness, width"
    param_2_name = "num sensors"
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)
    baseline_args = {
      "finger_thickness" : param_1[0],
      "finger_width" : param_1[1],
      "sensors" : param_2,
      "sensor_steps" : 3,
      "state_steps" : 3,
    }

    # run long trainings
    model.params.num_episodes = 100_000

    # run slightly longer tests
    model.env.params.test_trials_per_object = 3

    # test less often
    model.params.test_freq = 4000
    model.params.save_freq = 4000

  elif training_type == "paper_baseline_2_extra_noise":

    vary_1 = [
      (0.9e-3, 28e-3),
      (1.0e-3, 24e-3),
      (1.0e-3, 28e-3),
    ]
    vary_2 = [0, 1, 2, 3]
    vary_3 = None
    repeats = 10
    param_1_name = "thickness, width"
    param_2_name = "num sensors"
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)
    baseline_args = {
      "finger_thickness" : param_1[0],
      "finger_width" : param_1[1],
      "sensors" : param_2,
      "sensor_noise" : 0.05,
      "sensor_steps" : 3,
      "state_steps" : 3,
    }

    # run long trainings
    model.params.num_episodes = 100_000

    # run slightly longer tests
    model.env.params.test_trials_per_object = 3

    # test less often
    model.params.test_freq = 4000
    model.params.save_freq = 4000

  elif training_type == "avoid_overfit":

    vary_1 = [1, 3, 5, 7]
    vary_2 = None
    vary_3 = None
    repeats = 5
    param_1_name = "state steps"
    param_2_name = None
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)
    baseline_args = {
      "finger_thickness" : 0.9e-3,
      "finger_width" : 28e-3,
      "sensors" : 3,
      "sensor_noise" : 0.025,
      "sensor_steps" : 1,
      "state_steps" : param_1,
    }

    # don't run long trainings
    model.params.num_episodes = 48_000

    # run slightly longer tests
    model.env.params.test_trials_per_object = 3

    # test less often
    model.params.test_freq = 4000
    model.params.save_freq = 4000

  elif training_type == "test_exceed_limits_termination":

    param_1 = None
    param_2 = None
    param_3 = None
    param_1_name = None
    param_2_name = None
    param_3_name = None
    
    baseline_args = {
      "finger_thickness" : 0.9e-3,
      "sensors" : 3,
      "sensor_steps" : 3,
      "state_steps" : 3,
      "reward_options" : ["terminate_on_exceed_limits"]
    }

    # run long trainings
    model.params.num_episodes = 100_000

    # run longer tests
    model.env.params.test_trials_per_object = 5

  elif training_type == "test_prevent_table_impacts":

    param_1 = None
    param_2 = None
    param_3 = None
    param_1_name = None
    param_2_name = None
    param_3_name = None
    
    baseline_args = {
      "finger_thickness" : 0.9e-3,
      "sensors" : 3,
      "sensor_steps" : 3,
      "state_steps" : 3
    }

    # run long trainings
    model.params.num_episodes = 100_000

    # run longer tests
    model.env.params.test_trials_per_object = 5

    # # run fewer tests
    # model.params.test_freq = 4000

    # # reduce fingertip minimum from -12.5 to -10.0
    # model.env.mj.set.fingertip_min_mm = -10.0 # MOVEMENT BELOW THIS SETS within_limits=false;

    # prevent gripper from going lower than -12.5mm (see myfunctions.cpp for variable hardcoding)
    model.env.mj.prevent_table_impacts(True)

  elif training_type == "vary_others":

    vary_1 = None
    vary_2 = None
    vary_3 = None
    repeats = None
    param_1_name = None
    param_2_name = None
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)
    baseline_args = {
      "param_1_arg" : param_1,
      "param_2_arg" : param_2,
      "param_3_arg" : param_3
    }

  else: raise RuntimeError(f"array_training_DQN.py: training_type of '{training_type}' not recognised")

  # note and printing information
  param_1_string = f"\t{param_1_name} is {param_1}\n" if param_1 is not None else ""
  param_2_string = f"\t{param_2_name} is {param_2}\n" if param_2 is not None else ""
  param_3_string = f"\t{param_3_name} is {param_3}\n" if param_3 is not None else ""
  model.wandb_note += param_1_string + param_2_string + param_3_string

  if not args.print and not args.print_results:

    # apply settings
    model = baseline_settings(model, **baseline_args)

    # train
    if not args.heuristic: model.train()

    # test
    model = test(model, trials_per_obj=10, heuristic=args.heuristic)

    # finishing time, how long did everything take
    finishing_time = datetime.now()
    time_taken = finishing_time - starting_time
    d = divmod(time_taken.total_seconds(), 86400)
    h = divmod(d[1], 3600)
    m = divmod(h[1], 60)
    s = m[1]
    print("\nStarted at:", starting_time.strftime(datestr))
    print("Finished at:", datetime.now().strftime(datestr))
    print(f"Time taken was {d[0]:.0f} days {h[0]:.0f} hrs {m[0]:.0f} mins {s:.0f} secs\n")

  # prepare to print final results, check if files exist which recorded best performance
  best_sr, best_ep = model.read_best_performance_from_text(silence=True)
  if best_sr is not None:
    extra_info_string += f"\tTraining time best success rate = {best_sr} at episode = {best_ep}\n"
  full_sr, _ = model.read_best_performance_from_text(silence=True, fulltest=True)
  if full_sr is not None:
    extra_info_string += f"\tFinal full test success rate = {full_sr}\n"

  if args.print_results:

    filepath = model.savedir + model.group_name + "/results.txt"
    new_file_text = f"Input arg {args.job}\n"
    new_file_text += param_1_string
    new_file_text += param_2_string
    new_file_text += param_3_string
    new_file_text += extra_info_string
    with open(filepath, 'a') as f:
      f.write(new_file_text)

  else:
    print("Input arg", args.job)
    print(param_1_string, end="")
    print(param_2_string, end="")
    print(param_3_string, end="")
    print(extra_info_string)

  # ----- END ----- #
  