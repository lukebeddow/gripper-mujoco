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
from dataclasses import dataclass
import functools

def set_penalties(model, value, done=False, trigger=1, make_binary=None):
  """
  Set penalty rewards with given value, alongside defaults
  """

  # penalties                            reward   done   trigger  min   max  overshoot
  model.env.mj.set.exceed_limits.set     (value,  done,  trigger)
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

def set_sensor_reward_thresholds(model, exceed_style=None, min_style=None):
  """
  Determine the reward thresholds
  """

  printout = True

  @dataclass
  class RewardThresholds:
    # m=minimum, g=good, x=exceed, d=dangerous
    mBend = 0.0
    gBend = model.env.mj.set.stable_finger_force
    xBend = model.env.mj.set.stable_finger_force_lim
    dBend = model.env.yield_load()

    mPalm = 0.0
    gPalm = model.env.mj.set.stable_palm_force
    xPalm = model.env.mj.set.stable_palm_force_lim
    dPalm = 30.0

    xWrist = 5.0
    dWrist = 10.0

  global RT
  RT = RewardThresholds()

  # check if minimum handling is specified
  if isinstance(min_style, float):
    RT.mBend = min_style
    RT.mPalm = min_style
  elif isinstance(min_style, list) and len(min_style) == 2:
    RT.mBend = min_style[0]
    RT.mPalm = min_style[1]
  elif min_style == "binary":
    RT.mBend = RT.gBend
    RT.mPalm = RT.gPalm
  elif min_style == "middle":
    RT.mBend = 0.5 * RT.gBend
    RT.mPalm = 0.5 * RT.gPalm
  elif min_style is not None: 
    raise RuntimeError(f"set_sensor_reward_thresholds() got invalid 'min_style' of {min_style}")

  # check if we are given how 'exceed bend' and 'exceed palm' should work
  if isinstance(exceed_style, float):
    RT.xBend = exceed_style
    RT.xPalm = exceed_style
  elif isinstance(exceed_style, list) and len(exceed_style) == 2:
    RT.xBend = exceed_style[0]
    RT.xPalm = exceed_style[1]
  elif exceed_style == "dangerous":
    RT.xBend = RT.dBend
    RT.xPalm = RT.dPalm
  elif exceed_style == "middle":
    RT.xBend = RT.gBend + 0.5 * (RT.dBend - RT.gBend)
    RT.xPalm = RT.gPalm + 0.5 * (RT.dPalm - RT.gPalm)
  elif exceed_style == "factor_0.8":
    RT.xBend = RT.gBend + 0.8 * (RT.dBend - RT.gBend)
    RT.xPalm = RT.gPalm + 0.8 * (RT.dPalm - RT.gPalm)
  elif exceed_style is not None: 
    raise RuntimeError(f"set_sensor_reward_thresholds() got invalid 'exceed_style' of {exceed_style}")

  # confirm that the thresholds make sense
  if RT.mBend > RT.gBend:
    raise RuntimeError(f"set_sensor_reward_thresholds() finds mBend > gBend, {RT.mBend:.3f} > {RT.gBend:.3f}")
  if RT.mPalm > RT.gPalm:
    raise RuntimeError(f"set_sensor_reward_thresholds() finds mPalm > gPalm, {RT.mPalm:.3f} > {RT.gPalm:.3f}")
  if RT.gBend > RT.xBend:
    raise RuntimeError(f"set_sensor_reward_thresholds() finds gBend > xBend, {RT.gBend:.3f} > {RT.xBend:.3f}")
  if RT.gPalm > RT.xPalm:
    raise RuntimeError(f"set_sensor_reward_thresholds() finds gPalm > xPalm, {RT.gPalm:.3f} > {RT.xPalm:.3f}")
  if RT.xBend > RT.dBend:
    raise RuntimeError(f"set_sensor_reward_thresholds() finds xBend > dBend, {RT.xBend:.3f} > {RT.dBend:.3f}")
  if RT.xPalm > RT.dPalm:
    raise RuntimeError(f"set_sensor_reward_thresholds() finds xPalm > dPalm, {RT.xPalm:.3f} > {RT.dPalm:.3f}")
  if RT.xWrist > RT.dWrist:
    raise RuntimeError(f"set_sensor_reward_thresholds() finds xWrist > dWrist, {RT.xWrist:.3f} > {RT.dWrist:.3f}")

  if printout:
    print("Reward Thresholds\n")
    print(f"  -> mBend = {RT.mBend:.3f}")
    print(f"  -> gBend = {RT.gBend:.3f}")
    print(f"  -> xBend = {RT.xBend:.3f}")
    print(f"  -> dBend = {RT.dBend:.3f}\n")
    print(f"  -> mPalm = {RT.mPalm:.3f}")
    print(f"  -> gPalm = {RT.gPalm:.3f}")
    print(f"  -> xPalm = {RT.xPalm:.3f}")
    print(f"  -> dPalm = {RT.dPalm:.3f}\n")
    print(f"  -> xWrist = {RT.xWrist:.3f}")
    print(f"  -> dWrist = {RT.dWrist:.3f}\n")

def set_sensor_bonuses(model, value):
  """
  Set bonus rewards with a given value
  """

  # rewards                             reward   done   trigger  min       max     overshoot
  model.env.mj.set.lifted.set           (value,  False,   1)
  model.env.mj.set.target_height.set    (value,  False,   1)
  model.env.mj.set.object_stable.set    (value,  False,   1)
  model.env.mj.set.good_bend_sensor.set (value,  False,   1,     RT.mBend, RT.gBend,  -1)
  model.env.mj.set.good_palm_sensor.set (value,  False,   1,     RT.mPalm, RT.gPalm,  -1)

  return model

def set_sensor_penalties(model, value):
  """
  Set penalty rewards with given value, alongside defaults
  """

  # penalties                              reward   done   trigger  min        max     overshoot
  model.env.mj.set.exceed_limits.set       (value,  False,    1)
  model.env.mj.set.exceed_bend_sensor.set  (value,  False,    1,    RT.xBend,  RT.dBend,  -1)
  model.env.mj.set.exceed_palm_sensor.set  (value,  False,    1,    RT.xPalm,  RT.dPalm,  -1)
  model.env.mj.set.exceed_wrist_sensor.set (value,  False,    1,    RT.xWrist, RT.dWrist, -1)

  return model

def set_sensor_terminations(model, value=-1.0, done=True, trigger=1):
  """
  Set terminations based on dangerous sensor readings
  """

  # terminations                              reward   done   trigger  min        max     overshoot
  model.env.mj.set.dangerous_bend_sensor.set  (value,  done,  trigger, RT.dBend,  RT.dBend,  -1)
  model.env.mj.set.dangerous_palm_sensor.set  (value,  done,  trigger, RT.dPalm,  RT.dPalm,  -1)
  model.env.mj.set.dangerous_wrist_sensor.set (value,  done,  trigger, RT.dWrist, RT.dWrist, -1)

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
                          done=5 if "terminate_early" in options else False)
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
                          make_binary=True if "make_binary" in options else None)
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
                          make_binary=True if "make_binary" in options else None)
    # scale based on steps allowed per episode
    model.env.mj.set.scale_rewards(100 / model.env.params.max_episode_steps)
    # end criteria                         reward   done   trigger
    model.env.mj.set.stable_height.set     (1.0,    True,    1)
    model.env.mj.set.oob.set               (-1.0,   True,    1)

  elif style == "sensor_mixed":
    # prepare reward thresholds
    if (model.env.mj.set.stable_finger_force_lim > 99.0 and
        model.env.mj.set.stable_palm_force_lim > 99.0):
      exceed_style = [3.0, 10.0]
    else: exceed_style = None
    set_sensor_reward_thresholds(model, exceed_style=exceed_style,
                                 min_style=None)
    # reward each step                     reward   done   trigger
    model.env.mj.set.step_num.set          (-0.01,  False,   1)
    # penalties and bonuses
    model = set_sensor_bonuses(model, 0.002 * scale_rewards)
    model = set_sensor_penalties(model, -0.002 * scale_penalties)
    # scale based on steps allowed per episode
    model.env.mj.set.scale_rewards(100 / model.env.params.max_episode_steps)
    # end criteria                         reward   done   trigger
    model.env.mj.set.stable_height.set     (1.0,    True,    1)
    model.env.mj.set.oob.set               (-1.0,   True,    1)
    if penalty_termination:
      model = set_sensor_terminations(model)

  elif style == "sparse":
    # reward each step                     reward   done   trigger
    model.env.mj.set.step_num.set          (0.0,    False,   1)
    # penalties and bonuses
    model = set_bonuses(model, 0.0)
    model = set_penalties(model, 0.0,  
                          done=5 if "terminate_early" in options else False)
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
                          done=5 if "terminate_early" in options else False)
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
  model.env.mj.set.use_quit_on_reward = True

  model.wandb_note += f"Reward style: '{style}', options: [ "
  for extra in options: model.wandb_note += f"'{extra}' "
  model.wandb_note += "]\n"

  return model

def add_sensors(model, num=None, sensor_mode=1, state_mode=0, sensor_steps=1,
  state_steps=2, sensor_noise_std=0.025, state_noise_std=0.025, sensor_noise_mu=0.0,
  state_noise_mu=0.0, z_state=None, palm_norm=10.0, wrist_norm=10.0):
  """
  Add a number of sensors
  """

  if num is None: num = 10 # default, include all sensors

  # define the normalised range of palm and wrist (bend gauges are automatic)
  model.env.mj.set.palm_sensor.normalise = palm_norm
  model.env.mj.set.wrist_sensor_Z.normalise = wrist_norm

  # enable noise and normalisation for every sensor (should be enabled by default anyway)
  model.env.mj.set_sensor_noise_and_normalisation_to(True)

  # what sensing mode (0=raw data, 1=change, 2=average, 3=median)
  model.env.mj.set.sensor_sample_mode = sensor_mode
  model.env.mj.set.state_sample_mode = state_mode

  # set the same noise to regular sensors and state sensors
  model.env.mj.set.sensor_noise_std = sensor_noise_std
  model.env.mj.set.sensor_noise_mu = sensor_noise_mu
  model.env.mj.set.state_noise_std = state_noise_std
  model.env.mj.set.state_noise_mu = state_noise_mu

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
  if num >= 5 or z_state is True: model.env.mj.set.base_state_sensor_Z.in_use = True

  model.wandb_note += (
    f"Num sensors: {num}, state mode: {state_mode}, sensor mode: {sensor_mode}"
    + f", state steps: {state_steps}, sensor steps: {sensor_steps}"
    + f", sensor noise std: {sensor_noise_std}, state noise std: {state_noise_std}"
    + f", sensor_noise mu: {sensor_noise_mu}, state noise mu: {state_noise_mu}\n"
  )

  return model

def set_actions(model, discrete=True, base_XY=False, action_values=None):
  """
  Configure the actions to use in training
  """

  if action_values == None:
    action_values = [
      1e-3, # [0] gripper_prismatic_X
      0.01, # [1] gripper_revolute_Y
      2e-3, # [2] gripper_Z
      2e-3, # [3] base_Z
      2e-3  # [4] base_XY
    ]

  elif len(action_values) == 4 and not base_XY:
    pass
  elif len(action_values) == 5 and base_XY:
    pass
  else: raise RuntimeError(f"set_actions() got {len(action_values)} actions")

  # enable and configure the core actions with default settings
  model.env.mj.set.gripper_prismatic_X.in_use = True
  model.env.mj.set.gripper_revolute_Y.in_use = True
  model.env.mj.set.gripper_Z.in_use = True
  model.env.mj.set.base_Z.in_use = True
  model.env.mj.set.gripper_prismatic_X.value = action_values[0]
  model.env.mj.set.gripper_revolute_Y.value = action_values[1]
  model.env.mj.set.gripper_Z.value = action_values[2]
  model.env.mj.set.base_Z.value = action_values[3]
  model.env.mj.set.gripper_prismatic_X.sign = -1
  model.env.mj.set.gripper_revolute_Y.sign = -1
  model.env.mj.set.gripper_Z.sign = 1
  model.env.mj.set.base_Z.sign = 1

  if base_XY:

    # add in XY base movements
    model.env.mj.set.base_Z.in_use = True
    model.env.mj.set.base_Z.value = action_values[4]
    model.env.mj.set.base_Z.sign = 1

  # are we using continous actions
  if not discrete: model.env.mj.set.set_all_action_continous(True)

  return model

def curriculum_step_size(self, i):
  """
  Curriculum which changes the step size over time
  """

  if self.curriculum_params["finished"]: return

  stage = 0

  # determine the curriculum metric
  if self.curriculum_params["metric"] == "episode_number":

    for t in self.curriculum_params["thresholds"]:
      if i >= t:
        stage += 1
      else: break

    if stage == self.curriculum_params["stage"]: return

  elif self.curriculum_params["metric"] == "success_rate":

    # get the most recent success rate
    if len(self.track.avg_stable_height) > 0:
      success_rate = self.track.avg_stable_height[-1]
    else: success_rate = 0.0

    # determine if we have passed the required threshold
    for t in self.curriculum_params["thresholds"]:
      if success_rate >= t:
        stage += 1

    if stage <= self.curriculum_params["stage"]: return

  # if the metric is not recognised
  else: return

  # now set the step sizes
  self.env.mj.set.gripper_prismatic_X.value = self.curriculum_params["step_sizes"][stage][0]
  self.env.mj.set.gripper_revolute_Y.value = self.curriculum_params["step_sizes"][stage][1]
  self.env.mj.set.gripper_Z.value = self.curriculum_params["step_sizes"][stage][2]
  self.env.mj.set.base_Z.value = self.curriculum_params["step_sizes"][stage][3]

  # now adjust the time per action to match the step sizes
  self.env.mj.set.time_for_action = self.curriculum_params["step_sizes"][stage][4]

  print(f"Episode = {i}, stage = {stage}, curriculum is changing")

  self.curriculum_params["stage"] = stage
  if stage == len(self.curriculum_params["thresholds"]): self.curriculum_params["finished"] = True

  # now save a text file to reflect the changes
  labelstr = f"Hyperparameters after curriculum change which occured at episode {i}\n"
  name = f"hyperparameters_curriculum_stage_{stage}"
  self.save_hyperparameters(labelstr, name, print_out=False)
  
  return

def curriculum_punishments(self, i):
  """
  Curriculum which increases penalties for unsafe behaviour over time
  """

  pass

def apply_to_all_models(model):
  """
  Settings we want to apply to every single running model. This can also be used
  as a reference for which options are possible to change. Many options are first
  wiped here and later set by 'baseline_settings'
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
  model.params.target_update = 50
  model.params.num_episodes = 60_000
  model.params.optimiser = "adam"
  model.params.adam_beta1 = 0.9
  model.params.adam_beta2 = 0.999

  # memory replay and HER
  model.params.memory_replay = 75_000
  model.params.min_memory_replay = 5_000
  model.params.use_HER = False # python setting OVERRIDES cpp
  model.params.HER_mode = "final"
  model.params.HER_k = 4

  # curriculum learning
  model.params.use_curriculum = False

  # data loggings
  model.params.save_freq = 4_000
  model.params.test_freq = 4_000
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
  model.env.mj.set.time_for_action = 0.2
  model.env.mj.set.saturation_yield_factor = 1.0
  model.env.mj.set.exceed_lat_min_factor = 0.75
  model.env.mj.set.exceed_lat_max_factor = 1.5

  # define lengths and forces
  model.env.mj.set.oob_distance = 75e-3
  model.env.mj.set.done_height = 15e-3
  model.env.mj.set.stable_finger_force = 1.0
  model.env.mj.set.stable_palm_force = 1.0
  model.env.mj.set.stable_finger_force_lim = 100.0
  model.env.mj.set.stable_palm_force_lim = 100.0
  model.env.mj.set.fingertip_min_mm = -12.5 # below (from start position) sets within_limits=false;

  # wipe and disable all actions, these can be set in the set_actions(...) function
  model.env.mj.set.set_all_action_use(False)
  model.env.mj.set.set_all_action_continous(False)
  model.env.mj.set.set_all_action_value(0.0)
  model.env.mj.set.set_all_action_sign(1)

  # what sensing mode (0=raw data, 1=change, 2=average, 3=median, 4=change)
  model.env.mj.set.sensor_sample_mode = 1
  model.env.mj.set.state_sample_mode = 4

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
  model.env.mj.set.use_quit_on_reward = False

  # disable use of all sensors
  model.env.mj.set.disable_sensors()
  model.env.mj.set.sensor_n_prev_steps = 1 # lookback only 1 step
  model.env.mj.set.state_n_prev_steps = 1 # lookback only 1 step

  # ensure state sensors only give one reading per step (read_rate < 0)
  model.env.mj.set.motor_state_sensor.read_rate = -1
  model.env.mj.set.base_state_sensor_XY.read_rate = -1
  model.env.mj.set.base_state_sensor_Z.read_rate = -1

  # wipe sensor noise options
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

  new_endpoint = 100_000
  model.wandb_note += f"Continuing training until new endpoint of {new_endpoint} episodes\n"

  # extra_episodes = 48_000
  # model.wandb_note += f"Continuing training with an extra {extra_episodes} episodes\n"
  
  model.continue_training(run_name, model.savedir + group_name + "/",
                          new_endpoint=new_endpoint, extra_episodes=extra_episodes,
                          object_set=object_set, overridelib=args.override_lib)
  
  test(model)

  print("Continuing training has now finished")

  # finishing time, how long did everything take
  global starting_time
  finishing_time = datetime.now()
  time_taken = finishing_time - starting_time
  d = divmod(time_taken.total_seconds(), 86400)
  h = divmod(d[1], 3600)
  m = divmod(h[1], 60)
  s = m[1]
  print("\nStarted at:", starting_time.strftime(datestr))
  print("Finished at:", datetime.now().strftime(datestr))
  print(f"Time taken was {d[0]:.0f} days {h[0]:.0f} hrs {m[0]:.0f} mins {s:.0f} secs\n")

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

def baseline_settings(model, lr=5e-5, eps_decay=4000, sensors=3, network=[150, 100, 50], target_update=50, 
                      memory_replay=75_000, state_steps=3, sensor_steps=3, z_state=True, sensor_mode=2,
                      state_mode=4, sensor_noise=0.025, state_noise=0.0, sensor_mu=0.05,
                      state_mu=0.025, reward_style="sensor_mixed", reward_options=[], 
                      scale_rewards=1.0, scale_penalties=1.0, penalty_termination=False,
                      num_segments=8, finger_thickness=0.9e-3, finger_width=28e-3,
                      max_episode_steps=250, eval_me=None, base_XY_actions=False, action_values=None,
                      object_set="set8_fullset_1500"):
  """
  Applies baseline settings to the model when run without any arguments
  """

  # set key model hyperparameters
  model.params.learning_rate = lr
  model.params.eps_decay = eps_decay
  model.params.memory_replay = memory_replay
  model.params.target_update = target_update

  # set key environment parameters
  model.params.object_set = object_set
  model.env.params.max_episode_steps = max_episode_steps   # after this number of steps, is_done=True
  model.env.load_next.num_segments = num_segments          # 8 gives good speed/accuracy balance
  model.env.load_next.finger_thickness = finger_thickness  # options are 0.8e-3, 0.9e-3, 1.0e-3
  model.env.load_next.finger_width = finger_width          # options are 24e-3, 28e-3

  # wandb notes
  model.wandb_note += f"Learning rate {lr}\n"
  model.wandb_note += f"eps_decay = {eps_decay}\n"
  model.wandb_note += f"num_segments = {num_segments}\n"
  
  # configure rewards and sensors
  model = create_reward_function(model, style=reward_style, options=reward_options,
                                 scale_rewards=scale_rewards, scale_penalties=scale_penalties,
                                 penalty_termination=penalty_termination)
  model = add_sensors(model, num=sensors, sensor_mode=sensor_mode, state_mode=state_mode,
                      state_steps=state_steps, sensor_steps=sensor_steps,
                      z_state=z_state, sensor_noise_std=sensor_noise, sensor_noise_mu=sensor_mu,
                      state_noise_std=state_noise, state_noise_mu=state_mu)
  model = setup_HER(model, use=False)

  # configure actions
  model = set_actions(model, base_XY=base_XY_actions, action_values=action_values)

  # can perform special operations here
  if eval_me is not None: 
    if isinstance(eval_me, list):
      for eval_str in eval_me:
        exec(eval_str)
    else: exec(eval_me)

  # finish initialisation of model
  if network is not None:
    model.init(network)

  return model

def heuristic_test(model, inputarg=None, render=False):
  """
  THIS FUNCTION IS NEVER CALLED ANYWHERE AND NEVER USED

  Do a heuristic test with baseline settings. Most of these settings are irrelevant
  (RL hyperparameters) but matter, like sensor setup. Best to be safe
  """

  # # temporary override!
  model.env.params.test_objects = 3
  model.env.params.test_trials_per_object = 3

  vary_1 = [
    (0.9e-3, 28e-3),
    (1.0e-3, 24e-3),
    (1.0e-3, 28e-3),
  ]
  vary_2 = [0, 1, 2, 3]
  vary_3 = None
  repeats = 5
  param_1_name = "thickness, width"
  param_2_name = "num sensors"
  param_3_name = None
  param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                              param_3=vary_3, repeats=repeats)

  wrist_mu = 0.01             # large chance of zero error with the wrist
  wrist_std = 0.075           # wrist has a lot of noise, this is 15% coverage +-2stdevs

  baseline_args = {
    "finger_thickness" : param_1[0],
    "finger_width" : param_1[1],
    "sensors" : param_2,
    "sensor_noise" : 0.025,   # medium noise on sensor readings
    "state_noise" : 0.0,      # no noise on state readings, this is required for sign mode
    "sensor_mu" : 0.05,       # can be +- 5% from 0
    "state_mu" : 0.025,       # just a gentle zero error noise on state readings
    "sensor_steps" : 1,       # limit this since sensor data is unreliable
    "state_steps" : 5,        # this data stream is clean, so take a lot of it
    "sensor_mode" : 2,        # average sample, leave as before
    "state_mode" : 4,         # state sign mode, -1,0,+1 for motor state change
    "eval_me" : f"model.env.mj.set.wrist_sensor_Z.set_gaussian_noise({wrist_mu}, {wrist_std})",
    "scale_rewards" : 2.5,    # stronger reward signal aids training
    "scale_penalties" : 2.5,  # we do want to discourage dangerous actions
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
  if True or render: model.env.disable_rendering = False
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

def test_and_load(model, demo=False, render=False, pause=False, id=None, best_id=None):
  """
  Test overload where we load a specific model
  """

  # set up the object set
  model.env.mj.model_folder_path = "/home/luke/mymujoco/mjcf"

  folderpath = model.savedir + model.group_name + "/"
  foldername = model.run_name

  # from ModelSaver import ModelSaver
  # model.run_name = foldername
  # model.modelsaver = ModelSaver(folderpath)

  # load the most recent model in the given folder
  model.load(foldername=foldername, folderpath=folderpath, id=id, best_id=best_id)
  # model.modelsaver.enter_folder(foldername, folderpath=folderpath)

  return test(model, demo=demo, render=render, pause=pause, load=False)

def test(model, heuristic=False, trials_per_obj=10, render=False, pause=False, demo=False, id=None,
         load=True):
  """
  Perform a thorough test on the model, including loading the best performing network
  """

  print("\nPreparing to perform a model test, heuristic =", heuristic)

  # load the best performing network
  if load and not heuristic: 
    if id is None: model.load(best_id=True)
    else: model.load(id=id)

  # adjust settingss
  if demo:
    model.env.params.test_trials_per_object = 1
    model.env.params.test_objects = 30
  else:
    model.env.params.test_trials_per_object = trials_per_obj
  if render: model.env.disable_rendering = False

  # perform the test
  test_data = model.test(heuristic=heuristic, pause_each_episode=pause)
  test_report = model.create_test_report(test_data)

  # save data to a text file
  savetxt = f"array_training_DQN.test(...) final success rate = {model.last_test_success_rate}\n"
  savetxt += "\n" + test_report
  if heuristic: savename = "heuristic_test_"
  elif demo: savename = "demo_test_"
  else: savename = "full_test_"
  currenttime = datetime.now().strftime(datestr)
  model.modelsaver.save(savename + currenttime, txtonly=True, txtstr=savetxt)

  return model

def print_results(model, filename="results.txt", savefile="table.txt"):
  """
  Create a results table, presumes a file called 'results.txt' which is the terminal
  output from running './pc_job -j "X:Y" -t DD-MM-YY-HR:MN --program xxxxx --print
  """

  # filepath = model.savedir + model.group_name + "/" + filename

  use_timestamp = True

  fileroot = model.savedir + model.group_name
  if args.heuristic: fileroot += "/heuristic"
  filepath = fileroot + "/" + filename

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

      # if we are using timestamps, extract this first
      if use_timestamp:
        two_parts = line.split(";")
        line = two_parts[0]
        temp_headings.append("Timestamp    ") # 4 spaces to ensure the string is long enough
        new_elem.append("'" + two_parts[1].split("=")[-1].strip("\n")) # ' for excel import

      splits = line.split(" ")
      # int(...) drops the newline character
      new_elem.append(int(splits[-1]))

      temp_headings.append("Input arg")
      done_first_elem = True
        
    elif line.startswith("\t"):
      if line.startswith("\tTraining time best"):
        splits = line.split(" = ")
        new_elem.append(float(splits[1].split(" at ")[0]))
        new_elem.append(int(splits[2].split(";")[0]))
        temp_headings.append("Train best SR")
        temp_headings.append("Train best episode")
        if len(splits) == 4:
          temp_headings.append("Trained to")
          new_elem.append(int(splits[3]))
    
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
  savepath = fileroot + "/" + savefile
  with open(savepath, 'w') as f:
    f.write(print_str)

def print_time_taken():
  """
  Print the time taken since the training started
  """

  finishing_time = datetime.now()
  time_taken = finishing_time - starting_time
  d = divmod(time_taken.total_seconds(), 86400)
  h = divmod(d[1], 3600)
  m = divmod(h[1], 60)
  s = m[1]
  print("\nStarted at:", starting_time.strftime(datestr))
  print("Finished at:", datetime.now().strftime(datestr))
  print(f"Time taken was {d[0]:.0f} days {h[0]:.0f} hrs {m[0]:.0f} mins {s:.0f} secs\n")

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
  use_wandb = False
  no_plot = True
  datestr = "%d-%m-%y_%H-%M" # all date inputs must follow this format

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
  parser.add_argument("-w", "--wandb",        action="store_true") # use wandb logging
  parser.add_argument("-n", "--no-wandb",     action="store_true") # prevent wandb logging
  parser.add_argument("-H", "--heuristic",    action="store_true") # run a test using heuristic actions
  parser.add_argument("-r", "--render",       action="store_true") # render window during training
  parser.add_argument("--program",            default=None)        # program name to select from if..else if
  parser.add_argument("--device",             default=None)        # override device
  parser.add_argument("--savedir",            default=None)        # override save/load directory
  parser.add_argument("--print",              action="store_true") # don't train, print help
  parser.add_argument("--log-level",          type=int, default=1) # set script log level
  parser.add_argument("--best-performance",   action="store_true") # print best run performance
  parser.add_argument("--override-lib",       action="store_true") # override bind.so library with loaded data
  parser.add_argument("--no-delay",           action="store_true") # prevent a sleep(...) to seperate processes
  parser.add_argument("--test",               action="store_true") # run a thorough test on existing model
  parser.add_argument("--demo",               action="store_true") # run a demo test on model, can specify id number
  parser.add_argument("--results",            action="store_true") # print a table of results.txt
  parser.add_argument("--print-results",      action="store_true") # prepare and print all results
  parser.add_argument("--delete-results",     action="store_true") # delete any results.txt data

  args = parser.parse_args()

  # # parse arguments but allow unknown arguments
  # args, unknown = parser.parse_known_args()

  # extract primary inputs
  inputarg = args.job
  timestamp = args.timestamp if args.timestamp else datetime.now().strftime(datestr)
  if args.wandb: use_wandb = True
  if args.no_wandb: use_wandb = False

  if args.heuristic and args.program is None:
    args.program = "heuristic"

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

  # are we rendering
  if args.render is True: model.env.disable_rendering = False

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
      fileroot = model.savedir + model.group_name
      if args.heuristic: fileroot += "/heuristic"
      filepath = fileroot + "/results.txt"
      with open(filepath, 'w') as f:
        f.write("")
    except FileNotFoundError as e:
      print("No results table found")
    exit()

  if args.test or args.demo and not args.heuristic:
    if log_level > 0: print("Running a test")
    # first load the model
    if args.test:
      test_and_load(model, best_id=True)
    elif args.demo:
      test_and_load(model, demo=True, render=True, pause=False, best_id=True)
    exit()

  # ----- BEGIN TRAININGS ----- #


  # CONFIGURE KEY SETTINGS (take care that baseline_settings(...) does not overwrite)
  model.params.use_curriculum = False
  model.params.num_episodes = 60_000
  # model.params.object_set = "set6_fullset_800_50i" # baseline setting added

  if args.program is None:
    # cannot have whitespace
    raise RuntimeError("must specify the program to run with '--program xxxxxxx'")

  # OLD, avoid: defaults to pass into baseline_settings(...)
  this_segments = 8 # was 8, change to 6 for speed
  this_noise = 0.025 # was 0.05, change to 0.025 for stability

  extra_info_string = ""

  if args.program == "paper_baseline_1":

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

  elif args.program == "paper_baseline_1.5":

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

  elif args.program == "paper_baseline_2":

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

  elif args.program == "paper_baseline_2_extra_noise":

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

  elif args.program == "avoid_overfit":

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

  elif args.program == "new_sensor_styles":

    vary_1 = [
      (1, 3),
      (1, 5),
      (1, 7),
      (3, 3) # as baseline
    ]
    vary_2 = None
    vary_3 = None
    repeats = 10
    param_1_name = "sensor/state steps"
    param_2_name = None
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    wrist_mu = 0.01             # large chance of zero error with the wrist
    wrist_std = 0.075           # wrist has a lot of noise, this is 15% coverage +-2stdevs

    baseline_args = {
      "finger_thickness" : 0.9e-3,
      "finger_width" : 28e-3,
      "sensors" : 3,
      "sensor_noise" : 0.025,   # medium noise on sensor readings
      "state_noise" : 0.0,      # no noise on state readings, this is required for sign mode
      "sensor_mu" : 0.05,       # can be +- 5% from 0
      "state_mu" : 0.025,       # just a gentle zero error noise on state readings
      "sensor_steps" : param_1[0],       
      "state_steps" : param_1[1],
      "sensor_mode" : 2,        # average sample, leave as before
      "state_mode" : 4,         # state sign mode, -1,0,+1 for motor state change
      "eval_me" : f"model.env.mj.set.wrist_sensor_Z.set_gaussian_noise({wrist_mu}, {wrist_std})"
    }

    # don't run long trainings
    model.params.num_episodes = 48_000

    # run slightly longer tests
    model.env.params.test_trials_per_object = 3

    # test less often
    model.params.test_freq = 4000
    model.params.save_freq = 4000

  elif args.program == "new_sensor_styles_continued":

    print("Continuing training in group:", model.group_name)
    print("Continuing training of run:", model.run_name)

    # set up the object set
    model.env.mj.model_folder_path = "/home/luke/mymujoco/mjcf"

    extra_episodes = 48_000
    model.wandb_note += f"Continuing training with an extra {extra_episodes} episodes\n"

    folderpath = model.savedir + model.group_name + "/"
    foldername = model.run_name

    from ModelSaver import ModelSaver
    model.run_name = foldername + "_continued"
    model.modelsaver = ModelSaver(folderpath)

    # load the most recent model in the given folder
    model.load(foldername=foldername, folderpath=folderpath)
    model.modelsaver.enter_folder(foldername, folderpath=folderpath)

    # add extra episodes on to what has already been done
    model.params.num_episodes = model.params.num_episodes + extra_episodes

    # NOW SCALE UP THE PENALTIES
    model.env.mj.set.exceed_limits.reward = 2.5 * model.env.mj.set.exceed_limits.reward
    model.env.mj.set.exceed_axial.reward = 2.5 * model.env.mj.set.exceed_axial.reward
    model.env.mj.set.exceed_lateral.reward = 2.5 * model.env.mj.set.exceed_lateral.reward
    model.env.mj.set.exceed_palm.reward = 2.5 * model.env.mj.set.exceed_palm.reward

    # begin the training at the given starting point (always uses most recent pickle)
    model.train(i_start=model.track.episodes_done)

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

    # skip train/test below
    args.print = True

  elif args.program == "new_sensor_styles_extended":

    vary_1 = [
      (1, 3),
      (1, 5)
    ]
    vary_2 = [
      (2.5, 1.0), # baseline
      (2.5, 2.5),
    ]
    vary_3 = None
    repeats = 10
    param_1_name = "sensor/state steps"
    param_2_name = None
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    wrist_mu = 0.01             # large chance of zero error with the wrist
    wrist_std = 0.075           # wrist has a lot of noise, this is 15% coverage +-2stdevs

    baseline_args = {
      "finger_thickness" : 0.9e-3,
      "finger_width" : 28e-3,
      "sensors" : 3,
      "sensor_noise" : 0.025,   # medium noise on sensor readings
      "state_noise" : 0.0,      # no noise on state readings, this is required for sign mode
      "sensor_mu" : 0.05,       # can be +- 5% from 0
      "state_mu" : 0.025,       # just a gentle zero error noise on state readings
      "sensor_steps" : param_1[0],       
      "state_steps" : param_1[1],
      "sensor_mode" : 2,        # average sample, leave as before
      "state_mode" : 4,         # state sign mode, -1,0,+1 for motor state change
      "eval_me" : f"model.env.mj.set.wrist_sensor_Z.set_gaussian_noise({wrist_mu}, {wrist_std})",
      "scale_rewards" : param_2[0],
      "scale_penalties" : param_2[1],
    }

    # don't run long trainings
    model.params.num_episodes = 48_000

    # run slightly longer tests
    model.env.params.test_trials_per_object = 3

    # test less often
    model.params.test_freq = 4000
    model.params.save_freq = 4000

  elif args.program == "paper_baseline_3":

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

    wrist_mu = 0.01             # large chance of zero error with the wrist
    wrist_std = 0.075           # wrist has a lot of noise, this is 15% coverage +-2stdevs

    baseline_args = {
      "finger_thickness" : param_1[0],
      "finger_width" : param_1[1],
      "sensors" : param_2,
      "sensor_noise" : 0.025,   # medium noise on sensor readings
      "state_noise" : 0.0,      # no noise on state readings, this is required for sign mode
      "sensor_mu" : 0.05,       # can be +- 5% from 0
      "state_mu" : 0.025,       # just a gentle zero error noise on state readings
      "sensor_steps" : 1,       # limit this since sensor data is unreliable
      "state_steps" : 5,        # this data stream is clean, so take a lot of it
      "sensor_mode" : 2,        # average sample, leave as before
      "state_mode" : 4,         # state sign mode, -1,0,+1 for motor state change
      "eval_me" : f"model.env.mj.set.wrist_sensor_Z.set_gaussian_noise({wrist_mu}, {wrist_std})",
      "scale_rewards" : 2.5,    # stronger reward signal aids training
      "scale_penalties" : 2.5,  # we do want to discourage dangerous actions
    }

    # don't run long trainings
    model.params.num_episodes = 60_000

    # run slightly longer tests
    model.env.params.test_trials_per_object = 3

    # test less often
    model.params.test_freq = 4000
    model.params.save_freq = 4000

  elif args.program == "paper_baseline_3.1":

    vary_1 = [
      (0.9e-3, 28e-3),
      (1.0e-3, 24e-3),
      (1.0e-3, 28e-3),
    ]
    vary_2 = [0, 1, 2, 3]
    vary_3 = None
    repeats = 20
    param_1_name = "thickness, width"
    param_2_name = "num sensors"
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    wrist_mu = 0.01             # large chance of zero error with the wrist
    wrist_std = 0.075           # wrist has a lot of noise, this is 15% coverage +-2stdevs

    baseline_args = {
      "finger_thickness" : param_1[0],
      "finger_width" : param_1[1],
      "sensors" : param_2,
      "sensor_noise" : 0.025,   # medium noise on sensor readings
      "state_noise" : 0.0,      # no noise on state readings, this is required for sign mode
      "sensor_mu" : 0.05,       # can be +- 5% from 0
      "state_mu" : 0.025,       # just a gentle zero error noise on state readings
      "sensor_steps" : 1,       # limit this since sensor data is unreliable
      "state_steps" : 5,        # this data stream is clean, so take a lot of it
      "sensor_mode" : 2,        # average sample, leave as before
      "state_mode" : 4,         # state sign mode, -1,0,+1 for motor state change
      "eval_me" : f"model.env.mj.set.wrist_sensor_Z.set_gaussian_noise({wrist_mu}, {wrist_std})",
      "scale_rewards" : 2.5,    # stronger reward signal aids training
      "scale_penalties" : 2.5,  # we do want to discourage dangerous actions
    }

    # don't run long trainings
    model.params.num_episodes = 60_000

    # run slightly longer tests
    model.env.params.test_trials_per_object = 3

    # test less often
    model.params.test_freq = 4000
    model.params.save_freq = 4000

  elif args.program == "paper_baseline_3.1_heuristic":

    vary_1 = [
      (0.9e-3, 28e-3),
      (1.0e-3, 24e-3),
      (1.0e-3, 28e-3),
    ]
    vary_2 = [0, 1, 2, 3]
    vary_3 = None
    repeats = 5
    param_1_name = "thickness, width"
    param_2_name = "num sensors"
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    wrist_mu = 0.01             # large chance of zero error with the wrist
    wrist_std = 0.075           # wrist has a lot of noise, this is 15% coverage +-2stdevs

    baseline_args = {
      "finger_thickness" : param_1[0],
      "finger_width" : param_1[1],
      "sensors" : param_2,
      "sensor_noise" : 0.025,   # medium noise on sensor readings
      "state_noise" : 0.0,      # no noise on state readings, this is required for sign mode
      "sensor_mu" : 0.05,       # can be +- 5% from 0
      "state_mu" : 0.025,       # just a gentle zero error noise on state readings
      "sensor_steps" : 1,       # limit this since sensor data is unreliable
      "state_steps" : 5,        # this data stream is clean, so take a lot of it
      "sensor_mode" : 2,        # average sample, leave as before
      "state_mode" : 4,         # state sign mode, -1,0,+1 for motor state change
      "eval_me" : f"model.env.mj.set.wrist_sensor_Z.set_gaussian_noise({wrist_mu}, {wrist_std})",
      "scale_rewards" : 2.5,    # stronger reward signal aids training
      "scale_penalties" : 2.5,  # we do want to discourage dangerous actions
    }

  elif args.program == "pb4_testing":

    # vary_1 = [
    #   (0.9e-3, 28e-3),
    #   (1.0e-3, 24e-3),
    #   (1.0e-3, 28e-3),
    # ]
    # vary_2 = [0, 1, 2, 3]
    vary_1 = [4000, 8000]
    vary_2 = [1.0, 2.5]
    vary_3 = None
    repeats = 10
    param_1_name = "eps decay"
    param_2_name = "rew/pen scaling"
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    eval_me = []

    wrist_mu = 0.01             # large chance of zero error with the wrist
    wrist_std = 0.075           # wrist has a lot of noise, this is 15% coverage +-2stdevs
    eval_me.append(f"model.env.mj.set.wrist_sensor_Z.set_gaussian_noise({wrist_mu}, {wrist_std})")

    exceed_lat_max_factor = 1.1 # penalty reaches maximum at this factor of yield load
    eval_me.append(f"model.env.mj.set.exceed_lat_max_factor = {exceed_lat_max_factor}")

    baseline_args = {
      "finger_thickness" : 0.9e-3,
      "finger_width" : 28e-3,
      "sensors" : 3,
      "sensor_noise" : 0.025,         # medium noise on sensor readings
      "state_noise" : 0.0,            # no noise on state readings, this is required for sign mode
      "sensor_mu" : 0.05,             # can be +- 5% from 0
      "state_mu" : 0.025,             # just a gentle zero error noise on state readings
      "sensor_steps" : 1,             # limit this since sensor data is unreliable
      "state_steps" : 5,              # this data stream is clean, so take a lot of it
      "sensor_mode" : 2,              # average sample, leave as before
      "state_mode" : 4,               # state sign mode, -1,0,+1 for motor state change
      "eval_me" : eval_me,            # extra settings tweaks
      "scale_rewards" : param_2,          # stronger reward signal aids training
      "scale_penalties" : param_2,        # we do want to discourage dangerous actions
      "eps_decay" : param_1,             # add extra exploration -> eps=8k gives 10k=29%, 20k=8%, 30k=2%, 40k=0.7%
    }

    # use the new object set
    model.params.object_set = "set7_fullset_1500_50i"

    # run medium length trainings
    model.params.num_episodes = 60_000

    # run slightly longer tests during training
    model.env.params.test_trials_per_object = 3

    # test less often
    model.params.test_freq = 4000
    model.params.save_freq = 4000

  elif args.program == "paper_baseline_4":

    vary_1 = [
      (0.9e-3, 28e-3),
      (1.0e-3, 24e-3),
      (1.0e-3, 28e-3),
    ]
    vary_2 = [0, 1, 2, 3]
    vary_3 = None
    repeats = 20
    param_1_name = "finger thickness/width"
    param_2_name = "num sensors"
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    eval_me = []

    wrist_mu = 0.01             # large chance of zero error with the wrist
    wrist_std = 0.075           # wrist has a lot of noise, this is 15% coverage +-2stdevs
    eval_me.append(f"model.env.mj.set.wrist_sensor_Z.set_gaussian_noise({wrist_mu}, {wrist_std})")

    exceed_lat_max_factor = 1.1 # penalty reaches maximum at this factor of yield load
    eval_me.append(f"model.env.mj.set.exceed_lat_max_factor = {exceed_lat_max_factor}")

    baseline_args = {
      "finger_thickness" : param_1[0],
      "finger_width" : param_1[1],
      "sensors" : param_2,
      "sensor_noise" : 0.025,         # medium noise on sensor readings
      "state_noise" : 0.0,            # no noise on state readings, this is required for sign mode
      "sensor_mu" : 0.05,             # can be +- 5% from 0
      "state_mu" : 0.025,             # just a gentle zero error noise on state readings
      "sensor_steps" : 1,             # limit this since sensor data is unreliable
      "state_steps" : 5,              # this data stream is clean, so take a lot of it
      "sensor_mode" : 2,              # average sample, leave as before
      "state_mode" : 4,               # state sign mode, -1,0,+1 for motor state change
      "eval_me" : eval_me,            # extra settings tweaks
      "scale_rewards" : 1.0,          # stronger reward signal aids training
      "scale_penalties" : 1.0,        # we do want to discourage dangerous actions
    }

    # use the new object set
    model.params.object_set = "set7_fullset_1500_50i"

    # run long trainings
    model.params.num_episodes = 100_000

    # run slightly longer tests during training
    model.env.params.test_trials_per_object = 3

    # test less often
    model.params.test_freq = 4000
    model.params.save_freq = 4000

  elif args.program == "paper_baseline_4_heuristic":

    vary_1 = [
      (0.9e-3, 28e-3),
      (1.0e-3, 24e-3),
      (1.0e-3, 28e-3),
    ]
    vary_2 = [0, 1, 2, 3]
    vary_3 = None
    repeats = 5
    param_1_name = "finger thickness/width"
    param_2_name = "num sensors"
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    eval_me = []

    wrist_mu = 0.01             # large chance of zero error with the wrist
    wrist_std = 0.075           # wrist has a lot of noise, this is 15% coverage +-2stdevs
    eval_me.append(f"model.env.mj.set.wrist_sensor_Z.set_gaussian_noise({wrist_mu}, {wrist_std})")

    exceed_lat_max_factor = 1.1 # penalty reaches maximum at this factor of yield load
    eval_me.append(f"model.env.mj.set.exceed_lat_max_factor = {exceed_lat_max_factor}")

    baseline_args = {
      "finger_thickness" : param_1[0],
      "finger_width" : param_1[1],
      "sensors" : param_2,
      "sensor_noise" : 0.025,         # medium noise on sensor readings
      "state_noise" : 0.0,            # no noise on state readings, this is required for sign mode
      "sensor_mu" : 0.05,             # can be +- 5% from 0
      "state_mu" : 0.025,             # just a gentle zero error noise on state readings
      "sensor_steps" : 1,             # limit this since sensor data is unreliable
      "state_steps" : 5,              # this data stream is clean, so take a lot of it
      "sensor_mode" : 2,              # average sample, leave as before
      "state_mode" : 4,               # state sign mode, -1,0,+1 for motor state change
      "eval_me" : eval_me,            # extra settings tweaks

      "scale_rewards" : 1.0,          # stronger reward signal aids training
      "scale_penalties" : 1.0,        # we do want to discourage dangerous actions
      "exceed_lims_multiplier" : 1.0, # disable extra attention to avoiding the table
    }

    # use the new object set
    model.params.object_set = "set7_fullset_1500_50i"

    # run long trainings
    model.params.num_episodes = 100_000

    # run slightly longer tests during training
    model.env.params.test_trials_per_object = 3

    # test less often
    model.params.test_freq = 4000
    model.params.save_freq = 4000

  elif args.program == "heuristic_grid_search":

    vary_1 = [False, True]
    vary_2 = [
      (1.5, 1.5),
      (2, 2),
      (2.5, 2.5),
    ]
    vary_3 = [
      (10, True), 
      (15, True), 
      (20, True),
      (None, False),
    ]
    repeats = 3
    param_1_name = "final squeeze"
    param_2_name = "force target"
    param_3_name = "finger_angle"
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    eval_me = []

    wrist_mu = 0.01             # large chance of zero error with the wrist
    wrist_std = 0.075           # wrist has a lot of noise, this is 15% coverage +-2stdevs
    eval_me.append(f"model.env.mj.set.wrist_sensor_Z.set_gaussian_noise({wrist_mu}, {wrist_std})")

    exceed_lat_max_factor = 1.1 # penalty reaches maximum at this factor of yield load
    eval_me.append(f"model.env.mj.set.exceed_lat_max_factor = {exceed_lat_max_factor}")

    baseline_args = {
      "finger_thickness" : 0.9e-3,
      "finger_width" : 28e-3,
      "sensors" : 3,
      "sensor_noise" : 0.025,         # medium noise on sensor readings
      "state_noise" : 0.0,            # no noise on state readings, this is required for sign mode
      "sensor_mu" : 0.05,             # can be +- 5% from 0
      "state_mu" : 0.025,             # just a gentle zero error noise on state readings
      "sensor_steps" : 1,             # limit this since sensor data is unreliable
      "state_steps" : 5,              # this data stream is clean, so take a lot of it
      "sensor_mode" : 2,              # average sample, leave as before
      "state_mode" : 4,               # state sign mode, -1,0,+1 for motor state change
      "eval_me" : eval_me,            # extra settings tweaks
      "scale_rewards" : 1.0,          # stronger reward signal aids training
      "scale_penalties" : 1.0,        # we do want to discourage dangerous actions
      "exceed_lims_multiplier" : 1.0, # disable extra attention to avoiding the table
    }

    # set the heuristic parameters
    model.env.heuristic_params["final_squeeze"] = param_1
    model.env.heuristic_params["initial_bend_target_N"] = param_2[0]
    model.env.heuristic_params["initial_palm_target_N"] = param_2[1]
    model.env.heuristic_params["final_bend_target_N"] = param_2[0]
    model.env.heuristic_params["final_palm_target_N"] = param_2[1]
    model.env.heuristic_params["target_angle_deg"] = param_3[0]
    model.env.heuristic_params["fixed_angle"] = param_3[1]

    # use the new object set
    model.params.object_set = "set7_fullset_1500_50i"

    # run long trainings
    model.params.num_episodes = 100_000

    # run slightly longer tests during training
    model.env.params.test_trials_per_object = 3

    # test less often
    model.params.test_freq = 4000
    model.params.save_freq = 4000

  elif args.program == "test_exceed_limits_termination":

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

  elif args.program == "test_prevent_table_impacts":

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

  elif args.program == "new_sensor_rewards":

    vary_1 = [False, True]
    vary_2 = [(100.0, 100), (3.0, 10.0)]
    vary_3 = None
    repeats = 10
    param_1_name = "penalty termination"
    param_2_name = "stable force limit"
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    eval_me = []

    # # half wrist noise as the normalisation has been doubled from 5 -> 10N
    # wrist_mu = 0.01 * 0.5            # large chance of zero error with the wrist
    # wrist_std = 0.075 * 0.5          # wrist has a lot of noise, this is 15% coverage +-2stdevs
    # eval_me.append(f"model.env.mj.set.wrist_sensor_Z.set_gaussian_noise({wrist_mu}, {wrist_std})")

    # do we limit stable grasps to a maximum allowable force
    model.env.mj.set.stable_finger_force_lim = param_2[0]
    model.env.mj.set.stable_palm_force_lim = param_2[1]

    baseline_args = {

      "finger_thickness" : 0.9e-3,
      "finger_width" : 28e-3,
      "sensors" : 3,

      "penalty_termination" : param_1, # do we end episodes on dangerous readings

      "sensor_noise" : 0.025,          # medium noise on sensor readings
      "state_noise" : 0.0,             # no noise on state readings, this is required for sign mode
      "sensor_mu" : 0.05,              # can be +- 5% from 0
      "state_mu" : 0.025,              # just a gentle zero error noise on state readings
      "sensor_steps" : 3,              # limit this since sensor data is unreliable
      "state_steps" : 3,               # this data stream is clean, so take a lot of it
      "sensor_mode" : 2,               # average sample, leave as before
      "state_mode" : 4,                # state sign mode, -1,0,+1 for motor state change
      "eval_me" : eval_me,             # extra settings tweaks
      "scale_rewards" : 1.0,           # stronger reward signal aids training
      "scale_penalties" : 1.0,         # we do want to discourage dangerous actions
      "reward_style" : "sensor_mixed", # what reward function do we want


      # FOR TESTING - delete before any proper trainings
      "num_segments" : 8
    }

    # use the new object set
    model.params.object_set = "set7_fullset_1500_50i"

    # run medium length trainings
    model.params.num_episodes = 60_000

    # run slightly longer tests during training
    model.env.params.test_trials_per_object = 3

    # test less often
    model.params.test_freq = 4000
    model.params.save_freq = 4000

  elif args.program == "new_sensor_hypers":

    vary_1 = [1e-5, 5e-5, 10e-5]
    vary_2 = [1.0, 2.5]
    vary_3 = None
    repeats = 8
    param_1_name = "learning rate"
    param_2_name = "reward scaling"
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    # do we limit stable grasps to a maximum allowable force
    model.env.mj.set.stable_finger_force_lim = 100
    model.env.mj.set.stable_palm_force_lim = 100

    baseline_args = {

      # key finger parameters
      "finger_thickness" : 0.9e-3,
      "finger_width" : 28e-3,
      "sensors" : 3,
      "num_segments" : 8, # dont forget to set to 8 for proper trainings

      # hyperparemeters
      "lr" : param_1,

      # reward features
      "penalty_termination" : False, # do we end episodes on dangerous readings
      "scale_rewards" : param_2,           # stronger reward signal aids training
      "scale_penalties" : 1.0,         # we do want to discourage dangerous actions
      "reward_style" : "sensor_mixed", # what reward function do we want

      # sensor details
      "sensor_noise" : 0.025,          # medium noise on sensor readings
      "state_noise" : 0.0,             # no noise on state readings, this is required for sign mode
      "sensor_mu" : 0.05,              # can be +- 5% from 0
      "state_mu" : 0.025,              # just a gentle zero error noise on state readings
      "sensor_steps" : 3,              # limit this since sensor data is unreliable
      "state_steps" : 3,               # this data stream is clean, so take a lot of it
      "sensor_mode" : 2,               # average sample, leave as before
      "state_mode" : 4,                # state sign mode, -1,0,+1 for motor state change 
    }

    # use the new object set
    model.params.object_set = "set7_fullset_1500_50i"

    # run medium length trainings
    model.params.num_episodes = 60_000

    # run slightly longer tests during training
    model.env.params.test_trials_per_object = 3

    # test less often
    model.params.test_freq = 4000
    model.params.save_freq = 4000

  elif args.program == "new_sensor_hypers_2":

    vary_1 = [25_000, 50_000, 75_000, 100_000]
    vary_2 = [50, 100, 200, 500]
    vary_3 = None
    repeats = 8
    param_1_name = "memory replay"
    param_2_name = "target update"
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    # do we limit stable grasps to a maximum allowable force
    model.env.mj.set.stable_finger_force_lim = 100
    model.env.mj.set.stable_palm_force_lim = 100

    baseline_args = {

      # key finger parameters
      "finger_thickness" : 0.9e-3,
      "finger_width" : 28e-3,
      "sensors" : 3,
      "num_segments" : 8,

      # hyperparemeters
      "memory_replay" : param_1,
      "target_update" : param_2,

      # reward features
      "penalty_termination" : False, # do we end episodes on dangerous readings
      # "scale_rewards" : 1.0,           # stronger reward signal aids training
      # "scale_penalties" : 1.0,         # we do want to discourage dangerous actions
      "reward_style" : "sensor_mixed", # what reward function do we want

      # sensor details
      "sensor_noise" : 0.025,          # medium noise on sensor readings
      "state_noise" : 0.0,             # no noise on state readings, this is required for sign mode
      "sensor_mu" : 0.05,              # can be +- 5% from 0
      "state_mu" : 0.025,              # just a gentle zero error noise on state readings
      "sensor_steps" : 3,              # limit this since sensor data is unreliable
      "state_steps" : 3,               # this data stream is clean, so take a lot of it
      "sensor_mode" : 2,               # average sample, leave as before
      "state_mode" : 4,                # state sign mode, -1,0,+1 for motor state change 
    }

    # use the new object set
    model.params.object_set = "set7_fullset_1500_50i"

    # run medium length trainings
    model.params.num_episodes = 60_000

    # run slightly longer tests during training
    model.env.params.test_trials_per_object = 3

    # test less often
    model.params.test_freq = 4000
    model.params.save_freq = 4000

  elif args.program == "new_sensor_hypers_2_extended":

    vary_1 = [25_000, 50_000, 75_000, 100_000]
    vary_2 = [1, 25]
    vary_3 = None
    repeats = 8
    param_1_name = "memory replay"
    param_2_name = "target update"
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    # do we limit stable grasps to a maximum allowable force
    model.env.mj.set.stable_finger_force_lim = 100
    model.env.mj.set.stable_palm_force_lim = 100

    baseline_args = {

      # key finger parameters
      "finger_thickness" : 0.9e-3,
      "finger_width" : 28e-3,
      "sensors" : 3,
      "num_segments" : 8,

      # hyperparemeters
      "memory_replay" : param_1,
      "target_update" : param_2,

      # reward features
      "penalty_termination" : False, # do we end episodes on dangerous readings
      # "scale_rewards" : 1.0,           # stronger reward signal aids training
      # "scale_penalties" : 1.0,         # we do want to discourage dangerous actions
      "reward_style" : "sensor_mixed", # what reward function do we want

      # sensor details
      "sensor_noise" : 0.025,          # medium noise on sensor readings
      "state_noise" : 0.0,             # no noise on state readings, this is required for sign mode
      "sensor_mu" : 0.05,              # can be +- 5% from 0
      "state_mu" : 0.025,              # just a gentle zero error noise on state readings
      "sensor_steps" : 3,              # limit this since sensor data is unreliable
      "state_steps" : 3,               # this data stream is clean, so take a lot of it
      "sensor_mode" : 2,               # average sample, leave as before
      "state_mode" : 4,                # state sign mode, -1,0,+1 for motor state change 
    }

    # use the new object set
    model.params.object_set = "set7_fullset_1500_50i"

    # run medium length trainings
    model.params.num_episodes = 60_000

    # run slightly longer tests during training
    model.env.params.test_trials_per_object = 3

    # test less often
    model.params.test_freq = 4000
    model.params.save_freq = 4000

  elif args.program == "heavy_test":

    vary_1 = [1.0, 2.5]
    vary_2 = [3, 5]
    vary_3 = None
    repeats = 15
    param_1_name = "scale rewards"
    param_2_name = "state steps"
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    # do we limit stable grasps to a maximum allowable force
    model.env.mj.set.stable_finger_force_lim = 100
    model.env.mj.set.stable_palm_force_lim = 100

    baseline_args = {

      # key finger parameters
      "finger_thickness" : 0.9e-3,
      "finger_width" : 28e-3,
      "sensors" : 3,
      "num_segments" : 8,

      # hyperparemeters
      "memory_replay" : 50_000,
      "target_update" : 25,

      # reward features
      "penalty_termination" : False, # do we end episodes on dangerous readings
      "scale_rewards" : param_1,           # stronger reward signal aids training
      # "scale_penalties" : 1.0,         # we do want to discourage dangerous actions
      "reward_style" : "sensor_mixed", # what reward function do we want

      # sensor details
      "sensor_noise" : 0.025,          # medium noise on sensor readings
      "state_noise" : 0.0,             # no noise on state readings, this is required for sign mode
      "sensor_mu" : 0.05,              # can be +- 5% from 0
      "state_mu" : 0.025,              # just a gentle zero error noise on state readings
      "sensor_steps" : 3,              # limit this since sensor data is unreliable
      "state_steps" : param_2,               # this data stream is clean, so take a lot of it
      "sensor_mode" : 2,               # average sample, leave as before
      "state_mode" : 4,                # state sign mode, -1,0,+1 for motor state change 
    }

    # use the new heavy object set
    model.params.object_set = "set7_fullset_1500_heavy"

    # run long length trainings
    model.params.num_episodes = 100_000

    # run slightly longer tests during training
    model.env.params.test_trials_per_object = 3

    # test less often
    model.params.test_freq = 4000
    model.params.save_freq = 4000

  elif args.program == "step_size_curriculum":

    # define step size levels
    levels_A = [
      [8e-3, 0.025, 8e-3, 2e-3, 0.8],
      [4e-3, 0.02,  6e-3, 2e-3, 0.4],
      [2e-3, 0.015, 4e-3, 2e-3, 0.2],
      [1e-3, 0.01,  2e-3, 2e-3, 0.2],
    ]
    levels_B = [
      [4e-3, 0.025, 8e-3, 2e-3, 0.4],
      [3e-3, 0.02,  6e-3, 2e-3, 0.3],
      [2e-3, 0.015, 4e-3, 2e-3, 0.2],
      [1e-3, 0.01,  2e-3, 2e-3, 0.2],
    ]

    thresholds_A = [10_000, 25_000, 40_000]
    thresholds_B = [20_000, 30_000, 40_000]

    vary_1 = [
        (False, None, None),
        (True, levels_A, thresholds_A),
        (True, levels_A, thresholds_B),
        (True, levels_B, thresholds_A),
        (True, levels_B, thresholds_B),

    ]
    vary_2 = None
    vary_3 = None
    repeats = 10
    param_1_name = "curriculum/levels/thresholds"
    param_2_name = None
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    # do we limit stable grasps to a maximum allowable force
    model.env.mj.set.stable_finger_force_lim = 100
    model.env.mj.set.stable_palm_force_lim = 100

    # set up the curriculum
    if param_1[0]:
      model.params.use_curriculum = True
      model.curriculum_params["step_sizes"] = param_1[1]
      model.curriculum_params["thresholds"] = param_1[2]
      model.curriculum_fcn = functools.partial(curriculum_step_size, model)

    baseline_args = {

      # key finger parameters
      "finger_thickness" : 0.9e-3,
      "finger_width" : 28e-3,
      "sensors" : 3,
      "num_segments" : 8,

      # reward features
      "penalty_termination" : False, # do we end episodes on dangerous readings
      "reward_style" : "sensor_mixed", # use new sensor reward function

      # sensor details
      "sensor_steps" : 3,
      "state_steps" : 3,
    }

    # use the new object set
    model.params.object_set = "set7_fullset_1500_50i"

    # run medium length trainings
    model.params.num_episodes = 60_000

    # run slightly longer tests during training
    model.env.params.test_trials_per_object = 3

    # test less often
    model.params.test_freq = 4000
    model.params.save_freq = 4000

  elif args.program == "step_size_curriculum_2":

    # define step size levels
    levels_A = [
      [8e-3, 0.025, 8e-3, 2e-3, 0.8],
      [4e-3, 0.02,  6e-3, 2e-3, 0.4],
      [2e-3, 0.015, 4e-3, 2e-3, 0.2],
      [1e-3, 0.01,  2e-3, 2e-3, 0.2],
    ]
    levels_B = [
      [4e-3, 0.025, 8e-3, 2e-3, 0.4],
      [3e-3, 0.02,  6e-3, 2e-3, 0.3],
      [2e-3, 0.015, 4e-3, 2e-3, 0.2],
      [1e-3, 0.01,  2e-3, 2e-3, 0.2],
    ]

    thresholds_A = [10_000, 25_000, 50_000]
    thresholds_B = [20_000, 35_000, 50_000]

    vary_1 = [levels_A, levels_B]
    vary_2 = [thresholds_A, thresholds_B]
    vary_3 = None
    repeats = 10
    param_1_name = "step size levels"
    param_2_name = "episode thresholds"
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    # do we limit stable grasps to a maximum allowable force
    model.env.mj.set.stable_finger_force_lim = 100
    model.env.mj.set.stable_palm_force_lim = 100

    # set up the curriculum
    model.params.use_curriculum = True
    model.curriculum_params["step_sizes"] = param_1
    model.curriculum_params["thresholds"] = param_2
    model.curriculum_fcn = functools.partial(curriculum_step_size, model)

    baseline_args = {

      "reward_style" : "sensor_mixed", # use new sensor reward function

      # sensor details
      "sensor_steps" : 3,
      "state_steps" : 3,
    }

    # use the new object set
    model.params.object_set = "set7_fullset_1500_50i"

    # run medium length trainings
    model.params.num_episodes = 60_000

  elif args.program == "cnn_trial_1":

    vary_1 = [
      [150, 100, 50],
      "CNN_25_25",
      "CNN_50_50"
    ]
    vary_2 = None
    vary_3 = None
    repeats = 10
    param_1_name = "network style"
    param_2_name = None
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    # do we limit stable grasps to a maximum allowable force
    model.env.mj.set.stable_finger_force_lim = 100
    model.env.mj.set.stable_palm_force_lim = 100

    baseline_args = {

      "network" : param_1,

      "reward_style" : "sensor_mixed", # use new sensor reward function

      # sensor details
      "sensor_steps" : 3,
      "state_steps" : 3,
    }

    # use the new object set
    model.params.object_set = "set7_xycamera_50i"

  elif args.program == "cnn_trial_2":

    # setup up a step size curriculum
    levels_A = [
      [8e-3, 0.025, 8e-3, 2e-3, 0.8],
      [4e-3, 0.02,  6e-3, 2e-3, 0.4],
      [2e-3, 0.015, 4e-3, 2e-3, 0.2],
      [1e-3, 0.01,  2e-3, 2e-3, 0.2],
    ]
    thresholds_A = [10_000, 25_000, 50_000]
    model.params.use_curriculum = True
    model.curriculum_params["step_sizes"] = levels_A
    model.curriculum_params["thresholds"] = thresholds_A
    model.curriculum_params["metric"] = "episode_number"
    model.curriculum_fcn = functools.partial(curriculum_step_size, model)

    vary_1 = [
      "CNN_25_25",
      "CNN_50_50"
    ]
    vary_2 = None
    vary_3 = None
    repeats = 4
    param_1_name = "network"
    param_2_name = None
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    # do we limit stable grasps to a maximum allowable force
    model.env.mj.set.stable_finger_force_lim = 100
    model.env.mj.set.stable_palm_force_lim = 100

    baseline_args = {
      "network" : param_1,
      "sensor_steps" : 3,
      "state_steps" : 3,
    }

    # use the new object set
    model.params.object_set = "set7_fullset_1500_50i_updated"

    # test more often
    model.params.test_freq = 2000
    model.params.save_freq = 2000

  elif args.program == "cnn_trial_3":

    # setup up a step size curriculum
    levels_A = [
      [8e-3, 0.025, 8e-3, 2e-3, 0.8],
      [4e-3, 0.02,  6e-3, 2e-3, 0.4],
      [2e-3, 0.015, 4e-3, 2e-3, 0.2],
      [1e-3, 0.01,  2e-3, 2e-3, 0.2],
    ]
    thresholds_A = [0.5, 0.65, 0.8]
    model.params.use_curriculum = True
    model.curriculum_params["step_sizes"] = levels_A
    model.curriculum_params["thresholds"] = thresholds_A
    model.curriculum_params["metric"] = "success_rate"
    model.curriculum_fcn = functools.partial(curriculum_step_size, model)

    vary_1 = [
      "CNN_25_25",
      # "CNN_50_50"
    ]
    vary_2 = [5e-5, 1e-4]
    vary_3 = None
    repeats = 5
    param_1_name = "network"
    param_2_name = "learning rate"
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    # do we limit stable grasps to a maximum allowable force
    model.env.mj.set.stable_finger_force_lim = 100
    model.env.mj.set.stable_palm_force_lim = 100

    baseline_args = {
      "network" : param_1,
      "lr" : param_2,
      "sensor_steps" : 3,
      "state_steps" : 3,
    }

    # use the new object set
    model.params.object_set = "set7_fullset_1500_50i_updated"

    # normal testing
    model.params.test_freq = 4000
    model.params.save_freq = 4000

  elif args.program == "image_collection":

    vary_1 = [True, False]
    vary_2 = ["image_collection_100_100", "image_collection_50_50"]
    vary_3 = None
    repeats = 1
    param_1_name = "use_curriculum"
    param_2_name = None
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    # do we limit stable grasps to a maximum allowable force
    model.env.mj.set.stable_finger_force_lim = 100
    model.env.mj.set.stable_palm_force_lim = 100

    baseline_args = {
      "network" : param_2,
      "sensor_steps" : 3,
      "state_steps" : 3,
    }

    # 100x100 images appear to be about 100MB each in memory replay
    # the computer has 60GB RAM so I can store 600_000 maximum

    if param_1:
      # setup up a step size curriculum
      levels_A = [
        [8e-3, 0.025, 8e-3, 2e-3, 0.8],
        [4e-3, 0.02,  6e-3, 2e-3, 0.4],
        [2e-3, 0.015, 4e-3, 2e-3, 0.2],
        [1e-3, 0.01,  2e-3, 2e-3, 0.2],
      ]
      thresholds_A = [10_000, 25_000, 50_000]
      model.params.use_curriculum = True
      model.curriculum_params["step_sizes"] = levels_A
      model.curriculum_params["thresholds"] = thresholds_A
      model.curriculum_params["metric"] = "episode_number"
      model.curriculum_fcn = functools.partial(curriculum_step_size, model)
    else:
      model.params.use_curriculum = False

    # use the new object set
    model.params.object_set = "set8_fullset_1500"

    # normal testing
    model.params.test_freq = 4000
    model.params.save_freq = 4000

    # 75k -> 75000/200 = 375 episodes
    model.params.image_save_freq = 500 # save memory replay frequently

  elif args.program == "profile_cnn":

    model = baseline_settings(model)
    model.env.disable_rendering = True
    model.params.object_set = "set7_fullset_1500_50i_updated"

    # vary_1 = ["CNN_25_25", "CNN_50_50", "CNN_75_75", "CNN_100_100"]
    vary_1 = [[150, 100, 50]]
    vary_2 = ["cpu", "cuda"]
    vary_3 = None
    repeats = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    net = param_1
    dev = param_2

    model.set_device(dev)
    model.profile(saveas=f"py_profile_150x100x50_{dev}.xyz", network=net, path="/home/luke/luke-gripper-mujoco")

    exit()

  elif args.program == "set8_baseline":

    vary_1 = [False, True]
    vary_2 = [(1, 5), (3, 3)]
    vary_3 = None
    repeats = 10
    param_1_name = "use_curriculum"
    param_2_name = "sensor/state steps"
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)
    
    if param_1:
      # setup up a step size curriculum
      levels_A = [
        [8e-3, 0.025, 8e-3, 2e-3, 0.8],
        [4e-3, 0.02,  6e-3, 2e-3, 0.4],
        [2e-3, 0.015, 4e-3, 2e-3, 0.2],
        [1e-3, 0.01,  2e-3, 2e-3, 0.2],
      ]
      thresholds_A = [10_000, 25_000, 50_000]
      model.params.use_curriculum = True
      model.curriculum_params["step_sizes"] = levels_A
      model.curriculum_params["thresholds"] = thresholds_A
      model.curriculum_params["metric"] = "episode_number"
      model.curriculum_fcn = functools.partial(curriculum_step_size, model)
    else:
      model.params.use_curriculum = False

    model.params.object_set = "set8_fullset_1500"

    baseline_args = {
      # I've used (3, 3) in all recent trainings despite not testing if it is better
      # than the pb4 (1, 5) split
      "sensor_steps" : param_2[0],
      "state_steps" : param_2[1],
    }

  elif args.program.startswith("offline_train_v1-"):

    vary_1 = [
      (50, 5), 
      (125, 2),
      (250, 1)
    ]
    vary_2 = [False, True]
    vary_3 = None #[False, True]
    repeats = 3
    param_1_name = "iter_per_file"
    param_2_name = "random order"
    param_3_name = None #"disable_sensor_data"
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    # which offline data are we loading
    if args.program.endswith("v1-1"):
      dataset = "/home/luke/luke-gripper-mujoco/rl/models/dqn/11-08-23/operator-PC_16:33_A1"
      net = "CNN2_100_100"
    elif args.program.endswith("v1-2"):
      dataset = "/home/luke/luke-gripper-mujoco/rl/models/dqn/18-08-23/operator-PC_16:32_A1"
      net = "CNN2_100_100"
    elif args.program.endswith("v1-3"):
      dataset = "/home/luke/luke-gripper-mujoco/rl/models/dqn/18-08-23/operator-PC_16:32_A3"
      net = "CNN2_50_50"

    baseline_args = {
      "network" : net,
      "sensor_steps" : 3,
      "state_steps" : 3,
    }

    # use the new object set
    model.params.object_set = "set8_fullset_1500"

    model.params.no_sensor_data = False

    # normal testing
    model.params.test_freq = 4000
    model.params.save_freq = 4000

    if not args.print and not args.print_results:

      model = baseline_settings(model, **baseline_args)
      model.train_offline(dataset, iter_per_file=param_1[0], random_order=param_2, 
                          epochs=param_1[1])
      # model = test(model, trials_per_obj=10, heuristic=args.heuristic, demo=args.demo)

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

      exit()

  elif args.program.startswith("cnn_from_pretrain_v1"):

    vary_1 = [5e-5, 1e-5]
    vary_2 = None
    vary_3 = None
    repeats = 3
    param_1_name = None
    param_2_name = None
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    if not args.print and not args.print_results:

      # which offline data are we loading
      if args.program.endswith("v1-1"):
        # # image only pretrained network
        # path = "/home/luke/luke-gripper-mujoco/rl/models/dqn/15-08-23/operator-PC_16:52_A3"
        # id = 3
        # both inputs pretrained network
        path = "/home/luke/luke-gripper-mujoco/rl/models/dqn/14-08-23/operator-PC_17:50_A1"
        net = "CNN2_100_100"
        id = 4
      elif args.program.endswith("v1-2"):
        path = "/home/luke/luke-gripper-mujoco/rl/models/dqn/22-08-23/operator-PC_09:55_A5"
        net = "CNN2_50_50"
        folderpath = "/home/luke/luke-gripper-mujoco/rl/models/dqn/22-08-23/"
        foldername = "operator-PC_09:55_A5"
        id = 8

      model.load(id=id, folderpath=folderpath, foldername=foldername)
      model.params.learning_rate = param_1

      # optional: enable this for some trainings for monitoring
      if inputarg % 3 == 1:
        model.params.test_freq = 500 # close eye on performance

      # # remove the pretrained model from memory
      # import gc
      # del pretrain_model
      # gc.collect()

      # now proceed with training
      model.train()

      # test
      model = test(model, trials_per_obj=10, heuristic=args.heuristic, demo=args.demo)

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

      exit()

  elif args.program == "finger_angle_test":

    vary_1 = [30, 45, 60, 75]
    vary_2 = None
    vary_3 = None
    repeats = 10
    param_1_name = "finger_angle"
    param_2_name = None
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)
    
    if param_1:
      # setup up a step size curriculum
      levels_A = [
        [8e-3, 0.025, 8e-3, 2e-3, 0.8],
        [4e-3, 0.02,  6e-3, 2e-3, 0.4],
        [2e-3, 0.015, 4e-3, 2e-3, 0.2],
        [1e-3, 0.01,  2e-3, 2e-3, 0.2],
      ]
      thresholds_A = [10_000, 25_000, 50_000]
      model.params.use_curriculum = True
      model.curriculum_params["step_sizes"] = levels_A
      model.curriculum_params["thresholds"] = thresholds_A
      model.curriculum_params["metric"] = "episode_number"
      model.curriculum_fcn = functools.partial(curriculum_step_size, model)
    else:
      model.params.use_curriculum = False

    model.params.object_set = "set8_fullset_1500"
    model.env.load_next.finger_hook_angle_degrees = param_1

    baseline_args = {
      "sensor_steps" : 3,
      "state_steps" : 3,
    }

  elif args.program == "set8_baseline_extended":

    thresholds_A = [10_000, 25_000, 50_000]
    thresholds_B = [10_000, 25_000, 75_000]
    thresholds_C = [25_000, 50_000, 75_000]
    thresholds_D = [50_000, 75_000, 101_000]

    vary_1 = [thresholds_A, thresholds_B, thresholds_C, thresholds_D]
    vary_2 = None
    vary_3 = None
    repeats = 10
    param_1_name = "threshold"
    param_2_name = None
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)
    
    if param_1:
      # setup up a step size curriculum
      levels_A = [
        [8e-3, 0.025, 8e-3, 2e-3, 0.8],
        [4e-3, 0.02,  6e-3, 2e-3, 0.4],
        [2e-3, 0.015, 4e-3, 2e-3, 0.2],
        [1e-3, 0.01,  2e-3, 2e-3, 0.2],
      ]
      model.params.use_curriculum = True
      model.curriculum_params["step_sizes"] = levels_A
      model.curriculum_params["thresholds"] = param_1
      model.curriculum_params["metric"] = "episode_number"
      model.curriculum_fcn = functools.partial(curriculum_step_size, model)
    else:
      model.params.use_curriculum = False

    model.params.object_set = "set8_fullset_1500"
    # model.env.load_next.finger_hook_angle_degrees = 90

    # long trainings
    model.params.num_episodes = 100_000

    baseline_args = {
      "sensor_steps" : 3,
      "state_steps" : 3,
    }

  elif args.program.startswith("full_offline_training_v1"):

    # vary_1 = [
    #   (50, 5), 
    #   (125, 2),
    #   (250, 1)
    # ]
    # vary_2 = [False, True]
    # vary_3 = None #[False, True]
    # repeats = 3
    # param_1_name = "iter_per_file"
    # param_2_name = "random order"
    # param_3_name = None #"disable_sensor_data"
    # param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
    #                                             param_3=vary_3, repeats=repeats)

    # which offline data are we loading
    if args.program.endswith("v1-1"):
      dataset = "/home/luke/luke-gripper-mujoco/rl/models/dqn/11-08-23/operator-PC_16:33_A1"
      net = "CNN2_100_100"
    elif args.program.endswith("v1-2"):
      dataset = "/home/luke/luke-gripper-mujoco/rl/models/dqn/18-08-23/operator-PC_16:32_A1"
      net = "CNN2_100_100"
    elif args.program.endswith("v1-3"):
      dataset = "/home/luke/luke-gripper-mujoco/rl/models/dqn/18-08-23/operator-PC_16:32_A3"
      net = "CNN2_50_50"

    baseline_args = {
      "network" : net,
      "sensor_steps" : 3,
      "state_steps" : 3,
    }

    # use the new object set
    model.params.object_set = "set8_fullset_1500"

    # HARDCODED PARAMETERS BEWARE
    model.params.no_sensor_data = False
    model.params.offline_use_cql = False # test no CQL!
    iter_per_file = 125
    random_order = False
    epochs = 2
    file_cap = None

    # normal testing
    model.params.test_freq = 4000
    model.params.save_freq = 4000

    if not args.print and not args.print_results:

      model = baseline_settings(model, **baseline_args)
      model.train_offline(dataset, iter_per_file=iter_per_file, random_order=random_order, 
                          epochs=epochs, file_cap=file_cap)

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

      # optional: enable this for some trainings for monitoring
      if inputarg % 3 == 1:
        model.params.test_freq = 500 # close eye on performance

      # now proceed with online training
      model.train()

      # test
      model = test(model, trials_per_obj=10, heuristic=args.heuristic, demo=args.demo)

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

      exit()

  elif args.program == "cnn_trial_4":

    vary_1 = [
      "CNN_25_25",
      # "CNN_50_50"
    ]
    vary_2 = [5e-6, 5e-5, 5e-4]
    vary_3 = None
    repeats = 3
    param_1_name = "network"
    param_2_name = "learning rate"
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    baseline_args = {
      "network" : param_1,
      "lr" : param_2,
      "action_values" : [2e-3, 0.015, 4e-3, 2e-3]
    }

    # use the new object set
    model.params.object_set = "set8_fullset_1500"

    # very long trainings
    model.params.num_episodes = 200_000

  elif args.program == "larger_actions":

    vary_1 = None
    vary_2 = None
    vary_3 = None
    repeats = 10
    param_1_name = None
    param_2_name = None
    param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)

    baseline_args = {
      "action_values" : [2e-3, 0.015, 4e-3, 2e-3]
    }

    # use the new object set
    model.params.object_set = "set8_fullset_1500"

    # very long trainings
    model.params.num_episodes = 100_000

  elif args.program == "test_training_manager":

    from TrainingManager import TrainingManager
    from agents.DQN import Agent_DQN

    # key settings
    rngseed = None
    device = "cpu"
    log_level = 1

    run_name = f"luke-PC_{save_suffix}"
    group_name = timestamp[:8] # include only day-month-year

    # create the training manager
    tm = TrainingManager(rngseed=rngseed, device=device, log_level=log_level,
                         group_name=group_name, run_name=run_name)

    # # choose settings
    tm.settings["print_avg_return"] = True
    # tm.settings["trainer"]["num_episodes"] = 15
    # tm.settings["trainer"]["test_freq"] = 5
    # tm.settings["trainer"]["save_freq"] = 5
    # tm.settings["final_test_trials_per_object"] = 1
    # tm.settings["env"]["test_objects"] = 3
    # tm.settings["env"]["max_episode_steps"] = 1
    # tm.settings["episode_log_rate"] = 1
    # tm.settings["track_avg_num"] = 5

    # create the environment
    env = tm.make_env()

    # make the agent
    layers = [env.n_obs, 150, 100, 50, env.n_actions]
    network = networks.VariableNetwork(layers, device=device)
    agent = Agent_DQN(device=device)
    agent.init(network)

    # input into the training manager and train
    tm.run_training(agent, env)
    print_time_taken()
    exit()

  elif args.program == "debug_tm":

    from TrainingManager import TrainingManager
    from agents.DQN import Agent_DQN

    # key settings
    rngseed = None
    device = "cpu"
    log_level = 1

    # create the training manager
    tm = TrainingManager(rngseed=rngseed, device=device, log_level=log_level)
    tm.set_group_run_name(job_num=inputarg, timestamp=timestamp)

    vary_1 = [100, 200]
    vary_2 = [250, 500]
    vary_3 = None
    repeats = 3
    tm.param_1_name = "target_update"
    tm.param_2_name = "eps_decay"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(inputarg, param_1=vary_1, param_2=vary_2,
                                                param_3=vary_3, repeats=repeats)
    
    tm.settings["Agent_DQN"]["target_update"] = tm.param_1
    tm.settings["Agent_DQN"]["eps_decay"] = tm.param_2

    # # choose settings
    tm.settings["save"] = True
    tm.settings["trainer"]["num_episodes"] = 15
    tm.settings["trainer"]["test_freq"] = 5
    tm.settings["trainer"]["save_freq"] = 5
    tm.settings["final_test_trials_per_object"] = 1
    tm.settings["env"]["test_objects"] = 3
    tm.settings["env"]["max_episode_steps"] = 1
    tm.settings["episode_log_rate"] = 5
    tm.settings["track_avg_num"] = 3
    tm.settings["Agent_DQN"]["target_update"] = 10

    # create the environment
    env = tm.make_env()

    # make the agent
    layers = [env.n_obs, 10, env.n_actions]
    network = networks.VariableNetwork(layers, device=device)
    agent = Agent_DQN(device=device)
    agent.init(network)

    # input into the training manager and train
    tm.run_training(agent, env)
    print_time_taken()
    exit()

  elif args.program == "baseline_settings":

    # placeholder program where only the current baseline settings are used
    param_1_name = "baseline settings"
    param_2_name = None
    param_3_name = None
    param_1 = True
    param_2 = None
    param_3 = None
    baseline_args = {} # use only baseline settings

  elif args.program == "example_template":

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

  else: raise RuntimeError(f"array_training_DQN.py: args.program of '{args.program}' not recognised")

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
    else:
      model.modelsaver.new_folder(name="heuristic/" + model.run_name, notimestamp=True)
      model.save_hyperparameters()

    # test
    model = test(model, trials_per_obj=10, heuristic=args.heuristic, demo=args.demo)

    # finishing time, how long did everything take
    print_time_taken()

  # prepare to print final results, check if files exist which recorded best performance
  best_sr, best_ep = model.read_best_performance_from_text(silence=True)
  test_performance = model.read_test_performance()
  if best_sr is not None:
    extra_info_string += f"\tTraining time best success rate = {best_sr} at episode = {best_ep}; trained up to episode = {int(test_performance[0, -1])}\n"
  full_sr, _ = model.read_best_performance_from_text(silence=True, fulltest=True, heuristic=args.heuristic)
  if full_sr is not None:
    extra_info_string += f"\tFinal full test success rate = {full_sr}\n"

  if args.print_results:

    fileroot = model.savedir + model.group_name
    if args.heuristic: fileroot += "/heuristic"
    filepath = fileroot + "/results.txt"
    new_file_text = f"Input arg {args.job}; timestamp={timestamp}\n"
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
  