#!/usr/bin/env python3

# fix for cluster, numpy causes segfault
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
from datetime import datetime
from dataclasses import dataclass

from Trainer import MujocoTrainer
from env.MjEnv import MjEnv
from agents.DQN import Agent_DQN
import networks

datestr = "%d-%m-%y_%H-%M" # all date inputs must follow this format

class TrainingManager():

  # baseline settings for all components
  settings = {

    # trainer settings
    "trainer" : {
      "num_episodes" : 60_000,
      "test_freq" : 4000,
      "save_freq" : 4000,
    },

    # agent hyperparameters
    "Agent_DQN" : {
      "learning_rate" : 5e-5,
      "gamma" : 0.999,
      "batch_size" : 128,
      "eps_start" : 0.9,
      "eps_end" : 0.05,
      "eps_decay" : 4000,
      "target_update" : 50,
      "optimiser" : "adam",
      "adam_beta1" : 0.9,
      "adam_beta2" : 0.999,
      "min_memory_replay" : 5000,
      "memory_replay" : 75_000,
      "soft_target_update" : False,
      "soft_target_tau" : 0.05,
      "grad_clamp_value" : 1.0,
      "loss_criterion" : "smoothL1Loss"
    },

    # environment hyperparameters
    "env" : {
      "max_episode_steps" : 250,
      "object_position_noise_mm" : 10,
      "object_rotation_noise_deg" : 5,
      "test_obj_per_file" : 20,
      "task_reload_chance" : 0.05,
      "test_trials_per_object" : 3,
      "test_objects" : 100,
      "object_set_name" : "set8_fullset_1500",
      "num_segments" : 8,
      "finger_thickness" : 0.9e-3,
      "finger_length" : 235e-3,
      "finger_width" : 28e-3,
      "finger_modulus" : 193e9,
      "depth_camera" : False,
      "XY_base_actions" : False,
      "fixed_finger_hook" : True,
      "finger_hook_angle_degrees" : 90.0,
      "finger_hook_length" : 35e-3,
      "segment_inertia_scaling" : 50.0,
      "fingertip_clearance" : 0.01,
    },

    # cpp simulation settings
    "cpp" : {
      "randomise_colours" : False,
      "time_for_action" : 0.2,
      "saturation_yield_factor" : 1.0,
      "sensor_sample_mode" : 2,
      "state_sample_mode" : 4,
      "sensor_n_prev_steps" : 3,
      "state_n_prev_steps" : 3,
      "sensor_noise_mu" : 0.05,
      "sensor_noise_std" : 0.025,
      "state_noise_mu" : 0.025,
      "state_noise_std" : 0.0,
      "oob_distance" : 75e-3,
      "done_height" : 15e-3,
      "stable_finger_force" : 1.0,
      "stable_palm_force" : 1.0,
      "stable_finger_force_lim" : 100,
      "stable_palm_force_lim" : 100,
      "fingertip_min_mm" : -12.5, # below (from start position) sets within_limits=false;
      "action" : {
        "continous" : False,
        "gripper_prismatic_X" : { 
          "in_use" : True, 
          "value" : 1e-3,
          "sign" : -1
        },
        "gripper_revolute_Y" : { 
          "in_use" : True, 
          "value" : 0.01,
          "sign" : -1
        },
        "gripper_Z" : { 
          "in_use" : True, 
          "value" : 2e-3,
          "sign" : 1
        },
        "base_X" : { 
          "in_use" : False, 
          "value" : 2e-3,
          "sign" : 1
        },
        "base_Y" : { 
          "in_use" : False, 
          "value" : 2e-3,
          "sign" : 1
        },
        "base_Z" : { 
          "in_use" : True, 
          "value" : 2e-3,
          "sign" : 1
        },
      },
      "sensor" : {
        # to override noise, use list [mu, std] instead of None
        "motor_state_sensor" : {
          "in_use" : True,
          "normalise" : 0.0,
          "read_rate" : -1,
          "noise_override" : None
        },
        "base_state_sensor_Z" : {
          "in_use" : True,
          "normalise" : 0.0,
          "read_rate" : -1,
          "noise_override" : None
        },
        "base_state_sensor_XY" : {
          "in_use" : False,
          "normalise" : 0.0,
          "read_rate" : -1,
          "noise_override" : None
        },
        "bending_gauge" : {
          "in_use" : True,
          "normalise" : 20.0, # auto generated at runtime in cpp
          "read_rate" : 10,
          "noise_override" : None
        },
        "palm_sensor" : {
          "in_use" : True,
          "normalise" : 10.0,
          "read_rate" : 10,
          "noise_override" : None
        },
        "wrist_sensor_XY" : {
          "in_use" : False,
          "normalise" : 5.0,
          "read_rate" : 10,
          "noise_override" : None
        },
        "wrist_sensor_Z" : {
          "in_use" : True,
          "normalise" : 10.0,
          "read_rate" : 10,
          "noise_override" : None
        }
      }
    },

    # this class reward settings
    "reward_style" : "sensor_mixed_v1",
    "reward_options" : [],
    "scale_rewards" : 1.0,
    "scale_penalties" : 1.0,
    "penalty_termination" : False,
    
    # this class other settings
    "episode_log_rate" : 250,
    "track_avg_num" : 250,
    "print_avg_return" : False,
    "savedir" : "models",
    "save" : True,
    "plot" : False,
    "render" : False,
    "final_test_trials_per_object" : 10
  }

  def __init__(self, rngseed=None, device="cpu", log_level=1,
               group_name="default_%d-%m-%y", run_name="default_run_%H-%M"):
    """
    Class to launch trainings ensuring baseline parameters, manage batches of
    multiple trainings, and handle printing results
    """

    self.log_level = log_level
    self.group_name = group_name
    self.run_name = run_name
    self.device = device

    # are we strictly handling seeding
    self.rngseed = rngseed
    if rngseed:
      self.strict_seed = True
      torch.manual_seed(rngseed)
    else: self.strict_seed = False

  def run_training(self, agent, env):
    """
    Initialise the agent, apply settings, and make everything ready for training
    """

    # apply agent settings and initialise
    agent = self.apply_agent_settings(agent)
    agent.set_device(self.device)

    # create the trainer
    self.trainer = MujocoTrainer(agent, env, rngseed=self.rngseed, device=self.device,
                                 log_level=self.log_level, plot=self.settings["plot"], 
                                 render=self.settings["render"], group_name=self.group_name, 
                                 run_name=self.run_name, save=self.settings["save"],
                                 savedir=self.settings["savedir"], episode_log_rate=self.settings["episode_log_rate"],
                                 strict_seed=self.strict_seed, track_avg_num=self.settings["track_avg_num"])

    # apply trainer settings
    self.trainer.params.update(self.settings["trainer"])

    # now train
    self.trainer.train()

    # finish with an in depth test
    self.run_test(trials_per_obj=self.settings["final_test_trials_per_object"])

  def run_test(self, heuristic=False, trials_per_obj=10, render=False, pause=False, demo=False, id=None,
               load=True):
    """
    Perform a thorough test on the model, including loading the best performing network
    """

    print("\nPreparing to perform a model test, heuristic =", heuristic)

    # load the best performing network
    if load and not heuristic: 
      if id is None: self.trainer.load_best_id()
      else: self.trainer.load(id=id)

    # adjust settingss
    if demo:
      self.trainer.env.params.test_trials_per_object = 1
      self.trainer.env.params.test_objects = 30
    else:
      self.trainer.env.params.test_trials_per_object = trials_per_obj
    if render: self.trainer.env.disable_rendering = False

    # perform the test
    test_data = self.trainer.test(save=False, heuristic=heuristic, pause_each_episode=pause)
    test_report = self.trainer.create_test_report(test_data, print_out=False)

    # save data to a text file
    savetxt = f"array_training_DQN.test(...) final success rate = {self.trainer.last_test_success_rate}\n"
    savetxt += "\n" + test_report
    if heuristic: savename = "heuristic_test_"
    elif demo: savename = "demo_test_"
    else: savename = "full_test_"
    currenttime = datetime.now().strftime(datestr)
    self.trainer.modelsaver.save(savename + currenttime, txtonly=True, txtstr=savetxt)

  def wipe_cpp_settings(self, env):
    """
    Wipe and reset several cpp settings to ensure we start with a blank slate
    """

    # configure auto calibrations
    env.mj.set.auto_set_timestep = True
    env.mj.set.auto_calibrate_gauges = True
    env.mj.set.auto_sim_steps = True
    env.mj.set.auto_exceed_lateral_lim = False # no longer in_use

    # general settings
    env.mj.set.debug = False
    env.mj.set.curve_validation = False
    env.mj.set.render_on_step = False

    # wipe and disable all actions
    env.mj.set.set_all_action_use(False)
    env.mj.set.set_all_action_continous(False)
    env.mj.set.set_all_action_value(0.0)
    env.mj.set.set_all_action_sign(1)

    # remove any rewards and ensure none trigger
    env.mj.set.wipe_rewards()
    env.mj.set.quit_on_reward_below = -1e6
    env.mj.set.quit_on_reward_above = 1e6
    env.mj.set.use_quit_on_reward = False
    env.mj.set.use_HER = False

    # disable use of all sensors
    env.mj.set.disable_sensors()
    env.mj.set_sensor_noise_and_normalisation_to(True) # use both noise and normalisation
    env.mj.set.sensor_n_prev_steps = 1 # lookback only 1 step
    env.mj.set.state_n_prev_steps = 1 # lookback only 1 step

    # ensure state sensors only give one reading per step (read_rate < 0)
    env.mj.set.motor_state_sensor.read_rate = -1
    env.mj.set.base_state_sensor_XY.read_rate = -1
    env.mj.set.base_state_sensor_Z.read_rate = -1

    # wipe sensor noise options
    env.mj.set.sensor_noise_mag = 0
    env.mj.set.sensor_noise_mu = 0
    env.mj.set.sensor_noise_std = 0
    env.mj.set.state_noise_mag = 0
    env.mj.set.state_noise_mu = 0
    env.mj.set.state_noise_std = 0

    return env

  def apply_agent_settings(self, agent):
    """
    Apply all the settings given in a dictionary of settings
    """

    # apply agent settings
    if agent.name == "Agent_DQN":
      agent.params.update(self.settings["Agent_DQN"])
    else:
      raise RuntimeError(f"TrainingManager.apply_trainer_agent_settings() has agent with unrecognised name = {agent.name}")
    
    return agent

  def make_env(self):
    """
    Create an MjEnv environment
    """

    env = MjEnv(log_level=self.log_level)
    env = self.configure_env(env)
    env.load()

    return env

  def apply_env_settings(self, env, set):
    """
    Apply all the settings given in a dictionary of settings
    """
    
    # apply MjEnv settings
    env.params.update(set["env"])

    # apply cpp settings - general
    env.mj.set.randomise_colours = set["cpp"]["randomise_colours"]
    env.mj.set.time_for_action = set["cpp"]["time_for_action"]
    env.mj.set.saturation_yield_factor = set["cpp"]["saturation_yield_factor"]
    env.mj.set.exceed_lat_min_factor = set["cpp"]["exceed_lat_min_factor"]
    env.mj.set.exceed_lat_max_factor = set["cpp"]["exceed_lat_max_factor"]

    env.mj.set.sensor_sample_mode = set["cpp"]["sensor_sample_mode"]
    env.mj.set.state_sample_mode = set["cpp"]["state_sample_mode"]
    env.mj.set.sensor_n_prev_steps = set["cpp"]["sensor_n_prev_steps"]
    env.mj.set.state_n_prev_steps = set["cpp"]["state_n_prev_steps"]
    env.mj.set.sensor_noise_mu = set["cpp"]["sensor_noise_mu"]
    env.mj.set.sensor_noise_std = set["cpp"]["sensor_noise_std"]
    env.mj.set.state_noise_mu = set["cpp"]["state_noise_mu"]
    env.mj.set.state_noise_std = set["cpp"]["state_noise_std"]

    env.mj.set.oob_distance = set["cpp"]["oob_distance"]
    env.mj.set.done_height = set["cpp"]["done_height"]
    env.mj.set.stable_finger_force = set["cpp"]["stable_finger_force"]
    env.mj.set.stable_palm_force = set["cpp"]["stable_palm_force"]
    env.mj.set.stable_finger_force_lim = set["cpp"]["stable_finger_force_lim"]
    env.mj.set.stable_palm_force_lim = set["cpp"]["stable_palm_force_lim"]
    env.mj.set.fingertip_min_mm = set["cpp"]["fingertip_min_mm"]

    # apply cpp settings - actions
    env.mj.set.gripper_prismatic_X.in_use = set["cpp"]["action"]["gripper_prismatic_X"]["in_use"]
    env.mj.set.gripper_revolute_Y.in_use = set["cpp"]["action"]["gripper_revolute_Y"]["in_use"]
    env.mj.set.gripper_Z.in_use = set["cpp"]["action"]["gripper_Z"]["in_use"]
    env.mj.set.base_X.in_use = set["cpp"]["action"]["base_X"]["in_use"]
    env.mj.set.base_Y.in_use = set["cpp"]["action"]["base_Y"]["in_use"]
    env.mj.set.base_Z.in_use = set["cpp"]["action"]["base_Z"]["in_use"]

    env.mj.set.set_all_action_continous(set["cpp"]["action"]["continous"])

    env.mj.set.gripper_prismatic_X.value = set["cpp"]["action"]["gripper_prismatic_X"]["value"]
    env.mj.set.gripper_revolute_Y.value = set["cpp"]["action"]["gripper_revolute_Y"]["value"]
    env.mj.set.gripper_Z.value = set["cpp"]["action"]["gripper_Z"]["value"]
    env.mj.set.base_X.value = set["cpp"]["action"]["base_X"]["value"]
    env.mj.set.base_Y.value = set["cpp"]["action"]["base_Y"]["value"]
    env.mj.set.base_Z.value = set["cpp"]["action"]["base_Z"]["value"]

    env.mj.set.gripper_prismatic_X.sign = set["cpp"]["action"]["gripper_prismatic_X"]["sign"]
    env.mj.set.gripper_revolute_Y.sign = set["cpp"]["action"]["gripper_revolute_Y"]["sign"]
    env.mj.set.gripper_Z.sign = set["cpp"]["action"]["gripper_Z"]["sign"]
    env.mj.set.base_X.sign = set["cpp"]["action"]["base_X"]["sign"]
    env.mj.set.base_Y.sign = set["cpp"]["action"]["base_Y"]["sign"]
    env.mj.set.base_Z.sign = set["cpp"]["action"]["base_Z"]["sign"]

    # apply cpp settings - sensors
    env.mj.set.motor_state_sensor.in_use = set["cpp"]["sensor"]["motor_state_sensor"]["in_use"]
    env.mj.set.base_state_sensor_Z.in_use = set["cpp"]["sensor"]["base_state_sensor_Z"]["in_use"]
    env.mj.set.base_state_sensor_XY.in_use = set["cpp"]["sensor"]["base_state_sensor_XY"]["in_use"]
    env.mj.set.bending_gauge.in_use = set["cpp"]["sensor"]["bending_gauge"]["in_use"]
    env.mj.set.palm_sensor.in_use = set["cpp"]["sensor"]["palm_sensor"]["in_use"]
    env.mj.set.wrist_sensor_XY.in_use = set["cpp"]["sensor"]["wrist_sensor_XY"]["in_use"]
    env.mj.set.wrist_sensor_Z.in_use = set["cpp"]["sensor"]["wrist_sensor_Z"]["in_use"]

    env.mj.set.motor_state_sensor.normalise = set["cpp"]["sensor"]["motor_state_sensor"]["normalise"]
    env.mj.set.base_state_sensor_Z.normalise = set["cpp"]["sensor"]["base_state_sensor_Z"]["normalise"]
    env.mj.set.base_state_sensor_XY.normalise = set["cpp"]["sensor"]["base_state_sensor_XY"]["normalise"]
    env.mj.set.bending_gauge.normalise = set["cpp"]["sensor"]["bending_gauge"]["normalise"]
    env.mj.set.palm_sensor.normalise = set["cpp"]["sensor"]["palm_sensor"]["normalise"]
    env.mj.set.wrist_sensor_XY.normalise = set["cpp"]["sensor"]["wrist_sensor_XY"]["normalise"]
    env.mj.set.wrist_sensor_Z.normalise = set["cpp"]["sensor"]["wrist_sensor_Z"]["normalise"]

    env.mj.set.motor_state_sensor.read_rate = set["cpp"]["sensor"]["motor_state_sensor"]["read_rate"]
    env.mj.set.base_state_sensor_Z.read_rate = set["cpp"]["sensor"]["base_state_sensor_Z"]["read_rate"]
    env.mj.set.base_state_sensor_XY.read_rate = set["cpp"]["sensor"]["base_state_sensor_XY"]["read_rate"]
    env.mj.set.bending_gauge.read_rate = set["cpp"]["sensor"]["bending_gauge"]["read_rate"]
    env.mj.set.palm_sensor.read_rate = set["cpp"]["sensor"]["palm_sensor"]["read_rate"]
    env.mj.set.wrist_sensor_XY.read_rate = set["cpp"]["sensor"]["wrist_sensor_XY"]["read_rate"]
    env.mj.set.wrist_sensor_Z.read_rate = set["cpp"]["sensor"]["wrist_sensor_Z"]["read_rate"]

    if set["cpp"]["sensor"]["motor_state_sensor"]["noise_override"] is not None:
      env.mj.set.motor_state_sensor.set_gaussian_noise(*set["cpp"]["sensor"]["motor_state_sensor"]["noise_override"])
    if set["cpp"]["sensor"]["base_state_sensor_Z"]["noise_override"] is not None:
      env.mj.set.base_state_sensor_Z.set_gaussian_noise(*set["cpp"]["sensor"]["base_state_sensor_Z"]["noise_override"])
    if set["cpp"]["sensor"]["base_state_sensor_XY"]["noise_override"] is not None:
      env.mj.set.base_state_sensor_XY.set_gaussian_noise(*set["cpp"]["sensor"]["base_state_sensor_XY"]["noise_override"])
    if set["cpp"]["sensor"]["bending_gauge"]["noise_override"] is not None:
      env.mj.set.bending_gauge.set_gaussian_noise(*set["cpp"]["sensor"]["bending_gauge"]["noise_override"])
    if set["cpp"]["sensor"]["palm_sensor"]["noise_override"] is not None:
      env.mj.set.palm_sensor.set_gaussian_noise(*set["cpp"]["sensor"]["palm_sensor"]["noise_override"])
    if set["cpp"]["sensor"]["wrist_sensor_XY"]["noise_override"] is not None:
      env.mj.set.wrist_sensor_XY.set_gaussian_noise(*set["cpp"]["sensor"]["wrist_sensor_XY"]["noise_override"])
    if set["cpp"]["sensor"]["wrist_sensor_Z"]["noise_override"] is not None:
      env.mj.set.wrist_sensor_Z.set_gaussian_noise(*set["cpp"]["sensor"]["wrist_sensor_Z"]["noise_override"])

    # rewards are the exception, they are not set in this function

    return env

  def configure_env(self, env):
    """
    Apply all settings and configurations to the environment
    """

    env = self.wipe_cpp_settings(env)
    env = self.apply_env_settings(env, self.settings)
    env = self.create_reward_function(env, self.settings["reward_style"],
                                      self.settings["reward_options"],
                                      self.settings["scale_rewards"],
                                      self.settings["scale_penalties"],
                                      self.settings["penalty_termination"])
    
    return env

  def set_sensor_reward_thresholds(self, env, exceed_style=None, min_style=None):
    """
    Determine the reward thresholds
    """

    printout = True

    @dataclass
    class RewardThresholds:
      # m=minimum, g=good, x=exceed, d=dangerous
      mBend = 0.0
      gBend = env.mj.set.stable_finger_force
      xBend = env.mj.set.stable_finger_force_lim
      dBend = env.yield_load()

      mPalm = 0.0
      gPalm = env.mj.set.stable_palm_force
      xPalm = env.mj.set.stable_palm_force_lim
      dPalm = 30.0

      xWrist = 5.0
      dWrist = 10.0

    self.RT = RewardThresholds()

    # check if minimum handling is specified
    if isinstance(min_style, float):
      self.RT.mBend = min_style
      self.RT.mPalm = min_style
    elif isinstance(min_style, list) and len(min_style) == 2:
      self.RT.mBend = min_style[0]
      self.RT.mPalm = min_style[1]
    elif min_style == "binary":
      self.RT.mBend = self.RT.gBend
      self.RT.mPalm = self.RT.gPalm
    elif min_style == "middle":
      self.RT.mBend = 0.5 * self.RT.gBend
      self.RT.mPalm = 0.5 * self.RT.gPalm
    elif min_style is not None: 
      raise RuntimeError(f"set_sensor_reward_thresholds() got invalid 'min_style' of {min_style}")

    # check if we are given how 'exceed bend' and 'exceed palm' should work
    if isinstance(exceed_style, float):
      self.RT.xBend = exceed_style
      self.RT.xPalm = exceed_style
    elif isinstance(exceed_style, list) and len(exceed_style) == 2:
      self.RT.xBend = exceed_style[0]
      self.RT.xPalm = exceed_style[1]
    elif exceed_style == "dangerous":
      self.RT.xBend = self.RT.dBend
      self.RT.xPalm = self.RT.dPalm
    elif exceed_style == "middle":
      self.RT.xBend = self.RT.gBend + 0.5 * (self.RT.dBend - self.RT.gBend)
      self.RT.xPalm = self.RT.gPalm + 0.5 * (self.RT.dPalm - self.RT.gPalm)
    elif exceed_style == "factor_0.8":
      self.RT.xBend = self.RT.gBend + 0.8 * (self.RT.dBend - self.RT.gBend)
      self.RT.xPalm = self.RT.gPalm + 0.8 * (self.RT.dPalm - self.RT.gPalm)
    elif exceed_style is not None: 
      raise RuntimeError(f"set_sensor_reward_thresholds() got invalid 'exceed_style' of {exceed_style}")

    # confirm that the thresholds make sense
    if self.RT.mBend > self.RT.gBend:
      raise RuntimeError(f"set_sensor_reward_thresholds() finds mBend > gBend, {self.RT.mBend:.3f} > {self.RT.gBend:.3f}")
    if self.RT.mPalm > self.RT.gPalm:
      raise RuntimeError(f"set_sensor_reward_thresholds() finds mPalm > gPalm, {self.RT.mPalm:.3f} > {self.RT.gPalm:.3f}")
    if self.RT.gBend > self.RT.xBend:
      raise RuntimeError(f"set_sensor_reward_thresholds() finds gBend > xBend, {self.RT.gBend:.3f} > {self.RT.xBend:.3f}")
    if self.RT.gPalm > self.RT.xPalm:
      raise RuntimeError(f"set_sensor_reward_thresholds() finds gPalm > xPalm, {self.RT.gPalm:.3f} > {self.RT.xPalm:.3f}")
    if self.RT.xBend > self.RT.dBend:
      raise RuntimeError(f"set_sensor_reward_thresholds() finds xBend > dBend, {self.RT.xBend:.3f} > {self.RT.dBend:.3f}")
    if self.RT.xPalm > self.RT.dPalm:
      raise RuntimeError(f"set_sensor_reward_thresholds() finds xPalm > dPalm, {self.RT.xPalm:.3f} > {self.RT.dPalm:.3f}")
    if self.RT.xWrist > self.RT.dWrist:
      raise RuntimeError(f"set_sensor_reward_thresholds() finds xWrist > dWrist, {self.RT.xWrist:.3f} > {self.RT.dWrist:.3f}")

    if printout:
      print("Reward Thresholds\n")
      print(f"  -> mBend = {self.RT.mBend:.3f}")
      print(f"  -> gBend = {self.RT.gBend:.3f}")
      print(f"  -> xBend = {self.RT.xBend:.3f}")
      print(f"  -> dBend = {self.RT.dBend:.3f}\n")
      print(f"  -> mPalm = {self.RT.mPalm:.3f}")
      print(f"  -> gPalm = {self.RT.gPalm:.3f}")
      print(f"  -> xPalm = {self.RT.xPalm:.3f}")
      print(f"  -> dPalm = {self.RT.dPalm:.3f}\n")
      print(f"  -> xWrist = {self.RT.xWrist:.3f}")
      print(f"  -> dWrist = {self.RT.dWrist:.3f}\n")

  def set_sensor_bonuses(self, env, value):
    """
    Set bonus rewards with a given value
    """

    # rewards                             reward   done   trigger    min             max      overshoot
    env.mj.set.lifted.set           (value,  False,   1)
    env.mj.set.target_height.set    (value,  False,   1)
    env.mj.set.object_stable.set    (value,  False,   1)
    env.mj.set.good_bend_sensor.set (value,  False,   1,     self.RT.mBend, self.RT.gBend,  -1)
    env.mj.set.good_palm_sensor.set (value,  False,   1,     self.RT.mPalm, self.RT.gPalm,  -1)

    return env

  def set_sensor_penalties(self, env, value):
    """
    Set penalty rewards with given value, alongside defaults
    """

    # penalties                              reward   done   trigger  min               max        overshoot
    env.mj.set.exceed_limits.set       (value,  False,    1)
    env.mj.set.exceed_bend_sensor.set  (value,  False,    1,    self.RT.xBend,  self.RT.dBend,  -1)
    env.mj.set.exceed_palm_sensor.set  (value,  False,    1,    self.RT.xPalm,  self.RT.dPalm,  -1)
    env.mj.set.exceed_wrist_sensor.set (value,  False,    1,    self.RT.xWrist, self.RT.dWrist, -1)

    return env

  def set_sensor_terminations(self, env, value=-1.0, done=True, trigger=1):
    """
    Set terminations based on dangerous sensor readings
    """

    # terminations                              reward   done   trigger  min               max       overshoot
    env.mj.set.dangerous_bend_sensor.set  (value,  done,  trigger, self.RT.dBend,  self.RT.dBend,  -1)
    env.mj.set.dangerous_palm_sensor.set  (value,  done,  trigger, self.RT.dPalm,  self.RT.dPalm,  -1)
    env.mj.set.dangerous_wrist_sensor.set (value,  done,  trigger, self.RT.dWrist, self.RT.dWrist, -1)

    return env
  
  def create_reward_function(self, env, style, options=[], scale_rewards=1, scale_penalties=1,
                             penalty_termination=False):
    """
    Set the reward structure for the learning, with different style options
    """

    if style == "sensor_mixed_v1":
      # prepare reward thresholds
      if (env.mj.set.stable_finger_force_lim > 99.0 and
          env.mj.set.stable_palm_force_lim > 99.0):
        exceed_style = [3.0, 10.0]
      else: exceed_style = None
      self.set_sensor_reward_thresholds(env, exceed_style=exceed_style, min_style=None)
      # reward each step                     reward   done   trigger
      env.mj.set.step_num.set          (-0.01,  False,   1)
      # penalties and bonuses
      env = self.set_sensor_bonuses(env, 0.002 * scale_rewards)
      env = self.set_sensor_penalties(env, -0.002 * scale_penalties)
      # scale based on steps allowed per episode
      env.mj.set.scale_rewards(100 / env.params.max_episode_steps)
      # end criteria                         reward   done   trigger
      env.mj.set.stable_height.set     (1.0,    True,    1)
      env.mj.set.oob.set               (-1.0,   True,    1)
      if penalty_termination:
        env = self.set_sensor_terminations(env)
    
    else:
      raise RuntimeError(f"style={style} is not a valid option in TrainingManager.create_reward_function()")

    # specific options
    if "terminate_on_exceed_limits" in options:
      # reward each step                     reward   done   trigger
      env.mj.set.exceed_limits.set     (-1.0,   True,    3)

    # termination on specific reward
    env.mj.set.quit_on_reward_below = -1.0 if "neg_cap" in options else -1e6
    env.mj.set.quit_on_reward_above = +1.0 if "pos_cap" in options else 1e6
    env.mj.set.use_quit_on_reward = True

    return env

if __name__ == "__main__":

  # key settings
  rngseed = None
  device = "cpu"
  log_level = 1
  save = False

  # create the training manager
  tm = TrainingManager(rngseed=rngseed, device=device, log_level=log_level)

  # # choose settings
  tm.settings["save"] = save
  # tm.settings["trainer"]["num_episodes"] = 15
  # tm.settings["trainer"]["test_freq"] = 5
  # tm.settings["trainer"]["save_freq"] = 5
  # tm.settings["final_test_trials_per_object"] = 1
  # tm.settings["env"]["test_objects"] = 3
  tm.settings["env"]["max_episode_steps"] = 1
  tm.settings["episode_log_rate"] = 5
  # tm.settings["track_avg_num"] = 3
  tm.settings["Agent_DQN"]["target_update"] = 10

  # create the environment
  env = tm.make_env()

  # make the agent
  layers = [env.n_obs, 150, 100, 50, env.n_actions]
  network = networks.VariableNetwork(layers, device=device)
  agent = Agent_DQN(device=device)
  agent.init(network)

  # input into the training manager and train
  tm.run_training(agent, env)

