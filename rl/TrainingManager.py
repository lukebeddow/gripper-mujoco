#!/usr/bin/env python3

import torch
import numpy as np
from datetime import datetime
from dataclasses import dataclass
import yaml
import os
import functools

from Trainer import MujocoTrainer
from env.MjEnv import MjEnv
from agents.DQN import Agent_DQN
from agents.ActorCritic import Agent_SAC
from agents.PolicyGradient import Agent_PPO
import networks

class TrainingManager():

  # baseline settings for all components
  settings = {

    # trainer settings
    "trainer" : {
      "num_episodes" : 120_000,
      "test_freq" : 5000,
      "save_freq" : 5000,
      "use_curriculum" : False,
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

    "Agent_SAC" : {
      "learning_rate" : 5e-5,
      "gamma" : 0.999,
      "alpha" : 0.2,
      "batch_size" : 128,
      "update_after_steps" : 1000,
      "update_every_steps" : 50,
      "random_start_episodes" : 1000,
      "optimiser" : "adam",
      "adam_beta1" : 0.9,
      "adam_beta2" : 0.999,
      "min_memory_replay" : 5000,
      "memory_replay" : 75_000,
      "soft_target_tau" : 0.05,
    },

    "Agent_PPO" : {
      "learning_rate_pi" : 5e-5,
      "learning_rate_vf" : 5e-5,
      "gamma" : 0.99,
      "steps_per_epoch" : 6000,
      "clip_ratio" : 0.2,
      "train_pi_iters" : 80,
      "train_vf_iters" : 80,
      "lam" : 0.97,
      "target_kl" : 0.01,
      "max_kl_ratio" : 1.5,
      "use_random_action_noise" : True,
      "random_action_noise_size" : 0.05,
      "optimiser" : "adam",
      "adam_beta1" : 0.9,
      "adam_beta2" : 0.999,
      "grad_clamp_value" : None,
    },

    "Agent_PPO_Discriminator" : {
      "learning_rate_discrim" : 5e-5,
      "loss_criterion_discrim" : "MSELoss",
      "learning_rate_pi" : 5e-5,
      "learning_rate_vf" : 5e-5,
      "gamma" : 0.99,
      "steps_per_epoch" : 6000,
      "clip_ratio" : 0.2,
      "train_pi_iters" : 80,
      "train_vf_iters" : 80,
      "lam" : 0.97,
      "target_kl" : 0.01,
      "max_kl_ratio" : 1.5,
      "use_random_action_noise" : True,
      "random_action_noise_size" : 0.05,
      "optimiser" : "adam",
      "adam_beta1" : 0.9,
      "adam_beta2" : 0.999,
    },

    # environment hyperparameters
    "env" : {

      # training parameters
      "max_episode_steps" : 250,
      "object_position_noise_mm" : 10,
      "object_rotation_noise_deg" : 5,
      "base_lim_X_mm" : 300,
      "base_lim_Y_mm" : 200,
      "base_lim_Z_mm" : 30,
      "base_lim_yaw_rad" : np.pi / 4,

      # camera grasping settings
      "use_rgb_in_observation" : False,
      "use_depth_in_observation" : False,
      "use_rgb_rendering" : False,
      "rgb_rendering_method" : "CUT",
      "image_width" : 50,
      "image_height" : 50,

      # image preprocessing settings
      "use_standard_transform": False,
      "transform_resize_square": 144,
      "transform_crop_size": 128,

      # grasping sene parameters
      "use_scene_settings" : False,
      "num_objects_in_scene" : 1,
      "scene_grasp_target" : 1,
      "origin_noise_X_mm" : 150,
      "origin_noise_Y_mm" : 50,
      "scene_X_dimension_mm" : 300,
      "scene_Y_dimension_mm" : 200, 

      # file and testing parameters
      "test_obj_per_file" : 20,
      "task_reload_chance" : 0.05,
      "test_trials_per_object" : 3,
      "test_objects" : 100,

      # model parameters (for loading xml files)
      "object_set_name" : "set9_nosharp_smallspheres",
      "num_segments" : 8,
      "finger_thickness" : 0.9e-3,
      "finger_length" : 235e-3,
      "finger_width" : 28e-3,
      "finger_modulus" : 193e9,
      "depth_camera" : False,
      "XY_base_actions" : False,
      "Z_base_rotation" : False,
      "fixed_finger_hook" : True,
      "finger_hook_angle_degrees" : 75.0,
      "finger_hook_length" : 35e-3,
      "segment_inertia_scaling" : 50.0,
      "fingertip_clearance" : 0.01,

    },

    # cpp simulation settings
    "cpp" : {
      "randomise_colours" : False,
      "time_for_action" : 0.2,
      "saturation_yield_factor" : 1.5,
      "sensor_sample_mode" : 2,
      "state_sample_mode" : 4,
      "sensor_n_prev_steps" : 3,
      "state_n_prev_steps" : 3,
      "sensor_noise_mu" : 0.05,
      "sensor_noise_std" : 0.025,
      "state_noise_mu" : 0.025,
      "state_noise_std" : 0.0,
      "base_position_noise" : 5e-3,
      "oob_distance" : 75e-3,
      "lift_height" : 15e-3,
      "gripper_target_height" : 20e-3,
      "stable_finger_force" : 1.0,
      "stable_palm_force" : 1.0,
      "stable_finger_force_lim" : 4.0,
      "stable_palm_force_lim" : 4.0,
      "XY_distance_threshold" : 10e-3,
      "cap_reward" : False,
      "fingertip_min_mm" : -12.5, # below (from start position) sets within_limits=false;
      "continous_actions" : True,
      "use_termination_action" : False,
      "termination_threshold" : 0.9,
      "action" : {
        "gripper_prismatic_X" : { 
          "in_use" : True, 
          "value" : 2e-3, # value = size of discrete, max size of continous
          "sign" : -1
        },
        "gripper_revolute_Y" : { 
          "in_use" : True, 
          "value" : 0.015,
          "sign" : -1
        },
        "gripper_Z" : { 
          "in_use" : True, 
          "value" : 4e-3,
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
        "base_yaw" : { 
          "in_use" : False, 
          "value" : 5e-3,
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
          "noise_override" : [0.1, 0.0]
        },
        "base_state_sensor_XY" : {
          "in_use" : False,
          "normalise" : 0.0,
          "read_rate" : -1,
          "noise_override" : None
        },
        "base_state_sensor_yaw" : {
          "in_use" : True,
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
          "normalise" : 6.0,
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
    "reward" : {
      "style" : "sensor_mixed_v1",
      "options" : [],
      "scale_rewards" : 1.0,
      "scale_penalties" : 1.0,
      "penalty_termination" : True,
      "stable_trigger" : 4,
      "dangerous_trigger" : 1,
      "bend" : {
        "min" : 0.0,
        "good" : "stable",        # can use "stable"
        "exceed" : "stable_lim",  # can use "stable_lim"
        "dangerous" : 5.0,        # can use "yield"
      },
      "palm" : {
        "min" : 0.0,
        "good" : "stable",        # can use "stable"
        "exceed" : "stable_lim",  # can use "stable_lim"
        "dangerous" : 5.0,
      },
      "wrist" : {
        "exceed" : 6.0,
        "dangerous" : 8.0,
      },
      "action_pen_lin" : {
        "used" : False,
        "scaling" : 2,
        "min" : 0.1,
        "max" : 2.0,
      },
      "action_pen_sq" : {
        "used" : True,
        "scaling" : 2,
        "min" : 0.1,
        "max" : 3.0,
      },
      "object_XY_distance" : {
        "used" : False,
        "min" : -50e-3,
        "max" : -10e-3,
      },
    },

    # curriculum settings
    "curriculum" : {
      "metric_name" : "",
      "metric_thresholds" : [],
      "param_values" : [],
      "change_fcn" : None,
    },
    
    # this class other settings
    "episode_log_rate" : 250,
    "track_avg_num" : 250,
    "print_avg_return" : False,
    "savedir" : "models",
    "save" : True,
    "plot" : False,
    "render" : False,
    "final_test_trials_per_object" : 10,
    "final_test_max_stage" : True,
    "final_test_only_stage" : None,
    "env_image_collection" : False,
    "env_image_collection_chance" : 0.001,
    "env_image_collection_batch_size" : 1000,
    "env_image_collection_max_batches" : 10,
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
    self.baseline_folder = "baseline_settings"
    self.baseline_file_name = "baseline_{}.yaml"

    # key training information
    self.run_name_prefix = "run"
    self.summary_filename = "training_summary.txt"
    self.summary_section_seperator = "\n--- * ----\n"
    self.datestr = "%d-%m-%y_%H-%M" # all date inputs must follow this format
    self.init_training_summary()

    # are we strictly handling seeding
    self.rngseed = rngseed
    if rngseed:
      self.strict_seed = True
      torch.manual_seed(rngseed)
    else: self.strict_seed = False

  def set_group_run_name(self, job_num=None, timestamp=None, prefix=None):
    """
    Set a default group and run name given a job number and a timestamp
    """

    if prefix is None: prefix = self.run_name_prefix
    else: self.run_name_prefix = prefix

    timestamp = timestamp if timestamp is not None else datetime.now().strftime(self.datestr)
    job_suffix = f"_A{job_num}" if job_num is not None else ""

    self.group_name = timestamp[:8]
    self.run_name = f"{prefix}_{timestamp[-5:]}{job_suffix}"

    # apply changes to any trainer we have
    if hasattr(self, "trainer"):
      self.trainer.run_name = self.run_name
      self.trainer.group_name = self.group_name

    # save details
    if job_num is not None: self.job_number = job_num
    self.timestamp = timestamp

  def save_baseline_params(self):
    """
    Save the settings dictionary as a set of baseline parameters. Checks to see if
    these settings already exist first.
    """

    create_new = True

    if not os.path.exists(self.baseline_folder):
      try:
        os.makedirs(self.baseline_folder)
      except FileExistsError:
        # file must have just been created
        pass

    baseline_files = [x for x in os.listdir(self.baseline_folder)
                      if x.endswith(".yaml")]
    
    if len(baseline_files) != 0:

      # with open(self.baseline_folder + "/temp_settings", "w") as outfile:
      #   yaml.dump(self.settings, outfile, default_flow_style=False)
      # with open(self.baseline_folder + "/temp_settings") as file:
      #   my_settings = yaml.safe_load(file)

      for f in baseline_files:
        with open(self.baseline_folder + "/" + f) as file:
          loaded_settings = yaml.safe_load(file)
        if loaded_settings == self.settings:
          create_new = False
          break

    if create_new:
      new_filename = self.baseline_file_name.format(datetime.now().strftime(self.datestr))
      if self.log_level > 0:
        print("TrainingManager.save_baseline_params(): Creating a new baseline settings yaml file:", self.baseline_folder + "/" + new_filename)
      with open(self.baseline_folder + "/" + new_filename, "w") as outfile:
        yaml.dump(self.settings, outfile, default_flow_style=False)
    else:
      if self.log_level > 0:
        print("TrainingManager.save_baseline_params(): Not creating a new baseline settings yaml file, these settings already exist")

    return create_new

  def run_training(self, agent, env):
    """
    Initialise the agent, apply settings, and make everything ready for training
    """

    # apply agent settings and initialise
    agent = self.apply_agent_settings(agent)
    agent.set_device(self.device)

    # create the trainer
    self.trainer = self.make_trainer(agent, env)

    # save initial training summary (requires creating the training folder before train())
    if self.settings["save"] and not self.trainer.modelsaver.in_folder:
      self.trainer.modelsaver.new_folder(name=self.trainer.run_name, notimestamp=True)
    self.save_training_summary(printout=True, load_existing=False)

    # now train
    self.trainer.train()

    # finish by loading the best training and running an in depth test on it
    self.run_test(trials_per_obj=self.settings["final_test_trials_per_object"],
                  load_best_id=True)
      
    # save final summary of training
    self.save_training_summary()

  def run_test(self, heuristic=False, trials_per_obj=10, render=False, pause=False,
               demo=False, different_object_set=None, load_best_id=False):
    """
    Perform a thorough test on the model, including loading the best performing network
    """

    if self.log_level > 0:
      print(f"\nPreparing to perform a model test, trials_per_obj = {trials_per_obj}, load_best_id = {load_best_id}, heuristic = {heuristic}")

    # are we loading the best performing model before this test
    if load_best_id:
      # check if we should only finalise tests on trainings that reached a certain stage
      if self.settings["final_test_only_stage"] is not None:
        stage = self.settings["final_test_only_stage"]
      elif self.settings["final_test_max_stage"]:
        stage = "max"
      else: stage = None
      # try to load the best training
      found = self.trainer.load_best_id(self.run_name, stage=stage)
      if not found:
        if self.log_level > 0:
          print(f"TrainingManager.run_test() not run, as no best id was found, likely as all success rates were zero or training did not statisfy stage = {stage}")
        return
      elif self.log_level > 0:
        print("TrainingMananger.run_test() has loaded best_id =", self.trainer.last_loaded_agent_id)

    # adjust settingss
    if demo:
      self.trainer.env.params.test_trials_per_object = 1
      self.trainer.env.params.test_objects = demo
    else:
      self.trainer.env.params.test_trials_per_object = trials_per_obj
    if render:
      self.trainer.render = True 
      self.trainer.env.render_window = True
    if different_object_set is not None and isinstance(different_object_set, str):
      if self.log_level > 0:
        print(f"Loading a different object set for the test, name = {different_object_set}")
      self.trainer.env.load(object_set_name=different_object_set)

    # perform the test
    test_data = self.trainer.test(save=False, heuristic=heuristic, pause_each_episode=pause)
    test_report = self.trainer.create_test_report(test_data, print_out=False)

    # save data to a text file
    if self.settings["save"]:
      savetxt = f"TrainingMananger.run_test() final success rate = {self.trainer.last_test_success_rate}\n"
      savetxt += "\n" + test_report
      if heuristic: savename = "heuristic_test_"
      elif demo: savename = "demo_test_"
      else: savename = "full_test_"
      if different_object_set is not None and isinstance(different_object_set, str):
        savename = different_object_set + "_" + savename
      savename += datetime.now().strftime(self.datestr)
      self.trainer.modelsaver.save(savename, txtonly=True, txtstr=savetxt)

  def continue_training(self, new_endpoint=None, extra_episodes=None):
    """
    Continue a training (already loaded), either to a new endpoint, or simply adding a
    given number of episodes. A trainer must be loaded before this function is called.
    If neither new_endpoint or extra_episodes are set, the training will continue to the
    original endpoint. set 'with_test=False' to prevent from rerunning
    """

    if not hasattr(self, "trainer") or self.trainer is None:
      raise RuntimeError("TrainingManager.continue_training() has been called but no trainer is loaded. The load() function must be run before calling this function")

    self.save_training_summary(printout=True, load_existing=False)
    self.trainer.train(num_episodes_abs=new_endpoint, num_episodes_extra=extra_episodes)
    self.run_test(trials_per_obj=self.settings["final_test_trials_per_object"],
                  load_best_id=True)
    self.save_training_summary()

  def load(self, job_num=None, timestamp=None, run_name=None, group_name=None, 
           best_id=False, id=None, path_to_run_folder=None, use_abs_path=False,
           load_into_new_training=False):
    """
    Load the training currently specified. Either pass:
      1) nothing - run_name and group_name are already set in the class
      2) job_num and timestamp
      3) run_name and group_name
      4) run_name and path_to_run_folder
    Alongside all pass either the id to load, or best_id=True to find the id with
    the best test performance and load that one.
    """

    if load_into_new_training:
      # save details which will be restored after load
      keep_job_num = self.job_number
      keep_timestamp = self.timestamp
      keep_run_name = self.run_name
      keep_group_name = self.group_name

    if job_num is not None and timestamp is not None:
      self.set_group_run_name(job_num=job_num, timestamp=timestamp)
    elif run_name is not None and group_name is not None:
      if path_to_run_folder is not None:
        if self.log_level > 0:
          print("TrainingManager.load() warning: path_to_run_folder being ignored as run_name and group_name are set")
      path_to_run_folder = None
      self.run_name = run_name
      self.group_name = group_name
    elif run_name and path_to_run_folder is not None:
      self.run_name = run_name
      if path_to_run_folder[-1] == "/": path_to_run_folder = path_to_run_folder[:-1]
      self.group_name = path_to_run_folder.split("/")[-1]
  
    # make the trainer (overwrite any existing trainer)
    self.trainer = self.make_trainer(None, self.make_env(load=False))

    # special case, we want to remake the modelsaver in load()
    if run_name and path_to_run_folder is not None:
      delattr(self.trainer, "modelsaver")

    # now load the specified model
    if best_id:
      if self.settings["final_test_only_stage"] is not None:
        stage = self.settings["final_test_only_stage"]
      elif self.settings["final_test_max_stage"]:
        stage = "max"
      else: stage = None
      found = self.trainer.load_best_id(self.run_name, group_name=self.group_name,
                                        path_to_run_folder=path_to_run_folder, stage=stage)
      if not found:
        if self.log_level > 0:
          print(f"TrainingMananger.load() warning: load_best_id failed (stage = {stage}). Loading most recent id")
        self.trainer.load(self.run_name, group_name=self.group_name, id=id,
                        path_to_run_folder=path_to_run_folder)
        # raise RuntimeError(f"TrainingMananger.load() error: load_best_id failed (stage = {stage})")
    else:
      self.trainer.load(self.run_name, group_name=self.group_name, id=id,
                        path_to_run_folder=path_to_run_folder)

    if load_into_new_training:
      # apply new training details, for saving in a new folder
      self.run_name = keep_run_name
      self.group_name = keep_group_name
      self.job_number = keep_job_num
      self.timestamp = keep_timestamp
      self.trainer.setup_saving(run_name=keep_run_name, group_name=keep_group_name)
      # save a copy of the model we just loaded in our new folder
      self.trainer.save(force_save_number=self.trainer.last_loaded_agent_id)
    else:
      # see if we can load the existing training summary
      self.load_training_summary()

  def init_training_summary(self):
    """
    Remove any existing training summary data
    """

    self.job_number = None
    self.timestamp = None
    self.program = None
    self.param_1 = None
    self.param_2 = None
    self.param_3 = None
    self.param_1_name = None
    self.param_2_name = None
    self.param_3_name = None
    self.train_best_sr = None
    self.train_best_ep = None
    self.full_test_sr = None
    self.trained_to = None

  def get_training_summary(self, filepath=None, load_existing=True):
    """
    Get a string summary of training details (not to be confused with hyperparameters,
    see Trainer.py to print or get any hyperparameters. Alternatively print the settings
    dictionary from this class)
    """

    if filepath is None:
      filepath = self.trainer.savedir + "/" + self.trainer.group_name + "/" + self.trainer.run_name

    # try to load any existing information first
    if load_existing:
      exists = self.load_training_summary(filepath=filepath)
      if not exists:
        if self.log_level >= 2:
          print("TrainingManager did not find an existing training summary to load")

    job_string = f"Job number is {self.job_number}\n" if self.job_number is not None else ""
    timestamp_string = f"\tTimestamp is {self.timestamp}\n" if self.timestamp is not None else ""
    program_string = f"\tProgram is {self.program}\n" if self.program is not None else ""
    param_1_string = f"\tParam 1: {self.param_1_name} is {self.param_1}\n" if self.param_1 is not None else ""
    param_2_string = f"\tParam 2: {self.param_2_name} is {self.param_2}\n" if self.param_2 is not None else ""
    param_3_string = f"\tParam 3: {self.param_3_name} is {self.param_3}\n" if self.param_3 is not None else ""

    traintime_test_np = self.trainer.read_test_performance()
    best_index = np.argmax(traintime_test_np[2])
    best_traintime_success = traintime_test_np[2][best_index]
    best_traintime_episode = int(traintime_test_np[1][best_index])
    best_traintime_str = f"\tBest training time test performance = {best_traintime_success} at episode = {best_traintime_episode}\n"
    trained_to_str = f"\tTrained to episode = {int(traintime_test_np[1][-1])}\n"

    # include the table of test performances
    if load_existing:
      test_table = self.trainer.read_test_performance(as_string=True)
    else: test_table = "No test table loaded as load_existing=False"

    best_fulltest_sr, best_ep = self.trainer.read_best_performance_from_text(fulltest=True, silence=True)
    if best_fulltest_sr is not None:
      fulltest_str = f"\tMost recent fulltest performance = {best_fulltest_sr:.3f}\n"
    else: fulltest_str = ""

    # assemble our output
    output = job_string + timestamp_string + program_string + param_1_string + param_2_string + param_3_string
    output += trained_to_str + best_traintime_str + fulltest_str
    output += self.summary_section_seperator
    output += "\n" + test_table

    return output

  def load_training_summary(self, filepath=None):
    """
    Check for an existing training summary to load information from. Returns True if a
    training summary is found, False if not
    """

    if filepath is None:
      filepath = self.trainer.savedir + "/" + self.trainer.group_name + "/" + self.trainer.run_name

    try:
      with open(filepath + "/" + self.summary_filename, "r") as f:
        txt = f.read()
    except FileNotFoundError as e:
      if self.log_level >= 2: print("load_training_summary() failed with error:", e)
      return False
    
    sections = txt.split(self.summary_section_seperator)
    lines = sections[0].split("\n")

    for line in lines:

      if line.startswith("Job number"):
        splits = line.split(" is ")
        # int(...) drops the newline character
        self.job_number = int(splits[-1])
          
      elif line.startswith("\t"):

        if line.startswith("\tBest training time test performance"):
          splits = line.split(" = ")
          self.train_best_sr = float(splits[1].split(" at ")[0])
          self.train_best_ep = int(splits[2])
      
        elif line.startswith("\tMost recent fulltest performance"):
          splits = line.split(" = ")
          self.full_test_sr = float(splits[-1])

        elif line.startswith("\tTrained to episode"):
          splits = line.split(" = ")
          self.trained_to = int(splits[-1])

        elif line.startswith("\tTimestamp is"):
          splits = line.split(" is ")
          self.timestamp = splits[-1].strip("\n")

        elif line.startswith("\tProgram is"):
          splits = line.split(" is ")
          self.program = splits[-1].strip("\n")

        elif line.startswith("\tParam"):
          splits = line.split(" is ")
          item = splits[-1]
          if item.endswith("\n"): item = item.strip("\n")
          if line.startswith("\tParam 1"):
            self.param_1 = item
            self.param_1_name = splits[0].split(": ")[-1]
          elif line.startswith("\tParam 2"):
            self.param_2 = item
            self.param_2_name = splits[0].split(": ")[-1]
          elif line.startswith("\tParam 3"):
            self.param_3 = item
            self.param_3_name = splits[0].split(": ")[-1]

    return True

  def save_training_summary(self, filepath=None, force=True, printout=False,
                            load_existing=True):
    """
    Save a text file summarising the whole training
    """

    if self.settings["save"] is False and force is False: 
      if self.log_level > 0:
        print("TrainingMananger.save_training_summary() warning: trainer.enable_saving = False, nothing saved")
      return

    if filepath is None:
      filepath = self.trainer.savedir + "/" + self.trainer.group_name + "/" + self.trainer.run_name

    summary_string = self.get_training_summary(filepath=filepath, load_existing=load_existing)

    if printout and self.log_level > 0:
      print(summary_string)

    # save to a textfile
    with open(filepath + "/" + self.summary_filename, 'w') as f:
      f.write(summary_string)

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
    # env.mj.set.debug = False # MjEnv handles this
    env.mj.set.curve_validation = False
    env.mj.set.render_on_step = False

    # wipe and disable all actions
    env.mj.set.set_all_action_use(False)
    env.mj.set.set_all_action_continous(False)
    env.mj.set.set_all_action_value(0.0)
    env.mj.set.set_all_action_sign(1)
    env.mj.set.use_termination_action = False

    # remove any rewards and ensure none trigger
    env.mj.set.wipe_rewards()
    env.mj.set.cap_reward = False
    env.mj.set.quit_if_cap_exceeded = False
    env.mj.set.reward_cap_lower_bound = -1e6
    env.mj.set.reward_cap_upper_bound = 1e6
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
    env.mj.set.base_position_noise = 0

    return env

  def apply_agent_settings(self, agent):
    """
    Apply all the settings given in a dictionary of settings
    """

    # # apply agent settings
    # if agent.name == "Agent_DQN":
    #   agent.params.update(self.settings["Agent_DQN"])
    # else:
    #   raise RuntimeError(f"TrainingManager.apply_trainer_agent_settings() has agent with unrecognised name = {agent.name}")
    
    # update using the agent name, which should match an entry in the settings dict
    agent.params.update(self.settings[agent.name])

    # reinitialise the agent with the new settings, but network is not reloaded
    agent.init(network="loaded")

    return agent

  def make_env(self, load=True):
    """
    Create an MjEnv environment
    """

    env = MjEnv(log_level=self.log_level, render=self.settings["render"],
                device=self.device)
    env = self.configure_env(env)
    if load: env.load()

    return env
  
  def make_trainer(self, agent=None, env=None):
    """
    Create a MujocoTrainer
    """

    trainer = MujocoTrainer(agent, env, rngseed=self.rngseed, device=self.device,
                            log_level=self.log_level, plot=self.settings["plot"], 
                            render=self.settings["render"], group_name=self.group_name, 
                            run_name=self.run_name, save=self.settings["save"],
                            savedir=self.settings["savedir"], episode_log_rate=self.settings["episode_log_rate"],
                            strict_seed=self.strict_seed, track_avg_num=self.settings["track_avg_num"])
    
    # useful warnings
    if self.log_level > 0:
      if not self.settings["save"]:
        print("TrainingMananger.make_trainer() warning: saving is disabled, not model data will be saved")

    # apply trainer settings
    trainer.params.update(self.settings["trainer"])
    trainer.curriculum_dict["metric_name"] = self.settings["curriculum"]["metric_name"]
    trainer.curriculum_dict["metric_thresholds"] = self.settings["curriculum"]["metric_thresholds"]
    trainer.curriculum_dict["param_values"] = self.settings["curriculum"]["param_values"]
    if trainer.params.use_curriculum and trainer.curriculum_change is not None:
      trainer.curriculum_change = functools.partial(self.settings["curriculum"]["change_fcn"], trainer)
      trainer.curriculum_change(trainer.curriculum_dict["stage"]) # apply initial stage settings
    if self.settings["env_image_collection"]:
      if env is not None:
        env.collect_images = True
        if not env.params.depth_camera:
          if self.log_level >= 1:
            print("TrainingMananger.make_trainer() warning: env_image_collection=True but env.params.depth_camera=False. Hence image collection will NOT work")
        env.image_collection_chance = self.settings["env_image_collection_chance"]
        trainer.images_collected = 0
        trainer.image_batches_collected = 0
        trainer.image_list = []
        trainer.image_collection_num_per_batch = self.settings["env_image_collection_batch_size"]
        trainer.image_collection_max_batches = self.settings["env_image_collection_max_batches"]
        trainer.episode_fcn = functools.partial(trainer.image_collection_fcn)
      else:
        raise RuntimeError("TrainingMananger.make_trainer() error: env_image_collection=True but env=None")

    return trainer

  def apply_env_settings(self, env, set):
    """
    Apply all the settings given in a dictionary of settings
    """
    
    # apply MjEnv settings
    env.params.update(set["env"])
    env.load_next.update(set["env"]) # params to load in

    # apply cpp settings - general
    env.mj.set.randomise_colours = set["cpp"]["randomise_colours"]
    env.mj.set.time_for_action = set["cpp"]["time_for_action"]
    env.mj.set.saturation_yield_factor = set["cpp"]["saturation_yield_factor"]

    env.mj.set.sensor_sample_mode = set["cpp"]["sensor_sample_mode"]
    env.mj.set.state_sample_mode = set["cpp"]["state_sample_mode"]
    env.mj.set.sensor_n_prev_steps = set["cpp"]["sensor_n_prev_steps"]
    env.mj.set.state_n_prev_steps = set["cpp"]["state_n_prev_steps"]
    env.mj.set.sensor_noise_mu = set["cpp"]["sensor_noise_mu"]
    env.mj.set.sensor_noise_std = set["cpp"]["sensor_noise_std"]
    env.mj.set.state_noise_mu = set["cpp"]["state_noise_mu"]
    env.mj.set.state_noise_std = set["cpp"]["state_noise_std"]
    env.mj.set.base_position_noise = set["cpp"]["base_position_noise"]

    env.mj.set.oob_distance = set["cpp"]["oob_distance"]
    env.mj.set.lift_height = set["cpp"]["lift_height"]
    env.mj.set.gripper_target_height = set["cpp"]["gripper_target_height"]
    env.mj.set.stable_finger_force = set["cpp"]["stable_finger_force"]
    env.mj.set.stable_palm_force = set["cpp"]["stable_palm_force"]
    env.mj.set.stable_finger_force_lim = set["cpp"]["stable_finger_force_lim"]
    env.mj.set.stable_palm_force_lim = set["cpp"]["stable_palm_force_lim"]
    env.mj.set.XY_distance_threshold = set["cpp"]["XY_distance_threshold"]
    env.mj.set.fingertip_min_mm = set["cpp"]["fingertip_min_mm"]
    env.mj.set.continous_actions = set["cpp"]["continous_actions"]
    env.mj.set.use_termination_action = set["cpp"]["use_termination_action"]
    env.mj.set.termination_threshold = set["cpp"]["termination_threshold"]

    # apply cpp settings - actions
    env.mj.set.gripper_prismatic_X.in_use = set["cpp"]["action"]["gripper_prismatic_X"]["in_use"]
    env.mj.set.gripper_revolute_Y.in_use = set["cpp"]["action"]["gripper_revolute_Y"]["in_use"]
    env.mj.set.gripper_Z.in_use = set["cpp"]["action"]["gripper_Z"]["in_use"]
    env.mj.set.base_X.in_use = set["cpp"]["action"]["base_X"]["in_use"]
    env.mj.set.base_Y.in_use = set["cpp"]["action"]["base_Y"]["in_use"]
    env.mj.set.base_Z.in_use = set["cpp"]["action"]["base_Z"]["in_use"]
    env.mj.set.base_yaw.in_use = set["cpp"]["action"]["base_yaw"]["in_use"]

    env.mj.set.gripper_prismatic_X.value = set["cpp"]["action"]["gripper_prismatic_X"]["value"]
    env.mj.set.gripper_revolute_Y.value = set["cpp"]["action"]["gripper_revolute_Y"]["value"]
    env.mj.set.gripper_Z.value = set["cpp"]["action"]["gripper_Z"]["value"]
    env.mj.set.base_X.value = set["cpp"]["action"]["base_X"]["value"]
    env.mj.set.base_Y.value = set["cpp"]["action"]["base_Y"]["value"]
    env.mj.set.base_Z.value = set["cpp"]["action"]["base_Z"]["value"]
    env.mj.set.base_yaw.value = set["cpp"]["action"]["base_yaw"]["value"]

    env.mj.set.gripper_prismatic_X.sign = set["cpp"]["action"]["gripper_prismatic_X"]["sign"]
    env.mj.set.gripper_revolute_Y.sign = set["cpp"]["action"]["gripper_revolute_Y"]["sign"]
    env.mj.set.gripper_Z.sign = set["cpp"]["action"]["gripper_Z"]["sign"]
    env.mj.set.base_X.sign = set["cpp"]["action"]["base_X"]["sign"]
    env.mj.set.base_Y.sign = set["cpp"]["action"]["base_Y"]["sign"]
    env.mj.set.base_Z.sign = set["cpp"]["action"]["base_Z"]["sign"]
    env.mj.set.base_yaw.sign = set["cpp"]["action"]["base_yaw"]["sign"]

    # apply cpp settings - sensors
    env.mj.set.motor_state_sensor.in_use = set["cpp"]["sensor"]["motor_state_sensor"]["in_use"]
    env.mj.set.base_state_sensor_Z.in_use = set["cpp"]["sensor"]["base_state_sensor_Z"]["in_use"]
    env.mj.set.base_state_sensor_XY.in_use = set["cpp"]["sensor"]["base_state_sensor_XY"]["in_use"]
    env.mj.set.base_state_sensor_yaw.in_use = set["cpp"]["sensor"]["base_state_sensor_yaw"]["in_use"]
    env.mj.set.bending_gauge.in_use = set["cpp"]["sensor"]["bending_gauge"]["in_use"]
    env.mj.set.palm_sensor.in_use = set["cpp"]["sensor"]["palm_sensor"]["in_use"]
    env.mj.set.wrist_sensor_XY.in_use = set["cpp"]["sensor"]["wrist_sensor_XY"]["in_use"]
    env.mj.set.wrist_sensor_Z.in_use = set["cpp"]["sensor"]["wrist_sensor_Z"]["in_use"]

    env.mj.set.motor_state_sensor.normalise = set["cpp"]["sensor"]["motor_state_sensor"]["normalise"]
    env.mj.set.base_state_sensor_Z.normalise = set["cpp"]["sensor"]["base_state_sensor_Z"]["normalise"]
    env.mj.set.base_state_sensor_XY.normalise = set["cpp"]["sensor"]["base_state_sensor_XY"]["normalise"]
    env.mj.set.base_state_sensor_yaw.normalise = set["cpp"]["sensor"]["base_state_sensor_yaw"]["normalise"]
    env.mj.set.bending_gauge.normalise = set["cpp"]["sensor"]["bending_gauge"]["normalise"]
    env.mj.set.palm_sensor.normalise = set["cpp"]["sensor"]["palm_sensor"]["normalise"]
    env.mj.set.wrist_sensor_XY.normalise = set["cpp"]["sensor"]["wrist_sensor_XY"]["normalise"]
    env.mj.set.wrist_sensor_Z.normalise = set["cpp"]["sensor"]["wrist_sensor_Z"]["normalise"]

    env.mj.set.motor_state_sensor.read_rate = set["cpp"]["sensor"]["motor_state_sensor"]["read_rate"]
    env.mj.set.base_state_sensor_Z.read_rate = set["cpp"]["sensor"]["base_state_sensor_Z"]["read_rate"]
    env.mj.set.base_state_sensor_XY.read_rate = set["cpp"]["sensor"]["base_state_sensor_XY"]["read_rate"]
    env.mj.set.base_state_sensor_yaw.read_rate = set["cpp"]["sensor"]["base_state_sensor_yaw"]["read_rate"]
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
    if set["cpp"]["sensor"]["base_state_sensor_yaw"]["noise_override"] is not None:
      env.mj.set.base_state_sensor_yaw.set_gaussian_noise(*set["cpp"]["sensor"]["base_state_sensor_yaw"]["noise_override"])
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
    env = self.create_reward_function(env)
    
    return env

  def set_sensor_reward_thresholds(self, env):
    """
    Determine the reward thresholds
    """

    printout = True if self.log_level >= 2 else False

    if self.settings["reward"]["bend"]["good"] == "stable":
      set_gBend = env.mj.set.stable_finger_force
    else: set_gBend = self.settings["reward"]["bend"]["good"]
    if self.settings["reward"]["bend"]["exceed"] == "stable_lim":
      set_xBend = env.mj.set.stable_finger_force_lim
    else: set_xBend = self.settings["reward"]["bend"]["exceed"]
    if self.settings["reward"]["bend"]["dangerous"] == "yield":
      set_dBend = env.yield_load()
    else: set_dBend = self.settings["reward"]["bend"]["dangerous"]

    if self.settings["reward"]["palm"]["good"] == "stable":
      set_gPalm = env.mj.set.stable_palm_force
    else: set_gPalm = self.settings["reward"]["palm"]["good"]
    if self.settings["reward"]["palm"]["exceed"] == "stable_lim":
      set_xPalm = env.mj.set.stable_palm_force_lim
    else: set_xPalm = self.settings["reward"]["palm"]["exceed"]

    @dataclass
    class RewardThresholds:
      # m=minimum, g=good, x=exceed, d=dangerous
      mBend = self.settings["reward"]["bend"]["min"]
      gBend = set_gBend
      xBend = set_xBend
      dBend = set_dBend

      mPalm = self.settings["reward"]["palm"]["min"]
      gPalm = set_gPalm
      xPalm = set_xPalm
      dPalm = self.settings["reward"]["palm"]["dangerous"]

      xWrist = self.settings["reward"]["wrist"]["exceed"]
      dWrist = self.settings["reward"]["wrist"]["dangerous"]

    self.RT = RewardThresholds()

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

    # rewards                        reward   done   trigger    min             max      overshoot
    env.mj.set.lifted.set           (value,  False,   1)
    env.mj.set.lifted_to_height.set (value,  False,   1)
    env.mj.set.object_stable.set    (value,  False,   1)
    env.mj.set.good_bend_sensor.set (value,  False,   1,     self.RT.mBend, self.RT.gBend,  -1)
    env.mj.set.good_palm_sensor.set (value,  False,   1,     self.RT.mPalm, self.RT.gPalm,  -1)

    if self.settings["reward"]["object_XY_distance"]["used"]:
      d1 = self.settings["reward"]["object_XY_distance"]["min"]
      d2 = self.settings["reward"]["object_XY_distance"]["max"]
      env.mj.set.object_XY_distance.set (value,  False,   1,     d1, d2,  -1)

    return env

  def set_sensor_penalties(self, env, value):
    """
    Set penalty rewards with given value, alongside defaults
    """

    # penalties                        reward   done   trigger  min               max        overshoot
    env.mj.set.exceed_limits.set       (value,  False,    1)
    env.mj.set.exceed_bend_sensor.set  (value,  False,    1,    self.RT.xBend,  self.RT.dBend,  -1)
    env.mj.set.exceed_palm_sensor.set  (value,  False,    1,    self.RT.xPalm,  self.RT.dPalm,  -1)
    env.mj.set.exceed_wrist_sensor.set (value,  False,    1,    self.RT.xWrist, self.RT.dWrist, -1)

    if self.settings["reward"]["action_pen_lin"]["used"]:
      x = self.settings["reward"]["action_pen_lin"]["scaling"]
      amin = self.settings["reward"]["action_pen_lin"]["min"]
      amax = self.settings["reward"]["action_pen_lin"]["max"]
      # rewards                          reward     done  trigger  min  max  overshoot
      env.mj.set.action_penalty_lin.set (value * x, False,   1,   amin, amax,  -1)

    if self.settings["reward"]["action_pen_sq"]["used"]:
      x = self.settings["reward"]["action_pen_sq"]["scaling"]
      amin = self.settings["reward"]["action_pen_sq"]["min"]
      amax = self.settings["reward"]["action_pen_sq"]["max"]
      # rewards                         reward     done  trigger  min  max  overshoot
      env.mj.set.action_penalty_sq.set (value * x, False,   1,   amin, amax,  -1)

    return env

  def set_sensor_terminations(self, env, value=-1.0, done=True, trigger=1):
    """
    Set terminations based on dangerous sensor readings
    """

    # terminations                        reward   done   trigger  min               max       overshoot
    env.mj.set.dangerous_bend_sensor.set  (value,  done,  trigger, self.RT.dBend,  self.RT.dBend,  -1)
    env.mj.set.dangerous_palm_sensor.set  (value,  done,  trigger, self.RT.dPalm,  self.RT.dPalm,  -1)
    env.mj.set.dangerous_wrist_sensor.set (value,  done,  trigger, self.RT.dWrist, self.RT.dWrist, -1)

    return env

  def create_reward_function(self, env):
    """
    Set the reward structure for the learning, with different style options
    """

    if self.settings["reward"]["style"] == "sensor_mixed_v1":
      # prepare reward thresholds
      self.set_sensor_reward_thresholds(env)
      # reward each step               reward   done   trigger
      env.mj.set.step_num.set          (-0.01,  False,   1)
      # penalties and bonuses
      env = self.set_sensor_bonuses(env, 0.002 * self.settings["reward"]["scale_rewards"])
      env = self.set_sensor_penalties(env, -0.002 * self.settings["reward"]["scale_penalties"])
      # scale based on steps allowed per episode
      env.mj.set.scale_rewards(100 / env.params.max_episode_steps)
      # end criteria                   reward   done   trigger
      env.mj.set.stable_height.set     (1.0,    True,    1)
      env.mj.set.oob.set               (-1.0,   True,    1)
      if self.settings["reward"]["penalty_termination"]:
        env = self.set_sensor_terminations(env, trigger=self.settings["reward"]["dangerous_trigger"])

    elif self.settings["reward"]["style"] == "termination_action_v1":
      # prepare reward thresholds
      self.set_sensor_reward_thresholds(env)
      # reward each step               reward   done   trigger
      env.mj.set.step_num.set          (-0.01,  False,   1)
      # penalties and bonuses
      env = self.set_sensor_bonuses(env, 0.002 * self.settings["reward"]["scale_rewards"])
      env = self.set_sensor_penalties(env, -0.002 * self.settings["reward"]["scale_penalties"])
      # scale based on steps allowed per episode
      env.mj.set.scale_rewards(100 / env.params.max_episode_steps)
      # end criteria                        reward   done   trigger
      env.mj.set.stable_termination.set     (1.0,    True,   1)
      env.mj.set.failed_termination.set     (-1.0,   True,   1)
      env.mj.set.oob.set                    (-1.0,   True,   1)
      if self.settings["reward"]["penalty_termination"]:
        env = self.set_sensor_terminations(env, trigger=self.settings["reward"]["dangerous_trigger"])
    
    else:
      raise RuntimeError(f"style={self.settings['reward']['style']} is not a valid option in TrainingManager.create_reward_function()")

    # specific options
    if "terminate_on_exceed_limits" in self.settings["reward"]["options"]:
      # reward each step                     reward   done   trigger
      env.mj.set.exceed_limits.set     (-1.0,   True,    3)
    
    # how many stable steps before 'object_stable' triggers
    env.mj.set.object_stable.trigger = self.settings["reward"]["stable_trigger"]

    return env

if __name__ == "__main__":

  # key settings
  rngseed = None
  device = "cpu"
  log_level = 1
  save = True

  timestamp = "26-09-23_13-42"
  jobstr = None # "1:12"

  # create the training manager
  tm = TrainingManager(rngseed=rngseed, device=device, log_level=log_level)

  # # choose settings
  tm.settings["save"] = save
  tm.settings["trainer"]["num_episodes"] = 15
  tm.settings["trainer"]["test_freq"] = 5
  tm.settings["trainer"]["save_freq"] = 5
  tm.settings["final_test_trials_per_object"] = 1
  tm.settings["env"]["test_objects"] = 3
  tm.settings["env"]["max_episode_steps"] = 1
  tm.settings["episode_log_rate"] = 5
  tm.settings["track_avg_num"] = 3
  tm.settings["Agent_DQN"]["target_update"] = 10

  # log training details
  tm.job_number = 1
  tm.timestamp = "26-06-38_12:34"
  tm.param_1 = 5e-5
  tm.param_1_name = "learning rate"
  tm.param_2 = 4000
  tm.param_2_name = "eps decay"

  # create the environment
  env = tm.make_env()

  # make the agent
  layers = [env.n_obs, 150, 100, 50, env.n_actions]
  network = networks.VariableNetwork(layers, device=device)
  agent = Agent_DQN(device=device)
  agent.init(network)

  # input into the training manager and train
  tm.run_training(agent, env)

