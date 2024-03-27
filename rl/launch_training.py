#!/usr/bin/env python3

# fix for cluster, numpy causes segfault
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from datetime import datetime
import argparse
from time import sleep
from random import random
import torch
import functools
import numpy as np

from Trainer import MujocoTrainer
from TrainingManager import TrainingManager
from agents.DQN import Agent_DQN
from agents.ActorCritic import MLPActorCriticAC, Agent_SAC
from agents.PolicyGradient import MLPActorCriticPG, Agent_PPO
from agents.PolicyGradient import CNNActorCriticPG, NetActorCriticPG, MixedNetworkFromEncoder
from agents.PolicyGradient import Agent_PPO_MAT, MATActorCriticPG
import networks

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

def parse_job_string(jobstr):
  """
  Get a list of job numbers from a job string, eg "a:e" returns [a,b,c,d,e] and
  "a e f" returns [a,e,f]
  """

  # parse the job string, assume either "x:y" or "x y z"
  if ":" in jobstr:
    splits = jobstr.split(":")
    job_start = int(splits[0])
    job_end = int(splits[-1])
    job_numbers = list(range(job_start, job_end + 1))
  else:
    splits = jobstr.split(" ")
    job_numbers = []
    for s in splits:
      job_numbers.append(int(s))

  return job_numbers

def get_jobs_from_timestamp(timestamp, run_name_prefix=None):
  """
  Find all the jobs with a particular timestamp
  """

  tm = make_training_manager_from_args(args, silent=True)
  tm.set_group_run_name(job_num=1, timestamp=timestamp, prefix=run_name_prefix)
  match_str = tm.run_name[:-3]

  # get all the run folder corresponding to this timestamp
  group_path = tm.trainer.savedir + "/" + tm.group_name
  run_folders = [x for x in os.listdir(group_path) if x.startswith(match_str)]

  job_nums = []

  for folder in run_folders:
    num = folder.split(match_str)[-1][2:] # from _A5 -> 5"
    job_nums.append(int(num))

  # sort into numeric ascending order
  job_nums.sort()

  # check for failures
  if len(job_nums) == 0:
    print(f"launch_training.py warning: get_jobs_from_timestamp found zero trainings matching '{match_str}'")

  return job_nums

def update_training_summaries(timestamp, jobstr=None, job_numbers=None, run_name_prefix=None,
                              silent=True):
  """
  Regenerate training_summaries from a currently running training, or any trainings where
  there is no up to date training_summary.txt files
  """

  if jobstr is None and job_numbers is None:
    job_numbers = get_jobs_from_timestamp(timestamp, run_name_prefix=run_name_prefix)
    if len(job_numbers) == 0:
      raise RuntimeError(f"print_results_table() cannot find any job numbers for timestamp {timestamp}")

  if jobstr is not None:
    job_numbers = parse_job_string(jobstr)

  tm = make_training_manager_from_args(args, silent=silent)

  for j in job_numbers:

    # determine the path required for this job
    tm.init_training_summary()
    tm.set_group_run_name(job_num=j, timestamp=timestamp, prefix=run_name_prefix)
    tm.save_training_summary(force=True)

def print_results_table(timestamp, jobstr=None, job_numbers=None, run_name_prefix=None,
                        min_ep=None, max_ep=None, silent=True, min_stage=None, max_stage=None,
                        print_test=None):
  """
  Print a table of results for a training
  """

  if jobstr is None and job_numbers is None:
    job_numbers = get_jobs_from_timestamp(timestamp, run_name_prefix=run_name_prefix)
    if len(job_numbers) == 0:
      raise RuntimeError(f"print_results_table() cannot find any job numbers for timestamp {timestamp}")

  if jobstr is not None:
    job_numbers = parse_job_string(jobstr)

  tm = make_training_manager_from_args(args, silent=silent)

  # prepare to find information from training_summary files
  headings = []
  table = []
  new_elem = []
  program_names = [None]

  found_job_number = False
  found_timestamp = False
  found_param_1 = False
  found_param_2 = False
  found_param_3 = False
  found_trained_to = False
  found_train_best_ep = False
  found_train_best_sr = False
  found_full_test_sr = False

  if min_ep is not None or max_ep is not None:
    do_min_max_ep = True
  else: do_min_max_ep = False
  if min_stage is not None or max_stage is not None:
    do_min_max_stage = True
  else: do_min_max_stage = False

  for j in job_numbers:

    # determine the path required for this job
    tm.set_group_run_name(job_num=j, timestamp=timestamp, prefix=run_name_prefix)
    filepath = tm.trainer.savedir + "/" + tm.group_name + "/" + tm.run_name + "/"
    
    # load information from the training summary
    tm.init_training_summary() # wipe data from previous loop
    exists = tm.load_training_summary(filepath=filepath)
    if tm.program not in program_names:
      program_names.append(tm.program)

    if not exists:
      # try to create the file
      tm.trainer = MujocoTrainer(None, None, log_level=0, run_name=tm.run_name,
                                group_name=tm.group_name)
      tm.save_training_summary(filepath=filepath)
      exists = tm.load_training_summary(filepath=filepath)
      if not exists:
        raise RuntimeError("print_results_table() failed to make training summary at", filepath)
      
    if tm.job_number is not None: found_job_number = True
    if tm.timestamp is not None: found_timestamp = True
    if tm.param_1 is not None: found_param_1 = True
    if tm.param_2 is not None: found_param_2 = True
    if tm.param_3 is not None: found_param_3 = True
    if tm.trained_to is not None: found_trained_to = True
    if tm.train_best_ep is not None: found_train_best_ep = True
    if tm.train_best_sr is not None: found_train_best_sr = True
    if tm.full_test_sr is not None: found_full_test_sr = True

  # get the program name
  if len(program_names) == 2:
    program_str = f"Program: {program_names[1]}\n\n"
  elif len(program_names) == 1:
    program_str = ""
  else:
    program_str = "Multiple program names found: "
    for i in range(1, len(program_names) - 1):
      program_str += f"{program_names[i]}; "
    program_str += f"{program_names[-1]}\n\n"

  if found_job_number: headings.append("Job num")
  if found_timestamp: headings.append("Timestamp    ") # 4xspace for heading
  if found_param_1: headings.append(tm.param_1_name)
  if found_param_2: headings.append(tm.param_2_name)
  if found_param_3: headings.append(tm.param_3_name)
  if found_trained_to: headings.append("Trained to")
  if do_min_max_ep:
    if min_ep is not None and max_ep is not None:
      headings.append(f"Best SR range {min_ep} - {max_ep}")
    elif min_ep is not None:
      headings.append(f"Best SR from ep {min_ep}")
    elif max_ep is not None:
      headings.append(f"Best SR up to ep {max_ep}")
    else: raise RuntimeError("code error in print_results_table()")
    headings.append("At episode")
  if do_min_max_stage:
    if min_stage is not None and max_stage is not None:
      if min_stage == max_stage:
        headings.append(f"Best SR during stage {min_stage}")
      else:
        headings.append(f"Best SR stages {min_stage} - {max_stage}")
    elif min_stage is not None:
      headings.append(f"Best SR from stage {min_stage}")
    elif max_stage is not None:
      headings.append(f"Best SR up to stage {max_stage}")
    else: raise RuntimeError("code error in print_results_table()")
    headings.append("At episode")
  if found_train_best_ep: headings.append("Train best episode")
  if found_train_best_sr: headings.append("Train best SR")
  if found_full_test_sr: headings.append("Final test SR")
  if print_test is not None: headings.append(print_test)

  for j in job_numbers:

    # determine the path required for this job
    tm.set_group_run_name(job_num=j, timestamp=timestamp, prefix=run_name_prefix)
    filepath = tm.trainer.savedir + "/" + tm.group_name + "/" + tm.run_name + "/"
    
    # load information from the training summary
    tm.init_training_summary() # wipe data from previous loop
    exists = tm.load_training_summary(filepath=filepath)

    if not exists:
      # try to create the file
      tm.trainer = MujocoTrainer(None, None, log_level=0, run_name=tm.run_name,
                                group_name=tm.group_name)
      tm.save_training_summary(filepath=filepath)
      exists = tm.load_training_summary(filepath=filepath)
      if not exists:
        raise RuntimeError("print_results_table() failed to make training summary at", filepath)

    if found_job_number:
      if tm.job_number is not None:
        new_elem.append(tm.job_number)
      else: new_elem.append("nodata")
    if found_timestamp:
      if tm.timestamp is not None:
        new_elem.append(tm.timestamp)
      else: new_elem.append("nodata")
    if found_param_1:
      if tm.param_1 is not None:
        new_elem.append(tm.param_1)
      else: new_elem.append("nodata")
    if found_param_2:
      if tm.param_2 is not None:
        new_elem.append(tm.param_2)
      else: new_elem.append("nodata")
    if found_param_3:
      if tm.param_3 is not None:
        new_elem.append(tm.param_3)
      else: new_elem.append("nodata")
    if found_trained_to:
      if tm.trained_to is not None:
        new_elem.append(tm.trained_to)
      else: new_elem.append("nodata")
    if do_min_max_ep:
      data = tm.trainer.read_test_performance()
      sr, ep = tm.trainer.calc_best_performance(from_episode=min_ep, to_episode=max_ep,
                                                success_rate_vector=data[2,:],
                                                episodes_vector=data[1,:],
                                                stages_vector=data[5,:])
      if sr == 0 and ep == 0:
        new_elem.append("nodata")
        new_elem.append("nodata")
      else:
        new_elem.append(sr)
        new_elem.append(int(ep))
    if do_min_max_stage:
      data = tm.trainer.read_test_performance()
      sr, ep = tm.trainer.calc_best_performance(from_stage=min_stage, to_stage=max_stage,
                                                success_rate_vector=data[2,:],
                                                episodes_vector=data[1,:],
                                                stages_vector=data[5,:])
      if sr == 0 and ep == 0:
        new_elem.append("nodata")
        new_elem.append("nodata")
      else:
        new_elem.append(sr)
        new_elem.append(int(ep))
    if found_train_best_ep:
      if tm.train_best_ep is not None:
        new_elem.append(tm.train_best_ep)
      else: new_elem.append("nodata")
    if found_train_best_sr:
      if tm.train_best_sr is not None:
        new_elem.append(tm.train_best_sr)
      else: new_elem.append("nodata")
    if found_full_test_sr:
      if tm.full_test_sr is not None:
        new_elem.append(tm.full_test_sr)
      else: new_elem.append("nodata")
    if print_test is not None:
      sr, ep = tm.trainer.read_best_performance_from_text(silence=True, fulltest=True,
                                                          fulltestname=print_test)
      if sr is not None:
        new_elem.append(sr)
      else: new_elem.append("nodata")

    table.append(new_elem[:])
    new_elem = []

  # now prepare to print the table
  print_str = """""" + program_str
  heading_str = ""
  for x in range(len(headings) - 1): heading_str += "{" + str(x) + "} | "
  heading_str += "{" + str(len(headings) - 1) + "}"
  row_str = heading_str[:]
  heading_formatters = []
  row_formatters = []
  for x in range(len(headings)):
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
        table[i][j] = "{:.3f}".format(elem)
    # print(row_str.format(*table[i]))
    print_str += row_str.format(*table[i]) + "\n"

  # print and save the table
  print("\n" + print_str)
  tablepath = tm.trainer.savedir + "/" + tm.group_name + "/" + "results_table.txt"
  with open(tablepath, 'w') as f:
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

def print_training_info(include_all=False):
  """
  Print basic parameter information about a training
  """

  mj = MujocoTrainer(None, None, log_level=0)
  tm.trainer = mj
  txt = tm.get_training_summary().split(tm.summary_section_seperator)[0]

  if include_all:
    print(txt)
  else:
    lines = txt.splitlines(keepends=True)
    to_print = """"""
    for line in lines:
      if (line.startswith("\tTimestamp") or
          line.startswith("\tTrained to") or
          line.startswith("\tBest training") or
          line.startswith("\tMost recent fulltest")):
        continue
      to_print += line
    print(to_print)
  exit()

def make_training_manager_from_args(args, silent=False, save=True):
  """
  Create a training manager given the command line arguments
  """

  log_level = args.log_level
  if silent: log_level = 0

  tm = TrainingManager(rngseed=args.rngseed, device=args.device, log_level=log_level)

  # input any command line settings
  tm.settings["plot"] = args.plot
  tm.settings["render"] = args.render
  tm.settings["save"] = not args.no_saving
  if args.savedir is not None: tm.settings["savedir"] = args.savedir
  tm.program = args.program

  # now create an underlying trainer without an agent or environment
  tm.trainer = tm.make_trainer(None, None)

  return tm

def save_expert_network(tm, timestamp, job):
  """
  Get the network out of training and save it under rl/env/expert
  """

  save_path = "env/expert/"
  save_folder = f"T{timestamp}_J{job}/"
  filename = "net.pth"

  tm.load(job_num=job, timestamp=timestamp, best_id=True, agentonly=True)
  
  if tm.trainer.agent.name != "Agent_PPO":
    raise RuntimeError("save_expert_network() only supports Agent_PPO currently")
  
  net = tm.trainer.agent.mlp_ac.pi.mu_net

  fullpath = save_path + save_folder
  if not os.path.exists(fullpath):
    try:
      os.makedirs(fullpath)
    except FileExistsError:
      pass

  print(f"save_expert_network() saving at: {fullpath + filename} ...", end="", flush=True)
  torch.save(net, fullpath + filename)
  print("finished")

def curriculum_change_termination(self, stage):
  """
  Toggle the termination threshold over stages. Below -1.0 or above 1.0 is
  impossible to trigger.
  """
  if stage == 0:
    self.env.mj.set.termination_threshold = self.curriculum_dict["param_values"][stage]
  elif stage == 1:
    self.env.mj.set.termination_threshold = self.curriculum_dict["param_values"][stage]
    # disable stable height and enable stable_termination, penalise failed temrination
    self.env.mj.set.stable_height.set          (0.0,    False,  1000)
    self.env.mj.set.stable_termination.set     (1.0,    True,   1)
    self.env.mj.set.failed_termination.set     (-0.05,  False,  1)
  elif stage == 2:
    # set failed termination as a complete failure
    self.env.mj.set.failed_termination.set     (-1.0,   True,  1)

def curriculum_change_step_sizes(self, stage):
  """
  Change the step sizes
  """

  # set the action step sizes
  self.env.mj.set.gripper_prismatic_X.value = self.curriculum_dict["param_values"][stage][0]
  self.env.mj.set.gripper_revolute_Y.value = self.curriculum_dict["param_values"][stage][1]
  self.env.mj.set.gripper_Z.value = self.curriculum_dict["param_values"][stage][2]
  self.env.mj.set.base_Z.value = self.curriculum_dict["param_values"][stage][3]

  # now adjust the time per action to match the step sizes
  self.env.mj.set.time_for_action = self.curriculum_dict["param_values"][stage][4]

def curriculum_change_successful_grasp(self, stage):
  """
  Make the gripper height required for a successful grasp change over the stages
  """
  self.env.mj.set.gripper_target_height = self.curriculum_dict["param_values"][stage][0]
  env.mj.set.object_stable.trigger = self.curriculum_dict["param_values"][stage][1]

def curriculum_change_navigation_grasp(self, stage):
  """
  Curriculum for learning to navigate to objects and then grasp. Makes the grasping
  requirements increase each stage.

  Stage 0: navigate to the object only
  Stage 1: lift the object up
  Stage 2: lift the object stably
  Stage 3: lift to target height and maintain stability
  """

  # get the value of a standard shaped reward
  rew = self.env.mj.set.good_bend_sensor.reward

  if stage == 0:
    self.env.mj.set.within_XY_distance.set  (1.0,    True,   1)
    self.env.mj.set.lifted.set              (rew,    False,  1)
    self.env.mj.set.object_stable.set       (rew,    False,  1)
    self.env.mj.set.stable_height.set       (rew,    False,  1)

  if stage == 1:
    self.env.mj.set.within_XY_distance.set  (rew,    False,  1)
    self.env.mj.set.lifted.set              (1.0,    True,   1)
    self.env.mj.set.object_stable.set       (rew,    False,  1)
    self.env.mj.set.stable_height.set       (rew,    False,  1)

  if stage == 2:
    self.env.mj.set.within_XY_distance.set  (rew,    False,  1)
    self.env.mj.set.lifted.set              (rew,    False,  1)
    self.env.mj.set.object_stable.set       (1.0,    True,   1)
    self.env.mj.set.stable_height.set       (rew,    False,  1)

  if stage == 3:
    self.env.mj.set.within_XY_distance.set  (rew,    False,  1)
    self.env.mj.set.lifted.set              (rew,    False,  1)
    self.env.mj.set.object_stable.set       (rew,    False,  1)
    self.env.mj.set.stable_height.set       (1.0,    True,   1)

def curriculum_change_object_noise(self, stage):
  """
  Change the curriculum to increase the object noise so they spawn further from
  the gripper and are harder to grasp
  """

  self.env.params.object_position_noise_mm = self.curriculum_dict["param_values"][stage]
  
  # don't use this, it causes trainings to avoid using fingers
  # self.env.mj.set.oob_distance = (15 + self.curriculum_dict["param_values"][stage]) * 1e-3

def curriculum_fcn_MAT(self, i):
  """
  Change the step sizes
  """

  if not hasattr(self, "MAT_last_step_update_test_length"):
    self.MAT_last_step_update_test_length = 0

  if i != 1 and self.MAT_last_step_update_test_length == len(self.track.avg_successful_grasp):
    return
  
  self.MAT_last_step_update_test_length = len(self.track.avg_successful_grasp)

  # get the current maximum success rate achieved at test time
  if len(self.track.avg_successful_grasp) > 0:
    sr = max(self.track.avg_successful_grasp)
  else:
    sr = 0.0

  xmin = self.curriculum_dict["param_values"][0][0]
  xmax = self.curriculum_dict["param_values"][0][1]
  ymin = self.curriculum_dict["param_values"][1][0]
  ymax = self.curriculum_dict["param_values"][1][1]
  zmin = self.curriculum_dict["param_values"][2][0]
  zmax = self.curriculum_dict["param_values"][2][1]
  bmin = self.curriculum_dict["param_values"][3][0]
  bmax = self.curriculum_dict["param_values"][3][1]
  tmin = self.curriculum_dict["param_values"][4][0]
  tmax = self.curriculum_dict["param_values"][4][1]

  # set the action step sizes
  self.env.mj.set.gripper_prismatic_X.value = xmin + (xmax - xmin) * (1 - sr)
  self.env.mj.set.gripper_revolute_Y.value = ymin + (ymax - ymin) * (1 - sr)
  self.env.mj.set.gripper_Z.value = zmin + (zmax - zmin) * (1 - sr)
  self.env.mj.set.base_Z.value = bmin + (bmax - bmin) * (1 - sr)

  # now adjust the time per action to match the step sizes
  self.env.mj.set.time_for_action = tmin + (tmax - tmin) * (1 - sr)

  if self.log_level > 0:
    print(f"Step sizes at episode {i} being updated based on max success rate = {sr:.3f}")
    print(f"x action size is {self.env.mj.set.gripper_prismatic_X.value * 1000:.1f} mm")
    print(f"y action size is {self.env.mj.set.gripper_revolute_Y.value:.3f} rad")
    print(f"z action size is {self.env.mj.set.gripper_Z.value * 1000:.1f} mm")
    print(f"base z action size is {self.env.mj.set.base_Z.value * 1000:.1f} mm")
    print(f"time per action is {self.env.mj.set.time_for_action:.3f} s")

if __name__ == "__main__":

  """
  This script launches a mujoco grasping training, using the TrainingManager class.
  
  Basic usage: 
    1. source a virtual environment
    2. create a training program and add it to the bottom of this file in the if...elif
    3. run this file with the two required command line arguments:
        > python launch_training.py --program myprogram --job 1

  To print the results afterwards use the training timestamp to idenfity it: 
    > python launch_training.py --timestamp dd-mm-yy_hr-mn
  
  This script is designed to be called repeatedly with different job numbers. Say you
  want to compare a training with a learning rate of 1e-3 and 5e-3. You make your
  program which varies learning rate depending on the job number (either 1 or 2).
  Then you can call this script twice, with job numbers 1 and 2. You can add repeats,
  so maybe jobs 1-10 are the first learning rate and jobs 11-20 are the second. You
  can vary other parameters etc, this is how you build up large training comparisons.
  """

  # starting time
  starting_time = datetime.now()

  # key default settings
  datestr = "%d-%m-%y_%H-%M" # all date inputs must follow this format

  # define arguments and parse them
  parser = argparse.ArgumentParser()
  parser.add_argument("-j", "--job",          default=None, type=int) # job input number
  parser.add_argument("-t", "--timestamp",    default=None)           # timestamp
  parser.add_argument("-p", "--program",      default=None)           # program name to select from if..else if
  parser.add_argument("-d", "--device",       default=None)           # override device
  parser.add_argument("-r", "--render",       action="store_true")    # render window during training
  parser.add_argument("-g", "--plot",         action="store_true")    # plot to wandb job
  parser.add_argument("-c", "--continue",     action="store_true", dest="resume") # continue training
  parser.add_argument("-H", "--heuristic",    action="store_true")    # run a test using heuristic actions
  parser.add_argument("-id", "--load-id",     default=None)           # id to load in case we are testing/plotting
  parser.add_argument("--name-prefix",        default=None)           # run name prefix eg run_12-32 for training or loading
  parser.add_argument("--job-string",         default=None)           # job string for print results eg "1:10" or "1 2 3 6"
  parser.add_argument("--print-results",      action="store_true")    # prepare and print all 
  parser.add_argument("--print-from-ep",      default=None, type=int) # print best SR from a specific episode
  parser.add_argument("--print-up-to-ep",     default=None, type=int) # print best SR up until a specific episode
  parser.add_argument("--print-stage",        default=None, type=int) # print best SR during a specific stage
  parser.add_argument("--print-from-stage",   default=None, type=int) # print best SR from a specific curriculum stage
  parser.add_argument("--print-up-to-stage",  default=None, type=int) # print best SR up until a specific curriculum stage
  parser.add_argument("--print-test",         default=None)           # print full_test results on a given test set (if available)
  parser.add_argument("--rngseed",            default=None)           # turns on reproducible training with given seed (slower)
  parser.add_argument("--log-level",          type=int, default=1)    # set script log level
  parser.add_argument("--no-delay",           action="store_true")    # prevent a sleep(...) to seperate processes
  parser.add_argument("--print",              action="store_true")    # don't train, print job options
  parser.add_argument("--no-saving",          action="store_true")    # do we save any data from this training
  parser.add_argument("--savedir",            default=None)           # override save/load directory
  parser.add_argument("--pause",              default=False)          # pause between episodes in a test
  parser.add_argument("--test", default=0, const="saved", nargs="?")  # run a thorough test on existing model, default training set, can choose object set with arg
  parser.add_argument("--demo", default=0, const=30, nargs="?", type=int)  # run a demo test on model, default 30 trials, can set number with arg
  parser.add_argument("--new-endpoint",       default=None, type=int) # new episode target for continuing training
  parser.add_argument("--extra-episodes",     default=None, type=int) # extra episodes to run for continuing training
  parser.add_argument("--smallest-job-num",   default=1, type=int)    # only used to reduce sleep time to seperate processes
  parser.add_argument("--torch-threads",      default=1, type=int)    # maximum number of allowed pytorch threads, set 0 for no limit
  parser.add_argument("--save-expert",        action="store_true")    # load a training and save the network as an expert
  # parser.add_argument("--override-lib",       action="store_true")    # override bind.so library with loaded data

  args = parser.parse_args()

  # exit()

  timestamp = args.timestamp if args.timestamp else datetime.now().strftime(datestr)

  # default device
  if args.device is None:
    args.device = "cpu"

  if args.torch_threads > 0:
    torch.set_num_threads(args.torch_threads)

  if args.print: 
    args.log_level = 0

  # disable delays for in cases where we are not training
  if args.demo or args.print_results:
    args.no_delay = True

  # echo these inputs
  if args.log_level > 0:
    print("launch_training.py is preparing to train:")
    print(" -> Job number:", args.job)
    print(" -> Timestamp:", timestamp)
    print(" -> Program name:", args.program)
    print(" -> Device:", args.device)

  # seperate process for safety when running a training program
  if (not args.no_delay):
    if (args.job is not None and not args.plot and
        (args.program is not None or args.resume is not None or args.test)):
      sleep_for = args.job - args.smallest_job_num
      if sleep_for < 0: sleep_for = args.job # in case of jobstr "4 5 6 1"
      if args.log_level > 0:
        print(f"Sleeping for {sleep_for} seconds to seperate process for safety")
      sleep(sleep_for)
      sleep(0.25 * random()) # extra safety

  # ----- special cases ----- #

  if args.print_results:

    if args.timestamp is None:
      raise RuntimeError(f"launch_training.py: a timestamp [-t, --timestamp] in the following format '{datestr}' is required to load existing trainigs")

    if args.log_level > 0: 
      print("\nPreparing to print a results table in launch_training.py")
    silent = True if args.log_level <= 1 else False
    if not silent: print("Not silent, log_level set to:", args.log_level)
    if args.print_stage is not None:
      args.print_from_stage = args.print_stage
      args.print_up_to_stage = args.print_stage
    update_training_summaries(args.timestamp, jobstr=args.job_string, run_name_prefix=args.name_prefix,
                              silent=silent)
    print_results_table(args.timestamp, jobstr=args.job_string, run_name_prefix=args.name_prefix,
                        min_ep=args.print_from_ep, max_ep=args.print_up_to_ep, silent=silent,
                        min_stage=args.print_from_stage, max_stage=args.print_up_to_stage,
                        print_test=args.print_test)
    exit()

  if args.job is None:
    raise RuntimeError("launch_training.py: your options require a job number [-j, --job], either to identify an existing training for loading, or to correspond to your selected program")

  # create a training manager
  tm = make_training_manager_from_args(args, silent=True)
  tm.log_level = args.log_level

  if args.plot:

    if args.timestamp is None:
      raise RuntimeError(f"launch_training.py: a timestamp [-t, --timestamp] in the following format '{datestr}' is required to load existing trainigs")

    if args.log_level > 0: print("launch_training.py will plot a training")

    tm.load(job_num=args.job, timestamp=args.timestamp, id=args.load_id)
    tm.trainer.track.plot_train_avg = True
    tm.trainer.track.plot_test_raw = True
    tm.trainer.track.add_test_metrics(metrics_to_add=["Success rate"], 
                                      values=[tm.trainer.track.avg_successful_grasp])
    tm.trainer.track.plot_test_metrics = True
    tm.trainer.track.plot(plttitle=tm.group_name + "/" + tm.run_name)
    input("Press enter to quit plotting windows and terminate program")
    exit()

  if args.test or args.demo:

    if args.timestamp is None:
      raise RuntimeError(f"launch_training.py: a timestamp [-t, --timestamp] in the following format '{datestr}' is required to load existing trainigs")

    # sort out input arguments
    if args.demo: args.render = True # always render for demonstration tests
    best_id = True if args.load_id is None else False
    if args.test == "saved": args.test = None # don't load a new object set

    if args.log_level > 0: 
      print("launch_training.py is running a test")
      if args.test is not None: print(" -> Object set changed to", args.test)
      print(" -> ID set to", "best_id" if best_id else args.load_id)
      print(" -> Render set to", args.render)
      print(" -> Heuristic set to", args.heuristic)
      if args.demo: print(" -> Demo set, number of trials is", args.demo)

    tm.load(job_num=args.job, timestamp=args.timestamp, best_id=best_id, id=args.load_id)
    tm.run_test(heuristic=args.heuristic, demo=args.demo, render=args.render, pause=args.pause,
                different_object_set=args.test)
    exit()

  if args.resume:

    if args.timestamp is None:
      raise RuntimeError(f"launch_training.py: a timestamp [-t, --timestamp] in the following format '{datestr}' is required to load existing trainigs")
    if args.new_endpoint is None and args.extra_episodes is None:
      if args.log_level > 0:
        print("launch_training.py warning: [-c, --continue] has been used without either [--new-endpoint] or [--extra-episodes]. Original training endpoint will be used")
    
    if args.log_level > 0: print(f"launch_training.py is continuing a traing, new_endpoint={args.new_endpoint}, extra_episodes={args.extra_episodes}")

    tm.load(job_num=args.job, timestamp=args.timestamp, id=args.load_id)

    # adjust command line settings as load overrides them
    tm.settings["plot"] = args.plot
    tm.settings["render"] = args.render
    if args.savedir is not None: tm.settings["savedir"] = args.savedir

    tm.continue_training(new_endpoint=args.new_endpoint, extra_episodes=args.extra_episodes)
    exit()

  if args.save_expert:
    if args.log_level > 0:
      print(f"launch_training.py is saving an expert model for timestamp={args.timestamp} and job={args.job}")
    save_expert_network(tm, args.timestamp, args.job)
    exit()

  # ----- regular training ----- #

  if args.program is None:
    raise RuntimeError("launch_training.py: normal trainings require that [-p, --program] be set with training name corresponding to an option in this file")

  # set the name of this training in the training manager
  tm.set_group_run_name(job_num=args.job, timestamp=timestamp, prefix=args.name_prefix)

  # save the baseline settings (only in case they have changed)
  tm.save_baseline_params()

  if args.program == "baseline_basic":

    # create the environment
    env = tm.make_env()

    # baseline network size
    net = [150, 100, 50]

    # make the agent, may depend on variable settings above
    layers = [env.n_obs, *net, env.n_actions]
    network = networks.VariableNetwork(layers, device=args.device)
    agent = Agent_DQN(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "test_1":

    # define what to vary this training, dependent on job number
    vary_1 = [100, 200]
    vary_2 = [250, 500]
    vary_3 = None
    repeats = None
    tm.param_1_name = "target_update"
    tm.param_2_name = "eps_decay"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()
    
    # apply the varied settings (very important!)
    tm.settings["Agent_DQN"]["target_update"] = tm.param_1
    tm.settings["Agent_DQN"]["eps_decay"] = tm.param_2

    # choose any additional settings to change
    tm.settings["trainer"]["num_episodes"] = 15
    tm.settings["trainer"]["test_freq"] = 5
    tm.settings["trainer"]["save_freq"] = 5
    tm.settings["final_test_trials_per_object"] = 1
    tm.settings["env"]["test_objects"] = 3
    tm.settings["env"]["max_episode_steps"] = 5
    tm.settings["episode_log_rate"] = 5
    tm.settings["track_avg_num"] = 3
    tm.settings["Agent_DQN"]["target_update"] = 10
    tm.settings["Agent_DQN"]["min_memory_replay"] = 1

    # create the environment
    env = tm.make_env()

    # make the agent, may depend on variable settings above
    layers = [env.n_obs, 64, 64, env.n_actions]
    network = networks.VariableNetwork(layers, device=args.device)
    agent = Agent_DQN(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "test_2":

    # define what to vary this training, dependent on job number
    vary_1 = [1e-5, 5e-5]
    vary_2 = [0.1, 0.2]
    vary_3 = None
    repeats = None
    tm.param_1_name = "learning_rate"
    tm.param_2_name = "clip_ratio"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()
    
    # apply the varied settings (very important!)
    tm.settings["Agent_PPO"]["learning_rate_pi"] = tm.param_1
    tm.settings["Agent_PPO"]["learning_rate_vf"] = tm.param_1
    tm.settings["Agent_PPO"]["clip_ratio"] = tm.param_2

    # choose any additional settings to change
    tm.settings["trainer"]["num_episodes"] = 18
    tm.settings["trainer"]["test_freq"] = 6
    tm.settings["trainer"]["save_freq"] = 4
    tm.settings["final_test_trials_per_object"] = 1
    tm.settings["env"]["test_objects"] = 3
    tm.settings["env"]["max_episode_steps"] = 5
    tm.settings["episode_log_rate"] = 5
    tm.settings["track_avg_num"] = 3
    tm.settings["Agent_PPO"]["steps_per_epoch"] = 15
    tm.settings["cpp"]["continous_actions"] = True

    # create the environment
    env = tm.make_env()

    # make the agent, may depend on variable settings above
    layers = [20, 20]
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                               continous_actions=True)
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "sac_test_1":

    # define what to vary this training, dependent on job number
    vary_1 = [1e-6, 1e-5, 1e-4, 1e-3]
    vary_2 = [None]
    vary_3 = None
    repeats = 5
    tm.param_1_name = "learning_rate"
    tm.param_2_name = "Agent_SAC"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()
    
    # apply the varied settings
    tm.settings["Agent_SAC"]["learning_rate"] = tm.param_1

    # apply other essential settings
    tm.settings["cpp"]["continous_actions"] = True
    
    # # for debugging do NOT use for training
    # tm.settings["Agent_SAC"]["random_start_episodes"] = 1
    # tm.settings["Agent_SAC"]["update_after_steps"] = 1
    # tm.settings["Agent_SAC"]["update_every_steps"] = 1
    # tm.settings["Agent_SAC"]["min_memory_replay"] = 1
    # tm.settings["Agent_SAC"]["batch_size"] = 5

    # create the environment
    env = tm.make_env()

    # make the agent, may depend on variable settings above
    layers = [256, 256]
    network = MLPActorCriticAC(env.n_obs, env.n_actions, layers)
    agent = Agent_SAC(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "ppo_test_1":

    # define what to vary this training, dependent on job number
    vary_1 = [False, True]
    vary_2 = [None]
    vary_3 = None
    repeats = 1
    tm.param_1_name = "continous_actions"
    tm.param_2_name = "Agent_PPO"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()
    
    # # apply the varied settings
    # tm.settings["Agent_PPO"]["learning_rate"] = tm.param_1

    # apply other essential settings
    tm.settings["cpp"]["continous_actions"] = tm.param_1
    
    # # for debugging do NOT use for training
    # tm.settings["Agent_PPO"]["steps_per_epoch"] = 10

    # create the environment
    env = tm.make_env()

    # make the agent, may depend on variable settings above
    layers = [20, 20]
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                               continous_actions=tm.param_1)
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "ppo_and_sac_test_1":

    # define what to vary this training, dependent on job number
    vary_1 = [1.0, 1.5]
    vary_2 = [1e-6, 1e-5, 1e-4, 1e-3]
    vary_3 = [Agent_SAC, Agent_PPO]
    repeats = 2
    tm.param_1_name = "action_scale"
    tm.param_2_name = "learning_rate"
    tm.param_3_name = "agent"
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply environment dependent settings
    tm.settings["cpp"]["continous_actions"] = True
    tm.settings["scale_rewards"] = 2.5
    tm.settings["cpp"]["action"]["gripper_prismatic_X"]["value"] = 2e-3 * tm.param_1
    tm.settings["cpp"]["action"]["gripper_revolute_Y"]["value"] = 0.015 * tm.param_1
    tm.settings["cpp"]["action"]["gripper_Z"]["value"] = 4e-3 * tm.param_1
    tm.settings["cpp"]["action"]["base_Z"]["value"] = 2e-3 * tm.param_1

    # create the environment
    env = tm.make_env()
    
    # apply the agent settings
    layers = [64, 64]
    if tm.param_3.name == "Agent_SAC":
      tm.settings["Agent_SAC"]["learning_rate"] = tm.param_2
      network = MLPActorCriticAC(env.n_obs, env.n_actions, hidden_sizes=layers)
    elif tm.param_3.name == "Agent_PPO":
      tm.settings["Agent_PPO"]["learning_rate_pi"] = tm.param_2
      tm.settings["Agent_PPO"]["learning_rate_vf"] = tm.param_2
      network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                 continous_actions=True)

    # make the agent
    agent = tm.param_3(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "ppo_hypers_1":

    # define what to vary this training, dependent on job number
    vary_1 = [1.0, 2.5]
    vary_2 = [(64, 64), (128, 128)]
    vary_3 = None
    repeats = 8
    tm.param_1_name = "action_scaling"
    tm.param_2_name = None
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply environment dependent settings
    tm.settings["cpp"]["continous_actions"] = True
    tm.settings["scale_rewards"] = tm.param_1
    tm.settings["cpp"]["action"]["gripper_prismatic_X"]["value"] = 2e-3
    tm.settings["cpp"]["action"]["gripper_revolute_Y"]["value"] = 0.015
    tm.settings["cpp"]["action"]["gripper_Z"]["value"] = 4e-3
    tm.settings["cpp"]["action"]["base_Z"]["value"] = 2e-3

    # create the environment
    env = tm.make_env()
    
    # apply the agent settings
    layers = tm.param_2
    tm.settings["Agent_PPO"]["learning_rate_pi"] = 1e-5
    tm.settings["Agent_PPO"]["learning_rate_vf"] = 1e-5
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)

    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "ppo_force_limit":

    # define what to vary this training, dependent on job number
    vary_1 = [2.0, 3.0, 3.458, 5.0]
    vary_2 = [10.0, 30.0]
    vary_3 = None
    repeats = 4
    tm.param_1_name = "bending frc limit"
    tm.param_2_name = "palm frc limit"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply environment dependent settings
    tm.settings["cpp"]["continous_actions"] = True
    tm.settings["cpp"]["action"]["gripper_prismatic_X"]["value"] = 2e-3
    tm.settings["cpp"]["action"]["gripper_revolute_Y"]["value"] = 0.015
    tm.settings["cpp"]["action"]["gripper_Z"]["value"] = 4e-3
    tm.settings["cpp"]["action"]["base_Z"]["value"] = 2e-3

    # apply bending force limit
    tm.settings["cpp"]["stable_finger_force_lim"] = tm.param_1
    tm.settings["cpp"]["stable_palm_force_lim"] = tm.param_2

    # create the environment
    env = tm.make_env()
    
    # apply the agent settings
    layers = [128, 128]
    tm.settings["Agent_PPO"]["learning_rate_pi"] = 1e-5
    tm.settings["Agent_PPO"]["learning_rate_vf"] = 1e-5
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)

    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "ppo_force_limit_2":

    # define what to vary this training, dependent on job number
    vary_1 = [2.0, 3.0, 3.458, 5.0]
    vary_2 = [10.0, 30.0]
    vary_3 = None
    repeats = 4
    tm.param_1_name = "bending frc limit"
    tm.param_2_name = "palm frc limit"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply environment dependent settings
    tm.settings["cpp"]["continous_actions"] = True
    tm.settings["cpp"]["action"]["gripper_prismatic_X"]["value"] = 2e-3
    tm.settings["cpp"]["action"]["gripper_revolute_Y"]["value"] = 0.015
    tm.settings["cpp"]["action"]["gripper_Z"]["value"] = 4e-3
    tm.settings["cpp"]["action"]["base_Z"]["value"] = 2e-3

    # apply bending force limit
    tm.settings["cpp"]["stable_finger_force_lim"] = tm.param_1
    tm.settings["cpp"]["stable_palm_force_lim"] = tm.param_2

    # create the environment
    env = tm.make_env()
    
    # apply the agent settings
    layers = [128, 128, 128]
    tm.settings["Agent_PPO"]["learning_rate_pi"] = 1e-5
    tm.settings["Agent_PPO"]["learning_rate_vf"] = 1e-5
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)

    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "set9_baseline_1":

    # define what to vary this training, dependent on job number
    vary_1 = [3.0, 100.0]
    vary_2 = [10, 15, 20]
    vary_3 = [2e-6, 1e-5, 5e-5]
    repeats = 5
    tm.param_1_name = "bending frc limit"
    tm.param_2_name = "object pos noise"
    tm.param_3_name = "learning rate"
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply environment dependent settings
    tm.settings["cpp"]["continous_actions"] = True
    tm.settings["cpp"]["action"]["gripper_prismatic_X"]["value"] = 2e-3
    tm.settings["cpp"]["action"]["gripper_revolute_Y"]["value"] = 0.015
    tm.settings["cpp"]["action"]["gripper_Z"]["value"] = 4e-3
    tm.settings["cpp"]["action"]["base_Z"]["value"] = 2e-3

    # apply training specific settings
    tm.settings["cpp"]["stable_finger_force_lim"] = tm.param_1
    tm.settings["cpp"]["stable_palm_force_lim"] = 10.0
    tm.settings["env"]["object_position_noise_mm"] = tm.param_2
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["object_set_name"] = "set9_fullset"

    # create the environment
    env = tm.make_env()
    
    # apply the agent settings
    layers = [128, 128, 128]
    tm.settings["Agent_PPO"]["learning_rate_pi"] = tm.param_3
    tm.settings["Agent_PPO"]["learning_rate_vf"] = tm.param_3
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)

    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "set8_vs_set9":

    if args.job == 1:

      job = 68
      timestamp = "06-10-23_16-57"

    elif args.job == 2:

      job = 77
      timestamp = "06-10-23_16-57"

    elif args.job == 3:

      job = 87
      timestamp = "06-10-23_16-57"

    else:
      raise RuntimeError("args.job not set to valid number")

    tm.load(job_num=job, timestamp=timestamp, best_id=True)

    # now run test with set8
    tm.trainer.env.load("set8_fullset_1500")
    tm.run_test(trials_per_obj=10)
    
  elif args.program == "set9_baseline_fingertips":

    # define what to vary this training, dependent on job number
    vary_1 = [3.0, 100.0]
    vary_2 = [10, 20]
    vary_3 = [45, 60]
    repeats = 5
    tm.param_1_name = "bending frc limit"
    tm.param_2_name = "object pos noise"
    tm.param_3_name = "fingertip angle"
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply environment dependent settings
    tm.settings["cpp"]["continous_actions"] = True
    tm.settings["cpp"]["action"]["gripper_prismatic_X"]["value"] = 2e-3
    tm.settings["cpp"]["action"]["gripper_revolute_Y"]["value"] = 0.015
    tm.settings["cpp"]["action"]["gripper_Z"]["value"] = 4e-3
    tm.settings["cpp"]["action"]["base_Z"]["value"] = 2e-3

    # apply training specific settings
    tm.settings["cpp"]["stable_finger_force_lim"] = tm.param_1
    tm.settings["cpp"]["stable_palm_force_lim"] = 10.0
    tm.settings["env"]["object_position_noise_mm"] = tm.param_2
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["object_set_name"] = "set9_fullset"
    tm.settings["env"]["finger_hook_angle_degrees"] = tm.param_3

    # create the environment
    env = tm.make_env()
    
    # apply the agent settings
    layers = [128, 128, 128]
    tm.settings["Agent_PPO"]["learning_rate_pi"] = 5e-5
    tm.settings["Agent_PPO"]["learning_rate_vf"] = 5e-5
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)

    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "set9_baseline_fingertips_2":

    # define what to vary this training, dependent on job number
    vary_1 = [3.5, 4.0]
    vary_2 = [0.9e-3, 1.0e-3]
    vary_3 = [45, 60]
    repeats = 5
    tm.param_1_name = "bending frc limit"
    tm.param_2_name = "finger thickness"
    tm.param_3_name = "fingertip angle"
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply environment dependent settings
    tm.settings["cpp"]["continous_actions"] = True
    tm.settings["cpp"]["action"]["gripper_prismatic_X"]["value"] = 2e-3
    tm.settings["cpp"]["action"]["gripper_revolute_Y"]["value"] = 0.015
    tm.settings["cpp"]["action"]["gripper_Z"]["value"] = 4e-3
    tm.settings["cpp"]["action"]["base_Z"]["value"] = 2e-3

    # apply training specific settings
    tm.settings["cpp"]["stable_finger_force_lim"] = tm.param_1
    tm.settings["cpp"]["stable_palm_force_lim"] = 10.0
    tm.settings["env"]["object_position_noise_mm"] = 10
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["object_set_name"] = "set9_fullset"
    tm.settings["env"]["finger_hook_angle_degrees"] = tm.param_3
    tm.settings["env"]["finger_thickness"] = tm.param_2

    # create the environment
    env = tm.make_env()
    
    # apply the agent settings
    layers = [128, 128, 128]
    tm.settings["Agent_PPO"]["learning_rate_pi"] = 5e-5
    tm.settings["Agent_PPO"]["learning_rate_vf"] = 5e-5
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)

    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "set9_baseline_easier":

    # define what to vary this training, dependent on job number
    vary_1 = [1.0, 1.5]
    vary_2 = [3.0, 4.0]
    vary_3 = [45, 60, 90]
    repeats = 5
    tm.param_1_name = "saturation factor"
    tm.param_2_name = "bending frc limit"
    tm.param_3_name = "fingertip angle"
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply environment dependent settings
    tm.settings["cpp"]["continous_actions"] = True
    tm.settings["cpp"]["action"]["gripper_prismatic_X"]["value"] = 2e-3
    tm.settings["cpp"]["action"]["gripper_revolute_Y"]["value"] = 0.015
    tm.settings["cpp"]["action"]["gripper_Z"]["value"] = 4e-3
    tm.settings["cpp"]["action"]["base_Z"]["value"] = 2e-3

    # apply training specific settings
    tm.settings["cpp"]["saturation_yield_factor"] = tm.param_1
    tm.settings["cpp"]["stable_finger_force_lim"] = tm.param_2
    tm.settings["cpp"]["stable_palm_force_lim"] = 10.0
    tm.settings["env"]["object_position_noise_mm"] = 10
    tm.settings["trainer"]["num_episodes"] = 60_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp" # "set9_fullset"
    tm.settings["env"]["finger_hook_angle_degrees"] = tm.param_3

    # create the environment
    env = tm.make_env()
    
    # apply the agent settings
    layers = [128, 128, 128]
    tm.settings["Agent_PPO"]["learning_rate_pi"] = 5e-5
    tm.settings["Agent_PPO"]["learning_rate_vf"] = 5e-5
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)

    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "set9_baseline_easier_hypers":

    # define what to vary this training, dependent on job number
    vary_1 = [0.5, 1.0, 1.5]
    vary_2 = [1e-5, 5e-5]
    vary_3 = [45, 60, 90]
    repeats = 5
    tm.param_1_name = "action_size_scale"
    tm.param_2_name = "learning rate"
    tm.param_3_name = "fingertip angle"
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply environment dependent settings
    tm.settings["cpp"]["continous_actions"] = True
    tm.settings["cpp"]["action"]["gripper_prismatic_X"]["value"] = 2e-3 * tm.param_1
    tm.settings["cpp"]["action"]["gripper_revolute_Y"]["value"] = 0.015 * tm.param_1
    tm.settings["cpp"]["action"]["gripper_Z"]["value"] = 4e-3 * tm.param_1
    tm.settings["cpp"]["action"]["base_Z"]["value"] = 2e-3 * tm.param_1
    tm.settings["cpp"]["time_for_action"] = 0.2 * tm.param_1

    # apply training specific settings
    tm.settings["cpp"]["saturation_yield_factor"] = 1.5
    tm.settings["cpp"]["stable_finger_force_lim"] = 4.0
    tm.settings["cpp"]["stable_palm_force_lim"] = 10.0
    tm.settings["env"]["object_position_noise_mm"] = 10
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp" # "set9_fullset"
    tm.settings["env"]["finger_hook_angle_degrees"] = tm.param_3

    # create the environment
    env = tm.make_env()
    
    # apply the agent settings
    layers = [128, 128, 128]
    tm.settings["Agent_PPO"]["learning_rate_pi"] = tm.param_2
    tm.settings["Agent_PPO"]["learning_rate_vf"] = tm.param_2
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)

    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "tune_table_hit":

    timestamp = "20-10-23_17-45"

    if args.job == 1: job = 65
    elif args.job == 2: job = 74
    elif args.job == 3: job = 77
    elif args.job == 4: job = 85
    elif args.job == 5: job = 165
    elif args.job == 6: job = 174
    elif args.job == 7: job = 177
    elif args.job == 8: job = 185
    else:
      raise RuntimeError("args.job not set to valid number")

    tm.load(job_num=job, timestamp=timestamp) #, best_id=True)

    # add in punishment for dangerous forces
    # terminations                                     reward done  trigger  min   max  overshoot
    tm.trainer.env.mj.set.dangerous_wrist_sensor.set   (-1,   True, 1,      10.0, 10.0,   -1)
    if args.job >= 5:
      tm.trainer.env.mj.set.dangerous_bend_sensor.set  (-1, True,  1, 5.0,   5.0,   -1)
      tm.trainer.env.mj.set.dangerous_palm_sensor.set  (-1, True,  1, 10.0,  10.0,  -1)

    # now run test with set8
    tm.continue_training(extra_episodes=40_000)
    print_time_taken()

  elif args.program == "test_termination_action":

    # define what to vary this training, dependent on job number
    vary_1 = [1e-5, 5e-5]
    vary_2 = None
    vary_3 = None
    repeats = 5
    tm.param_1_name = "learning rate"
    tm.param_2_name = None
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply environment dependent settings
    tm.settings["cpp"]["continous_actions"] = True
    tm.settings["cpp"]["action"]["gripper_prismatic_X"]["value"] = 2e-3
    tm.settings["cpp"]["action"]["gripper_revolute_Y"]["value"] = 0.015
    tm.settings["cpp"]["action"]["gripper_Z"]["value"] = 4e-3
    tm.settings["cpp"]["action"]["base_Z"]["value"] = 2e-3
    tm.settings["cpp"]["time_for_action"] = 0.2

    # apply training specific settings
    tm.settings["reward_style"] = "termination_action_v1"
    tm.settings["cpp"]["use_termination_action"] = True
    tm.settings["cpp"]["saturation_yield_factor"] = 1.5
    tm.settings["cpp"]["stable_finger_force_lim"] = 4.0
    tm.settings["cpp"]["stable_palm_force_lim"] = 10.0
    tm.settings["env"]["object_position_noise_mm"] = 10
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp" # "set9_fullset"
    tm.settings["env"]["finger_hook_angle_degrees"] = 90

    # create the environment
    env = tm.make_env()
    
    # apply the agent settings
    layers = [128, 128, 128]
    tm.settings["Agent_PPO"]["learning_rate_pi"] = tm.param_1
    tm.settings["Agent_PPO"]["learning_rate_vf"] = tm.param_1
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)

    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "test_dangerous_terminations":

    # define what to vary this training, dependent on job number
    vary_1 = [1e-5, 5e-5]
    vary_2 = None
    vary_3 = None
    repeats = 5
    tm.param_1_name = "learning rate"
    tm.param_2_name = None
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply environment dependent settings
    tm.settings["cpp"]["continous_actions"] = True
    tm.settings["cpp"]["action"]["gripper_prismatic_X"]["value"] = 2e-3
    tm.settings["cpp"]["action"]["gripper_revolute_Y"]["value"] = 0.015
    tm.settings["cpp"]["action"]["gripper_Z"]["value"] = 4e-3
    tm.settings["cpp"]["action"]["base_Z"]["value"] = 2e-3
    tm.settings["cpp"]["time_for_action"] = 0.2

    # apply training specific settings
    tm.settings["penalty_termination"] = True
    tm.settings["danger_style"] = [5.0, 15.0, 10.0] # bend, palm, wrist
    tm.settings["cpp"]["saturation_yield_factor"] = 1.5
    tm.settings["cpp"]["stable_finger_force_lim"] = 4.0
    tm.settings["cpp"]["stable_palm_force_lim"] = 10.0
    tm.settings["env"]["object_position_noise_mm"] = 10
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp" # "set9_fullset"
    tm.settings["env"]["finger_hook_angle_degrees"] = 90

    # create the environment
    env = tm.make_env()
    
    # apply the agent settings
    layers = [128, 128, 128]
    tm.settings["Agent_PPO"]["learning_rate_pi"] = tm.param_1
    tm.settings["Agent_PPO"]["learning_rate_vf"] = tm.param_1
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)

    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "try_improve_transfer":

    # define what to vary this training, dependent on job number
    vary_1 = [1, 2, 4]
    vary_2 = [6.0, 10.0]
    vary_3 = [False, True]
    repeats = 5
    tm.param_1_name = "object_stable num"
    tm.param_2_name = "wrist sensor max"
    tm.param_3_name = "use action penalty"
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply environment dependent settings
    tm.settings["cpp"]["continous_actions"] = True
    tm.settings["cpp"]["action"]["gripper_prismatic_X"]["value"] = 2e-3
    tm.settings["cpp"]["action"]["gripper_revolute_Y"]["value"] = 0.015
    tm.settings["cpp"]["action"]["gripper_Z"]["value"] = 4e-3
    tm.settings["cpp"]["action"]["base_Z"]["value"] = 2e-3
    tm.settings["cpp"]["time_for_action"] = 0.2

    # apply training specific settings
    tm.settings["penalty_termination"] = True
    tm.settings["danger_style"] = [5.0, 15.0, tm.param_2] # bend, palm, wrist
    tm.settings["cpp"]["saturation_yield_factor"] = 1.5
    tm.settings["cpp"]["stable_finger_force_lim"] = 4.0
    tm.settings["cpp"]["stable_palm_force_lim"] = 10.0
    tm.settings["env"]["object_position_noise_mm"] = 10
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp" # "set9_fullset"
    tm.settings["env"]["finger_hook_angle_degrees"] = 90

    # create the environment
    env = tm.make_env()

    # how many steps in a row do we require stability
    env.mj.set.object_stable.trigger = tm.param_1

    # wrist normalisation
    env.mj.set.wrist_sensor_Z.normalise = tm.param_2 + 1

    # are we using an action penalty
    if tm.param_3:
      value = 1 * env.mj.set.exceed_limits.reward
      # rewards                      reward  done   trigger  min  max  overshoot
      env.mj.set.action_penalty_sq.set (value,  False,   1,     0.1, 1.5,  -1)
      
    # apply the agent settings
    layers = [128, 128, 128]
    tm.settings["Agent_PPO"]["learning_rate_pi"] = 5e-5
    tm.settings["Agent_PPO"]["learning_rate_vf"] = 5e-5
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)

    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "sensitive_wrist":

    # define what to vary this training, dependent on job number
    vary_1 = [0.5, 0.75, 1.0]
    vary_2 = [1.0, 2.0, 4.0, 6.0]
    vary_3 = None
    repeats = 5
    tm.param_1_name = "action scaling"
    tm.param_2_name = "wrist sensor max"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply environment dependent settings
    tm.settings["cpp"]["continous_actions"] = True
    tm.settings["cpp"]["action"]["gripper_prismatic_X"]["value"] = 2e-3 * tm.param_1
    tm.settings["cpp"]["action"]["gripper_revolute_Y"]["value"] = 0.015 * tm.param_1
    tm.settings["cpp"]["action"]["gripper_Z"]["value"] = 4e-3 * tm.param_1
    tm.settings["cpp"]["action"]["base_Z"]["value"] = 2e-3 * tm.param_1
    tm.settings["cpp"]["time_for_action"] = 0.2 * tm.param_1

    # apply training specific settings
    tm.settings["penalty_termination"] = True
    tm.settings["reward"]["bend"]["dangerous"] = 5.0
    tm.settings["reward"]["palm"]["dangerous"] = 15.0
    tm.settings["reward"]["wrist"]["exceed"] = tm.param_2 * 0.5
    tm.settings["reward"]["wrist"]["dangerous"] = tm.param_2
    tm.settings["cpp"]["saturation_yield_factor"] = 1.5
    tm.settings["cpp"]["stable_finger_force_lim"] = 4.0
    tm.settings["cpp"]["stable_palm_force_lim"] = 10.0
    tm.settings["env"]["object_position_noise_mm"] = 10
    tm.settings["trainer"]["num_episodes"] = 60_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp" # "set9_fullset"
    tm.settings["env"]["finger_hook_angle_degrees"] = 90

    # create the environment
    env = tm.make_env()

    # wrist normalisation
    env.mj.set.wrist_sensor_Z.normalise = tm.param_2 * 1.5

    # apply the agent settings
    layers = [128, 128, 128]
    tm.settings["Agent_PPO"]["learning_rate_pi"] = 5e-5
    tm.settings["Agent_PPO"]["learning_rate_vf"] = 5e-5
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)

    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "try_action_noise":

    # define what to vary this training, dependent on job number
    vary_1 = [0.5, 1.0]
    vary_2 = [0.05, 0.15, 0.3]
    vary_3 = [False, True]
    repeats = 5
    tm.param_1_name = "action scaling"
    tm.param_2_name = "action noise"
    tm.param_3_name = "action penalty"
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply environment dependent settings
    tm.settings["cpp"]["continous_actions"] = True
    tm.settings["cpp"]["action"]["gripper_prismatic_X"]["value"] = 2e-3 * tm.param_1
    tm.settings["cpp"]["action"]["gripper_revolute_Y"]["value"] = 0.015 * tm.param_1
    tm.settings["cpp"]["action"]["gripper_Z"]["value"] = 4e-3 * tm.param_1
    tm.settings["cpp"]["action"]["base_Z"]["value"] = 2e-3 * tm.param_1
    tm.settings["cpp"]["time_for_action"] = 0.2 * tm.param_1

    # apply training specific settings
    wrist_limit = 4
    tm.settings["penalty_termination"] = True
    tm.settings["reward"]["bend"]["dangerous"] = 5.0
    tm.settings["reward"]["palm"]["dangerous"] = 15.0
    tm.settings["reward"]["wrist"]["exceed"] = wrist_limit * 0.5
    tm.settings["reward"]["wrist"]["dangerous"] = wrist_limit
    tm.settings["cpp"]["saturation_yield_factor"] = 1.5
    tm.settings["cpp"]["stable_finger_force_lim"] = 4.0
    tm.settings["cpp"]["stable_palm_force_lim"] = 10.0
    tm.settings["env"]["object_position_noise_mm"] = 10
    tm.settings["trainer"]["num_episodes"] = 60_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp" # "set9_fullset"
    tm.settings["env"]["finger_hook_angle_degrees"] = 90

    # create the environment
    env = tm.make_env()

    # wrist normalisation
    env.mj.set.wrist_sensor_Z.normalise = wrist_limit * 1.5

    # are we using an action penalty
    if tm.param_3:
      value = 2 * env.mj.set.exceed_limits.reward
      # rewards                      reward  done   trigger  min  max  overshoot
      env.mj.set.action_penalty_sq.set (value,  False,   1,     0.1, 3.0,  -1)

    # apply the agent settings
    layers = [128, 128, 128]
    tm.settings["Agent_PPO"]["learning_rate_pi"] = 5e-5
    tm.settings["Agent_PPO"]["learning_rate_vf"] = 5e-5
    tm.settings["Agent_PPO"]["use_random_action_noise"] = True
    tm.settings["Agent_PPO"]["random_action_noise_size"] = tm.param_2
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)

    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "evaluate_action_noise":

    # define what to vary this training, dependent on job number
    vary_1 = [0.5, 1.0]
    vary_2 = [0.025, 0.05, 0.075]
    vary_3 = [4, 6, 10]
    repeats = 5
    tm.param_1_name = "action scaling"
    tm.param_2_name = "action noise"
    tm.param_3_name = "wrist limit"
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply environment dependent settings
    tm.settings["cpp"]["continous_actions"] = True
    tm.settings["cpp"]["action"]["gripper_prismatic_X"]["value"] = 2e-3 * tm.param_1
    tm.settings["cpp"]["action"]["gripper_revolute_Y"]["value"] = 0.015 * tm.param_1
    tm.settings["cpp"]["action"]["gripper_Z"]["value"] = 4e-3 * tm.param_1
    tm.settings["cpp"]["action"]["base_Z"]["value"] = 2e-3 * tm.param_1
    tm.settings["cpp"]["time_for_action"] = 0.2 * tm.param_1

    # apply training specific settings
    wrist_limit = tm.param_3
    tm.settings["penalty_termination"] = True
    tm.settings["reward"]["bend"]["dangerous"] = 5.0
    tm.settings["reward"]["palm"]["dangerous"] = 15.0
    tm.settings["reward"]["wrist"]["exceed"] = wrist_limit * 0.5
    tm.settings["reward"]["wrist"]["dangerous"] = wrist_limit
    tm.settings["cpp"]["saturation_yield_factor"] = 1.5
    tm.settings["cpp"]["stable_finger_force_lim"] = 4.0
    tm.settings["cpp"]["stable_palm_force_lim"] = 10.0
    tm.settings["env"]["object_position_noise_mm"] = 10
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp" # "set9_fullset"
    tm.settings["env"]["finger_hook_angle_degrees"] = 90

    # create the environment
    env = tm.make_env()

    # wrist normalisation
    env.mj.set.wrist_sensor_Z.normalise = wrist_limit + 2

    # are we using an action penalty
    if True:
      value = 2 * env.mj.set.exceed_limits.reward
      # rewards                      reward  done   trigger  min  max  overshoot
      env.mj.set.action_penalty_sq.set (value,  False,   1,     0.1, 3.0,  -1)

    # apply the agent settings
    layers = [128, 128, 128]
    tm.settings["Agent_PPO"]["learning_rate_pi"] = 5e-5
    tm.settings["Agent_PPO"]["learning_rate_vf"] = 5e-5
    tm.settings["Agent_PPO"]["use_random_action_noise"] = True
    tm.settings["Agent_PPO"]["random_action_noise_size"] = tm.param_2
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)

    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "try_z_noise":

    # define what to vary this training, dependent on job number
    vary_1 = [0, 2e-3, 5e-3]
    vary_2 = [
      (False, None),
      (True, None),
      (True, (0.05, 0.0)),
      (True, (0.10, 0.0))
    ]
    vary_3 = None
    repeats = 5
    tm.param_1_name = "base pos noise"
    tm.param_2_name = "(use Z sensor, noise override)"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply environment dependent settings
    action_scale = 1.0
    tm.settings["cpp"]["continous_actions"] = True
    tm.settings["cpp"]["action"]["gripper_prismatic_X"]["value"] = 2e-3 * action_scale
    tm.settings["cpp"]["action"]["gripper_revolute_Y"]["value"] = 0.015 * action_scale
    tm.settings["cpp"]["action"]["gripper_Z"]["value"] = 4e-3 * action_scale
    tm.settings["cpp"]["action"]["base_Z"]["value"] = 2e-3 * action_scale
    tm.settings["cpp"]["time_for_action"] = 0.2 * action_scale

    # apply training specific settings
    wrist_limit = 4
    tm.settings["penalty_termination"] = True
    tm.settings["reward"]["bend"]["dangerous"] = 5.0
    tm.settings["reward"]["palm"]["dangerous"] = 15.0
    tm.settings["reward"]["wrist"]["exceed"] = wrist_limit * 0.5
    tm.settings["reward"]["wrist"]["dangerous"] = wrist_limit
    tm.settings["cpp"]["saturation_yield_factor"] = 1.5
    tm.settings["cpp"]["stable_finger_force_lim"] = 4.0
    tm.settings["cpp"]["stable_palm_force_lim"] = 10.0
    tm.settings["env"]["object_position_noise_mm"] = 10
    tm.settings["trainer"]["num_episodes"] = 60_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp" # "set9_fullset"
    tm.settings["env"]["finger_hook_angle_degrees"] = 90

    # add in z base noise
    tm.settings["cpp"]["base_position_noise"] = tm.param_1

    # are we using z base state sensor
    if tm.param_2[0]:
      tm.settings["cpp"]["sensor"]["base_state_sensor_Z"]["in_use"] = True
      tm.settings["cpp"]["sensor"]["base_state_sensor_Z"]["noise_override"] = tm.param_2[1]
    else:
      tm.settings["cpp"]["sensor"]["base_state_sensor_Z"]["in_use"] = False

    # create the environment
    env = tm.make_env()

    # wrist normalisation
    env.mj.set.wrist_sensor_Z.normalise = wrist_limit + 2

    # use an action penalty
    value = 2 * env.mj.set.exceed_limits.reward
    # rewards                      reward  done   trigger  min  max  overshoot
    env.mj.set.action_penalty_sq.set (value,  False,   1,     0.1, 3.0,  -1)

    # apply the agent settings
    layers = [128, 128, 128]
    tm.settings["Agent_PPO"]["learning_rate_pi"] = 5e-5
    tm.settings["Agent_PPO"]["learning_rate_vf"] = 5e-5
    tm.settings["Agent_PPO"]["use_random_action_noise"] = True
    tm.settings["Agent_PPO"]["random_action_noise_size"] = 0.05
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)

    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "improve_on_z_noise":

    """
    Palm in simulation gives much smaller readings, in real life it reaches up to 10N much faster
    and leads to corrective actions by the policy interrupting the normal grasp. Should make
    the palm more sensitive in simultion, perhaps 2x as sensitive then manually adjust for
    real life grasping.

    Wrist sensor avoidance also seems to be working okay with these trainings. Perhaps more
    can be done, likely the wrist sensor moves faster in real life. Also, the fingers getting
    caught on the foam causes a sim2real gap as the policy is not used to that. Could I 
    greatly increase the friction of the ground to reflect this?

    Lastly, the raising to 30mm whilst keeping stable is not really working. Partly this is
    probably due to the palm being much more sensitive in real life. However, I should also
    enforce a slower trigger on the 'object stable' and perhaps rethink how trainings
    terminate (and in real life as well).

    Revisiting a termination signal would be nice, as easy grasps can then be completed 
    quicker. However, it does introduce another point of failure. Perhaps I could try a 
    training which has the termination signal but is not able to trigger it and just uses
    stable height. Then when it is performing well I change the criteria and keep training
    to try and get it to learn to use the termination signal when it thinks the grasp is
    stable.
      - beware though the termination signal introduces another point of failure - just
      imagine in real life how many good grasps would be terminated incorrectly and be
      chalked up as failures

    At what point do I start training with set9 with sharp edged objects? Would that help
    or hinder the learning? It may well improve the real life performance.
    """

    # define what to vary this training, dependent on job number
    vary_1 = [4, 8]
    vary_2 = [4, 8]
    vary_3 = [
      (False, 15e-3, 1),
      (False, 28e-3, 4),
      (True,  15e-3, 1),
    ]
    repeats = 5
    tm.param_1_name = "palm limit"
    tm.param_2_name = "wrist limit"
    tm.param_3_name = "curriculum/t.h/stb trigger"
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply environment dependent settings
    action_scale = 1.0
    tm.settings["cpp"]["continous_actions"] = True
    tm.settings["cpp"]["action"]["gripper_prismatic_X"]["value"] = 2e-3 * action_scale
    tm.settings["cpp"]["action"]["gripper_revolute_Y"]["value"] = 0.015 * action_scale
    tm.settings["cpp"]["action"]["gripper_Z"]["value"] = 4e-3 * action_scale
    tm.settings["cpp"]["action"]["base_Z"]["value"] = 2e-3 * action_scale
    tm.settings["cpp"]["time_for_action"] = 0.2 * action_scale

    # apply training specific settings
    palm_stable_lim = tm.param_1
    palm_danger_lim = palm_stable_lim * 1.25
    wrist_limit = tm.param_2
    tm.settings["penalty_termination"] = True
    tm.settings["reward"]["bend"]["dangerous"] = 5.0
    tm.settings["reward"]["palm"]["dangerous"] = palm_danger_lim
    tm.settings["reward"]["wrist"]["exceed"] = wrist_limit * 0.75
    tm.settings["reward"]["wrist"]["dangerous"] = wrist_limit
    tm.settings["cpp"]["saturation_yield_factor"] = 1.5
    tm.settings["cpp"]["stable_finger_force_lim"] = 4.0
    tm.settings["cpp"]["stable_palm_force_lim"] = palm_stable_lim
    tm.settings["env"]["object_position_noise_mm"] = 10
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp" # "set9_fullset"
    tm.settings["env"]["finger_hook_angle_degrees"] = 90

    # add in z base noise
    tm.settings["cpp"]["base_position_noise"] = 5e-3

    # add significant mean noise to z base state sensor
    tm.settings["cpp"]["sensor"]["base_state_sensor_Z"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["base_state_sensor_Z"]["noise_override"] = [0.1, 0.0]

    # create the environment
    env = tm.make_env()

    env.mj.set.gripper_target_height = tm.param_3[1]
    env.mj.set.object_stable.trigger = tm.param_3[2]

    # palm and wrist normalisation
    env.mj.set.palm_sensor.normalise = palm_danger_lim * (6/5)
    env.mj.set.wrist_sensor_Z.normalise = wrist_limit + 2

    # use an action penalty
    value = 2 * env.mj.set.exceed_limits.reward
    # rewards                      reward  done   trigger  min  max  overshoot
    env.mj.set.action_penalty_sq.set (value,  False,   1,     0.1, 3.0,  -1)

    # add in curriculum where grasping gets harder
    tm.settings["trainer"]["use_curriculum"] = tm.param_3[0]
    tm.settings["curriculum"]["metric_name"] = "success_rate"
    tm.settings["curriculum"]["metric_thresholds"] = [0.7]
    tm.settings["curriculum"]["param_values"] = [(15e-3, 1), (28e-3, 4)]
    tm.settings["curriculum"]["change_fcn"] = curriculum_change_successful_grasp

    # apply the agent settings
    layers = [128, 128, 128]
    tm.settings["Agent_PPO"]["learning_rate_pi"] = 5e-5
    tm.settings["Agent_PPO"]["learning_rate_vf"] = 5e-5
    tm.settings["Agent_PPO"]["use_random_action_noise"] = True
    tm.settings["Agent_PPO"]["random_action_noise_size"] = 0.05
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)

    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "hyperparam_search_1":

    # define what to vary this training, dependent on job number
    vary_1 = [
      (1e-5, 1e-5),
      (5e-5, 1e-5),
      (1e-5, 5e-5),
      (5e-5, 5e-5)
    ]
    vary_2 = [0.95, 0.97, 0.99]
    vary_3 = [
      [128, 128],
      [128, 128, 128],
      [128, 128, 128, 128]
    ]
    repeats = 2
    tm.param_1_name = "learning rate pi/vf"
    tm.param_2_name = "gamma"
    tm.param_3_name = "network layers"
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply environment dependent settings
    action_scale = 1.0
    tm.settings["cpp"]["continous_actions"] = True
    tm.settings["cpp"]["action"]["gripper_prismatic_X"]["value"] = 2e-3 * action_scale
    tm.settings["cpp"]["action"]["gripper_revolute_Y"]["value"] = 0.015 * action_scale
    tm.settings["cpp"]["action"]["gripper_Z"]["value"] = 4e-3 * action_scale
    tm.settings["cpp"]["action"]["base_Z"]["value"] = 2e-3 * action_scale
    tm.settings["cpp"]["time_for_action"] = 0.2 * action_scale

    # apply training specific settings
    palm_stable_lim = 4
    palm_danger_lim = palm_stable_lim * 1.25
    wrist_limit = 8
    tm.settings["penalty_termination"] = True
    tm.settings["reward"]["bend"]["dangerous"] = 5.0
    tm.settings["reward"]["palm"]["dangerous"] = palm_danger_lim
    tm.settings["reward"]["wrist"]["exceed"] = wrist_limit * 0.75
    tm.settings["reward"]["wrist"]["dangerous"] = wrist_limit
    tm.settings["cpp"]["saturation_yield_factor"] = 1.5
    tm.settings["cpp"]["stable_finger_force_lim"] = 4.0
    tm.settings["cpp"]["stable_palm_force_lim"] = palm_stable_lim
    tm.settings["env"]["object_position_noise_mm"] = 10
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp" # "set9_fullset"
    tm.settings["env"]["finger_hook_angle_degrees"] = 90

    # add in z base noise
    tm.settings["cpp"]["base_position_noise"] = 5e-3

    # add significant mean noise to z base state sensor
    tm.settings["cpp"]["sensor"]["base_state_sensor_Z"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["base_state_sensor_Z"]["noise_override"] = [0.1, 0.0]

    # create the environment
    env = tm.make_env()

    env.mj.set.gripper_target_height = 20e-3
    env.mj.set.object_stable.trigger = 4

    # palm and wrist normalisation
    env.mj.set.palm_sensor.normalise = palm_danger_lim * (6/5)
    env.mj.set.wrist_sensor_Z.normalise = wrist_limit + 2

    # use an action penalty
    value = 2 * env.mj.set.exceed_limits.reward
    # rewards                      reward  done   trigger  min  max  overshoot
    env.mj.set.action_penalty_sq.set (value,  False,   1,     0.1, 3.0,  -1)

    # # add in curriculum where grasping gets harder
    # tm.settings["trainer"]["use_curriculum"] = tm.param_3[0]
    # tm.settings["curriculum"]["metric_name"] = "success_rate"
    # tm.settings["curriculum"]["metric_thresholds"] = [0.7]
    # tm.settings["curriculum"]["param_values"] = [(15e-3, 1), (28e-3, 4)]
    # tm.settings["curriculum"]["change_fcn"] = curriculum_change_successful_grasp

    # apply the agent settings
    layers = tm.param_3
    tm.settings["Agent_PPO"]["learning_rate_pi"] = tm.param_1[0]
    tm.settings["Agent_PPO"]["learning_rate_vf"] = tm.param_1[1]
    tm.settings["Agent_PPO"]["gamma"] = tm.param_2
    tm.settings["Agent_PPO"]["use_random_action_noise"] = True
    tm.settings["Agent_PPO"]["random_action_noise_size"] = 0.05
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)

    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "curriculum_termination_action":

    # define what to vary this training, dependent on job number
    vary_1 = [1e-5, 5e-5]
    vary_2 = [
      (-1, -1),
      (60_000, 90_000), 
      (40_000, 90_000)
    ]
    vary_3 = None
    repeats = 2
    tm.param_1_name = "learning rate"
    tm.param_2_name = "ep thresholds"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply environment dependent settings
    tm.settings["cpp"]["continous_actions"] = True
    tm.settings["cpp"]["action"]["gripper_prismatic_X"]["value"] = 2e-3
    tm.settings["cpp"]["action"]["gripper_revolute_Y"]["value"] = 0.015
    tm.settings["cpp"]["action"]["gripper_Z"]["value"] = 4e-3
    tm.settings["cpp"]["action"]["base_Z"]["value"] = 2e-3
    tm.settings["cpp"]["time_for_action"] = 0.2

    # ensure we are training with the termination action
    tm.settings["cpp"]["use_termination_action"] = True

    # apply training specific settings
    palm_stable_lim = 8
    palm_danger_lim = palm_stable_lim * 1.25
    wrist_limit = 8
    tm.settings["penalty_termination"] = True
    tm.settings["reward"]["bend"]["dangerous"] = 5.0
    tm.settings["reward"]["palm"]["dangerous"] = palm_danger_lim
    tm.settings["reward"]["wrist"]["exceed"] = wrist_limit * 0.75
    tm.settings["reward"]["wrist"]["dangerous"] = wrist_limit
    tm.settings["cpp"]["saturation_yield_factor"] = 1.5
    tm.settings["cpp"]["stable_finger_force_lim"] = 4.0
    tm.settings["cpp"]["stable_palm_force_lim"] = palm_stable_lim
    tm.settings["env"]["object_position_noise_mm"] = 10
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp" # "set9_fullset"
    tm.settings["env"]["finger_hook_angle_degrees"] = 90

    # add in z base noise
    tm.settings["cpp"]["base_position_noise"] = 5e-3

    # create the environment
    env = tm.make_env()

    env.mj.set.gripper_target_height = 15e-3
    env.mj.set.object_stable.trigger = 1

    # palm and wrist normalisation
    env.mj.set.palm_sensor.normalise = palm_danger_lim * (6/5)
    env.mj.set.wrist_sensor_Z.normalise = wrist_limit + 2

    # add in curriculum where grasping gets harder
    tm.settings["trainer"]["use_curriculum"] = True
    tm.settings["curriculum"]["metric_name"] = "episode_number"
    tm.settings["curriculum"]["metric_thresholds"] = tm.param_2
    tm.settings["curriculum"]["param_values"] = [1.1, 0.8]
    tm.settings["curriculum"]["change_fcn"] = curriculum_change_termination
    
    # apply the agent settings
    layers = [128, 128, 128]
    tm.settings["Agent_PPO"]["learning_rate_pi"] = tm.param_1
    tm.settings["Agent_PPO"]["learning_rate_vf"] = tm.param_1
    tm.settings["Agent_PPO"]["use_random_action_noise"] = True
    tm.settings["Agent_PPO"]["random_action_noise_size"] = 0.05
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)

    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "finetune_test_1":

    # load the best training, but we continue from the end
    job = 60
    timestamp = "08-11-23_17-30"
    tm.load(job_num=job, timestamp=timestamp, best_id=True,
            load_into_new_training=True)

    # define what to vary this training, dependent on job number
    vary_1 = [0.01, 0.05]
    vary_2 = [4, 8]
    vary_3 = [1, 2]
    repeats = 2
    tm.param_1_name = "action_noise"
    tm.param_2_name = "stable trigger"
    tm.param_3_name = "action penalty scale"
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply parameter changes
    tm.trainer.agent.params.random_action_noise_size = tm.param_1
    tm.trainer.env.mj.set.object_stable.trigger = tm.param_2
    tm.trainer.env.mj.set.action_penalty_sq.reward *= tm.param_3

    # record that our curriculum has changed and final test should be new model only
    tm.trainer.curriculum_dict["stage"] += 1
    tm.settings["final_test_max_stage"] = True
    tm.settings["final_test_only_stage"] = None

    # now continue training
    extra_episodes = 40_000
    tm.continue_training(extra_episodes=extra_episodes)
    print_time_taken()

  elif args.program == "finetune_test_2":

    # load the best training, but we continue from the end
    job = 60
    timestamp = "08-11-23_17-30"
    tm.load(job_num=job, timestamp=timestamp, best_id=True,
            load_into_new_training=True)

    # define what to vary this training, dependent on job number
    vary_1 = [0.01, 0.05]
    vary_2 = [8, 12]
    vary_3 = [3, 5]
    repeats = 2
    tm.param_1_name = "action_noise"
    tm.param_2_name = "stable trigger"
    tm.param_3_name = "action penalty scale"
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply parameter changes
    tm.trainer.agent.params.random_action_noise_size = tm.param_1
    tm.trainer.env.mj.set.object_stable.trigger = tm.param_2
    tm.trainer.env.mj.set.action_penalty_sq.reward *= tm.param_3

    # record that our curriculum has changed and final test should be new model only
    tm.trainer.curriculum_dict["stage"] += 1
    tm.settings["final_test_max_stage"] = True
    tm.settings["final_test_only_stage"] = None

    # now continue training
    extra_episodes = 60_000
    tm.continue_training(extra_episodes=extra_episodes)
    print_time_taken()

  elif args.program == "hyperparam_search_2":

    # define what to vary this training, dependent on job number
    vary_1 = [
      (5e-5, 5e-5),
      (5e-5, 10e-5),
      (10e-5, 10e-5),
    ]
    vary_2 = [2000, 4000, 8000]
    vary_3 = [
      (1, 1),
      (3, 3),
      (3, 5),
      (5, 3)
    ]
    repeats = 2
    tm.param_1_name = "learning rate pi/vf"
    tm.param_2_name = "steps per epoch"
    tm.param_3_name = "sensor/state steps"
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply environment dependent settings
    action_scale = 1.0
    tm.settings["cpp"]["continous_actions"] = True
    tm.settings["cpp"]["action"]["gripper_prismatic_X"]["value"] *= action_scale
    tm.settings["cpp"]["action"]["gripper_revolute_Y"]["value"] *= action_scale
    tm.settings["cpp"]["action"]["gripper_Z"]["value"] *= action_scale
    tm.settings["cpp"]["action"]["base_Z"]["value"] *= action_scale
    tm.settings["cpp"]["time_for_action"] *= action_scale

    # apply training specific settings
    palm_stable_lim = 4
    palm_danger_lim = palm_stable_lim * 1.25
    wrist_limit = 8
    tm.settings["penalty_termination"] = True
    tm.settings["reward"]["bend"]["dangerous"] = 5.0
    tm.settings["reward"]["palm"]["dangerous"] = palm_danger_lim
    tm.settings["reward"]["wrist"]["exceed"] = wrist_limit * 0.75
    tm.settings["reward"]["wrist"]["dangerous"] = wrist_limit
    tm.settings["cpp"]["saturation_yield_factor"] = 1.5
    tm.settings["cpp"]["stable_finger_force_lim"] = 4.0
    tm.settings["cpp"]["stable_palm_force_lim"] = palm_stable_lim
    tm.settings["env"]["object_position_noise_mm"] = 10
    tm.settings["trainer"]["num_episodes"] = 80_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp" # "set9_fullset"
    tm.settings["env"]["finger_hook_angle_degrees"] = 90

    # add in z base noise
    tm.settings["cpp"]["base_position_noise"] = 5e-3

    # add significant mean noise to z base state sensor
    tm.settings["cpp"]["sensor"]["base_state_sensor_Z"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["base_state_sensor_Z"]["noise_override"] = [0.1, 0.0]

    # change sensor steps
    tm.settings["cpp"]["sensor_n_prev_steps"] = tm.param_3[0]
    tm.settings["cpp"]["state_n_prev_steps"] = tm.param_3[1]

    # create the environment
    env = tm.make_env()

    env.mj.set.gripper_target_height = 20e-3
    env.mj.set.object_stable.trigger = 4

    # palm and wrist normalisation
    env.mj.set.palm_sensor.normalise = palm_danger_lim * (6/5)
    env.mj.set.wrist_sensor_Z.normalise = wrist_limit + 2

    # use an action penalty
    value = 2 * env.mj.set.exceed_limits.reward
    # rewards                      reward  done   trigger  min  max  overshoot
    env.mj.set.action_penalty_sq.set (value,  False,   1,     0.1, 3.0,  -1)

    # apply the agent settings
    layers = [128, 128, 128, 128]
    tm.settings["Agent_PPO"]["learning_rate_pi"] = tm.param_1[0]
    tm.settings["Agent_PPO"]["learning_rate_vf"] = tm.param_1[1]
    tm.settings["Agent_PPO"]["use_random_action_noise"] = True
    tm.settings["Agent_PPO"]["random_action_noise_size"] = 0.05
    tm.settings["Agent_PPO"]["steps_per_epoch"] = tm.param_2
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)

    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "object_set_baseline":

    # define what to vary this training, dependent on job number
    vary_1 = [
      "set9_nosharp",
      "set9_fullset",
      "set9_nosharp_smallspheres",
      "set9_full_smallspheres",
    ]
    vary_2 = None
    vary_3 = None
    repeats = 10
    tm.param_1_name = "object_set"
    tm.param_2_name = None
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply training specific settings
    tm.settings["trainer"]["num_episodes"] = 80_000
    tm.settings["env"]["object_set_name"] = tm.param_1

    # create the environment
    env = tm.make_env()

    # apply the agent settings
    layers = [128, 128, 128, 128]
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)

    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "finger_angle_baseline":

    # define what to vary this training, dependent on job number
    vary_1 = [45, 60, 75, 90]
    vary_2 = [0.9e-3, 1.0e-3]
    vary_3 = None
    repeats = 10
    tm.param_1_name = "hook angle"
    tm.param_2_name = "finger thickness"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply training specific settings
    tm.settings["trainer"]["num_episodes"] = 80_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp_smallspheres"
    tm.settings["env"]["finger_hook_angle_degrees"] = tm.param_1
    tm.settings["env"]["finger_thickness"] = tm.param_2

    # create the environment
    env = tm.make_env()

    # apply the agent settings
    layers = [128, 128, 128, 128]
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)

    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "state_sensor_compare":

    # define what to vary this training, dependent on job number
    vary_1 = [1, 3, 5, 7]
    vary_2 = [0, 4, 5]
    vary_3 = None
    repeats = 5
    tm.param_1_name = "num state sense readings"
    tm.param_2_name = "sample mode"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply training specific settings
    tm.settings["trainer"]["num_episodes"] = 80_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp_smallspheres"
    tm.settings["cpp"]["state_n_prev_steps"] = tm.param_1
    tm.settings["cpp"]["state_sample_mode"] = tm.param_2

    # create the environment
    env = tm.make_env()

    # apply the agent settings
    layers = [128, 128, 128, 128]
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)

    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "hyperparam_search_3":

    # define what to vary this training, dependent on job number
    vary_1 = [40, 80, 120, 160]
    vary_2 = [4000, 6000, 8000, 10_000]
    vary_3 = None
    repeats = 5
    tm.param_1_name = "train iters"
    tm.param_2_name = "steps per epoch"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply training specific settings
    tm.settings["trainer"]["num_episodes"] = 80_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp_smallspheres"
    tm.settings["Agent_PPO"]["train_pi_iters"] = tm.param_1
    tm.settings["Agent_PPO"]["train_vf_iters"] = tm.param_1
    tm.settings["Agent_PPO"]["steps_per_epoch"] = tm.param_2

    # create the environment
    env = tm.make_env()

    # apply the agent settings
    layers = [128, 128, 128, 128]
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)

    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "discrim_test_1":

    # define what to vary this training, dependent on job number
    vary_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    vary_2 = None
    vary_3 = None
    repeats = 2
    tm.param_1_name = None
    tm.param_2_name = None
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply training specific settings
    tm.settings["trainer"]["num_episodes"] = 80_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp"
    tm.settings["Agent_PPO_Discriminator"]["learning_rate_discrim"] = tm.param_1

    # create the environment
    env = tm.make_env()

    # apply the agent settings
    n_discrim = 8 # must match the size of the target vector
    dlayers = [env.n_obs + n_discrim, 64, 64, n_discrim]
    discrim = networks.VariableNetwork(dlayers, args.device)
    layers = [128, 128, 128, 128]
    network = MLPActorCriticPG(env.n_obs + n_discrim, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)
    
    # make the agent
    agent = Agent_PPO_Discriminator(device=args.device)
    # agent.params.steps_per_epoch = 10 # for testing only!! Disable at runtime
    agent.init(network, discrim)
    agent.get_target_vector = env._object_discrimination_target

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "improve_small_spheres":

    # define what to vary this training, dependent on job number
    vary_1 = [False, True]
    vary_2 = [10, 20]
    vary_3 = [
      (1.0, False),
      (1.0, True),
      (1.5, False),
      (1.5, True),
    ]
    repeats = 5
    tm.param_1_name = "extra action penalty"
    tm.param_2_name = "object pos noise"
    tm.param_3_name = "palm action scale/Y motor"
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply training specific settings
    tm.settings["trainer"]["num_episodes"] = 80_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp_smallspheres"
    tm.settings["reward"]["action_pen_lin"]["used"] = tm.param_1
    tm.settings["env"]["object_position_noise_mm"] = tm.param_2
    tm.settings["cpp"]["action"]["gripper_Z"]["value"] *= tm.param_3[0]

    # create the environment
    env = tm.make_env()

    if tm.param_3[1]:

      # disable existing prismatic/revolute joints
      env.mj.set.gripper_prismatic_X.in_use = False
      env.mj.set.gripper_revolute_Y.in_use = False

      # enable direct motor control joints
      env.mj.set.gripper_X.set(True, 2e-3, -1)
      env.mj.set.gripper_Y.set(True, 2e-3, -1)

    # apply the agent settings
    layers = [128, 128, 128, 128]
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)
    
    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "pre_paper_baseline":

    # define what to vary this training, dependent on job number
    vary_1 = [60, 75, 90]
    vary_2 = [0.9e-3, 1.0e-3]
    vary_3 = None
    repeats = 10
    tm.param_1_name = "finger hook angle"
    tm.param_2_name = "finger thickness"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply training specific settings
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp_smallspheres"
    tm.settings["env"]["finger_hook_angle_degrees"] = tm.param_1
    tm.settings["env"]["finger_thickness"] = tm.param_2

    # create the environment
    env = tm.make_env()

    # apply the agent settings
    layers = [128, 128, 128, 128]
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)
    
    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "finetune_pre_paper_baseline":

    # Program: pre_paper_baseline
    timestamp = "01-12-23_17-23"

    # define what to vary this training, dependent on job number
    vary_1 = [
      5, 8,   # 60x0.9
      18, 20, # 75x0.9
      22, 25, # 90x0.9
    ]
    vary_2 = None
    vary_3 = None
    repeats = 5
    tm.param_1_name = "job num"
    tm.param_2_name = None
    tm.param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    tm.load(job_num=param_1, timestamp=timestamp, best_id=True,
            load_into_new_training=True)
    tm.param_1 = param_1
    tm.param_2 = param_2
    tm.param_3 = param_3

    # apply parameter changes
    wrist_limit = 4
    tm.settings["reward"]["wrist"]["exceed"] = wrist_limit * 0.75
    tm.settings["reward"]["wrist"]["dangerous"] = wrist_limit
    tm.trainer.env.mj.set.wrist_sensor_Z.normalise = wrist_limit * 1.5
    tm.create_reward_function(tm.trainer.env)

    # record that our curriculum has changed and final test should be new model only
    tm.trainer.curriculum_dict["stage"] += 1
    tm.settings["final_test_max_stage"] = True
    tm.settings["final_test_only_stage"] = None

    # now continue training
    extra_episodes = 60_000
    tm.continue_training(extra_episodes=extra_episodes)
    print_time_taken()

  elif args.program == "finetune_pre_paper_baseline_2":

    # Program: pre_paper_baseline
    timestamp = "01-12-23_17-23"

    # define what to vary this training, dependent on job number
    vary_1 = [
      5, 8,   # 60x0.9
      18, 20, # 75x0.9
      22, 25, # 90x0.9
    ]
    vary_2 = None
    vary_3 = None
    repeats = 5
    tm.param_1_name = "job num"
    tm.param_2_name = None
    tm.param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    tm.load(job_num=param_1, timestamp=timestamp, best_id=True,
            load_into_new_training=True)
    tm.param_1 = param_1
    tm.param_2 = param_2
    tm.param_3 = param_3

    # apply parameter changes
    wrist_limit = 4
    tm.settings["reward"]["wrist"]["exceed"] = wrist_limit * 0.75
    tm.settings["reward"]["wrist"]["dangerous"] = wrist_limit
    tm.settings["reward"]["action_pen_lin"]["used"] = True
    tm.create_reward_function(tm.trainer.env)
    tm.trainer.env.mj.set.wrist_sensor_Z.normalise = wrist_limit * 1.5
    tm.trainer.env.mj.set.object_stable.trigger = 8
    tm.trainer.env.mj.set.palm_sensor.normalise = 4

    # record that our curriculum has changed and final test should be new model only
    tm.trainer.curriculum_dict["stage"] += 1
    tm.settings["final_test_max_stage"] = True
    tm.settings["final_test_only_stage"] = None

    # now continue training
    extra_episodes = 60_000
    tm.continue_training(extra_episodes=extra_episodes)
    print_time_taken()

  elif args.program == "paper_testing":

    # define what to vary this training, dependent on job number
    vary_1 = [1, 3]
    vary_2 = [4, 6]
    vary_3 = None
    repeats = 5
    tm.param_1_name = "dangerous trigger"
    tm.param_2_name = "palm normalise"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply training specific settings
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp_smallspheres"
    tm.settings["env"]["finger_hook_angle_degrees"] = 75
    tm.settings["env"]["finger_thickness"] = 0.9e-3

    # try tuning
    wrist_limit = 4
    tm.settings["reward"]["wrist"]["exceed"] = wrist_limit * 0.75
    tm.settings["reward"]["wrist"]["dangerous"] = wrist_limit
    tm.settings["reward"]["action_pen_lin"]["used"] = True
    tm.settings["reward"]["dangerous_trigger"] = tm.param_1

    # create the environment
    env = tm.make_env()

    env.mj.set.wrist_sensor_Z.normalise = wrist_limit * 1.5
    env.mj.set.object_stable.trigger = 8
    env.mj.set.palm_sensor.normalise = tm.param_2

    # apply the agent settings
    layers = [128, 128, 128, 128]
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)
    
    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "paper_testing_2":

    # define what to vary this training, dependent on job number
    vary_1 = [1e-5, 5e-5]
    vary_2 = [2, 4, 6]
    vary_3 = None
    repeats = 5
    tm.param_1_name = "learning rate"
    tm.param_2_name = "num layers"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply training specific settings
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp_smallspheres"
    tm.settings["env"]["finger_hook_angle_degrees"] = 75
    tm.settings["env"]["finger_thickness"] = 0.9e-3

    # create the environment
    env = tm.make_env()

    # apply the agent settings
    layers = [128 for i in range(tm.param_2)]
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)
    
    # make the agent
    tm.settings["Agent_PPO"]["learning_rate_pi"] = tm.param_1
    tm.settings["Agent_PPO"]["learning_rate_vf"] = tm.param_1
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "paper_baseline_1":

    # define what to vary this training, dependent on job number
    vary_1 = [0, 1, 2, 3]
    vary_2 = None
    vary_3 = None
    repeats = 15
    tm.param_1_name = "num sensors"
    tm.param_2_name = None
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply training specific settings
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp_smallspheres"
    tm.settings["env"]["finger_hook_angle_degrees"] = 75
    tm.settings["env"]["finger_thickness"] = 0.9e-3

    if tm.param_1 < 1: tm.settings["cpp"]["sensor"]["bending_gauge"]["in_use"] = False
    if tm.param_1 < 2: tm.settings["cpp"]["sensor"]["palm_sensor"]["in_use"] = False
    if tm.param_1 < 3: tm.settings["cpp"]["sensor"]["wrist_sensor_Z"]["in_use"] = False

    # create the environment
    env = tm.make_env()

    # apply the agent settings
    layers = [128 for i in range(4)]
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)
    
    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)

    # add an extra test on the old object set
    tm.run_test(trials_per_obj=20, different_object_set="set8_fullset_1500",
                load_best_id=True)
    
    print_time_taken()

  elif args.program == "paper_baseline_1_extended":

    # 0-15      0 EI1
    # 16-30     1 EI1
    # 31-45     2 EI1
    # 46-60     3 EI1
    # 61-75     0 EI2
    # 76-90     1 EI2
    # 91-105    2 EI2
    # 106-120   3 EI2
    # 121-135   0 EI3
    # 136-150   1 EI3
    # 151-165   2 EI3
    # 166-180   3 EI3

    # define what to vary this training, dependent on job number
    vary_1 = [0, 1, 2, 3]
    vary_2 = [
      (0.9e-3, 28e-3),
      (1.0e-3, 24e-3),
      (1.0e-3, 28e-3)
    ]
    vary_3 = None
    repeats = 15
    tm.param_1_name = "num sensors"
    tm.param_2_name = "finger dimension"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply training specific settings
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp_smallspheres"
    tm.settings["env"]["finger_hook_angle_degrees"] = 75
    tm.settings["env"]["finger_thickness"] = tm.param_2[0]
    tm.settings["env"]["finger_width"] = tm.param_2[1]

    if tm.param_1 < 1: tm.settings["cpp"]["sensor"]["bending_gauge"]["in_use"] = False
    if tm.param_1 < 2: tm.settings["cpp"]["sensor"]["palm_sensor"]["in_use"] = False
    if tm.param_1 < 3: tm.settings["cpp"]["sensor"]["wrist_sensor_Z"]["in_use"] = False

    # create the environment
    env = tm.make_env()

    # apply the agent settings
    layers = [128 for i in range(4)]
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)
    
    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)

    # add an extra test on the old object set
    tm.run_test(trials_per_obj=20, different_object_set="set8_fullset_1500",
                load_best_id=True)
    
    print_time_taken()

  elif args.program == "test_cnn_ppo":

    # define what to vary this training, dependent on job number
    vary_1 = [1e-6, 1e-5, 1e-4]
    vary_2 = [(50, 50)]
    vary_3 = None
    repeats = None
    tm.param_1_name = "learning rate"
    tm.param_2_name = "image size"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply training specific settings
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp_smallspheres"
    tm.settings["env"]["finger_hook_angle_degrees"] = 75
    tm.settings["env"]["finger_thickness"] = 0.9e-3
    tm.settings["env"]["depth_camera"] = True
    tm.settings["env"]["use_rgb_in_observation"] = True

    # create the environment
    env = tm.make_env()
    env._set_rgbd_size(*tm.param_2)

    # apply the agent settings
    network = CNNActorCriticPG([3, *tm.param_2], env.n_obs, env.n_actions,
                               continous_actions=True, device=args.device)
    
    # make the agent
    tm.settings["Agent_PPO"]["learning_rate_pi"] = tm.param_1
    tm.settings["Agent_PPO"]["learning_rate_vf"] = tm.param_1
    tm.settings["Agent_PPO"]["steps_per_epoch"] = 6000
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "test_scene_grasping":

    # define what to vary this training, dependent on job number
    vary_1 = [False, True]
    vary_2 = None
    vary_3 = None
    repeats = None
    tm.param_1_name = "use termination"
    tm.param_2_name = None
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()
    
    # prepare the environment
    tm.settings["trainer"]["num_episodes"] = 200_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp_smallspheres"
    tm.settings["env"]["finger_hook_angle_degrees"] = 75
    tm.settings["env"]["finger_thickness"] = 0.9e-3
    tm.settings["env"]["XY_base_actions"] = True
    tm.settings["env"]["base_lim_X_mm"] = 150
    tm.settings["env"]["base_lim_Y_mm"] = 50
    tm.settings["env"]["depth_camera"] = True
    tm.settings["env"]["use_rgb_in_observation"] = True
    tm.settings["env"]["use_depth_in_observation"] = False

    # now prepare the grasping scene
    tm.settings["env"]["use_scene_settings"] = True
    tm.settings["env"]["object_position_noise_mm"] = 1000 # use min/max
    tm.settings["env"]["num_objects_in_scene"] = 3
    tm.settings["env"]["scene_X_dimension_mm"] = 300
    tm.settings["env"]["scene_Y_dimension_mm"] = 200

    # update the actions and sensors of the gripper
    tm.settings["cpp"]["oob_distance"] = 100 # never trigger
    tm.settings["cpp"]["action"]["base_X"]["in_use"] = True
    tm.settings["cpp"]["action"]["base_Y"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["base_state_sensor_XY"]["in_use"] = True

    # make grasping easier - no penalty termination for dangerous forces
    tm.settings["reward"]["penalty_termination"] = tm.param_1

    # initial testing parameters for PPO   
    rgb_size = (50, 50)
    lr = 1e-5

    # create the environment
    env = tm.make_env()
    env._set_rgbd_size(*rgb_size)

    # apply the agent settings
    network = CNNActorCriticPG([3, *rgb_size], env.n_obs, env.n_actions,
                               continous_actions=True, device=args.device)
    
    # make the agent
    tm.settings["Agent_PPO"]["learning_rate_pi"] = lr
    tm.settings["Agent_PPO"]["learning_rate_vf"] = lr
    tm.settings["Agent_PPO"]["steps_per_epoch"] = 6000
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "palm_vs_no_palm_1":

    # define what to vary this training, dependent on job number
    vary_1 = [45, 60, 75, 90]
    vary_2 = [False, True]
    vary_3 = [(0.9e-3, 28e-3), (1.0e-3, 24e-3), (1.0e-3, 28e-3)]
    repeats = 10
    tm.param_1_name = "fingertip angle"
    tm.param_2_name = "use palm"
    tm.param_3_name = "finger thickness/width"
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply training specific settings
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp_smallspheres"
    tm.settings["env"]["finger_hook_angle_degrees"] = tm.param_1
    tm.settings["env"]["finger_thickness"] = tm.param_3[0]
    tm.settings["env"]["finger_width"] = tm.param_3[1]

    if not tm.param_2:

      # remove palm action
      tm.settings["cpp"]["action"]["gripper_Z"]["in_use"] = False

      # remove palm sensor
      tm.settings["cpp"]["sensor"]["palm_sensor"]["in_use"] = False

      # remove palm requirement for stable grasp (-ve means it is always reached with 0 force)
      tm.settings["reward"]["palm"]["min"] = -1.1
      tm.settings["cpp"]["stable_palm_force"] = -1.0

    # create the environment
    env = tm.make_env()

    # apply the agent settings
    layers = [128 for i in range(4)]
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)
    
    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)

    # add an extra test on the old object set
    tm.run_test(trials_per_obj=20, different_object_set="set8_fullset_1500",
                load_best_id=True)

    print_time_taken()

  elif args.program == "debug_cnn_localisation":

    # define what to vary this training, dependent on job number
    vary_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    vary_2 = [(50, 50)]
    vary_3 = None
    repeats = None
    tm.param_1_name = "learning rate"
    tm.param_2_name = "image size"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # prepare the environment
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp_smallspheres"
    tm.settings["env"]["finger_hook_angle_degrees"] = 75
    tm.settings["env"]["finger_thickness"] = 0.9e-3
    tm.settings["env"]["XY_base_actions"] = True
    tm.settings["env"]["base_lim_X_mm"] = 50
    tm.settings["env"]["base_lim_Y_mm"] = 50
    tm.settings["env"]["depth_camera"] = True
    tm.settings["env"]["use_rgb_in_observation"] = True
    tm.settings["env"]["use_depth_in_observation"] = False

    # don't use scene, just add noise
    tm.settings["env"]["use_scene_settings"] = False
    tm.settings["env"]["object_position_noise_mm"] = 50

    # update the actions and sensors of the gripper
    tm.settings["cpp"]["oob_distance"] = 1e6
    tm.settings["cpp"]["action"]["base_X"]["in_use"] = True
    tm.settings["cpp"]["action"]["base_Y"]["in_use"] = True
    tm.settings["cpp"]["action"]["base_X"]["value"] = 4e-3
    tm.settings["cpp"]["action"]["base_Y"]["value"] = 4e-3
    tm.settings["cpp"]["sensor"]["base_state_sensor_XY"]["in_use"] = True

    # disable regular actions, only examine base movement
    tm.settings["cpp"]["action"]["gripper_prismatic_X"]["in_use"] = False
    tm.settings["cpp"]["action"]["gripper_revolute_Y"]["in_use"] = False
    tm.settings["cpp"]["action"]["gripper_Z"]["in_use"] = False
    tm.settings["cpp"]["action"]["base_Z"]["in_use"] = False

    # disable regular rewards and prepare for distance reward
    tm.settings["cpp"]["XY_distance_threshold"] = 10e-3
    tm.settings["reward"]["penalty_termination"] = False

    # create the environment
    env = tm.make_env()
    env._set_rgbd_size(*tm.param_2)

    # now create distance rewards
    env.mj.set.within_XY_distance.set(1.0, True, 1)
    env.mj.set.object_XY_distance.set(1.0/250, False, 1, -200e-3, -10e-3, -1)

    # apply the agent settings
    network = CNNActorCriticPG([3, *tm.param_2], env.n_obs, env.n_actions,
                               continous_actions=True, device=args.device)
    
    # make the agent
    tm.settings["Agent_PPO"]["learning_rate_pi"] = tm.param_1
    tm.settings["Agent_PPO"]["learning_rate_vf"] = tm.param_1
    tm.settings["Agent_PPO"]["steps_per_epoch"] = 6000
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "ppo_cnn_single_object":

    # define what to vary this training, dependent on job number
    vary_1 = [1e-6, 1e-5, 1e-4]
    vary_2 = [(25, 25), (50, 50)]
    vary_3 = None
    repeats = None
    tm.param_1_name = "learning rate"
    tm.param_2_name = "image size"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # add in the camera
    tm.settings["env"]["depth_camera"] = True
    tm.settings["env"]["use_rgb_in_observation"] = True
    tm.settings["env"]["image_width"] = tm.param_2[0]
    tm.settings["env"]["image_height"] = tm.param_2[1]

    # turn up noise and add in XY base actions
    tm.settings["env"]["object_position_noise_mm"] = 30
    tm.settings["env"]["XY_base_actions"] = True
    tm.settings["env"]["base_lim_X_mm"] = 50
    tm.settings["env"]["base_lim_Y_mm"] = 50

    # update oob, actions, and sensors of the gripper
    tm.settings["cpp"]["oob_distance"] = 60e-3
    tm.settings["cpp"]["action"]["base_X"]["in_use"] = True
    tm.settings["cpp"]["action"]["base_Y"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["base_state_sensor_XY"]["in_use"] = True

    # create the environment
    env = tm.make_env()

    # apply the agent settings
    network = CNNActorCriticPG((3, env.params.image_width, env.params.image_height), 
                               env.n_obs, env.n_actions, continous_actions=True, 
                               device=args.device)
    
    # make the agent
    tm.settings["Agent_PPO"]["learning_rate_pi"] = tm.param_1
    tm.settings["Agent_PPO"]["learning_rate_vf"] = tm.param_1
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "ppo_encoder_test":

    # define what to vary this training, dependent on job number
    vary_1 = [1e-6, 1e-5, 1e-4]
    vary_2 = [(200, 100)]
    vary_3 = None
    repeats = None
    tm.param_1_name = "learning rate"
    tm.param_2_name = "image size"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # add in the camera
    tm.settings["env"]["depth_camera"] = True
    tm.settings["env"]["use_rgb_in_observation"] = True
    tm.settings["env"]["image_width"] = tm.param_2[0]
    tm.settings["env"]["image_height"] = tm.param_2[1]

    # turn up noise and add in XY base actions
    tm.settings["env"]["object_position_noise_mm"] = 30
    tm.settings["env"]["XY_base_actions"] = True
    tm.settings["env"]["base_lim_X_mm"] = 50
    tm.settings["env"]["base_lim_Y_mm"] = 50

    # update oob, actions, and sensors of the gripper
    tm.settings["cpp"]["oob_distance"] = 60e-3
    tm.settings["cpp"]["action"]["base_X"]["in_use"] = True
    tm.settings["cpp"]["action"]["base_Y"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["base_state_sensor_XY"]["in_use"] = True

    # add in image rendering with the encoder
    tm.settings["env"]["use_rgb_rendering"] = True
    tm.settings["env"]["rgb_rendering_method"] = "cycleGAN_encoder"

    # create the environment
    env = tm.make_env()

    # apply the agent settings
    img_size = (256, 32, 32)
    network = NetActorCriticPG(MixedNetworkFromEncoder, img_size,
                               env.n_obs, env.n_actions, continous_actions=True, 
                               device=args.device)
    
    # make the agent
    tm.settings["Agent_PPO"]["learning_rate_pi"] = tm.param_1
    tm.settings["Agent_PPO"]["learning_rate_vf"] = tm.param_1
    tm.settings["Agent_PPO"]["steps_per_epoch"] = 4000
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "ppo_cnn_single_object_curriculum":

    # define what to vary this training, dependent on job number
    vary_1 = [5e-6, 1e-5, 5e-5]
    vary_2 = [False, True]
    vary_3 = None
    repeats = None
    tm.param_1_name = "learning rate"
    tm.param_2_name = "use encoder"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # add in the camera
    tm.settings["env"]["depth_camera"] = True
    tm.settings["env"]["use_rgb_in_observation"] = True
    tm.settings["env"]["image_width"] = 200
    tm.settings["env"]["image_height"] = 100

    # enable image transforms and set network image sizes
    tm.settings["env"]["use_standard_transform"] = True
    tm.settings["env"]["transform_resize_square"] = 58
    tm.settings["env"]["transform_crop_size"] = 52

    # turn up noise and add in XY base actions
    tm.settings["env"]["object_position_noise_mm"] = 30
    tm.settings["env"]["XY_base_actions"] = True
    tm.settings["env"]["base_lim_X_mm"] = 50
    tm.settings["env"]["base_lim_Y_mm"] = 50

    # update oob, actions, and sensors of the gripper
    tm.settings["cpp"]["oob_distance"] = 60e-3
    tm.settings["cpp"]["action"]["base_X"]["in_use"] = True
    tm.settings["cpp"]["action"]["base_Y"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["base_state_sensor_XY"]["in_use"] = True

    # add in curriculum where grasping gets harder
    tm.settings["trainer"]["use_curriculum"] = True
    tm.settings["curriculum"]["metric_name"] = "success_rate"
    tm.settings["curriculum"]["metric_thresholds"] = [0.5 for i in range(3)]
    tm.settings["curriculum"]["change_fcn"] = curriculum_change_navigation_grasp

    # add in image rendering with the encoder
    if tm.param_2:
      tm.settings["env"]["use_rgb_rendering"] = True
      tm.settings["env"]["rgb_rendering_method"] = "cycleGAN_encoder"

    # create the environment
    env = tm.make_env()

    # create the agent network
    if tm.param_2:
      obs_size = (256, 13, 13)
      network = NetActorCriticPG(MixedNetworkFromEncoder, obs_size,
                                 env.n_obs, env.n_actions, continous_actions=True, 
                                 device=args.device)
    else:
      obs_size = (3, env.params.transform_crop_size, 
                  env.params.transform_crop_size)
      network = CNNActorCriticPG(obs_size, env.n_obs, env.n_actions, 
                                continous_actions=True, device=args.device)
    
    # make the agent
    tm.settings["Agent_PPO"]["learning_rate_pi"] = tm.param_1
    tm.settings["Agent_PPO"]["learning_rate_vf"] = tm.param_1
    tm.settings["Agent_PPO"]["steps_per_epoch"] = 2800
    agent = Agent_PPO(device=args.device, steps=tm.settings["Agent_PPO"]["steps_per_epoch"])
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "ppo_cnn_single_object_curriculum_2":

    # define what to vary this training, dependent on job number
    vary_1 = [5e-5, 8e-5]
    vary_2 = [False, True]
    vary_3 = None
    repeats = 1
    tm.param_1_name = "learning rate"
    tm.param_2_name = "use encoder"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # add in the camera
    tm.settings["env"]["depth_camera"] = True
    tm.settings["env"]["use_rgb_in_observation"] = True
    tm.settings["env"]["image_width"] = 200
    tm.settings["env"]["image_height"] = 100

    # enable image transforms and set network image sizes
    tm.settings["env"]["use_standard_transform"] = True
    tm.settings["env"]["transform_resize_square"] = 58
    tm.settings["env"]["transform_crop_size"] = 52 # must be a multiple of 4 (mult in the network)

    # turn up noise and add in XY base actions
    tm.settings["env"]["object_position_noise_mm"] = 30
    tm.settings["env"]["XY_base_actions"] = True
    tm.settings["env"]["base_lim_X_mm"] = 50
    tm.settings["env"]["base_lim_Y_mm"] = 50

    # enable base yaw
    tm.settings["env"]["Z_base_rotation"] = True
    tm.settings["cpp"]["action"]["base_yaw"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["base_state_sensor_yaw"]["in_use"] = True

    # update oob, actions, and sensors of the gripper
    tm.settings["cpp"]["oob_distance"] = 75e-3
    tm.settings["cpp"]["action"]["base_X"]["in_use"] = True
    tm.settings["cpp"]["action"]["base_Y"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["base_state_sensor_XY"]["in_use"] = True

    # add in curriculum where grasping gets harder
    tm.settings["trainer"]["use_curriculum"] = True
    tm.settings["curriculum"]["metric_name"] = "success_rate"
    tm.settings["curriculum"]["metric_thresholds"] = [0.6 for i in range(4)]
    tm.settings["curriculum"]["param_values"] = [10, 20, 30, 40, 50]
    tm.settings["curriculum"]["change_fcn"] = curriculum_change_object_noise

    # enable reward for getting close to objects
    tm.settings["reward"]["object_XY_distance"]["used"] = False # TEST NOT USING THIS

    # TESTING: try using faster simulation
    tm.settings["env"]["num_segments"] = 6

    # add in image rendering with the encoder
    if tm.param_2:
      tm.settings["env"]["use_rgb_rendering"] = True
      tm.settings["env"]["rgb_rendering_method"] = "cycleGAN_encoder"

    # create the environment
    env = tm.make_env()

    # create the agent network
    if tm.param_2:
      ngf = 8 # set in the GAN training options
      mult = 4 # set in the GAN network itself
      bottleneck = 4
      channels = int((ngf * mult) / float(bottleneck))
      img_x = int(tm.settings["env"]["transform_crop_size"] / float(mult))
      obs_size = (channels, img_x, img_x)
      # network = NetActorCriticPG(MixedNetworkFromEncoder, obs_size,
      #                            env.n_obs, env.n_actions, continous_actions=True, 
      #                            device=args.device)
      network = NetActorCriticPG(networks.MixedNetworkFromEncoder2, obs_size,
                                 env.n_obs, env.n_actions, continous_actions=True, 
                                 device=args.device)
    else:
      obs_size = (3, env.params.transform_crop_size, 
                  env.params.transform_crop_size)
      network = CNNActorCriticPG(obs_size, env.n_obs, env.n_actions, 
                                 continous_actions=True, device=args.device)
    
    # make the agent
    tm.settings["Agent_PPO"]["learning_rate_pi"] = tm.param_1
    tm.settings["Agent_PPO"]["learning_rate_vf"] = tm.param_1
    tm.settings["Agent_PPO"]["steps_per_epoch"] = 4000
    agent = Agent_PPO(device=args.device, steps=tm.settings["Agent_PPO"]["steps_per_epoch"])
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "ppo_cnn_single_object_curriculum_3":

    # define what to vary this training, dependent on job number
    vary_1 = [5e-5, 8e-5]
    vary_2 = [1500, 3000, 6000]
    vary_3 = None
    repeats = 1
    tm.param_1_name = "learning rate"
    tm.param_2_name = "steps per epoch"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # add in the camera
    tm.settings["env"]["depth_camera"] = True
    tm.settings["env"]["use_rgb_in_observation"] = True
    tm.settings["env"]["image_width"] = 200
    tm.settings["env"]["image_height"] = 100

    # enable image transforms and set network image sizes
    tm.settings["env"]["use_standard_transform"] = True
    tm.settings["env"]["transform_resize_square"] = 58
    tm.settings["env"]["transform_crop_size"] = 52

    # turn up noise and add in XY base actions
    tm.settings["env"]["object_position_noise_mm"] = 30
    tm.settings["env"]["XY_base_actions"] = True
    tm.settings["env"]["base_lim_X_mm"] = 50
    tm.settings["env"]["base_lim_Y_mm"] = 50

    # enable base yaw
    tm.settings["env"]["Z_base_rotation"] = True
    tm.settings["cpp"]["action"]["base_yaw"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["base_state_sensor_yaw"]["in_use"] = True

    # update oob, actions, and sensors of the gripper
    # tm.settings["cpp"]["oob_distance"] = 70e-3
    tm.settings["cpp"]["action"]["base_X"]["in_use"] = True
    tm.settings["cpp"]["action"]["base_Y"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["base_state_sensor_XY"]["in_use"] = True

    # add in curriculum where grasping gets harder
    tm.settings["trainer"]["use_curriculum"] = True
    tm.settings["curriculum"]["metric_name"] = "success_rate"
    tm.settings["curriculum"]["metric_thresholds"] = [0.6 for i in range(4)]
    tm.settings["curriculum"]["param_values"] = [10, 20, 30, 40, 50]
    tm.settings["curriculum"]["change_fcn"] = curriculum_change_object_noise

    # enable reward for getting close to objects
    tm.settings["reward"]["object_XY_distance"]["used"] = True

    # # try using faster simulation
    # tm.settings["env"]["num_segments"] = 5

    # create the environment
    env = tm.make_env()

    # create the agent network
    obs_size = (3, env.params.transform_crop_size, env.params.transform_crop_size)
    network = CNNActorCriticPG(obs_size, env.n_obs, env.n_actions, 
                              continous_actions=True, device=args.device)
    
    # make the agent
    tm.settings["Agent_PPO"]["learning_rate_pi"] = tm.param_1
    tm.settings["Agent_PPO"]["learning_rate_vf"] = tm.param_1
    tm.settings["Agent_PPO"]["steps_per_epoch"] = tm.param_2
    agent = Agent_PPO(device=args.device, steps=tm.settings["Agent_PPO"]["steps_per_epoch"])
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "continue_good_curriculum":

    # from program: ppo_cnn_single_object_curriculum_2
    timestamp = "31-01-24_18-23"
    job_num = 5

    torch.set_default_device("cuda")

    # define what to vary this training, dependent on job number
    vary_1 = [False, True]
    vary_2 = None
    vary_3 = None
    repeats = 2
    tm.param_1_name = "best_id"

  elif args.program == "high_noise_image_collection":

    # load the best training, but we continue from the end
    job = 67
    timestamp = "19-01-24_16-54"
    tm.load(job_num=job, timestamp=timestamp, best_id=True,
            load_into_new_training=False)

    # increase object noise
    tm.trainer.env.params.object_position_noise_mm = 200
    tm.trainer.env.params.max_episode_steps = 100
    tm.trainer.env.mj.set.oob.done = False

    # setup the camera
    tm.trainer.env._init_rgbd()
    tm.trainer.env._set_rgbd_size(width=848, height=480)

    import functools

    # enable env image collection with relatively high chance
    tm.trainer.env.randomise_colours_every_step = True
    tm.trainer.env.collect_images = True
    tm.trainer.env.load_next.depth_camera = True
    tm.trainer.env.image_collection_chance = 1.0 / 100.0
    tm.trainer.images_collected = 0
    tm.trainer.image_batches_collected = 0
    tm.trainer.image_list = []
    tm.trainer.image_collection_num_per_batch = 1000
    tm.trainer.image_collection_max_batches = 10
    tm.trainer.episode_fcn = functools.partial(tm.trainer.image_collection_fcn)

    tm.trainer.env.load()

    # essentially disable learning so behaviour is constant
    tm.trainer.agent.params.learning_rate_pi = 1e-10
    tm.trainer.agent.params.learning_rate_vf = 1e-10

    # now continue training
    extra_episodes = 80_000
    tm.continue_training(extra_episodes=extra_episodes)
    print_time_taken()

  elif args.program == "good_grasps_image_collection":

    # Program: pre_paper_baseline
    timestamp = "19-01-24_16-54"

    # define what to vary this training, dependent on job number
    vary_1 = [
      46, 56, 66, 76,
    ]
    vary_2 = None
    vary_3 = None
    repeats = 1
    tm.param_1_name = "job num"
    tm.param_2_name = None
    tm.param_3_name = None
    param_1, param_2, param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    tm.load(job_num=param_1, timestamp=timestamp, best_id=True,
            load_into_new_training=True)
    tm.param_1 = param_1
    tm.param_2 = param_2
    tm.param_3 = param_3
    
    # shorten episodes as grasping is already learned
    tm.trainer.env.params.max_episode_steps = 70 #100

    # increase object noise
    tm.trainer.env.params.object_position_noise_mm = 200 #20
    tm.trainer.env.mj.set.oob.done = False

    import functools

    # enable env image collection with relatively high chance
    tm.trainer.env.randomise_colours_every_step = True
    tm.trainer.env.collect_images = True
    tm.trainer.env.image_collection_chance = 1.0 / 100.0
    tm.trainer.images_collected = 0
    tm.trainer.image_batches_collected = 0
    tm.trainer.image_list = []
    tm.trainer.image_collection_num_per_batch = 1000
    tm.trainer.image_collection_max_batches = 3
    tm.trainer.episode_fcn = functools.partial(tm.trainer.image_collection_fcn)

    # load in the depth camera
    tm.trainer.env.params.image_height = 480
    tm.trainer.env.params.image_width = 848
    tm.trainer.env.load(object_set_name="set9_fullset", depth_camera=True)

    # essentially disable learning so behaviour is constant
    tm.trainer.agent.params.learning_rate_pi = 1e-10
    tm.trainer.agent.params.learning_rate_vf = 1e-10

    # now continue training
    extra_episodes = 10_000
    tm.continue_training(extra_episodes=extra_episodes)
    print_time_taken()

  elif args.program == "paper_baseline_1_rigid_fingers":

    # define what to vary this training, dependent on job number
    vary_1 = [True]
    vary_2 = None
    vary_3 = None
    repeats = 15
    tm.param_1_name = "rigid fingers"
    tm.param_2_name = None
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply training specific settings
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp_smallspheres"
    tm.settings["env"]["finger_hook_angle_degrees"] = 75
    tm.settings["env"]["finger_thickness"] = 0.9e-3

    # make fingers rigid
    tm.settings["env"]["num_segments"] = 1

    # create the environment
    env = tm.make_env()

    # apply the agent settings
    layers = [128 for i in range(4)]
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                continous_actions=True)
    
    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)

    # add an extra test on the old object set
    tm.run_test(trials_per_obj=20, different_object_set="set8_fullset_1500",
                load_best_id=True)
    
    print_time_taken()

  elif args.program == "learning_check":

    # define what to vary this training, dependent on job number
    vary_1 = [1, 3, 4]
    vary_2 = [False, True]
    vary_3 = None
    repeats = 2
    tm.param_1_name = "num base actions"
    tm.param_2_name = "use vision"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # set very little object position noise and the old default oob distance
    tm.settings["env"]["object_position_noise_mm"] = 10
    tm.settings["cpp"]["oob_distance"] = 75e-3

    if tm.param_1 >= 3:
      # enable base XY movements
      tm.settings["env"]["XY_base_actions"] = True
      tm.settings["env"]["base_lim_X_mm"] = 50
      tm.settings["env"]["base_lim_Y_mm"] = 50
      tm.settings["cpp"]["action"]["base_X"]["in_use"] = True
      tm.settings["cpp"]["action"]["base_Y"]["in_use"] = True
      tm.settings["cpp"]["sensor"]["base_state_sensor_XY"]["in_use"] = True

    if tm.param_1 >= 4:
      # enable base yaw
      tm.settings["env"]["Z_base_rotation"] = True
      tm.settings["cpp"]["action"]["base_yaw"]["in_use"] = True
      tm.settings["cpp"]["sensor"]["base_state_sensor_yaw"]["in_use"] = True

    if tm.param_2:
      # add in the camera
      tm.settings["env"]["depth_camera"] = True
      tm.settings["env"]["use_rgb_in_observation"] = True
      tm.settings["env"]["image_width"] = 200
      tm.settings["env"]["image_height"] = 100
      tm.settings["env"]["use_standard_transform"] = True
      tm.settings["env"]["transform_resize_square"] = 58
      tm.settings["env"]["transform_crop_size"] = 52

    # create the environment
    env = tm.make_env()

    if tm.param_2:
      # create the agent CNN network
      obs_size = (3, env.params.transform_crop_size, env.params.transform_crop_size)
      network = CNNActorCriticPG(obs_size, env.n_obs, env.n_actions, 
                                continous_actions=True, device=args.device)
    else:
      # regular linear network
      layers = [128 for i in range(4)]
      network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                                  continous_actions=True)
    
    # make the agent
    tm.settings["Agent_PPO"]["learning_rate_pi"] = 5e-5
    tm.settings["Agent_PPO"]["learning_rate_vf"] = 5e-5
    tm.settings["Agent_PPO"]["steps_per_epoch"] = 4000
    agent = Agent_PPO(device=args.device, steps=tm.settings["Agent_PPO"]["steps_per_epoch"])
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "test_expert":

    # define what to vary this training, dependent on job number
    vary_1 = [True, False]
    vary_2 = None
    vary_3 = None
    repeats = 2
    tm.param_1_name = "use feedforward"
    tm.param_2_name = None
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # set very little object position noise and the old default oob distance
    tm.settings["env"]["object_position_noise_mm"] = 10
    tm.settings["cpp"]["oob_distance"] = 75e-3

    # enable base XY movements
    tm.settings["env"]["XY_base_actions"] = True
    tm.settings["env"]["base_lim_X_mm"] = 50
    tm.settings["env"]["base_lim_Y_mm"] = 50
    tm.settings["cpp"]["action"]["base_X"]["in_use"] = True
    tm.settings["cpp"]["action"]["base_Y"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["base_state_sensor_XY"]["in_use"] = True
  
    # enable base yaw
    tm.settings["env"]["Z_base_rotation"] = True
    tm.settings["cpp"]["action"]["base_yaw"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["base_state_sensor_yaw"]["in_use"] = True

    # add in the camera
    tm.settings["env"]["depth_camera"] = True
    tm.settings["env"]["use_rgb_in_observation"] = True
    tm.settings["env"]["image_width"] = 200
    tm.settings["env"]["image_height"] = 100
    tm.settings["env"]["use_standard_transform"] = True
    tm.settings["env"]["transform_resize_square"] = 58
    tm.settings["env"]["transform_crop_size"] = 52

    # create the environment
    env = tm.make_env()

    # load in the expert
    feedsize = 4 if tm.param_1 else 0
    env._load_expert_model(timestamp="08-12-23_19-19", job=53)

    # create the agent CNN network
    obs_size = (3, env.params.transform_crop_size, env.params.transform_crop_size)
    # network = CNNActorCriticPG(obs_size, env.n_obs, env.n_actions, 
    #                           continous_actions=True, device=args.device)
    network = NetActorCriticPG(networks.MxNetFeedforward,
                               obs_size, env.n_obs, env.n_actions, 
                               continous_actions=True, device=args.device,
                               netargs={ "feedforwardsize" : feedsize })
    
    # make the agent
    tm.settings["Agent_PPO"]["learning_rate_pi"] = 5e-5
    tm.settings["Agent_PPO"]["learning_rate_vf"] = 5e-5
    tm.settings["Agent_PPO"]["steps_per_epoch"] = 4000
    agent = Agent_PPO(device=args.device, steps=tm.settings["Agent_PPO"]["steps_per_epoch"])
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "expert_onto_curriculum":

    # from program: ppo_cnn_single_object_curriculum_2
    timestamp = "08-02-24_18-23"
    job_num = args.job

    tm.load(job_num=job_num, timestamp=timestamp, best_id=False,
            load_into_new_training=False)
    
    # # update: now handled in load
    # # load the expert, this is not handled automatically in load
    # tm.trainer.env._load_expert_model(timestamp="08-12-23_19-19", job=53)

    tm.trainer.params.use_curriculum = True
    tm.trainer.curriculum_dict["metric_name"] = "success_rate"
    tm.trainer.curriculum_dict["metric_thresholds"] = [0.6 for i in range(4)]
    tm.trainer.curriculum_dict["param_values"] = [10, 20, 30, 40, 50]
    tm.trainer.curriculum_change = functools.partial(curriculum_change_object_noise, tm.trainer)

    # now continue training
    tm.continue_training(new_endpoint=150_000)
    print_time_taken()

  elif args.program == "bottleneck_encoder":

    # define what to vary this training, dependent on job number
    vary_1 = [0, 1]
    vary_2 = None
    vary_3 = None
    repeats = 2
    tm.param_1_name = "encoder option"
    tm.param_2_name = None
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # set very little object position noise and the old default oob distance
    tm.settings["env"]["object_position_noise_mm"] = 10
    tm.settings["cpp"]["oob_distance"] = 75e-3

    # enable base XY movements
    tm.settings["env"]["XY_base_actions"] = True
    tm.settings["env"]["base_lim_X_mm"] = 50
    tm.settings["env"]["base_lim_Y_mm"] = 50
    tm.settings["cpp"]["action"]["base_X"]["in_use"] = True
    tm.settings["cpp"]["action"]["base_Y"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["base_state_sensor_XY"]["in_use"] = True

    # enable base yaw
    tm.settings["env"]["Z_base_rotation"] = True
    tm.settings["cpp"]["action"]["base_yaw"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["base_state_sensor_yaw"]["in_use"] = True

    # add in the camera
    tm.settings["env"]["depth_camera"] = True
    tm.settings["env"]["use_rgb_in_observation"] = True
    tm.settings["env"]["image_width"] = 200
    tm.settings["env"]["image_height"] = 100
    tm.settings["env"]["use_standard_transform"] = True
    tm.settings["env"]["transform_resize_square"] = 58
    tm.settings["env"]["transform_crop_size"] = 52

    # add in curriculum where grasping gets harder
    tm.settings["trainer"]["use_curriculum"] = True
    tm.settings["curriculum"]["metric_name"] = "success_rate"
    tm.settings["curriculum"]["metric_thresholds"] = [0.6 for i in range(4)]
    tm.settings["curriculum"]["param_values"] = [10, 20, 30, 40, 50]
    tm.settings["curriculum"]["change_fcn"] = curriculum_change_object_noise

    # add in image rendering with the encoder
    tm.settings["env"]["use_rgb_rendering"] = True
    tm.settings["env"]["rgb_rendering_method"] = "cycleGAN_encoder_" + str(tm.param_1)

    # create the environment
    env = tm.make_env()

    # create the agent network
    ngf = 8 # set in the GAN training options
    mult = 4 # set in the GAN network itself
    if tm.param_1 == 0:
      bottleneck = 4
    elif tm.param_1 == 1:
      bottleneck = 1
    channels = int((ngf * mult) / float(bottleneck))
    img_x = int(tm.settings["env"]["transform_crop_size"] / float(mult))
    obs_size = (channels, img_x, img_x)
    network = NetActorCriticPG(networks.MixedNetworkFromEncoder2, obs_size,
                                 env.n_obs, env.n_actions, continous_actions=True, 
                                 device=args.device)
    
    # make the agent
    tm.settings["Agent_PPO"]["learning_rate_pi"] = 5e-5
    tm.settings["Agent_PPO"]["learning_rate_vf"] = 5e-5
    tm.settings["Agent_PPO"]["steps_per_epoch"] = 4000
    agent = Agent_PPO(device=args.device, steps=tm.settings["Agent_PPO"]["steps_per_epoch"])
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "expert_with_encoder":

    # define what to vary this training, dependent on job number
    vary_1 = [0, 1]
    vary_2 = [False, True]
    vary_3 = None
    repeats = 2
    tm.param_1_name = "encoder option"
    tm.param_2_name = "use feedforward"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # set very little object position noise and the old default oob distance
    tm.settings["env"]["object_position_noise_mm"] = 10
    tm.settings["cpp"]["oob_distance"] = 75e-3

    # enable base XY movements
    tm.settings["env"]["XY_base_actions"] = True
    tm.settings["env"]["base_lim_X_mm"] = 50
    tm.settings["env"]["base_lim_Y_mm"] = 50
    tm.settings["cpp"]["action"]["base_X"]["in_use"] = True
    tm.settings["cpp"]["action"]["base_Y"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["base_state_sensor_XY"]["in_use"] = True
  
    # enable base yaw
    tm.settings["env"]["Z_base_rotation"] = True
    tm.settings["cpp"]["action"]["base_yaw"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["base_state_sensor_yaw"]["in_use"] = True

    # add in the camera
    tm.settings["env"]["depth_camera"] = True
    tm.settings["env"]["use_rgb_in_observation"] = True
    tm.settings["env"]["image_width"] = 200
    tm.settings["env"]["image_height"] = 100
    tm.settings["env"]["use_standard_transform"] = True
    tm.settings["env"]["transform_resize_square"] = 58
    tm.settings["env"]["transform_crop_size"] = 52

    # add in curriculum where grasping gets harder
    tm.settings["trainer"]["use_curriculum"] = True
    tm.settings["curriculum"]["metric_name"] = "success_rate"
    tm.settings["curriculum"]["metric_thresholds"] = [0.6 for i in range(4)]
    tm.settings["curriculum"]["param_values"] = [10, 20, 30, 40, 50]
    tm.settings["curriculum"]["change_fcn"] = curriculum_change_object_noise

    # add in image rendering with the encoder
    tm.settings["env"]["use_rgb_rendering"] = True
    tm.settings["env"]["rgb_rendering_method"] = "cycleGAN_encoder_" + str(tm.param_1)

    # create the environment
    env = tm.make_env()

    # create the agent network
    ngf = 8 # set in the GAN training options
    mult = 4 # set in the GAN network itself
    if tm.param_1 == 0:
      bottleneck = 4
    elif tm.param_1 == 1:
      bottleneck = 1
    channels = int((ngf * mult) / float(bottleneck))
    img_x = int(tm.settings["env"]["transform_crop_size"] / float(mult))
    obs_size = (channels, img_x, img_x)
    network = NetActorCriticPG(networks.MixedNetworkFromEncoder2, obs_size,
                                 env.n_obs, env.n_actions, continous_actions=True, 
                                 device=args.device)

    # load in the expert
    feedsize = 4 if tm.param_2 else 0
    env._load_expert_model(timestamp="08-12-23_19-19", job=53)
    
    # make the agent
    tm.settings["Agent_PPO"]["learning_rate_pi"] = 5e-5
    tm.settings["Agent_PPO"]["learning_rate_vf"] = 5e-5
    tm.settings["Agent_PPO"]["steps_per_epoch"] = 4000
    agent = Agent_PPO(device=args.device, steps=tm.settings["Agent_PPO"]["steps_per_epoch"])
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  elif args.program == "mat_without_extra_actions":

    # define what to vary this training, dependent on job number
    vary_1 = [5e-6, 1e-5, 3e-5, 6e-5, 1e-4]
    vary_2 = [1000, 3000]
    vary_3 = [False, True]
    repeats = 2
    tm.param_1_name = "learning rate"
    tm.param_2_name = "steps per epoch"
    tm.param_3_name = "use_Z"
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply env settings
    tm.settings["env"]["use_MAT"] = True
    tm.settings["env"]["MAT_use_reopen"] = False
    tm.settings["env"]["max_episode_steps"] = 250 # Horizon = 250 in paper
    # tm.settings["env"]["base_lim_yaw_rad"] = np.pi / 4 # reduce from [-pi, +pi] in paper due to symettry
    tm.settings["env"]["finger_thickness"] = 1.0e-3 # put same fingers on as TMech eval
    tm.settings["env"]["finger_width"] = 24e-3
    tm.settings["env"]["finger_hook_angle_degrees"] = 75
    tm.settings["env"]["XY_base_actions"] = True # enable for xml, but not actions
    tm.settings["env"]["Z_base_rotation"] = True # enable for xml, but not actions

    # apply state and sensor settings
    tm.settings["cpp"]["sensor_n_prev_steps"] = 20
    tm.settings["cpp"]["state_n_prev_steps"] = 20
    tm.settings["cpp"]["sensor_sample_mode"] = 6 # scaled square of change, saturates at 0.1 change
    tm.settings["cpp"]["state_sample_mode"] = 4 # binary change information, no threshold, 0.05 in paper
    # tm.settings["cpp"]["sensor_noise_std"] = 0.01 # reduce based on real life

    # apply action settings
    tm.settings["cpp"]["use_termination_action"] = True # for final lift

    # turn on or off Z height
    use_Z = tm.param_3
    tm.settings["cpp"]["sensor"]["base_state_sensor_Z"]["in_use"] = use_Z
    tm.settings["cpp"]["action"]["base_Z"]["in_use"] = use_Z
    tm.settings["cpp"]["sensor"]["wrist_sensor_Z"]["in_use"] = use_Z
    if not use_Z:
      tm.settings["env"]["fingertip_clearance"] = 5e-3 # closer to ground given no Z height changes. Original=10e-3
      tm.settings["cpp"]["base_position_noise"] = 0e-3 # disable base position noise? Original=5e-3

    # apply reward settings
    tm.settings["reward"]["style"] = "MAT_liftonly"
    tm.settings["reward"]["penalty_termination"] = True # do we end early for oob and dangerous forces
    tm.settings["reward"]["stable_trigger"] = 1

    # enable the curriculum of step size adjustments
    tm.settings["trainer"]["use_curriculum"] = True
    tm.settings["curriculum"]["whole_fcn_override"] = curriculum_fcn_MAT
    tm.settings["curriculum"]["param_values"] = [
      [2e-3, 8e-3], # gripper_X action min/max
      [0.015, 0.06], # gripper_Y action min/max
      [4e-3, 16e-3], # gripper_Z action min/max
      [2e-3, 8e-3], # base_Z action min/max
      [0.2, 0.8],   # time per action min/max
    ]

    # apply agent hyperparameters
    tm.settings["Agent_PPO"]["learning_rate_pi"] = tm.param_1
    tm.settings["Agent_PPO"]["learning_rate_vf"] = tm.param_1
    tm.settings["Agent_PPO"]["gamma"] = 0.999 # paper specified
    tm.settings["Agent_PPO"]["steps_per_epoch"] = tm.param_2
    tm.settings["Agent_PPO"]["clip_ratio"] = 0.2 # paper specified
    tm.settings["Agent_PPO"]["train_pi_iters"] = 80
    tm.settings["Agent_PPO"]["train_vf_iters"] = 80
    tm.settings["Agent_PPO"]["lam"] = 0.95 # paper specified
    tm.settings["Agent_PPO"]["target_kl"] = 0.01
    tm.settings["Agent_PPO"]["max_kl_ratio"] = 1.5
    tm.settings["Agent_PPO"]["use_random_action_noise"] = True
    tm.settings["Agent_PPO"]["random_action_noise_size"] = 0.05
    tm.settings["Agent_PPO"]["optimiser"] = "adam" # paper specified
    tm.settings["Agent_PPO"]["adam_beta1"] = 0.9 # implied paper specified
    tm.settings["Agent_PPO"]["adam_beta2"] = 0.999 # implied paper specified
    tm.settings["Agent_PPO"]["grad_clamp_value"] = 200 # paper specified

    # create the environment
    env = tm.make_env()

    # enable new sensors
    env.mj.set.cartesian_contacts_XYZ.in_use = True
    env._update_n_actions_obs()

    # apply the agent settings
    layers = [128 for i in range(6)]
    n = 3 if tm.settings["env"]["MAT_use_reopen"] else 0
    network = MLPActorCriticPG(env.n_obs, env.n_actions + n, hidden_sizes=layers,
                                continous_actions=True)
    
    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)

    # add an extra test on the old object set
    tm.run_test(trials_per_obj=20, different_object_set="set8_fullset_1500",
                load_best_id=True)
    
    print_time_taken()

  elif args.program == "mat_liftonly":

    # define what to vary this training, dependent on job number
    vary_1 = [5e-6, 1e-5, 3e-5, 6e-5, 1e-4]
    vary_2 = [300, 1000, 3000]
    vary_3 = [True, False]
    repeats = 2
    tm.param_1_name = "learning rate"
    tm.param_2_name = "steps per epoch"
    tm.param_3_name = "use extra actions"
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply env settings
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["use_MAT"] = True
    tm.settings["env"]["MAT_use_reopen"] = tm.param_3
    tm.settings["env"]["max_episode_steps"] = 250 # Horizon = 250 in paper
    # tm.settings["env"]["base_lim_yaw_rad"] = np.pi / 4 # reduce from [-pi, +pi] in paper due to symettry
    tm.settings["env"]["finger_thickness"] = 1.0e-3 # put same fingers on as TMech eval
    tm.settings["env"]["finger_width"] = 24e-3
    tm.settings["env"]["finger_hook_angle_degrees"] = 75
    tm.settings["env"]["XY_base_actions"] = True # enable for xml, but not actions
    tm.settings["env"]["Z_base_rotation"] = True # enable for xml, but not actions

    # apply state and sensor settings
    tm.settings["cpp"]["sensor_n_prev_steps"] = 20
    tm.settings["cpp"]["state_n_prev_steps"] = 20
    tm.settings["cpp"]["sensor_sample_mode"] = 6 # scaled square of change, saturates at 0.1 change
    tm.settings["cpp"]["state_sample_mode"] = 4 # binary change information, no threshold, 0.05 in paper
    tm.settings["cpp"]["sensor"]["cartesian_contacts_XYZ"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["cartesian_contacts_XYZ"]["noise_override"] = [0, 0]
    
    # # probably change this back later, but for now makes learning a bit easier
    # tm.settings["cpp"]["sensor_noise_std"] = 0.01 # reduce based on real life

    # apply action settings
    tm.settings["cpp"]["use_termination_action"] = True # for final lift

    # turn on or off Z height
    use_Z = False
    tm.settings["cpp"]["sensor"]["base_state_sensor_Z"]["in_use"] = use_Z
    tm.settings["cpp"]["action"]["base_Z"]["in_use"] = use_Z
    tm.settings["cpp"]["sensor"]["wrist_sensor_Z"]["in_use"] = use_Z
    if not use_Z:
      tm.settings["env"]["fingertip_clearance"] = 5e-3 # closer to ground given no Z height changes. Original=10e-3
      tm.settings["cpp"]["base_position_noise"] = 0e-3 # disable base position noise? Original=5e-3

    # apply reward settings
    tm.settings["reward"]["style"] = "MAT_liftonly"
    tm.settings["reward"]["penalty_termination"] = True # do we end early for oob and dangerous forces
    tm.settings["reward"]["stable_trigger"] = 1

    # enable the curriculum of step size adjustments
    tm.settings["trainer"]["use_curriculum"] = True
    tm.settings["curriculum"]["whole_fcn_override"] = curriculum_fcn_MAT
    tm.settings["curriculum"]["param_values"] = [
      [2e-3, 8e-3], # gripper_X action min/max
      [0.015, 0.06], # gripper_Y action min/max
      [4e-3, 16e-3], # gripper_Z action min/max
      [2e-3, 8e-3], # base_Z action min/max
      [0.2, 0.8],   # time per action min/max
    ]

    # apply agent hyperparameters
    tm.settings["Agent_PPO"]["learning_rate_pi"] = tm.param_1
    tm.settings["Agent_PPO"]["learning_rate_vf"] = tm.param_1
    tm.settings["Agent_PPO"]["gamma"] = 0.999 # paper specified
    tm.settings["Agent_PPO"]["steps_per_epoch"] = tm.param_2
    tm.settings["Agent_PPO"]["clip_ratio"] = 0.2 # paper specified
    tm.settings["Agent_PPO"]["train_pi_iters"] = 80
    tm.settings["Agent_PPO"]["train_vf_iters"] = 80
    tm.settings["Agent_PPO"]["lam"] = 0.95 # paper specified
    tm.settings["Agent_PPO"]["target_kl"] = 0.01
    tm.settings["Agent_PPO"]["max_kl_ratio"] = 1.5
    tm.settings["Agent_PPO"]["use_random_action_noise"] = True
    tm.settings["Agent_PPO"]["random_action_noise_size"] = 0.05
    tm.settings["Agent_PPO"]["optimiser"] = "adam" # paper specified
    tm.settings["Agent_PPO"]["adam_beta1"] = 0.9 # implied paper specified
    tm.settings["Agent_PPO"]["adam_beta2"] = 0.999 # implied paper specified
    tm.settings["Agent_PPO"]["grad_clamp_value"] = 200 # paper specified

    # create the environment
    env = tm.make_env()

    # apply the agent settings
    layers = [128 for i in range(6)]
    n = 3 if tm.settings["env"]["MAT_use_reopen"] else 0
    network = MLPActorCriticPG(env.n_obs, env.n_actions + n, hidden_sizes=layers,
                                continous_actions=True)
    
    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)

    # add an extra test on the old object set
    tm.run_test(trials_per_obj=20, different_object_set="set8_fullset_1500",
                load_best_id=True)
    
    print_time_taken()

  elif args.program == "mat_stable":

    # define what to vary this training, dependent on job number
    vary_1 = [5e-6, 1e-5, 3e-5, 6e-5, 1e-4]
    vary_2 = [300, 1000, 3000]
    vary_3 = [True, False]
    repeats = 2
    tm.param_1_name = "learning rate"
    tm.param_2_name = "steps per epoch"
    tm.param_3_name = "use extra actions"
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply env settings
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["use_MAT"] = True
    tm.settings["env"]["MAT_use_reopen"] = tm.param_3
    tm.settings["env"]["max_episode_steps"] = 250 # Horizon = 250 in paper
    # tm.settings["env"]["base_lim_yaw_rad"] = np.pi / 4 # reduce from [-pi, +pi] in paper due to symettry
    tm.settings["env"]["finger_thickness"] = 1.0e-3 # put same fingers on as TMech eval
    tm.settings["env"]["finger_width"] = 24e-3
    tm.settings["env"]["finger_hook_angle_degrees"] = 75
    tm.settings["env"]["XY_base_actions"] = True # enable for xml, but not actions
    tm.settings["env"]["Z_base_rotation"] = True # enable for xml, but not actions

    # apply state and sensor settings
    tm.settings["cpp"]["sensor_n_prev_steps"] = 20
    tm.settings["cpp"]["state_n_prev_steps"] = 20
    tm.settings["cpp"]["sensor_sample_mode"] = 6 # scaled square of change, saturates at 0.1 change
    tm.settings["cpp"]["state_sample_mode"] = 4 # binary change information, no threshold, 0.05 in paper
    tm.settings["cpp"]["sensor"]["cartesian_contacts_XYZ"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["cartesian_contacts_XYZ"]["noise_override"] = [0, 0]
    
    # # probably change this back later, but for now makes learning a bit easier
    # tm.settings["cpp"]["sensor_noise_std"] = 0.01 # reduce based on real life

    # apply action settings
    tm.settings["cpp"]["use_termination_action"] = True # for final lift

    # turn on or off Z height
    use_Z = False
    tm.settings["cpp"]["sensor"]["base_state_sensor_Z"]["in_use"] = use_Z
    tm.settings["cpp"]["action"]["base_Z"]["in_use"] = use_Z
    tm.settings["cpp"]["sensor"]["wrist_sensor_Z"]["in_use"] = use_Z
    if not use_Z:
      tm.settings["env"]["fingertip_clearance"] = 5e-3 # closer to ground given no Z height changes. Original=10e-3
      tm.settings["cpp"]["base_position_noise"] = 0e-3 # disable base position noise? Original=5e-3

    # apply reward settings
    tm.settings["reward"]["style"] = "MAT"
    tm.settings["reward"]["penalty_termination"] = True # do we end early for oob and dangerous forces
    tm.settings["reward"]["stable_trigger"] = 1

    # enable the curriculum of step size adjustments
    tm.settings["trainer"]["use_curriculum"] = True
    tm.settings["curriculum"]["whole_fcn_override"] = curriculum_fcn_MAT
    tm.settings["curriculum"]["param_values"] = [
      [2e-3, 8e-3], # gripper_X action min/max
      [0.015, 0.06], # gripper_Y action min/max
      [4e-3, 16e-3], # gripper_Z action min/max
      [2e-3, 8e-3], # base_Z action min/max
      [0.2, 0.8],   # time per action min/max
    ]

    # apply agent hyperparameters
    tm.settings["Agent_PPO"]["learning_rate_pi"] = tm.param_1
    tm.settings["Agent_PPO"]["learning_rate_vf"] = tm.param_1
    tm.settings["Agent_PPO"]["gamma"] = 0.999 # paper specified
    tm.settings["Agent_PPO"]["steps_per_epoch"] = tm.param_2
    tm.settings["Agent_PPO"]["clip_ratio"] = 0.2 # paper specified
    tm.settings["Agent_PPO"]["train_pi_iters"] = 80
    tm.settings["Agent_PPO"]["train_vf_iters"] = 80
    tm.settings["Agent_PPO"]["lam"] = 0.95 # paper specified
    tm.settings["Agent_PPO"]["target_kl"] = 0.01
    tm.settings["Agent_PPO"]["max_kl_ratio"] = 1.5
    tm.settings["Agent_PPO"]["use_random_action_noise"] = True
    tm.settings["Agent_PPO"]["random_action_noise_size"] = 0.05
    tm.settings["Agent_PPO"]["optimiser"] = "adam" # paper specified
    tm.settings["Agent_PPO"]["adam_beta1"] = 0.9 # implied paper specified
    tm.settings["Agent_PPO"]["adam_beta2"] = 0.999 # implied paper specified
    tm.settings["Agent_PPO"]["grad_clamp_value"] = 200 # paper specified

    # create the environment
    env = tm.make_env()

    # apply the agent settings
    layers = [128 for i in range(6)]
    n = 3 if tm.settings["env"]["MAT_use_reopen"] else 0
    network = MLPActorCriticPG(env.n_obs, env.n_actions + n, hidden_sizes=layers,
                                continous_actions=True)
    
    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)

    # add an extra test on the old object set
    tm.run_test(trials_per_obj=20, different_object_set="set8_fullset_1500",
                load_best_id=True)
    
    print_time_taken()

  elif args.program == "mat_stable_shaped":

    # define what to vary this training, dependent on job number
    vary_1 = [5e-6, 1e-5, 3e-5, 6e-5, 1e-4]
    vary_2 = [True, False]
    vary_3 = None
    repeats = 3
    tm.param_1_name = "learning rate"
    tm.param_2_name = "use extra actions"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply env settings
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["use_MAT"] = True
    tm.settings["env"]["MAT_use_reopen"] = tm.param_2
    tm.settings["env"]["max_episode_steps"] = 250 # Horizon = 250 in paper
    # tm.settings["env"]["base_lim_yaw_rad"] = np.pi / 4 # reduce from [-pi, +pi] in paper due to symettry
    tm.settings["env"]["finger_thickness"] = 1.0e-3 # put same fingers on as TMech eval
    tm.settings["env"]["finger_width"] = 24e-3
    tm.settings["env"]["finger_hook_angle_degrees"] = 75
    tm.settings["env"]["XY_base_actions"] = True # enable for xml, but not actions
    tm.settings["env"]["Z_base_rotation"] = True # enable for xml, but not actions

    # apply state and sensor settings
    tm.settings["cpp"]["sensor_n_prev_steps"] = 20
    tm.settings["cpp"]["state_n_prev_steps"] = 20
    tm.settings["cpp"]["sensor_sample_mode"] = 6 # scaled square of change, saturates at 0.1 change
    tm.settings["cpp"]["state_sample_mode"] = 4 # binary change information, no threshold, 0.05 in paper
    tm.settings["cpp"]["sensor"]["cartesian_contacts_XYZ"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["cartesian_contacts_XYZ"]["noise_override"] = [0, 0]
    
    # # probably change this back later, but for now makes learning a bit easier
    # tm.settings["cpp"]["sensor_noise_std"] = 0.01 # reduce based on real life

    # apply action settings
    tm.settings["cpp"]["use_termination_action"] = True # for final lift

    # turn on or off Z height
    use_Z = False
    tm.settings["cpp"]["sensor"]["base_state_sensor_Z"]["in_use"] = use_Z
    tm.settings["cpp"]["action"]["base_Z"]["in_use"] = use_Z
    tm.settings["cpp"]["sensor"]["wrist_sensor_Z"]["in_use"] = use_Z
    if not use_Z:
      tm.settings["env"]["fingertip_clearance"] = 5e-3 # closer to ground given no Z height changes. Original=10e-3
      tm.settings["cpp"]["base_position_noise"] = 0e-3 # disable base position noise? Original=5e-3

    # apply reward settings
    tm.settings["reward"]["style"] = "MAT_shaped"
    tm.settings["reward"]["penalty_termination"] = True # do we end early for oob and dangerous forces
    tm.settings["reward"]["stable_trigger"] = 1

    # enable the curriculum of step size adjustments
    tm.settings["trainer"]["use_curriculum"] = True
    tm.settings["curriculum"]["whole_fcn_override"] = curriculum_fcn_MAT
    tm.settings["curriculum"]["param_values"] = [
      [2e-3, 8e-3], # gripper_X action min/max
      [0.015, 0.06], # gripper_Y action min/max
      [4e-3, 16e-3], # gripper_Z action min/max
      [2e-3, 8e-3], # base_Z action min/max
      [0.2, 0.8],   # time per action min/max
    ]

    # apply agent hyperparameters
    tm.settings["Agent_PPO"]["learning_rate_pi"] = tm.param_1
    tm.settings["Agent_PPO"]["learning_rate_vf"] = tm.param_1
    tm.settings["Agent_PPO"]["gamma"] = 0.999 # paper specified
    tm.settings["Agent_PPO"]["steps_per_epoch"] = 300
    tm.settings["Agent_PPO"]["clip_ratio"] = 0.2 # paper specified
    tm.settings["Agent_PPO"]["train_pi_iters"] = 80
    tm.settings["Agent_PPO"]["train_vf_iters"] = 80
    tm.settings["Agent_PPO"]["lam"] = 0.95 # paper specified
    tm.settings["Agent_PPO"]["target_kl"] = 0.01
    tm.settings["Agent_PPO"]["max_kl_ratio"] = 1.5
    tm.settings["Agent_PPO"]["use_random_action_noise"] = True
    tm.settings["Agent_PPO"]["random_action_noise_size"] = 0.05
    tm.settings["Agent_PPO"]["optimiser"] = "adam" # paper specified
    tm.settings["Agent_PPO"]["adam_beta1"] = 0.9 # implied paper specified
    tm.settings["Agent_PPO"]["adam_beta2"] = 0.999 # implied paper specified
    tm.settings["Agent_PPO"]["grad_clamp_value"] = 200 # paper specified

    # create the environment
    env = tm.make_env()

    # apply the agent settings
    layers = [128 for i in range(6)]
    n = 3 if tm.settings["env"]["MAT_use_reopen"] else 0
    network = MLPActorCriticPG(env.n_obs, env.n_actions + n, hidden_sizes=layers,
                                continous_actions=True)
    
    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)

    # add an extra test on the old object set
    tm.run_test(trials_per_obj=20, different_object_set="set8_fullset_1500",
                load_best_id=True)
    
    print_time_taken()

  elif args.program == "mat_ppo_liftonly":

    # define what to vary this training, dependent on job number
    vary_1 = [3e-6, 1e-5, 3e-5, 1e-4, 3e-4]
    vary_2 = [True, False]
    vary_3 = None
    repeats = 3
    tm.param_1_name = "learning rate"
    tm.param_2_name = "use extra actions"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply env settings
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["use_MAT"] = True
    tm.settings["env"]["MAT_use_reopen"] = tm.param_2
    tm.settings["env"]["max_episode_steps"] = 250 # Horizon = 250 in paper, see eq.5
    # tm.settings["env"]["base_lim_yaw_rad"] = np.pi / 4 # reduce from [-pi, +pi] in paper due to symettry
    tm.settings["env"]["finger_thickness"] = 1.0e-3 # put same fingers on as TMech eval
    tm.settings["env"]["finger_width"] = 24e-3
    tm.settings["env"]["finger_hook_angle_degrees"] = 75
    tm.settings["env"]["XY_base_actions"] = True # enable for xml, but not actions
    tm.settings["env"]["Z_base_rotation"] = True # enable for xml, but not actions

    # apply state and sensor settings
    tm.settings["cpp"]["sensor_n_prev_steps"] = 20
    tm.settings["cpp"]["state_n_prev_steps"] = 20
    tm.settings["cpp"]["sensor_sample_mode"] = 6 # scaled square of change, saturates at 0.1 change
    tm.settings["cpp"]["state_sample_mode"] = 4 # binary change information, no threshold, 0.05 in paper
    tm.settings["cpp"]["sensor"]["cartesian_contacts_XYZ"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["cartesian_contacts_XYZ"]["noise_override"] = [0, 0]
    
    # # possible to change noise settings to improve learning
    # tm.settings["cpp"]["sensor_noise_std"] = 0.01 # reduce based on real life

    # apply action settings
    tm.settings["cpp"]["use_termination_action"] = True # for final lift

    # turn on or off Z height
    use_Z = False
    tm.settings["cpp"]["sensor"]["base_state_sensor_Z"]["in_use"] = use_Z
    tm.settings["cpp"]["action"]["base_Z"]["in_use"] = use_Z
    tm.settings["cpp"]["sensor"]["wrist_sensor_Z"]["in_use"] = use_Z
    if not use_Z:
      tm.settings["env"]["fingertip_clearance"] = 5e-3 # closer to ground given no Z height changes. Original=10e-3
      tm.settings["cpp"]["base_position_noise"] = 0e-3 # disable base position noise? Original=5e-3

    # apply reward settings
    tm.settings["reward"]["style"] = "MAT_liftonly"
    tm.settings["reward"]["penalty_termination"] = True # do we end early for oob and dangerous forces
    tm.settings["reward"]["stable_trigger"] = 1

    # enable the curriculum of step size adjustments
    tm.settings["trainer"]["use_curriculum"] = True
    tm.settings["curriculum"]["whole_fcn_override"] = curriculum_fcn_MAT
    tm.settings["curriculum"]["param_values"] = [
      [2e-3, 8e-3], # gripper_X action min/max
      [0.015, 0.06], # gripper_Y action min/max
      [4e-3, 16e-3], # gripper_Z action min/max
      [2e-3, 8e-3], # base_Z action min/max
      [0.2, 0.8],   # time per action min/max
    ]

    # apply agent hyperparameters
    tm.settings["Agent_PPO_MAT"]["use_extra_actions"] = tm.param_2
    tm.settings["Agent_PPO_MAT"]["learning_rate_pi"] = tm.param_1  # 1e-4 paper specified
    tm.settings["Agent_PPO_MAT"]["learning_rate_vf"] = tm.param_1  # 1e-4 paper specified

    # create the environment
    env = tm.make_env()

    # apply the agent settings
    layers = [128 for i in range(6)]
    n = 3 if tm.settings["env"]["MAT_use_reopen"] else 0
    network = MLPActorCriticPG(env.n_obs, env.n_actions + n, hidden_sizes=layers,
                                continous_actions=True)
    
    # make the agent
    agent = Agent_PPO_MAT(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)

    # add an extra test on the old object set
    tm.run_test(trials_per_obj=20, different_object_set="set8_fullset_1500",
                load_best_id=True)
    
    print_time_taken()

  elif args.program == "mat_ppo_stable_shaped":

    # define what to vary this training, dependent on job number
    vary_1 = [3e-6, 1e-5, 3e-5, 1e-4, 3e-4]
    vary_2 = [True, False]
    vary_3 = None
    repeats = 3
    tm.param_1_name = "learning rate"
    tm.param_2_name = "use extra actions"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply env settings
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["use_MAT"] = True
    tm.settings["env"]["MAT_use_reopen"] = tm.param_2
    tm.settings["env"]["max_episode_steps"] = 250 # Horizon = 250 in paper, see eq.5
    # tm.settings["env"]["base_lim_yaw_rad"] = np.pi / 4 # reduce from [-pi, +pi] in paper due to symettry
    tm.settings["env"]["finger_thickness"] = 1.0e-3 # put same fingers on as TMech eval
    tm.settings["env"]["finger_width"] = 24e-3
    tm.settings["env"]["finger_hook_angle_degrees"] = 75
    tm.settings["env"]["XY_base_actions"] = True # enable for xml, but not actions
    tm.settings["env"]["Z_base_rotation"] = True # enable for xml, but not actions

    # apply state and sensor settings
    tm.settings["cpp"]["sensor_n_prev_steps"] = 20
    tm.settings["cpp"]["state_n_prev_steps"] = 20
    tm.settings["cpp"]["sensor_sample_mode"] = 6 # scaled square of change, saturates at 0.1 change
    tm.settings["cpp"]["state_sample_mode"] = 4 # binary change information, no threshold, 0.05 in paper
    tm.settings["cpp"]["sensor"]["cartesian_contacts_XYZ"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["cartesian_contacts_XYZ"]["noise_override"] = [0, 0]
    
    # # possible to change noise settings to improve learning
    # tm.settings["cpp"]["sensor_noise_std"] = 0.01 # reduce based on real life

    # apply action settings
    tm.settings["cpp"]["use_termination_action"] = True # for final lift

    # turn on or off Z height
    use_Z = False
    tm.settings["cpp"]["sensor"]["base_state_sensor_Z"]["in_use"] = use_Z
    tm.settings["cpp"]["action"]["base_Z"]["in_use"] = use_Z
    tm.settings["cpp"]["sensor"]["wrist_sensor_Z"]["in_use"] = use_Z
    if not use_Z:
      tm.settings["env"]["fingertip_clearance"] = 5e-3 # closer to ground given no Z height changes. Original=10e-3
      tm.settings["cpp"]["base_position_noise"] = 0e-3 # disable base position noise? Original=5e-3

    # apply reward settings
    tm.settings["reward"]["style"] = "MAT_shaped"
    tm.settings["reward"]["penalty_termination"] = True # do we end early for oob and dangerous forces
    tm.settings["reward"]["stable_trigger"] = 1

    # enable the curriculum of step size adjustments
    tm.settings["trainer"]["use_curriculum"] = True
    tm.settings["curriculum"]["whole_fcn_override"] = curriculum_fcn_MAT
    tm.settings["curriculum"]["param_values"] = [
      [2e-3, 8e-3], # gripper_X action min/max
      [0.015, 0.06], # gripper_Y action min/max
      [4e-3, 16e-3], # gripper_Z action min/max
      [2e-3, 8e-3], # base_Z action min/max
      [0.2, 0.8],   # time per action min/max
    ]

    # apply agent hyperparameters
    tm.settings["Agent_PPO_MAT"]["use_extra_actions"] = tm.param_2
    tm.settings["Agent_PPO_MAT"]["learning_rate_pi"] = tm.param_1  # 1e-4 paper specified
    tm.settings["Agent_PPO_MAT"]["learning_rate_vf"] = tm.param_1  # 1e-4 paper specified

    # create the environment
    env = tm.make_env()

    # apply the agent settings
    layers = [128 for i in range(6)]
    n = 3 if tm.settings["env"]["MAT_use_reopen"] else 0
    network = MLPActorCriticPG(env.n_obs, env.n_actions + n, hidden_sizes=layers,
                                continous_actions=True)
    
    # make the agent
    agent = Agent_PPO_MAT(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)

    # add an extra test on the old object set
    tm.run_test(trials_per_obj=20, different_object_set="set8_fullset_1500",
                load_best_id=True)
    
    print_time_taken()

  elif args.program == "mat_ppo_grid":

    # define what to vary this training, dependent on job number
    vary_1 = [
      (1e-5, 5e-5),
      (5e-5, 5e-5),
      (1e-4, 5e-5),
      (2e-4, 5e-5),
      (1e-5, 5e-4),
      (5e-5, 5e-4),
      (1e-4, 5e-4),
      (2e-4, 5e-4),
      (1e-5, 5e-3),
      (5e-5, 5e-3),
      (1e-4, 5e-3),
      (2e-4, 5e-3),
    ]
    vary_2 = [150, 300, 600]
    vary_3 = [False, True]
    repeats = 3
    tm.param_1_name = "lr/temperature"
    tm.param_2_name = "steps per epoch"
    tm.param_3_name = "extra actions"
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply env settings
    use_extra_actions = False
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["use_MAT"] = True
    tm.settings["env"]["MAT_use_reopen"] = use_extra_actions
    tm.settings["env"]["max_episode_steps"] = 250 # Horizon = 250 in paper, see eq.5
    # tm.settings["env"]["base_lim_yaw_rad"] = np.pi / 4 # reduce from [-pi, +pi] in paper due to symettry
    tm.settings["env"]["finger_thickness"] = 1.0e-3 # put same fingers on as TMech eval
    tm.settings["env"]["finger_width"] = 24e-3
    tm.settings["env"]["finger_hook_angle_degrees"] = 75
    tm.settings["env"]["XY_base_actions"] = True # enable for xml, but not actions
    tm.settings["env"]["Z_base_rotation"] = True # enable for xml, but not actions

    # apply state and sensor settings
    tm.settings["cpp"]["sensor_n_prev_steps"] = 20
    tm.settings["cpp"]["state_n_prev_steps"] = 20
    tm.settings["cpp"]["sensor_sample_mode"] = 6 # scaled square of change, saturates at 0.1 change
    tm.settings["cpp"]["state_sample_mode"] = 4 # binary change information, no threshold, 0.05 in paper
    tm.settings["cpp"]["sensor"]["cartesian_contacts_XYZ"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["cartesian_contacts_XYZ"]["noise_override"] = [0, 0]
    
    # # possible to change noise settings to improve learning
    # tm.settings["cpp"]["sensor_noise_std"] = 0.01 # reduce based on real life

    # apply action settings
    tm.settings["cpp"]["use_termination_action"] = True # for final lift

    # turn on or off Z height
    use_Z = False
    tm.settings["cpp"]["sensor"]["base_state_sensor_Z"]["in_use"] = use_Z
    tm.settings["cpp"]["action"]["base_Z"]["in_use"] = use_Z
    tm.settings["cpp"]["sensor"]["wrist_sensor_Z"]["in_use"] = use_Z
    if not use_Z:
      tm.settings["env"]["fingertip_clearance"] = 5e-3 # closer to ground given no Z height changes. Original=10e-3
      tm.settings["cpp"]["base_position_noise"] = 0e-3 # disable base position noise? Original=5e-3

    # apply reward settings
    tm.settings["reward"]["style"] = "MAT_liftonly"
    tm.settings["reward"]["penalty_termination"] = True # do we end early for oob and dangerous forces
    tm.settings["reward"]["stable_trigger"] = 1

    # enable the curriculum of step size adjustments
    tm.settings["trainer"]["use_curriculum"] = True
    tm.settings["curriculum"]["whole_fcn_override"] = curriculum_fcn_MAT
    tm.settings["curriculum"]["param_values"] = [
      [2e-3, 8e-3], # gripper_X action min/max
      [0.015, 0.06], # gripper_Y action min/max
      [4e-3, 16e-3], # gripper_Z action min/max
      [2e-3, 8e-3], # base_Z action min/max
      [0.2, 0.8],   # time per action min/max
    ]

    # apply agent hyperparameters
    tm.settings["Agent_PPO_MAT"]["use_extra_actions"] = use_extra_actions
    tm.settings["Agent_PPO_MAT"]["learning_rate_pi"] = tm.param_1[0]  # 1e-4 paper specified
    tm.settings["Agent_PPO_MAT"]["learning_rate_vf"] = tm.param_1[0]  # 1e-4 paper specified
    tm.settings["Agent_PPO_MAT"]["steps_per_epoch"] = tm.param_2

    # create the environment
    env = tm.make_env()

    # apply the agent settings
    layers = [128 for i in range(6)] # only applies to pi, vf is set to [3*128]
    n = 2 if use_extra_actions else 0
    network = MATActorCriticPG(env.n_obs, env.n_actions + n, hidden_sizes=layers,
                                use_extra_actions=use_extra_actions)
    
    # make the agent
    agent = Agent_PPO_MAT(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)

    # add an extra test on the old object set
    tm.run_test(trials_per_obj=20, different_object_set="set8_fullset_1500",
                load_best_id=True)
    
    print_time_taken()

  elif args.program == "panic_check":

    # define what to vary this training, dependent on job number
    vary_1 = [
      (1e-5, 5e-5),
      (5e-5, 5e-5),
      (1e-4, 5e-5),
      (2e-4, 5e-5),
      (1e-5, 5e-4),
      (5e-5, 5e-4),
      (1e-4, 5e-4),
      (2e-4, 5e-4),
      (1e-5, 5e-3),
      (5e-5, 5e-3),
      (1e-4, 5e-3),
      (2e-4, 5e-3),
    ]
    vary_2 = [False, True]
    vary_3 = None
    repeats = 1
    tm.param_1_name = "lr/temperature"
    tm.param_2_name = "extra actions"
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()

    # apply env settings
    use_extra_actions = tm.param_2
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["use_MAT"] = True
    tm.settings["env"]["MAT_use_reopen"] = use_extra_actions
    tm.settings["env"]["max_episode_steps"] = 250 # Horizon = 250 in paper, see eq.5
    # tm.settings["env"]["base_lim_yaw_rad"] = np.pi / 4 # reduce from [-pi, +pi] in paper due to symettry
    tm.settings["env"]["finger_thickness"] = 1.0e-3 # put same fingers on as TMech eval
    tm.settings["env"]["finger_width"] = 24e-3
    tm.settings["env"]["finger_hook_angle_degrees"] = 75
    tm.settings["env"]["XY_base_actions"] = True # enable for xml, but not actions
    tm.settings["env"]["Z_base_rotation"] = True # enable for xml, but not actions

    # apply state and sensor settings
    tm.settings["cpp"]["sensor_n_prev_steps"] = 20
    tm.settings["cpp"]["state_n_prev_steps"] = 20
    tm.settings["cpp"]["sensor_sample_mode"] = 6 # scaled square of change, saturates at 0.1 change
    tm.settings["cpp"]["state_sample_mode"] = 4 # binary change information, no threshold, 0.05 in paper
    tm.settings["cpp"]["sensor"]["cartesian_contacts_XYZ"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["cartesian_contacts_XYZ"]["noise_override"] = [0, 0]
    
    # # possible to change noise settings to improve learning
    # tm.settings["cpp"]["sensor_noise_std"] = 0.01 # reduce based on real life

    # apply action settings
    tm.settings["cpp"]["use_termination_action"] = True # for final lift

    # turn on or off Z height
    use_Z = False
    tm.settings["cpp"]["sensor"]["base_state_sensor_Z"]["in_use"] = use_Z
    tm.settings["cpp"]["action"]["base_Z"]["in_use"] = use_Z
    tm.settings["cpp"]["sensor"]["wrist_sensor_Z"]["in_use"] = use_Z
    if not use_Z:
      tm.settings["env"]["fingertip_clearance"] = 5e-3 # closer to ground given no Z height changes. Original=10e-3
      tm.settings["cpp"]["base_position_noise"] = 0e-3 # disable base position noise? Original=5e-3

    # apply reward settings
    tm.settings["reward"]["style"] = "MAT_liftonly"
    tm.settings["reward"]["penalty_termination"] = True # do we end early for oob and dangerous forces
    tm.settings["reward"]["stable_trigger"] = 1

    # enable the curriculum of step size adjustments
    tm.settings["trainer"]["use_curriculum"] = True
    tm.settings["curriculum"]["whole_fcn_override"] = curriculum_fcn_MAT
    tm.settings["curriculum"]["param_values"] = [
      [2e-3, 8e-3], # gripper_X action min/max
      [0.015, 0.06], # gripper_Y action min/max
      [4e-3, 16e-3], # gripper_Z action min/max
      [2e-3, 8e-3], # base_Z action min/max
      [0.2, 0.8],   # time per action min/max
    ]

    # apply agent hyperparameters
    tm.settings["Agent_PPO_MAT"]["use_extra_actions"] = use_extra_actions
    tm.settings["Agent_PPO_MAT"]["learning_rate_pi"] = tm.param_1[0]  # 1e-4 paper specified
    tm.settings["Agent_PPO_MAT"]["learning_rate_vf"] = tm.param_1[0]  # 1e-4 paper specified
    tm.settings["Agent_PPO_MAT"]["steps_per_epoch"] = 300

    # create the environment
    env = tm.make_env()

    tm.settings["trainer"]["test_freq"] = 1000

    # apply the agent settings
    layers = [128 for i in range(6)] # only applies to pi, vf is set to [3*128]
    n = 2 if use_extra_actions else 0
    network = MATActorCriticPG(env.n_obs, env.n_actions + n, hidden_sizes=layers,
                                use_extra_actions=use_extra_actions)
    
    # make the agent
    agent = Agent_PPO_MAT(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)

    # add an extra test on the old object set
    tm.run_test(trials_per_obj=20, different_object_set="set8_fullset_1500",
                load_best_id=True)
    
    print_time_taken()

  elif args.program == "example_template":

    # define what to vary this training, dependent on job number
    vary_1 = None
    vary_2 = None
    vary_3 = None
    repeats = None
    tm.param_1_name = None
    tm.param_2_name = None
    tm.param_3_name = None
    tm.param_1, tm.param_2, tm.param_3 = vary_all_inputs(args.job, param_1=vary_1, param_2=vary_2,
                                                         param_3=vary_3, repeats=repeats)
    if args.print: print_training_info()
    
    # # apply the varied settings (very important!)
    # tm.settings["A"]["B"] = tm.param_1
    # tm.settings["C"]["D"] = tm.param_2
    # tm.settings["E"]["F"] = tm.param_3

    # # choose any additional settings to change
    # tm.settings["A"]["B"] = X
    # tm.settings["C"]["D"] = Y
    # tm.settings["E"]["F"] = Z

    # create the environment
    env = tm.make_env()

    # make the agent, may depend on variable settings above
    layers = [env.n_obs, 64, 64, env.n_actions]
    network = networks.VariableNetwork(layers, device=args.device)
    agent = Agent_DQN(device=args.device)
    agent.init(network)

    # complete the training
    tm.run_training(agent, env)
    print_time_taken()

  else:
    raise RuntimeError(f"launch_training.py error: program name of {args.program} not recognised")

# ----- end ----- #