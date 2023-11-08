#!/usr/bin/env python3

# fix for cluster, numpy causes segfault
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from datetime import datetime
import argparse
from time import sleep
from random import random
import torch

from Trainer import MujocoTrainer
from TrainingManager import TrainingManager
from agents.DQN import Agent_DQN
from agents.ActorCritic import MLPActorCriticAC, Agent_SAC
from agents.PolicyGradient import MLPActorCriticPG, Agent_PPO
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
                        min_ep=None, max_ep=None, silent=True, min_stage=None, max_stage=None):
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

def curriculum_change_termination(self, stage):
  """
  Toggle the termination threshold over the stages. Below -1.0 or above 1.0 is
  impossible to trigger.
  """

  self.env.mj.set.termination_threshold = self.curriculum_dict["param_values"][stage]

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

  # # print all the inputs we have received
  # print("array_training_DQN.py inputs are:", sys.argv[1:])

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
  # parser.add_argument("--override-lib",       action="store_true")    # override bind.so library with loaded data

  args = parser.parse_args()

  timestamp = args.timestamp if args.timestamp else datetime.now().strftime(datestr)

  # default device
  if args.device is None:
    args.device = "cpu"
  if args.device == "cpu": torch.set_num_threads(1)

  if args.print: 
    args.log_level = 0

  # disable delays for in cases where we are not training
  if args.demo or args.test or args.print_results:
    args.no_delay = True

  # echo these inputs
  if args.log_level > 0:
    print("launch_training.py is preparing to train:")
    print(" -> Job number:", args.job)
    print(" -> Timestamp:", timestamp)
    print(" -> Program name:", args.program)
    print(" -> Device:", args.device)

  # seperate process for safety when running a training program
  if not args.no_delay and args.job is not None and args.program is not None:
    sleep(args.job)
    sleep(0.25 * random())

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
                        min_stage=args.print_from_stage, max_stage=args.print_up_to_stage)
    exit()

  if args.job is None:
    raise RuntimeError("launch_training.py: your options require a job number [-j, --job], either to identify an existing training for loading, or to correspond to your selected program")

  # create a training manager
  tm = make_training_manager_from_args(args)

  if args.plot:

    if args.timestamp is None:
      raise RuntimeError(f"launch_training.py: a timestamp [-t, --timestamp] in the following format '{datestr}' is required to load existing trainigs")

    if args.log_level > 0: print("launch_training.py will plot a training")

    tm.load(job_num=args.job, timestamp=args.timestamp, id=args.load_id)
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
      raise RuntimeError("launch_training.py: [-c, --continue] must be used with either [--new-endpoint] or [--extra-episodes]")
    
    if args.log_level > 0: print(f"launch_training.py is continuing a traing, new_endpoint={args.new_endpoint}, extra_episodes={args.extra_episodes}")

    tm.load(job_num=args.job, timestamp=args.timestamp, id=args.load_id)

    # adjust command line settings as load overrides them
    tm.settings["plot"] = args.plot
    tm.settings["render"] = args.render
    if args.savedir is not None: tm.settings["savedir"] = args.savedir

    tm.continue_training(new_endpoint=args.new_endpoint, extra_episodes=args.extra_episodes)
    exit()

  # ----- regular training ----- #

  if args.program is None:
    raise RuntimeError("launch_training.py: normal trainings require that [-p, --program] be set with training name corresponding to an option in this file")

  # set the name of this training in the training manager
  tm.set_group_run_name(job_num=args.job, timestamp=timestamp, prefix=args.name_prefix)

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
      env.mj.set.action_penalty.set (value,  False,   1,     0.1, 1.5,  -1)
      
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
    tm.settings["exceed_style"] = "wrist_" + str(tm.param_2 * 0.5)
    tm.settings["danger_style"] = [5.0, 15.0, tm.param_2] # bend, palm, wrist
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
    tm.settings["exceed_style"] = "wrist_" + str(wrist_limit * 0.5)
    tm.settings["danger_style"] = [5.0, 15.0, wrist_limit] # bend, palm, wrist
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
      env.mj.set.action_penalty.set (value,  False,   1,     0.1, 3.0,  -1)

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
    tm.settings["exceed_style"] = "wrist_" + str(wrist_limit * 0.5)
    tm.settings["danger_style"] = [5.0, 15.0, wrist_limit] # bend, palm, wrist
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
      env.mj.set.action_penalty.set (value,  False,   1,     0.1, 3.0,  -1)

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
    tm.settings["exceed_style"] = "wrist_" + str(wrist_limit * 0.5)
    tm.settings["danger_style"] = [5.0, 15.0, wrist_limit] # bend, palm, wrist
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
    env.mj.set.action_penalty.set (value,  False,   1,     0.1, 3.0,  -1)

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
    vary_1 = None
    vary_2 = None
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
    palm_stable_lim = 4
    palm_danger_lim = 5
    wrist_limit = 4
    tm.settings["penalty_termination"] = True
    tm.settings["exceed_style"] = "wrist_" + str(wrist_limit * 0.5)
    tm.settings["danger_style"] = [5.0, 15.0, wrist_limit] # bend, palm, wrist
    tm.settings["cpp"]["saturation_yield_factor"] = 1.5
    tm.settings["cpp"]["stable_finger_force_lim"] = 4.0
    tm.settings["cpp"]["stable_palm_force_lim"] = 10.0
    tm.settings["env"]["object_position_noise_mm"] = 10
    tm.settings["trainer"]["num_episodes"] = 120_000
    tm.settings["env"]["object_set_name"] = "set9_nosharp" # "set9_fullset"
    tm.settings["env"]["finger_hook_angle_degrees"] = 90

    # add in z base noise
    tm.settings["cpp"]["base_position_noise"] = 5e-3

    # add significant mean noise to z base state sensor
    tm.settings["cpp"]["sensor"]["base_state_sensor_Z"]["in_use"] = True
    tm.settings["cpp"]["sensor"]["base_state_sensor_Z"]["noise_override"] = [0.1, 0.0]

    # FOR TESTING ONLY
    tm.settings["trainer"]["num_episodes"] = 18
    tm.settings["trainer"]["test_freq"] = 6
    tm.settings["trainer"]["save_freq"] = 4
    tm.settings["final_test_trials_per_object"] = 1
    tm.settings["env"]["test_objects"] = 3
    tm.settings["env"]["max_episode_steps"] = 5
    tm.settings["episode_log_rate"] = 5
    tm.settings["track_avg_num"] = 3
    tm.settings["Agent_PPO"]["steps_per_epoch"] = 15

    # create the environment
    env = tm.make_env()

    # # how many steps in a row do we require stability
    # env.mj.set.object_stable.trigger = 4

    # palm and wrist normalisation
    env.mj.set.palm_sensor.normalise = palm_danger_lim + 1
    env.mj.set.wrist_sensor_Z.normalise = wrist_limit + 2

    # use an action penalty
    value = 2 * env.mj.set.exceed_limits.reward
    # rewards                      reward  done   trigger  min  max  overshoot
    env.mj.set.action_penalty.set (value,  False,   1,     0.1, 3.0,  -1)

    # add in curriculum
    tm.settings["trainer"]["use_curriculum"] = True
    tm.settings["curriculum"]["metric_name"] = "episode_number"
    tm.settings["curriculum"]["metric_thresholds"] = [8]
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