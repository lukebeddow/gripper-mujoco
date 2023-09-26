#!/usr/bin/env python3

# fix for cluster, numpy causes segfault
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from datetime import datetime
import argparse
from time import sleep
from random import random

from Trainer import MujocoTrainer
from TrainingManager import TrainingManager
from agents.DQN import Agent_DQN
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

def get_jobs_from_timestamp(timestamp):
  """
  Find all the jobs with a particular timestamp
  """

  mj = MujocoTrainer(None, None, log_level=0)
  savedir = mj.savedir
  tm = TrainingManager(log_level=0)
  tm.set_group_run_name(job_num=1, timestamp=timestamp)

  # get all the run folder corresponding to this timestamp
  group_path = savedir + "/" + tm.group_name
  run_folders = [x for x in os.listdir(group_path) if x.startswith(tm.run_name[:-3])]
  
  job_nums = []

  for folder in run_folders:
    num = folder.split(tm.run_name[:-3])[-1][2:] # from _A5 -> 5"
    job_nums.append(int(num))

  # sort into numeric ascending order
  job_nums.sort()

  return job_nums

def update_training_summaries(timestamp, jobstr=None, job_numbers=None):
  """
  Regenerate training_summaries from a currently running training, or any trainings where
  there is no up to date training_summary.txt files
  """

  if jobstr is None and job_numbers is None:
    job_numbers = get_jobs_from_timestamp(timestamp)
    if len(job_numbers) == 0:
      raise RuntimeError(f"print_results_table() cannot find any job numbers for timestamp {timestamp}")

  if jobstr is not None:
    job_numbers = parse_job_string(jobstr)

  tm = TrainingManager(log_level=0)

  for j in job_numbers:

    # determine the path required for this job
    tm.init_training_summary()
    tm.set_group_run_name(job_num=j, timestamp=timestamp)
    tm.trainer = MujocoTrainer(None, None, log_level=0, run_name=tm.run_name,
                               group_name=tm.group_name)
    tm.save_training_summary()

def print_results_table(timestamp, jobstr=None, job_numbers=None):
  """
  Print a table of results for a training
  """

  if jobstr is None and job_numbers is None:
    job_numbers = get_jobs_from_timestamp(timestamp)
    if len(job_numbers) == 0:
      raise RuntimeError(f"print_results_table() cannot find any job numbers for timestamp {timestamp}")

  if jobstr is not None:
    job_numbers = parse_job_string(jobstr)

  mj = MujocoTrainer(None, None, log_level=0)
  savedir = mj.savedir
  tm = TrainingManager(log_level=0)

  # prepare to find information from training_summary files
  first_loop = True
  headings = []
  table = []
  new_elem = []

  for j in job_numbers:

    # determine the path required for this job
    tm.set_group_run_name(job_num=j, timestamp=timestamp)
    filepath = savedir + "/" + tm.group_name + "/" + tm.run_name + "/"
    
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

    if first_loop:
      if tm.job_number is not None: 
        found_job_number = True
        headings.append("Job num")
      else: found_job_number = False
      if tm.timestamp is not None: 
        found_timestamp = True
        headings.append("Timestamp    ") # 4xspace for heading
      else: found_timestamp = False
      if tm.param_1 is not None: 
        found_param_1 = True
        headings.append(tm.param_1_name)
      else: found_param_1 = False
      if tm.param_2 is not None: 
        found_param_2 = True
        headings.append(tm.param_2_name)
      else: found_param_2 = False
      if tm.param_3 is not None: 
        found_param_3 = True
        headings.append(tm.param_3_name)
      else: found_param_3 = False
      if tm.trained_to is not None: 
        found_trained_to = True
        headings.append("Trained to")
      else: found_trained_to = False
      if tm.train_best_ep is not None: 
        found_train_best_ep = True
        headings.append("Train best SR")
      else: found_train_best_ep = False
      if tm.train_best_sr is not None: 
        found_train_best_sr = True
        headings.append("Train best episode")
      else: found_train_best_sr = False
      if tm.full_test_sr is not None: 
        found_full_test_sr = True
        headings.append("Final test SR")
      else: found_full_test_sr = False
      first_loop = False

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
  print_str = """"""
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
        table[i][j] = "{:.4f}".format(elem)
    # print(row_str.format(*table[i]))
    print_str += row_str.format(*table[i]) + "\n"

  # print and save the table
  print("\n" + print_str)
  tablepath = savedir + "/" + tm.group_name + "/" + "results_table.txt"
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

if __name__ == "__main__":

  # starting time
  starting_time = datetime.now()

  # key default settings
  plot = False
  render = False
  datestr = "%d-%m-%y_%H-%M" # all date inputs must follow this format

  # # print all the inputs we have received
  # print("array_training_DQN.py inputs are:", sys.argv[1:])

  # define arguments and parse them
  parser = argparse.ArgumentParser()
  parser.add_argument("-j", "--job",          default=None, type=int) # job input number
  parser.add_argument("-t", "--timestamp",    default=None)           # timestamp
  parser.add_argument("-p", "--program",      default=None)           # program name to select from if..else if
  parser.add_argument("-d", "--device",       default=None)           # override device
  parser.add_argument("--print-results",      action="store_true")    # prepare and print all 
  parser.add_argument("--rngseed",            default=None)           # turns on reproducible training with given seed (slower)
  parser.add_argument("--log-level",          type=int, default=1)    # set script log level
  parser.add_argument("--no-delay",           action="store_true")    # prevent a sleep(...) to seperate processes
  parser.add_argument("--print",              action="store_true")    # don't train, print job options

  parser.add_argument("-c", "--continue",     action="store_true", dest="resume") # continue training

  parser.add_argument("-g", "--plot",         action="store_true") # plot to wandb job
  parser.add_argument("-H", "--heuristic",    action="store_true") # run a test using heuristic actions
  parser.add_argument("-r", "--render",       action="store_true") # render window during training
  
  parser.add_argument("--savedir",            default=None)        # override save/load directory
  
  
  parser.add_argument("--override-lib",       action="store_true") # override bind.so library with loaded data
  parser.add_argument("--test",               action="store_true") # run a thorough test on existing model
  parser.add_argument("--demo",               action="store_true") # run a demo test on model, can specify id number

  args = parser.parse_args()

  timestamp = args.timestamp if args.timestamp else datetime.now().strftime(datestr)

  # default device
  if args.device is None:
    args.device = "cpu"

  if args.print: 
    args.log_level = 0

  # echo these inputs
  if args.log_level > 0:
    print("launch_training.py is preparing to train:")
    print(" -> Job number:", args.job)
    print(" -> Timestamp:", timestamp)
    print(" -> Program name:", args.program)
    print(" -> Device:", args.device)

  # seperate process for safety
  if not args.no_delay and args.job is not None:
    sleep(args.job)
    sleep(0.25 * random())

  # ----- special cases ----- #

  if args.print_results:
    if args.log_level > 0: print("\nPreparing to print a results table in launch_training.py")
    if args.timestamp is None:
      raise RuntimeError("--print-results requires a timestamp (of the chosen training) be set")
    print_results_table(args.timestamp)
    exit()

  # ----- regular training ----- #

  if args.program is None:
    raise RuntimeError("launch_training.py: regular training requires that [-p, --program] be set with training name corresponding to an option in this file")
  if args.job is None:
    raise RuntimeError("launch_training.py: regular training requires a job number [-j, --job] which should correspond to options in the program")
  
  # create the training manager
  tm = TrainingManager(rngseed=args.rngseed, device=args.device, log_level=args.log_level)
  tm.set_group_run_name(job_num=args.job, timestamp=timestamp)

  # input any command line settings
  tm.settings["plot"] = plot
  tm.settings["render"] = render

  if args.program == "test_1":

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

    # make the agent, may depend on variable settings above
    layers = [env.n_obs, 64, 64, env.n_actions]
    network = networks.VariableNetwork(layers, device=args.device)
    agent = Agent_DQN(device=args.device)
    agent.init(network)

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

  # ----- run training ----- #

  # complete the training
  tm.run_training(agent, env)
  print_time_taken()