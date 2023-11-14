#!/usr/bin/env python3

import os
import torch
import numpy as np
from dataclasses import dataclass, asdict
from itertools import count
import time
import random
from datetime import datetime

from ModelSaver import ModelSaver
from agents.DQN import Agent_DQN
from agents.ActorCritic import MLPActorCriticAC, Agent_SAC
from agents.PolicyGradient import MLPActorCriticPG, Agent_PPO
from env.MjEnv import MjEnv
import networks


class TrackTraining:

  def __init__(self, test_metrics=None, avg_num=50, plt_frequency_seconds=30):
    """
    Track training data and generate matplotlib plots. To save test metrics pass
    in a list of metric names eg ["success_rate", "average_force", ...]. Metrics
    are stored as floats
    """
    self.numpy_float = np.float32
    self.avg_num = avg_num
    self.plt_frequency_seconds = plt_frequency_seconds
    # plotting options
    self.plot_episode_time = False
    self.plot_train_raw = False
    self.plot_train_avg = True
    self.plot_test_raw = False
    self.plot_test_metrics = False
    # general
    self.episodes_done = 0
    self.last_plot = 0
    self.per_action_time_taken = np.array([], dtype=self.numpy_float)
    self.avg_time_taken = np.array([], dtype=self.numpy_float)
    # training data
    self.train_episodes = np.array([], dtype=np.int32)
    self.train_rewards = np.array([], dtype=self.numpy_float)
    self.train_durations = np.array([], dtype=np.int32)
    self.train_avg_episodes = np.array([], dtype=np.int32)
    self.train_avg_rewards = np.array([], dtype=self.numpy_float)
    self.train_avg_durations = np.array([], dtype=np.int32)
    self.train_curriculum_stages = np.array([], dtype=np.int32)
    # testing data
    self.test_episodes = np.array([], dtype=np.int32)
    self.test_rewards = np.array([], dtype=self.numpy_float)
    self.test_durations = np.array([], dtype=np.int32)
    self.test_curriculum_stages = np.array([], dtype=np.int32)
    self.n_test_metrics = 0
    self.test_metric_names = []
    self.test_metric_values = []
    if test_metrics is not None: self.add_test_metrics(test_metrics)
    # misc
    self.fig = None
    self.axs = None

  def add_test_metrics(self, metrics_to_add, dtype=None):
    """
    Include additional test metrics
    """

    if metrics_to_add is None: return

    if dtype is None: dtype = self.numpy_float

    if isinstance(metrics_to_add, str):
      metrics_to_add = [metrics_to_add]

    for m in metrics_to_add:
      self.test_metric_names.append(m)
      self.test_metric_values.append(np.array([], dtype=dtype))

    self.n_test_metrics = len(self.test_metric_names)

  def get_test_metric(self, metric_name):
    """
    Return the array corresponding to a given metric_name
    """

    for i in range(len(self.test_metric_names)):
      if self.test_metric_names[i] == metric_name:
        return self.test_metrics[i]
    return None

  def log_training_episode(self, reward, duration, time_taken, curriculum_stage=0):
    """
    Log one training episode
    """

    self.train_episodes = np.append(self.train_episodes, self.episodes_done)
    self.train_durations = np.append(self.train_durations, duration)
    self.train_rewards = np.append(self.train_rewards, reward)
    self.train_curriculum_stages = np.append(self.train_curriculum_stages, curriculum_stage)
    self.per_action_time_taken = np.append(self.per_action_time_taken, time_taken)
    self.episodes_done += 1

    # update average information
    self.calc_static_average()

  def log_test_information(self, avg_reward, avg_duration, metrics=None):
    """
    Log information following a test
    """

    self.test_episodes = np.append(self.test_episodes, self.episodes_done)
    self.test_durations = np.append(self.test_durations, avg_duration)
    self.test_rewards = np.append(self.test_rewards, avg_reward)

    if metrics is not None:
      if len(metrics) != len(self.n_test_metrics):
        raise RuntimeError(f"TrackTraining.log_test_information got 'metrics' len={len(metrics)}, but self.n_test_metrics = {self.n_test_metrics}")
      for i in range(len(metrics)):
        self.test_metric_values[i] = np.append(self.test_metric_values[i], metrics[i])

  def calc_static_average(self):
    """
    Average rewards and durations to reduce data points
    """

    # find number of points we can average
    num_avg_points = len(self.train_avg_rewards) * self.avg_num

    # if we points which have not been averaged yet
    if num_avg_points + self.avg_num < len(self.train_episodes):

      # prepare to average rewards, durations, time taken
      unaveraged_r = self.train_rewards[num_avg_points:]
      unaveraged_d = self.train_durations[num_avg_points:]
      unaveraged_t = self.per_action_time_taken[num_avg_points:]

      num_points_to_avg = len(unaveraged_r) // self.avg_num

      for i in range(num_points_to_avg):
        # find average values
        avg_e = self.train_episodes[
          num_avg_points + (i * self.avg_num) + (self.avg_num // 2)]
        avg_r = np.mean(unaveraged_r[i * self.avg_num : (i + 1) * self.avg_num])
        avg_d = np.mean(unaveraged_d[i * self.avg_num : (i + 1) * self.avg_num])
        avg_t = np.mean(unaveraged_t[i * self.avg_num : (i + 1) * self.avg_num])
        # append to average lists
        self.train_avg_episodes = np.append(self.train_avg_episodes, avg_e)
        self.train_avg_rewards = np.append(self.train_avg_rewards, avg_r)
        self.train_avg_durations = np.append(self.train_avg_durations, avg_d)
        self.avg_time_taken = np.append(self.avg_time_taken, avg_t)

  def plot_matplotlib(self, xdata, ydata, ylabel, title, axs, label=None):
    """
    Plot a matplotlib 2x1 subplot
    """
    axs.plot(xdata, ydata, label=label)
    axs.set_title(title, fontstyle="italic")
    axs.set(ylabel=ylabel)

  def plot(self, plttitle=None, plt_frequency_seconds=None):
      """
      Plot training results figures, pass a frequency to plot only if enough
      time has elapsed
      """

      if plt_frequency_seconds is None:
        plt_frequency_seconds = self.plt_frequency_seconds

      # if not enough time has elapsed since the last plot
      if (self.last_plot + plt_frequency_seconds > time.time()):
        return

      self.plot_bar_chart = True

      if self.fig is None:
        # multiple figures
        self.fig = []
        self.axs = []
        if self.plot_train_raw: 
          fig1, axs1 = plt.subplots(2, 1)
          self.fig.append(fig1)
          self.axs.append(axs1)
        if self.plot_train_avg:
          fig2, axs2 = plt.subplots(2, 1)
          self.fig.append(fig2)
          self.axs.append(axs2)
        if self.plot_test_raw:
          fig3, axs3 = plt.subplots(2, 1)
          self.fig.append(fig3)
          self.axs.append(axs3)
        if self.plot_episode_time:
          fig4, axs4 = plt.subplots(1, 1)
          self.fig.append(fig4)
          self.axs.append([axs4, axs4]) # add paired to hold the pattern
        if self.plot_test_metrics:
          for i in range(self.n_test_metrics):
            fig5, axs5 = plt.subplots(2, 1)
            self.fig.append(fig5)
            self.axs.append([axs5, axs5]) # add paired to hold the pattern

      ind = 0

      E = "Episode"
      R = "Reward"
      D = "Duration"

      # clear all axes
      for i, pairs in enumerate(self.axs):
        if plttitle is not None:
          self.fig[i].suptitle(plttitle)
        for axis in pairs:
          axis.clear()

      if self.plot_train_raw:
        self.plot_matplotlib(self.train_episodes, self.train_durations, D,
                             "Raw durations", self.axs[ind][0])
        self.plot_matplotlib(self.train_episodes, self.train_rewards, R,
                             "Raw rewards", self.axs[ind][1])
        self.fig[ind].subplots_adjust(hspace=0.4)
        ind += 1

      if self.plot_train_avg:
        self.plot_matplotlib(self.train_avg_episodes, self.train_avg_durations, D,
                             f"Durations static average ({self.avg_num} samples)", 
                             self.axs[ind][0])
        self.plot_matplotlib(self.train_avg_episodes, self.train_avg_rewards, R,
                             f"Rewards static average ({self.avg_num} samples)", 
                             self.axs[ind][1])
        self.fig[ind].subplots_adjust(hspace=0.4)
        ind += 1

      if self.plot_test_raw:
        self.plot_matplotlib(self.test_episodes, self.test_durations, D,
                             "Test durations", self.axs[ind][0])
        self.plot_matplotlib(self.test_episodes, self.test_rewards, R,
                             "Test rewards", self.axs[ind][1])
        self.fig[ind].subplots_adjust(hspace=0.4)
        ind += 1

      # create plots for static average of time taken per step
      if self.plot_episode_time:
        self.plot_matplotlib(self.avgS_episodes, self.avg_time_taken, "Time per action / s",
          f"Time per action static average ({self.avg_num} samples)", self.axs[ind][0])
        ind += 1 

      if self.plot_test_metrics:
        for m, metric in enumerate(self.test_metric_names):
          self.plot_matplotlib(self.test_episodes, self.test_metric_values[m], f"{metric}",
            f"Test metric: {metric}", self.axs[ind][0])
          ind += 1 
        
      plt.pause(0.001)

      # save that we plotted
      self.last_plot = time.time()

      return

  def print_training(self):
    """
    Print out some training metrics
    """
    if self.episodes_done % self.avg_num == 0:
      if len(self.train_avg_rewards) == 0: return
      else: print(f"Episode {self.episodes_done}, avg_reward = {self.train_avg_rewards[-1]}")

  def get_avg_return(self):
    """
    Return the average reward only if the value has updated
    """

    if self.episodes_done % self.avg_num == 0:
      if len(self.train_avg_rewards) == 0: return None
      else: return self.train_avg_rewards[-1]

class Trainer:

  @dataclass
  class Parameters:

    num_episodes: int = 10_000
    test_freq: int = 1000
    save_freq: int = 50
    use_curriculum: bool = False

  def __init__(self, agent, env, rngseed=None, device="cpu", log_level=1, plot=False,
               render=False, group_name="default_%d-%m-%y", run_name="default_run_%H-%M",
               save=True, savedir="models", episode_log_rate=10, strict_seed=False,
               track_avg_num=50, print_avg_return=False):
    """
    Class that trains RL agents in an environment
    """

    # prepare class variables
    self.track = TrackTraining(avg_num=track_avg_num)
    self.params = Trainer.Parameters()
    self.agent = agent
    self.env = env
    self.saved_trainer_params = False
    self.last_loaded_agent_id = None
    self.last_saved_agent_id = None
    self.curriculum_dict = {
      "stage" : 0,
      "metric_name" : "",
      "metric_thresholds" : [],
      "param_values" : [],
      "finished" : False,
      "info" : "",
    }

    # input class options
    self.rngseed = rngseed
    self.device = torch.device(device)
    self.log_level = log_level
    self.plot = plot
    self.render = render
    self.log_rate_for_episodes = episode_log_rate
    self.print_avg_return = print_avg_return
    
    # set up saving
    self.train_param_savename = "Trainer_params"
    self.track_savename = "Tracking_info"
    self.setup_saving(run_name, group_name, savedir, enable_saving=save)

    # are we plotting
    if self.plot:
      global plt
      import matplotlib.pyplot as plt
      plt.ion()

    # seed the environment (skip if given None for agent and env)
    # training only reproducible if torch.manual_seed() set BEFORE agent network initialisation
    self.training_reproducible = strict_seed
    if agent is not None and env is not None: self.seed(strict=strict_seed)
    else:
      if strict_seed or rngseed is not None:
        raise RuntimeError("MujocoTrainer.__init__() error: agent and/or env is None, environment is not seeded by rngseed or strict_seed was set")
      elif self.log_level >= 2:
        print("MujocoTrainer.__init__() warning: agent and/or env is None and environment is NOT seeded")

    if self.log_level > 0:
      print("Trainer settings:")
      print(" -> Run name:", self.run_name)
      print(" -> Group name:", self.group_name)
      print(" -> Given seed:", rngseed)
      print(" -> Training reproducible:", self.training_reproducible)
      print(" -> Using device:", self.device)
      print(" -> Save enabled:", self.enable_saving)
      if self.enable_saving:
        print(" -> Save path:", self.modelsaver.path)

  def setup_saving(self, run_name="default_run_%H-%M", group_name="default_%d-%m-%y",
                   savedir="models", enable_saving=None, track_info_overwrite=True):
    """
    Provide saving information and enable saving of models during training. The
    save() function will not work without first running this function
    """

    if enable_saving is not None:
      self.enable_saving = enable_saving

    # check for default group and run names (use current time and date)
    if group_name.startswith("default_"):
      group_name = datetime.now().strftime(group_name[8:])
    if run_name.startswith("default_"):
      run_name = datetime.now().strftime(run_name[8:])
      
    # save information and create modelsaver to manage saving/loading
    self.group_name = group_name
    self.run_name = run_name
    self.savedir = savedir
    self.trackinfo_numbering = not track_info_overwrite

    if self.enable_saving:
      self.modelsaver = ModelSaver(self.savedir + "/" + self.group_name)
  
  def to_torch(self, data, dtype=torch.float32):
    return torch.tensor(data, device=self.device, dtype=dtype).unsqueeze(0)

  def seed(self, rngseed=None, strict=None):
    """
    Set a random seed for the entire environment
    """
    if rngseed is None:
      if self.rngseed is not None: rngseed = self.rngseed
      else: 
        self.training_reproducible = False
        rngseed = np.random.randint(0, 2_147_483_647)

    torch.manual_seed(rngseed)
    self.rngseed = rngseed
    self.agent.seed(rngseed)
    self.env.seed(rngseed)

    # if we want to ensure reproducitibilty at the cost of performance
    if strict is None: strict = self.training_reproducible
    if strict:
      os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # increases GPU usage by 24MiB, see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility and ctrl+f "CUBLAS_WORKSPACE_CONFIG"
      torch.backends.cudnn.benchmark = False
      torch.use_deterministic_algorithms(mode=True)
    else: self.training_reproducible = False

  def save(self, txtfilename=None, txtfilestr=None, extra_data=None, force_train_params=False,
           force_save_number=None):
    """
    Save the state of the current trainer and agent.
    """

    if not self.enable_saving: return

    if not self.modelsaver.in_folder:
      self.modelsaver.new_folder(name=self.run_name, notimestamp=True)

    # have we saved key information about the trainer
    if not self.saved_trainer_params or force_train_params:
      trainer_save = {
        "parameters" : self.params,
        "rngseed" : self.rngseed,
        "run_name" : self.run_name,
        "group_name" : self.group_name,
        "agent_name" : self.agent.name,
        "env_data" : self.env.get_save_state(),
        "curriculum_dict" : self.curriculum_dict,
        "extra_data" : extra_data
      }

      self.modelsaver.save(self.train_param_savename, pyobj=trainer_save)
      self.saved_trainer_params = True

    # determine what save_id to use
    save_id = self.get_save_id(self.track.episodes_done)
    save_id_agent = save_id if force_save_number is None else force_save_number
    save_id_track = save_id if self.trackinfo_numbering else None

    # save tracking information
    self.modelsaver.save(self.track_savename, pyobj=self.track, 
                         suffix_numbering=self.trackinfo_numbering,
                         force_suffix=save_id_track)

    # save the actual agent
    self.modelsaver.save(self.agent.name, pyobj=self.agent.get_save_state(),
                         txtstr=txtfilestr, txtlabel=txtfilename,
                         force_suffix=save_id_agent)
    self.last_saved_agent_id = self.modelsaver.last_saved_id

  def get_save_id(self, episode):
    """
    Return the save id associated with a given episode. Note: if the test_freq
    or save_freq is changed, this function will no longer output correct ids
    """
    first_save = 1
    if self.params.test_freq == self.params.save_freq:
      save_id = first_save + (episode // self.params.test_freq)
    else:
      save_id = first_save + (episode // self.params.test_freq
                              + episode // self.params.save_freq
                              - episode // (np.lcm(self.params.test_freq, 
                                                   self.params.save_freq)))
    return save_id

  def get_param_dict(self):
    """
    Return a dictionary of hyperparameters
    """
    param_dict = asdict(self.params)
    param_dict.update({
      "rngseed" : self.rngseed,
      "training_reproducible" : self.training_reproducible,
      "saving_enabled" : self.enable_saving,
    })
    return param_dict

  def save_hyperparameters(self, filename="hyperparameters", strheader=None, 
                           print_terminal=None):
    """
    Save the model hyperparameters
    """

    if print_terminal is None:
      if self.log_level > 0: print_terminal = True
      else: print_terminal = False

    hyper_str = """"""
    if strheader is not None: hyper_str += strheader + "\n"

    hyper_str += "Trainer hyperparameters:\n\n"
    hyper_str += str(self.get_param_dict()).replace(",", "\n") + "\n\n"

    hyper_str += "Agent hyperparameters:\n\n"
    hyper_str += str(self.agent.get_params_dict()).replace(",", "\n") + "\n\n"

    hyper_str += "Env hyperparameters:\n\n"
    hyper_str += str(self.env.get_params_dict()).replace(",", "\n") + "\n\n"

    if print_terminal: print(hyper_str)

    if self.enable_saving:
      self.modelsaver.save(filename, txtstr=hyper_str, txtonly=True)

  def load(self, run_name, id=None, group_name=None, path_to_run_folder=None):
    """
    Load a model given a path to it
    """

    # check if modelsaver is defined
    if not hasattr(self, "modelsaver"):
      if path_to_run_folder is not None:
        print(f"load not given a modelsaver, making one from path_to_group: {path_to_run_folder}")
        self.modelsaver = ModelSaver(path_to_run_folder)
      elif group_name is not None:
        # try to find the group from this folder
        pathhere = os.path.dirname(os.path.abspath(__file__))
        print(f"load not given modelsaver or path_to_group, assuming group is local at {pathhere + '/' + self.savedir}")
        self.modelsaver = ModelSaver(pathhere + "/" + self.savedir + "/" + self.group_name)
      else:
        raise RuntimeError("load not given a modelsaver and either of a) path_to_run_folder b) group_name (if group can be found locally)")
    
    # enter the run folder (exit if already in one)
    self.modelsaver.enter_folder(run_name)

    # folderpath ignores the current folder, so add that if necessary
    if path_to_run_folder is not None:
      path_to_run_folder += "/" + run_name

    load_agent = self.modelsaver.load(id=id, folderpath=path_to_run_folder,
                                      filenamestarts="Agent")
    self.last_loaded_agent_id = self.modelsaver.last_loaded_id
    load_train = self.modelsaver.load(folderpath=path_to_run_folder,
                                      filenamestarts=self.train_param_savename)
    try:
      load_track = self.modelsaver.load(id=id, folderpath=path_to_run_folder, 
                                      filenamestarts=self.track_savename,
                                      suffix_numbering=self.trackinfo_numbering)
    except FileNotFoundError as e:
      if self.log_level > 0:
        print("failed\nLoading track failed, trying again with alternative numbering. Error:", e)
      load_track = self.modelsaver.load(id=id, folderpath=path_to_run_folder, 
                                    filenamestarts=self.track_savename,
                                    suffix_numbering=not self.trackinfo_numbering)
    
    # extract loaded data
    self.params = load_train["parameters"]
    self.run_name = load_train["run_name"]
    self.group_name = load_train["group_name"]
    self.env.load_save_state(load_train["env_data"])
    self.track = load_track

    try:
      self.curriculum_dict = load_train["curriculum_dict"]
      if len(self.track.train_curriculum_stages) > 0:
        self.curriculum_dict["stage"] = self.track.train_curriculum_stages[-1]
      else: self.curriculum_dict["stage"] = 0
    except KeyError as e:
      print("curriculum_dict not found in loaded trainer_params, old code. Error:", e)
      self.curriculum_dict = {
        "stage" : 0,
        "metric_name" : "",
        "metric_thresholds" : [],
        "param_values" : [],
        "finished" : False,
        "info" : "",
      }

    # do we have the agent already, if not, create it
    if self.agent is None:
      to_exec = f"""self.agent = {load_train["agent_name"]}()"""
      exec(to_exec)
    self.agent.load_save_state(load_agent)

    # reseed - be aware this will not be contingous
    self.rngseed = load_train["rngseed"]
    self.training_reproducible = False # training no longer reproducible
    self.seed()

  def run_episode(self, i_episode, test=False):
    """
    Run one episode of RL
    """

    # initialise environment and state
    obs = self.env.reset()
    obs = self.to_torch(obs)

    ep_start = time.time()

    cumulative_reward = 0

    # count up through actions
    for t in count():

      if self.log_level >= 3: print("Episode", i_episode, "action", t)

      # select and perform an action
      if self.env.using_continous_actions():
        action = self.agent.select_action(obs, decay_num=i_episode, test=test)
        (new_obs, reward, terminated, truncated, info) = self.env.step((action.cpu()).numpy())
      else:
        action = self.agent.select_action(obs, decay_num=i_episode, test=test)
        (new_obs, reward, terminated, truncated, info) = self.env.step((action.cpu()).item())
   
      # render the new environment
      if self.render: self.env.render()

      if terminated or truncated: done = True
      else: done = False

      # convert data to torch tensors on specified device
      new_obs = self.to_torch(new_obs)
      reward = self.to_torch(reward)
      action = action.to(self.device).unsqueeze(0) # from Tensor([x]) -> Tensor([[x]])
      truncated = self.to_torch(truncated, dtype=torch.bool)

      # store if it was a terminal state (ie either terminated or truncated)
      done = self.to_torch(done, dtype=torch.bool)

      # perform one step of the optimisation on the policy network
      if not test:
        self.agent.update_step(obs, action, new_obs, reward, done, truncated)

      obs = new_obs
      cumulative_reward += reward.cpu()

      # check if this episode is over and log if we aren't testing
      if done:

        ep_end = time.time()
        time_per_step = (ep_end - ep_start) / float(t + 1)

        if self.log_level >= 3:
          print(f"Time for episode was {ep_end - ep_start:.3f}s"
            f", time per action was {time_per_step * 1e3:.3f} ms")

        # if we are testing, no data is logged
        if test: break

        # save training data
        self.track.log_training_episode(cumulative_reward, t + 1, time_per_step,
                                        curriculum_stage=0 if not self.params.use_curriculum
                                        else self.curriculum_dict["stage"])
        cumulative_reward = 0

        break

  def train(self, i_start=None, num_episodes_abs=None, num_episodes_extra=None):
    """
    Run a training
    """

    if i_start is None:
      i_start = self.track.episodes_done

    if num_episodes_abs is not None:
      self.params.num_episodes = num_episodes_abs

    if num_episodes_extra is not None:
      self.params.num_episodes = i_start + num_episodes_extra

    if num_episodes_abs is not None and num_episodes_extra is not None:
      if self.log_level > 0:
        print(f"Trainer.train() warning: num_episodes={num_episodes_abs} (ignored) and num_episodes_extra={num_episodes_extra} (used) were both set. Training endpoing set as {self.params.num_episodes}")

    if i_start >= self.params.num_episodes:
      raise RuntimeError(f"Trainer.train() error: training episode start = {i_start} is greater or equal to the target number of episodes = {self.params.num_episodes}")

    # if this is a fresh, new training
    if i_start == 0:
      # save starting network parameters and training settings
      self.save()
      self.save_hyperparameters()
    else:
      # save a record of the training restart
      continue_label = f"Training is continuing from episode {i_start} with these hyperparameters\n"
      hypername = f"hyperparameters_from_ep_{i_start}"
      self.save_hyperparameters(filename=hypername, strheader=continue_label)

    if self.log_level > 0:
      print(f"\nBegin training, target is {self.params.num_episodes} episodes\n", flush=True)

    # prepare the agent for training
    self.agent.set_device(self.device)
    self.agent.training_mode()
    
    # begin training episodes
    for i_episode in range(i_start + 1, self.params.num_episodes + 1):

      # check if we should adjust the training curriculum
      if self.params.use_curriculum: self.curriculum_fcn(i_episode)

      if self.log_level == 1 and (i_episode - 1) % self.log_rate_for_episodes == 0:
        print("Begin training episode", i_episode, flush=True)
      elif self.log_level > 1:
        avg_return = self.track.get_avg_return()
        if avg_return is not None: str_to_add = f". Average return = {avg_return}"
        else: str_to_add = ""
        print(f"Begin training episode {i_episode} at {datetime.now().strftime('%H:%M')}" + str_to_add, flush=True)

      self.run_episode(i_episode)

      # plot graphs to the screen
      if self.plot: self.track.plot(plt_frequency_seconds=1)
      if self.print_avg_return: self.track.print_training()

      # check if we need to do any episode level updates (eg target network)
      self.agent.update_episode(i_episode)

      # test the target network and then save it
      if i_episode % self.params.test_freq == 0 and i_episode != 0:
        self.test() # this function should save the network

      # or only save the network
      elif i_episode % self.params.save_freq == 0:
        self.save()

    # the agent may require final updating at the end
    self.agent.update_episode(i_episode, finished=True)

    # save, log and plot now we are finished
    if self.log_level > 0:
      print("\nTraining complete, finished", i_episode, "episodes\n")

    # wrap up
    if self.render: self.env.render()
    if self.plot:
      self.track.plot(force=True, end=True, hang=True) # leave plots on screen if we are plotting

  def test(self):
    """
    Empty test function, should be overriden for each environment
    """
    pass

  def curriculum_fcn(self, i):
    """
    Empty curriculum function, override if using a curriculum. Takes as input the
    current episode number, i. See below an example template which uses another
    function 'self.curriculum_change(stage)' to apply the stage-dependent changes:

    if self.curriculum_dict["finished"]: return

    # determine what stage we are at
    stage = self.curriculum_dict["stage"]

    # determine the curriculum metric
    if self.curriculum_dict["metric_name"] == "episode_number":

      for t in self.curriculum_dict["metric_thresholds"][stage:]:
        if i >= t:
          stage += 1
        else: break

    # example of using success rate as a metric
    elif self.curriculum_dict["metric_name"] == "success_rate":

      # get the most recent success rate
      success_rate = 0.0 # update this...

      # determine if we have passed the required threshold
      for t in self.curriculum_dict["metric_thresholds"][stage:]:
        if success_rate >= t:
          stage += 1

    # if the metric is not recognised
    else: raise RuntimeError(f"TrainingManager.curriculum_fcn() metric of {self.curriculum_dict['metric_name']} not recognised")

    # check if we are still at the same stage we were last episode
    if stage == self.curriculum_dict["stage"]:
      # check if we have finished the curriculum
      if stage == len(self.curriculum_dict["metric_thresholds"]): 
        self.curriculum_dict["finished"] = True
      return

    # update to the new stage
    if self.log_level > 0:
      print(f"Episode = {i}, curriculum is changing from stage {self.curriculum_dict['stage']} to stage {stage}")
    self.curriculum_dict["stage"] = stage

    # now apply the curriculum change (this function must be user overwritten for a training)
    self.curriculum_change(stage)

    # save a text file to reflect the changes
    labelstr = f"Hyperparameters after curriculum change which occured at episode {i}\n"
    name = f"hyperparameters_curriculum_stage_{stage}"
    self.save_hyperparameters(filename=name, strheader=labelstr, print_terminal=False)
    
    return
    """

    pass

class MujocoTrainer(Trainer):

  @dataclass
  class Parameters:
    num_episodes: int = 10_000
    test_freq: int = 1000
    save_freq: int = 1000
    use_curriculum: bool = False

    def update(self, newdict):
      for key, value in newdict.items():
        if hasattr(self, key):
          setattr(self, key, value)
        else: raise RuntimeError(f"incorrect key: {key}")

  def __init__(self, agent, mjenv, rngseed=None, device="cpu", log_level=1, plot=False,
               render=False, group_name="default_%d-%m-%y", run_name="default_run_%H-%M",
               save=True, savedir="models", episode_log_rate=10, strict_seed=False,
               track_avg_num=50, print_avg_return=False):
    """
    Trainer class for the gripper mujoco RL environment
    """

    super().__init__(agent, mjenv, rngseed=rngseed, device=device, log_level=log_level, 
                     plot=plot, render=render, group_name=group_name, run_name=run_name,
                     save=save, savedir=savedir, episode_log_rate=episode_log_rate, 
                     strict_seed=strict_seed, track_avg_num=track_avg_num, print_avg_return=print_avg_return)

    # override the parameters of the base class
    self.params = MujocoTrainer.Parameters()

    # class variables
    self.last_test_data = None
    self.test_performances_filename = "test_performance"
    self.test_result_filename = "test_results"

    # add variables to tracker
    numpy_float = np.float32
    self.track.avg_p_lifted = np.array([], dtype=numpy_float)
    self.track.avg_p_contact = np.array([], dtype=numpy_float)
    self.track.avg_p_palm_force = np.array([], dtype=numpy_float)
    self.track.avg_p_exceed_limits = np.array([], dtype=numpy_float)
    self.track.avg_p_exceed_bend  = np.array([], dtype=numpy_float)
    self.track.avg_p_exceed_palm = np.array([], dtype=numpy_float)
    self.track.avg_p_exceed_wrist = np.array([], dtype=numpy_float)
    self.track.avg_lifted = np.array([], dtype=numpy_float)
    self.track.avg_stable = np.array([], dtype=numpy_float)
    self.track.avg_oob = np.array([], dtype=numpy_float)
    self.track.avg_lifted_to_height = np.array([], dtype=numpy_float)
    self.track.avg_stable_height = np.array([], dtype=numpy_float)
    self.track.avg_successful_grasp = np.array([], dtype=numpy_float)
    self.track.avg_dangerous_bend = np.array([], dtype=numpy_float)
    self.track.avg_dangerous_palm = np.array([], dtype=numpy_float)
    self.track.avg_dangerous_wrist = np.array([], dtype=numpy_float)
    self.track.object_categories = []
    self.track.category_num = []
    self.track.category_stable = []
    self.track.category_lifted = []
    self.track.category_lifted_to_height = []
    self.track.category_stable_height = []
    self.track.category_successful_grasp = []
    self.track.category_dangerous_bend = []
    self.track.category_dangerous_palm = []
    self.track.category_dangerous_wrist = []

  def save_hyperparameters(self, filename="hyperparameters", strheader=None, 
                           print_terminal=None):
    """
    Save the model hyperparameters, overrride base class method to add cpp settings
    """

    if print_terminal is None:
      if self.log_level > 0: print_terminal = True
      else: print_terminal = False

    hyper_str = "\nTraining Hyperparameters\n\n"
    if strheader is not None: hyper_str += strheader + "\n"

    hyper_str += "Trainer hyperparameters:\n\n"
    hyper_str += str(self.get_param_dict()).replace(",", "\n") + "\n\n"

    hyper_str += "Agent hyperparameters:\n\n"
    hyper_str += str(self.agent.get_params_dict()).replace(",", "\n") + "\n\n"

    hyper_str += "Env hyperparameters:\n\n"
    hyper_str += str(self.env.get_params_dict()).replace(",", "\n") + "\n\n"

    hyper_str += "Cpp hyperparameters:\n\n"
    hyper_str += self.env._get_cpp_settings() + "\n\n"

    if print_terminal: print(hyper_str)

    if self.enable_saving:
      self.modelsaver.save(filename, txtstr=hyper_str, txtonly=True)

  def run_heuristic_episode(self, i_episode):
    """
    Run heuristic test episode
    """

    # initialise environment and state
    obs = self.env.reset()
    obs = self.to_torch(obs)

    ep_start = time.time()

    cumulative_reward = 0

    # count up through actions
    for t in count():

      if self.log_level >= 3: print("Episode", i_episode, "action", t)

      # human written action selection function
      action = self.env.get_heuristic_action()
      action = torch.tensor(action) # to fit with select_action(...)
      (new_obs, reward, terminated, truncated, info) = self.env.step(action.item())
   
      # render the new environment
      if self.render: self.env.render()

      if terminated or truncated: done = True
      else: done = False

      obs = new_obs
      cumulative_reward += reward.cpu()

      # check if this episode is over and log if we aren't testing
      if done:

        ep_end = time.time()
        time_per_step = (ep_end - ep_start) / float(t + 1)

        if self.log_level >= 3:
          print(f"Time for episode was {ep_end - ep_start:.3f}s"
            f", time per action was {time_per_step * 1e3:.3f} ms")
          
        break

  def test(self, save=True, pause_each_episode=False, heuristic=False):
    """
    Test the target net performance, return a test report. Set heuristic to True
    in order to use a human written function for selecting actions.
    """

    if self.log_level > 0: 
      print("Starting test with", 
            self.env.params.test_objects * self.env.params.test_trials_per_object,
            "trial episodes", 
            f"({self.env.params.test_objects} objects each with {self.env.params.test_trials_per_object} trials)")

    # begin test mode
    self.env.start_test()

    # switch the pytorch model into evaluation mode
    self.agent.testing_mode()

    # begin testing
    for i_episode in count(1):

      # check whether the test has completed
      if self.env.test_completed:
        i_episode -= 1 # we didn't finish this episode
        break

      if self.log_level == 1 and i_episode % self.log_rate_for_episodes == 1:
        print("Begin test episode", i_episode, flush=True)
      elif self.log_level > 1:
        print("Begin test episode", i_episode, flush=True)

      if pause_each_episode: input("Press enter to continue")

      if heuristic: 
        self.env.start_heuristic_grasping()
        self.run_heuristic_episode(i_episode)
      else:
        self.run_episode(i_episode, test=True)

    # get the test data out
    test_data = self.env.test_trial_data

    # plot to the screen
    if self.plot: self.track.plot(plt_frequency_seconds=0) # guarantees data will be plotted

    # save internally this test data
    self.last_test_data = test_data

    if self.log_level > 0: print("Testing complete, finished", i_episode, "episodes")

    # switch the network back into training mode
    self.agent.training_mode()

    # process test data
    test_report = self.create_test_report(test_data, i_episode=self.track.episodes_done)

    if save and self.enable_saving:
      # save the network along with the test report
      self.save(txtfilename=self.test_result_filename, txtfilestr=test_report, extra_data=(test_data))
      # save table of test performances
      self.modelsaver.save(self.test_performances_filename, txtonly=True, txtstr=self.get_test_performance())
    else:
      if self.log_level > 0:
        print(f"Trainer.test() warning: nothing saved following test, save={save}, enable_saving={self.enable_saving}")

    return test_data

  def get_test_performance(self):
    """
    Get a table of test time performance
    """

    # save table of test performances
    log_str = "Test time performance:\n\n"
    top_row = "{0:<10} | {1:<10} | {2:<15} | {3:<10} | {4:<10} | {5:<10}\n".format(
      "Save ID", "Episode", "Success rate", "Reward", "Avg steps", "Stage"
    )
    log_str += top_row
    row_str = "{0:<10} | {1:<10} | {2:<15.3f} | {3:<10.3f} | {4:<10.3f} | {5:<10}\n"
    for i in range(len(self.track.test_episodes)):
      # if self.params.test_freq == self.params.save_freq:
      #   save_id = 1 + (self.track.test_episodes[i] // self.params.test_freq)
      # else:
      #   save_id = 1 + (self.track.test_episodes[i] // self.params.test_freq
      #                   + self.track.test_episodes[i] // self.params.save_freq
      #                   - self.track.test_episodes[i] // (np.lcm(self.params.test_freq, self.params.save_freq)))

      log_str += row_str.format(
        self.get_save_id(self.track.test_episodes[i]),
        self.track.test_episodes[i], 
        self.track.avg_successful_grasp[i],
        self.track.test_rewards[i],
        self.track.test_durations[i],
        self.track.test_curriculum_stages[i],
      )

    return log_str

  def create_test_report(self, test_data, i_episode=None, print_out=True):
    """
    Process the test data from a finished test to get a test report string
    """

    # how much do we print from this function
    print_objects = False     # print results from every object
    print_categories = True   # print averages from object categories (eg cube/sphere)
    print_overall = True      # print overall average from all objects

    len_data = len(test_data)
    num_trials = self.env.params.test_trials_per_object
    num_obj = int(len_data / num_trials)

    # safety check
    if len_data % num_trials != 0:
      raise RuntimeError("incorrect test_data length")

    # create and initialise
    names = []
    avg_rewards = []

    # save all outputs in one place
    output_str = ""
    object_table = ""
    category_table = ""
    overall_avg_table = ""

    # create intro text and column header text
    start_str = f"Test report on {num_obj} objects, with {num_trials} trials each"
    if i_episode != None: start_str += f", after {i_episode} training steps"
    else: start_str += ", before any training steps"

    # define the number of columns for the print out table and group them into styles
    col_str = (
        "{0} | " * 1 # name
      + "{1} | " * 1 # number of instances
      + "{2} | " * 1 # success rate
      + "{3} | " * 5 # float fields - reward, steps, palm f, fing.f, avg. action
      + "{4} | " * 8 # end conditions - Lft, Stb, oob, t.h, s.h, dB, dP, dW
      + "{5} | " * 7 # percentages - pLft, pCon, pPlmFrc, pXLim, pXBend, pXPalm, pXWrist #pXAxial, pXlaT, pXPalm
      + "\n"
    )

    # insert string formatting information for each column style
    header_str = col_str.format    ("{}",     "{:<4}", "{:<7}",    "{:<7}",    "{:<4}",    "{:<3}")
    normal_row_str = col_str.format("{:<51}", "{:<4}", "{:<7}",    "{:<7.2f}", "{:<4}",    "{:<3.0f}")
    avg_row_str = col_str.format   ("{:<51}", "{:<4}", "{:<7.3f}", "{:<7.2f}", "{:<4.2f}", "{:<3.0f}")

    # insert the names into the top of each column - notice the grouping of styles
    table_header = header_str.format(
      "{:<51}",
      "Num",
      "Success",
      "Reward", "Steps", "Palm f", "Fing.f", "Act.pen",
      "lft", "stb", "oob", "l2h", "s.h", "dB", "dP", "dW",
      "%Lt", "%Cn", "%PF", "%XL", "%XB", "%XP", "%XW"
    )
    
    object_table += table_header.format("Object name")

    # create cpp counter objects
    total_counter = self.env._make_event_track()
    obj_counter = self.env._make_event_track()

    # create dictionary to store the object categories
    category_dict = {}

    # loop through the number of objects in the test
    for i in range(num_obj):

      j = i * num_trials

      names.append(test_data[j].object_name)
      total_rewards = 0

      # is this object in a new catergory, if so add an event tracker
      new_category = True
      for key in category_dict.keys():
        if key == test_data[j].object_category:
          new_category = False
          break
      if new_category:
        category_dict[test_data[j].object_category] = {
          "counter" : self.env._make_event_track(),
          "num" : 0.0,
          "reward" : 0.0
        }

      # loop through the number of trials for each object
      for k in range(num_trials):

        # add together the event counts for this object
        obj_counter = self.env._add_events(obj_counter, test_data[j+k].cnt)

        # add the counts for this category
        category_dict[test_data[j].object_category]["counter"] = self.env._add_events(
          category_dict[test_data[j].object_category]["counter"], test_data[j+k].cnt
        )
        category_dict[test_data[j].object_category]["num"] += 1
        category_dict[test_data[j].object_category]["reward"] += test_data[j + k].reward

        # sum end of episode rewards for this set of trials
        total_rewards += test_data[j + k].reward

      # calculate averages rewards for the set of trials
      avg_rewards.append(total_rewards / float(num_trials))
      
      # calculate the percentage of steps that events were active
      obj_counter.calculate_percentage()

      # save all data in a string to output to a test summary text file
      obj_row = normal_row_str.format(
        # name x1
        names[-1], 
        # number x1
        num_trials,
        # success rate x1
        obj_counter.successful_grasp.active_sum,
        # float style x5
        avg_rewards[-1], 
        obj_counter.step_num.abs / float(num_trials),
        obj_counter.palm_force.last_value / float(num_trials),
        obj_counter.finger_force.last_value / float(num_trials),
        obj_counter.action_penalty.last_value / float(num_trials),
        # end state style x8
        obj_counter.lifted.active_sum, 
        obj_counter.object_stable.active_sum, 
        obj_counter.oob.active_sum, 
        obj_counter.lifted_to_height.active_sum, 
        obj_counter.stable_height.active_sum,
        obj_counter.dangerous_bend_sensor.active_sum,
        obj_counter.dangerous_palm_sensor.active_sum,
        obj_counter.dangerous_wrist_sensor.active_sum,
        # perentage style x7
        obj_counter.lifted.percent,
        obj_counter.object_contact.percent,
        obj_counter.palm_force.percent,
        obj_counter.exceed_limits.percent,
        obj_counter.exceed_bend_sensor.percent,
        obj_counter.exceed_palm_sensor.percent,
        obj_counter.exceed_wrist_sensor.percent
      )

      object_table += obj_row

      # add these events to the total counter
      total_counter = self.env._add_events(total_counter, obj_counter)

      # reset the object counter
      obj_counter.reset()  

    # now get the mean reward over the entire test
    mean_reward = np.mean(np.array(avg_rewards))

    # update the percentage values for the entire test
    total_counter.calculate_percentage()

    # prepare to divide by total test episdoes to calculate float averges
    N = float(num_trials * num_obj)

    # add the overall averages to the test report string
    overall_avg_table += table_header.format("Overall average")
    overall_avg_table += avg_row_str.format(
      # name x1
      "All objects",
      # number x1
      int(N),
      # success rate x1 (averaged)
      total_counter.successful_grasp.active_sum / N,
      # float style x5
      mean_reward, 
      total_counter.step_num.abs / N,
      total_counter.palm_force.last_value / N,
      total_counter.finger_force.last_value / N,
      total_counter.action_penalty.last_value / N,
      # end state style (averaged) x8
      total_counter.lifted.active_sum / N, 
      total_counter.object_stable.active_sum / N, 
      total_counter.oob.active_sum / N, 
      total_counter.lifted_to_height.active_sum / N, 
      total_counter.stable_height.active_sum / N,
      total_counter.dangerous_bend_sensor.active_sum / N,
      total_counter.dangerous_palm_sensor.active_sum / N,
      total_counter.dangerous_wrist_sensor.active_sum / N,
      # percentage style x7
      total_counter.lifted.percent,
      total_counter.object_contact.percent,
      total_counter.palm_force.percent,
      total_counter.exceed_limits.percent,
      obj_counter.exceed_bend_sensor.percent,
      obj_counter.exceed_palm_sensor.percent,
      obj_counter.exceed_wrist_sensor.percent
    )

    # now extract category data

    # use try-catch for old code compatibility
    try:
      test = self.track.object_categories
    except Exception as e:
      print("Object categories fields did not exist in this model, old code")
      print(e)
      self.track.object_categories = []
      self.track.category_num = []
      self.track.category_stable = []
      self.track.category_lifted = []
      self.track.category_lifted_to_height = []
      self.track.category_stable_height = []
      self.track.category_successful_grasp = []

    try:
      test = self.track.avg_dangerous_bend
    except Exception as e:
      print("Dangerous fields did not exist in this model, old code")
      print(e)
      self.track.category_dangerous_bend = []
      self.track.category_dangerous_palm = []
      self.track.category_dangerous_wrist = []
      numpy_float = np.float32
      self.track.avg_successful_grasp = np.array([], dtype=numpy_float)
      self.track.avg_dangerous_bend = np.array([], dtype=numpy_float)
      self.track.avg_dangerous_palm = np.array([], dtype=numpy_float)
      self.track.avg_dangerous_wrist = np.array([], dtype=numpy_float)

    try:
      test = self.track.test_curriculum_stages
    except Exception as e:
      print("Curriculum stages not found in self.track, old code")
      print(e)
      self.track.train_curriculum_stages = np.array([], dtype=np.int32)
      self.track.test_curriculum_stages = np.array([], dtype=np.int32)

    self.track.object_categories = list(category_dict.keys())

    category_table += table_header.format("Overall averages by category")

    num_per_obj = []
    successful_grasp_per_obj = []
    reward_per_obj = []
    step_num_per_obj = []
    palm_force_per_obj = []
    finger_force_per_obj = []
    action_penalty_per_obj = []
    lifted_per_obj = []
    stable_per_obj = []
    oob_per_obj = []
    lifted_to_height_per_obj = []
    stable_height_per_obj = []
    lifted_percentage_per_obj = []
    contact_percentage_per_obj = []
    palm_force_percentage_per_obj = []
    exceed_limits_percentage_per_obj = []
    exceed_bend_percentage_per_obj = []
    exceed_palm_percentage_per_obj = []
    exceed_wrist_percentage_per_obj = []
    dangerous_bend_per_obj = []
    dangerous_palm_per_obj = []
    dangerous_wrist_per_obj = []

    for i, cat in enumerate(self.track.object_categories):

      category_dict[cat]["counter"].calculate_percentage()
      category_dict[cat]["reward"] /= category_dict[cat]["num"]

      num_per_obj.append(category_dict[cat]["num"])
      successful_grasp_per_obj.append(category_dict[cat]["counter"].successful_grasp.active_sum / category_dict[cat]["num"])
      reward_per_obj.append(category_dict[cat]["reward"])

      step_num_per_obj.append(category_dict[cat]["counter"].step_num.abs / category_dict[cat]["num"])
      palm_force_per_obj.append(category_dict[cat]["counter"].palm_force.last_value / category_dict[cat]["num"])
      finger_force_per_obj.append(category_dict[cat]["counter"].finger_force.last_value / category_dict[cat]["num"])
      action_penalty_per_obj.append(category_dict[cat]["counter"].action_penalty.last_value / category_dict[cat]["num"])

      lifted_per_obj.append(category_dict[cat]["counter"].lifted.active_sum / category_dict[cat]["num"])
      stable_per_obj.append(category_dict[cat]["counter"].object_stable.active_sum / category_dict[cat]["num"])
      oob_per_obj.append(category_dict[cat]["counter"].oob.active_sum / category_dict[cat]["num"])
      lifted_to_height_per_obj.append(category_dict[cat]["counter"].lifted_to_height.active_sum / category_dict[cat]["num"])
      stable_height_per_obj.append(category_dict[cat]["counter"].stable_height.active_sum / category_dict[cat]["num"])
      dangerous_bend_per_obj.append(category_dict[cat]["counter"].dangerous_bend_sensor.active_sum / category_dict[cat]["num"])
      dangerous_palm_per_obj.append(category_dict[cat]["counter"].dangerous_palm_sensor.active_sum / category_dict[cat]["num"])
      dangerous_wrist_per_obj.append(category_dict[cat]["counter"].dangerous_wrist_sensor.active_sum / category_dict[cat]["num"])

      lifted_percentage_per_obj.append(category_dict[cat]["counter"].lifted.percent)
      contact_percentage_per_obj.append(category_dict[cat]["counter"].object_contact.percent)
      palm_force_percentage_per_obj.append(category_dict[cat]["counter"].palm_force.percent)
      exceed_limits_percentage_per_obj.append(category_dict[cat]["counter"].exceed_limits.percent)
      exceed_bend_percentage_per_obj.append(category_dict[cat]["counter"].exceed_bend_sensor.percent)
      exceed_palm_percentage_per_obj.append(category_dict[cat]["counter"].exceed_palm_sensor.percent)
      exceed_wrist_percentage_per_obj.append(category_dict[cat]["counter"].exceed_wrist_sensor.percent)

    for c in range(len(self.track.object_categories)):

      cat_row = avg_row_str.format(
        # name x1
        self.track.object_categories[c], 
        # number x1
        int(category_dict[self.track.object_categories[c]]["num"]),
        # success rate
        successful_grasp_per_obj[c],
        # float style x5
        reward_per_obj[c], 
        step_num_per_obj[c],
        palm_force_per_obj[c],
        finger_force_per_obj[c],
        action_penalty_per_obj[c],
        # end state style x8
        lifted_per_obj[c], 
        stable_per_obj[c], 
        oob_per_obj[c], 
        lifted_to_height_per_obj[c], 
        stable_height_per_obj[c],
        dangerous_bend_per_obj[c],
        dangerous_palm_per_obj[c],
        dangerous_wrist_per_obj[c],
        # perentage style x7
        lifted_percentage_per_obj[c],
        contact_percentage_per_obj[c],
        palm_force_percentage_per_obj[c],
        exceed_limits_percentage_per_obj[c],
        exceed_bend_percentage_per_obj[c],
        exceed_palm_percentage_per_obj[c],
        exceed_wrist_percentage_per_obj[c]
      )

      category_table += cat_row

    # save test results if we are mid-training
    if i_episode != None:

      # overall results
      self.track.test_episodes = np.append(self.track.test_episodes, i_episode)
      self.track.test_durations = np.append(self.track.test_durations, total_counter.step_num.abs / N)
      self.track.test_rewards = np.append(self.track.test_rewards, mean_reward)
      self.track.test_curriculum_stages = np.append(self.track.test_curriculum_stages, self.curriculum_dict["stage"])
      self.track.avg_successful_grasp = np.append(self.track.avg_successful_grasp, total_counter.successful_grasp.active_sum / N)
      self.track.avg_p_lifted = np.append(self.track.avg_p_lifted, total_counter.lifted.percent)
      self.track.avg_p_contact = np.append(self.track.avg_p_contact, total_counter.object_contact.percent)
      self.track.avg_p_palm_force = np.append(self.track.avg_p_palm_force, total_counter.palm_force.percent)
      self.track.avg_p_exceed_limits = np.append(self.track.avg_p_exceed_limits, total_counter.exceed_limits.percent)
      self.track.avg_p_exceed_bend = np.append(self.track.avg_p_exceed_bend, total_counter.exceed_bend_sensor.percent)
      self.track.avg_p_exceed_palm = np.append(self.track.avg_p_exceed_palm, total_counter.exceed_palm_sensor.percent)
      self.track.avg_p_exceed_wrist = np.append(self.track.avg_p_exceed_wrist, total_counter.exceed_wrist_sensor.percent)
      self.track.avg_lifted = np.append(self.track.avg_lifted, total_counter.lifted.active_sum / N)
      self.track.avg_stable = np.append(self.track.avg_stable, total_counter.object_stable.active_sum / N)
      self.track.avg_oob = np.append(self.track.avg_oob, total_counter.oob.active_sum / N)
      self.track.avg_lifted_to_height = np.append(self.track.avg_lifted_to_height, total_counter.lifted_to_height.active_sum / N)
      self.track.avg_stable_height = np.append(self.track.avg_stable_height, total_counter.stable_height.active_sum / N)
      self.track.avg_dangerous_bend = np.append(self.track.avg_dangerous_bend, total_counter.dangerous_bend_sensor.active_sum / N)
      self.track.avg_dangerous_palm = np.append(self.track.avg_dangerous_palm, total_counter.dangerous_palm_sensor.active_sum / N)
      self.track.avg_dangerous_wrist = np.append(self.track.avg_dangerous_wrist, total_counter.dangerous_wrist_sensor.active_sum / N)

      # save only select category data
      self.track.category_num.append(num_per_obj)
      self.track.category_lifted.append(lifted_per_obj)
      self.track.category_stable.append(stable_per_obj)
      self.track.category_lifted_to_height.append(lifted_to_height_per_obj)
      self.track.category_stable_height.append(stable_height_per_obj)
      self.track.category_successful_grasp.append(successful_grasp_per_obj)
      self.track.category_dangerous_bend.append(dangerous_bend_per_obj)
      self.track.category_dangerous_palm.append(dangerous_palm_per_obj)
      self.track.category_dangerous_wrist.append(dangerous_wrist_per_obj)

    # finally, assembly the output string
    output_str += start_str + "\n"
    output_str += "\n" + object_table
    output_str += "\n" + category_table
    output_str += "\n" + overall_avg_table

    # print out information based on flags at top of function
    if print_out:
      print(start_str + "\n")
      if print_objects: print(object_table)
      if print_categories: print(category_table)
      if print_overall: print(overall_avg_table)

    # save a flag for final success rate
    self.last_test_success_rate = total_counter.object_stable.active_sum / N

    return output_str

  def calc_best_performance(self, from_episode=None, to_episode=None, return_id=None,
                            success_rate_vector=None, episodes_vector=None,
                            stages_vector=None, from_stage=None, to_stage=None):
    """
    Find the best success rate by the model, and what episode number it occured
    """

    if from_episode is None: from_episode = 0
    if to_episode is None: to_episode = 100_000_000
    if from_stage is None: from_stage = 0
    if to_stage is None: to_stage = 100_000_000

    if success_rate_vector is None:
      success_rate_vector = self.track.avg_stable_height

    if episodes_vector is None:
      episodes_vector = self.track.test_episodes

    if stages_vector is None:
      stages_vector = self.track.test_curriculum_stages

    best_sr = 0
    best_ep = 0
    best_id = 0

    # loop through, this is slower than numpy but lets us check for 'from_episode' etc
    for i, sr in enumerate(success_rate_vector):

      # get info
      this_ep = int(episodes_vector[i])
      this_stage = int(stages_vector[i])

      # check if this episode is past our minimum
      if this_ep < from_episode: continue

      # check if this episode is past our maximum
      if this_ep > to_episode: break

      # check if this stage is past our minimum
      if this_stage < from_stage: continue

      # check if this stage is past our maximum
      if this_stage > to_stage: break

      # see if this is best
      if sr > best_sr:
        best_sr = sr
        best_ep = this_ep
        best_id = i

    if return_id: return best_sr, best_ep, best_id

    return best_sr, best_ep

  def read_test_performance(self, as_string=False):
    """
    Read the test performance into a numpy array
    """

    try:

      readroot = self.savedir + "/" + self.group_name + "/"
      readpath = readroot + self.run_name + "/"
      readname = self.test_performances_filename + ".txt"

      with open(readpath + readname, "r") as f:
        txt = f.read()

    except Exception as e:
      if self.log_level > 0:
        print(f"TrainDQN.read_test_performance() error: {e}")
      if as_string:
        return f"TrainDQN.read_test_performance() error: {e}"
      else:
        return np.zeros((5,1))
    
    if as_string: return txt

    lines = txt.splitlines()

    save_ids = []
    episodes = []
    success_rates = []
    rewards = []
    avg_steps = []
    stages = []

    found_data_new = False
    found_data_old = False

    for l in lines:

      if found_data_new:

        splits = l.split("|")
        save_ids.append(int(splits[0]))
        episodes.append(int(splits[1]))
        success_rates.append(float(splits[2]))
        rewards.append(float(splits[3]))
        avg_steps.append(float(splits[4]))
        try:
          stages.append(int(splits[5]))
        except Exception as e:
          print("Error in read_test_performance, old code. Error", e)
          stages.append(0)

      elif found_data_old:

        splits = l.split("|")
        save_ids.append(0)
        episodes.append(int(splits[0]))
        success_rates.append(float(splits[1]))
        rewards.append(0)
        avg_steps.append(0)
        stages.append(0)

      if l.startswith("Save ID"):
        found_data_new = True
      elif l.startswith("Episode"):
        found_data_old = True

    if len(episodes) != len(success_rates):
      raise RuntimeError("TrainDQN.read_test_performance() found episode length != success rate length")

    if not found_data_new and not found_data_old:
      save_ids = [0]
      episodes = [0]
      success_rates = [0]
      rewards = [0]
      avg_steps = [0]
      stages = [0]

    # convert into a numpy matrix
    matrix = np.concatenate(
      (np.array([save_ids]), 
       np.array([episodes]),
       np.array([success_rates]),
       np.array([rewards]),
       np.array([avg_steps]),
       np.array([stages])), 
      axis=0
    )

    return matrix

  def read_best_performance_from_text(self, silence=False, fulltest=False, heuristic=False,
                                      stage=None):
    """
    Read a text file to get the best model performance. This function contains
    hardcoding. Stage can be None, or a number, or "max" to choose the highest
    """

    readroot = self.savedir + "/" + self.group_name + "/"

    if heuristic: 
      readroot += "heuristic/"
      fulltest_str = "heuristic_test"
    else:
      fulltest_str = "full_test"

    readpath = readroot + self.run_name + "/"

    try:

      # special case, get fulltest information
      if fulltest:

        test_files = [x for x in os.listdir(readpath) if x.startswith(fulltest_str) and x.endswith(".txt")]
        
        if len(test_files) == 0: return None, None
        
        elif len(test_files) > 1: 

          if not silence: print(f"Multiple '{fulltest_str}.txt' files found in read_best_performance_from_text(...)")

          # hardcoded date string
          datestr = "%d-%m-%y_%H-%M"
          ex_date = datetime.now().strftime(datestr)

          # remove the '.txt' extension
          to_remove = ".txt"
          no_txt = [x[:-len(to_remove)] for x in test_files[:]]

          # get only the dates
          date_strings = [x[-len(ex_date):] for x in no_txt[:]]

          # convert to datetime objects
          try:
            dates = [datetime.strptime(x, datestr) for x in date_strings[:]]
          except ValueError as e:
            if not silence: print("read_best_performance_from_text() datetime error:", e)
            if not silence: print("Trying again with another datestring") # OLD CODE compatible
            # try again with alternative datestring
            datestr = "%d-%m-%y-%H:%M"
            ex_date = datetime.now().strftime(datestr)
            date_strings = [x[-len(ex_date):] for x in no_txt[:]]
            dates = [datetime.strptime(x, datestr) for x in date_strings[:]]

          # loop over to find the index of the most recent
          recent_ind = 0
          recent_date = dates[0]
          for i in range(1, len(dates)):
            if recent_date < dates[i]:
              recent_date = dates[i]
              recent_ind = i

          # now finally select the most recent date
          readname = str(test_files[recent_ind])

          if not silence: print("Most recent fulltest selected:", readname)

        else: readname = str(test_files[0])

      else: 
        # return the best_sr, best_ep from the test performance file
        matrix = self.read_test_performance()
        if stage is not None:
          if matrix.shape[0] < 5:
            raise RuntimeError("read_best_performance_from_text() failed as 'stage' was given but old training is incompatible")
          if isinstance(stage, int):
            stage_indexes = np.nonzero(abs(matrix[5] - stage) < 1e-5)[0]
            if len(stage_indexes) == 0:
              raise RuntimeError(f"read_best_performance_from_text() failed as stage = {stage} but this was not reached in this training")
            matrix = matrix[:,stage_indexes[0]:stage_indexes[-1]]
          elif stage == "max":
            stage_index_min = np.argmax(matrix[5])
            matrix = matrix[:,stage_index_min:]
          else:
            raise RuntimeError(f"read_best_performance_from_text() failed as 'stage' was not None, int, or 'max', instead stage = {stage}")
        best_index = np.argmax(matrix[2])
        return matrix[2][best_index], int(matrix[1][best_index])

      if self.log_level > 0: print(f"Reading text file: {readpath + readname}")
      with open(readpath + readname, 'r') as openfile:
        file_txt = openfile.read()

    except FileNotFoundError as e:
      if not silence: print("read_best_performance_from_text() failed with error:", e)
      return None, None
    
    except RuntimeError as e:
      if not silence: print("read_best_performance_from_text() failed with error:", e)
      return None, None

    # special case, get the fulltest information
    if fulltest:
      lines = file_txt.splitlines()
      elem = lines[0].split(" ")
      try:
        best_sr = float(elem[-1])
      except ValueError as e:
        print("Error in read_best_performance_from_text() on fulltest:", e)
        exit()
      if self.log_level > 0: print(f"{readname} gives performance of {best_sr}")
      return best_sr, None

    # extract details based on hardcoded knowledge of txt file structure (see self.best_performance_template)
    lines = file_txt.splitlines()
    for i in range(len(lines)):
      lines[i] = lines[i].split(" ")

    # check for errors
    if len(lines) < 2:
      if not silence: print("Error in read_best_performance_from_text(), lines < 2")
      return None, None
    
    try:
      best_sr = float(lines[0][-1])
      best_ep = int(lines[1][-1])
    except ValueError as e:
      print("Error in read_best_performance_from_text():", e)
      exit()

    if self.log_level > 0:
      print(f"model.read_best_performance_from_text() gives best_sr={best_sr} and best_ep={best_ep}")

    return best_sr, best_ep

  def load_best_id(self, run_name, group_name=None, path_to_run_folder=None, stage=None):
    """
    Try to find the best performing agent and load that. Stage indicates requirements
    for which curriulum stage we can load. If an int, we load that stage, or if "max"
    we load the highest reached stage
    """

    id = None
    best_id_found = False

    if self.log_level > 0: 
      print(f"MujocoTrainer.load_best_id() is now trying to find the best agent id to load, stage = {stage}")

    best_sr, best_ep = self.read_best_performance_from_text(stage=stage)
    if best_sr is None or best_sr < 1e-5:
      if self.log_level > 0: 
        print("MujocoTrainer.load_best_id() cannot find best id as best success rate is zero")
    elif best_ep % self.params.save_freq != 0:
      if self.log_level > 0: 
        print(f"MujocoTrainer.load_best_id() cannot find best id as best_episode = {best_ep} and save_freq = {self.params.save_freq}, these are incompatible")
    else:
      id = self.get_save_id(best_ep)
      if self.log_level > 0: 
        print(f"id set to {id} with best_ep={best_ep}, save_freq={self.params.save_freq} and best_sr={best_sr}")
      best_id_found = True

    # try to load, if best id not found it loads most recent (ie id=None)
    self.load(run_name, id=id, group_name=group_name, path_to_run_folder=path_to_run_folder)

    if not best_id_found:
      if stage == "max": stage = self.curriculum_dict["stage"]
      best_sr, best_ep = self.calc_best_performance(from_stage=stage, to_stage=stage)
      if self.log_level > 0: 
        print(f"BEST_ID_FAILED  -> Preparing to reload with best id in model.load(...)")
      if best_sr < 1e-5:
        if self.log_level > 0: 
          print("BEST_ID_FAILED  -> load(...) cannot find best id as best success rate is zero")
      else:
        best_id_found = True
        best_id = self.get_save_id(best_ep)
        if self.log_level > 0: 
          print(f"BEST_ID_SUCCESS -> best_id set to {best_id} with best_ep={best_ep}, save_freq={self.params.save_freq} and best_sr={best_sr}")
        # try to load again
        self.load(run_name, id=best_id, group_name=group_name, path_to_run_folder=path_to_run_folder)

    return best_id_found

  def curriculum_fcn(self, i):
    """
    Curriculum function which updates which stage we are at given the episode number
    i, and calls another function 'self.curriculum_change(stage)' with the stage
    number to apply stage dependent parameter changes. This function must be user
    defined and overriden! Otherwise a NotImplementedError will be raised
    """

    # allow_backwards = False
    # allow_multistep = False

    if self.curriculum_dict["finished"]: return

    # determine what stage we are at
    stage = self.curriculum_dict["stage"]

    # determine the curriculum metric
    if self.curriculum_dict["metric_name"] == "episode_number":

      for t in self.curriculum_dict["metric_thresholds"][stage:]:
        if i >= t:
          stage += 1
        else: break

    elif self.curriculum_dict["metric_name"] == "success_rate":

      # get the most recent success rate
      if len(self.track.avg_successful_grasp) > 0:
        success_rate = self.track.avg_successful_grasp[-1]
      else: success_rate = 0.0

      # determine if we have passed the required threshold
      for t in self.curriculum_dict["metric_thresholds"][stage:]:
        if success_rate >= t:
          stage += 1

    # if the metric is not recognised
    else: raise RuntimeError(f"TrainingManager.curriculum_fcn() metric of {self.curriculum_dict['metric_name']} not recognised")

    # check if we are still at the same stage we were last episode
    if stage == self.curriculum_dict["stage"]:
      # check if we have finished the curriculum
      if stage == len(self.curriculum_dict["metric_thresholds"]): 
        self.curriculum_dict["finished"] = True
      return

    # update to the new stage
    if self.log_level > 0:
      print(f"Episode = {i}, curriculum is changing from stage {self.curriculum_dict['stage']} to stage {stage}")
    self.curriculum_dict["stage"] = stage

    # now apply the curriculum change (this function must be user overwritten for a training)
    self.curriculum_change(stage)

    # save a text file to reflect the changes
    labelstr = f"Hyperparameters after curriculum change which occured at episode {i}\n"
    name = f"hyperparameters_curriculum_stage_{stage}"
    self.save_hyperparameters(filename=name, strheader=labelstr, print_terminal=False)
    
    return

  def curriculum_change(self, stage):
    """
    Apply parameters to change the curriculum to the given stage. This function
    should be overwritten if using a curriculum (params.use_curriculum = True)
    """
    raise NotImplementedError("Trainer.curriculum_change() must be overwritten if using a curriculum")

if __name__ == "__main__":

  # master seed, torch seed must be set before network creation (random initialisation)
  rngseed = 2
  strict_seed = True
  if strict_seed:
    if rngseed is None: rngseed = random.randint(0, 2_147_483_647)
    torch.manual_seed(rngseed)
  
  render = True
  continous_actions = True
  log_level = 2
  
  # create the environment
  env = MjEnv(object_set="set9_fullset", log_level=log_level, render=render,
              continous_actions=continous_actions)
  # env.params.test_objects = 1
  # env.params.test_trials_per_object = 1
  env.params.max_episode_steps = 250
  env.params.object_position_noise_mm = 20

  # training device
  device = "cpu"

  # make the agent
  layers = [32, 32]
  network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                             continous_actions=continous_actions)
  agent = Agent_PPO(device=device)
  agent.init(network)

  # train the agent on the environment
  trainer = MujocoTrainer(agent, env, rngseed=rngseed, device=device, plot=False, save=True,
                          strict_seed=strict_seed, episode_log_rate=1, render=render,
                          track_avg_num=5, print_avg_return=True, log_level=log_level)
  
  # set training parameters then train
  trainer.params.num_episodes = 10000
  # trainer.params.save_freq = 5
  # trainer.params.test_freq = 5
  # trainer.load("run_16-37", path_to_run_folder="/home/luke/mujoco-devel/models/22-09-23", id=2)
  trainer.train()