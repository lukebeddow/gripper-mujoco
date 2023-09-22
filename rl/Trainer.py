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
    # testing data
    self.test_episodes = np.array([], dtype=np.int32)
    self.test_rewards = np.array([], dtype=self.numpy_float)
    self.test_durations = np.array([], dtype=np.int32)
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

    for m in metrics_to_add:
      self.test_metric_names.append(m)
      self.test_metric_values.append(np.array([], dtype=dtype))

    self.n_test_metrics = len(self.test_metric_names)

  def log_training_episode(self, reward, duration, time_taken):
    """
    Log one training episode
    """

    self.train_episodes = np.append(self.train_episodes, self.episodes_done)
    self.train_durations = np.append(self.train_durations, duration)
    self.train_rewards = np.append(self.train_rewards, reward)
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

class Trainer:

  @dataclass
  class Parameters:

    num_episodes: int = 10_000
    test_freq: int = 1000
    save_freq: int = 50

  def __init__(self, agent, env, rngseed=None, device="cpu", log_level=1, plot=False,
               render=False, group_name="default_%d-%m-%y", run_name="default_run_%H-%M",
               save=True, savedir="models", episode_log_rate=10, strict_seed=False):
    """
    Class that trains RL agents in an environment
    """

    # prepare class variables
    self.track = TrackTraining()
    self.params = Trainer.Parameters()
    self.agent = agent
    self.env = env
    self.saved_trainer_params = False

    # input class options
    self.rngseed = rngseed
    self.device = torch.device(device)
    self.agent.set_device(device)
    self.log_level = log_level
    self.plot = plot
    self.render = render
    self.log_rate_for_episodes = episode_log_rate
    
    # set up saving
    self.enable_saving = save
    self.train_param_savename = "Trainer_params"
    self.track_savename = "Tracking_info"
    self.setup_saving(run_name, group_name, savedir)

    # are we plotting
    if self.plot:
      global plt
      import matplotlib.pyplot as plt
      plt.ion()

    # seed the environment 
    # training only reproducible if torch.manual_seed() set BEFORE agent network initialisation
    self.training_reproducible = strict_seed
    self.seed(strict=strict_seed)

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
                   savedir="models", enable_saving=None):
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

  def save(self, txtfilename=None, txtfilestr=None, extra_data=None, force_train_params=False):
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
        "env_data" : self.env.get_save_state(),
        "extra_data" : extra_data
      }

      self.modelsaver.save(self.train_param_savename, pyobj=trainer_save)
      self.saved_trainer_params = True

    # save tracking information
    self.modelsaver.save(self.track_savename, pyobj=self.track)

    # save the actual agent
    self.modelsaver.save(self.agent.name, pyobj=self.agent.get_save_state(),
                         txtstr=txtfilestr, txtlabel=txtfilename)

  def get_param_dict(self):
    """
    Return a dictionary of hyperparameters
    """
    param_dict = asdict(self.params)
    param_dict.update({
      "rngseed" : self.rngseed,
      "training_reproducible" : self.training_reproducible
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

    hyper_str = "\nTraining Hyperparameters\n\n"
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

    load_agent = self.modelsaver.load(id=id, folderpath=path_to_run_folder,
                                      foldername=run_name, filenamestarts="Agent")
    load_track = self.modelsaver.load(id=id, folderpath=path_to_run_folder, 
                                      foldername=run_name, filenamestarts=self.track_savename)
    load_train = self.modelsaver.load(folderpath=path_to_run_folder,
                                      foldername=run_name, filenamestarts=self.train_param_savename)
    
    # extract loaded data
    self.params = load_train["parameters"]
    self.track = load_track
    self.agent.load_save_state(load_agent)
    self.env.load_save_state(load_train["env_data"])

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
      action = self.agent.select_action(obs, decay_num=i_episode, test=test)
      (new_obs, reward, terminated, truncated, info) = self.env.step(action.item())
   
      # render the new environment
      if self.render: self.env.render()

      if terminated or truncated: done = True
      else: done = False

      # convert data to torch tensors on specified device
      new_obs = self.to_torch(new_obs)
      reward = self.to_torch(reward)
      action = action.to(self.device).unsqueeze(0).unsqueeze(0) # from Tensor(x) -> Tensor([[x]])
      done_torch = self.to_torch(done, dtype=torch.bool)

      # perform one step of the optimisation on the policy network
      if test != True:
        self.agent.update_step(obs, action, new_obs, reward, done_torch)

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
        self.track.log_training_episode(cumulative_reward, t + 1, time_per_step)
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

    # put the agent into training mode
    self.agent.training_mode()
    
    # begin training episodes
    for i_episode in range(i_start + 1, self.params.num_episodes + 1):

      if self.log_level == 1 and (i_episode - 1) % self.log_rate_for_episodes == 0:
        print("Begin training episode", i_episode, flush=True)
      elif self.log_level > 1:
        print(f"Begin training episode {i_episode} at {datetime.now().strftime('%H:%M')}", flush=True)

      self.run_episode(i_episode)

      # plot graphs to the screen
      if self.plot: self.track.plot(plt_frequency_seconds=1)
      self.track.print_training()

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

    # end of training
    self.env.close()
    self.finish_training()

  def test(self):
    """
    Empty test function, should be overriden for each environment
    """
    pass

  def finish_training(self):
    """
    Override this function to run code at the end of a training
    """
    pass

class MujocoTrainer(Trainer):

  @dataclass
  class Parameters:
    num_episodes: int = 10_000
    test_freq: int = 1000
    save_freq: int = 1000

  def __init__(self, agent, mjenv, rngseed=None, device="cpu", log_level=1, plot=False,
               render=False, group_name="default_%d-%m-%y", run_name="default_run_%H-%M",
               save=True, savedir="models", episode_log_rate=10, strict_seed=False):
    """
    Trainer class for the gripper mujoco RL environment
    """

    super().__init__(agent, mjenv, rngseed=rngseed, device=device, log_level=log_level, 
                     plot=plot, render=render, group_name=group_name, run_name=run_name,
                     save=save, savedir=savedir, episode_log_rate=episode_log_rate, 
                     strict_seed=strict_seed)

    # override the parameters of the base class
    self.params = MujocoTrainer.Parameters()

    # class variables
    self.last_test_data = None
    self.test_performances_filename = "test_performance"
    self.test_result_filename = "test_results"

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

  def test(self, pause_each_episode=None, heuristic=None):
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
    test_report = self.create_test_report(test_data, i_episode=i_episode)

    # save the network along with the test report
    self.save(txtfilename=self.test_result_filename, txtfilestr=test_report, extra_data=(test_data))

    # save table of test performances
    log_str = "Test time performance (success rate metric = stable height):\n\n"
    top_row = "{0:<10} | {1:<15}\n".format("Episode", "Success rate")
    log_str += top_row
    row_str = "{0:<10} | {1:<15.3f}\n"
    for i in range(len(self.track.test_episodes)):
      log_str += row_str.format(self.track.test_episodes[i], self.track.avg_stable_height[i])
    self.modelsaver.save(self.test_performances_filename, txtonly=True, txtstr=log_str)

    return test_data

  def create_test_report(self, test_data, i_episode=None):
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
      + "{2} | " * 4 # float fields - reward, steps, palm f, fing.f
      + "{3} | " * 5 # end conditions - Lft, Stb, oob, t.h, s.h
      + "{4} | " * 7 # percentages - pLft, pCon, pPlmFrc, pXLim, pXAxial, pXlaT, pXPalm
      + "\n"
    )

    # insert string formatting information for each column style
    header_str = col_str.format    ("{}",     "{:<3}", "{:<6}",    "{:<4}",    "{:<3}")
    normal_row_str = col_str.format("{:<51}", "{:<3}", "{:<6.2f}", "{:<4}",    "{:<3.0f}")
    avg_row_str = col_str.format   ("{:<51}", "{:<3}", "{:<6.2f}", "{:<4.2f}", "{:<3.0f}")

    # insert the names into the top of each column - notice the grouping of styles
    table_header = header_str.format(
      "{:<51}",
      "Num",
      "Reward", "Steps", "Palm f", "Fing.f", 
      "lft", "stb", "oob", "t.h", "s.h", 
      "%Lt", "%Cn", "%PF", "%XL", "%XA", "%XT", "%XP"                                                                         
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
        # float style x4
        avg_rewards[-1], 
        obj_counter.step_num.abs / float(num_trials),
        obj_counter.palm_force.last_value / float(num_trials),
        obj_counter.finger_force.last_value / float(num_trials),
        # end state style x5
        obj_counter.lifted.last_value, 
        obj_counter.object_stable.last_value, 
        obj_counter.oob.last_value, 
        obj_counter.target_height.last_value, 
        obj_counter.stable_height.last_value,
        # perentage style x7
        obj_counter.lifted.percent,
        obj_counter.object_contact.percent,
        obj_counter.palm_force.percent,
        obj_counter.exceed_limits.percent,
        obj_counter.exceed_axial.percent,
        obj_counter.exceed_lateral.percent,
        obj_counter.exceed_palm.percent
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
      # float style x4
      mean_reward, 
      total_counter.step_num.abs / N,
      total_counter.palm_force.last_value / N,
      total_counter.finger_force.last_value / N,
      # end state style (averaged) x5
      total_counter.lifted.last_value / N, 
      total_counter.object_stable.last_value / N, 
      total_counter.oob.last_value / N, 
      total_counter.target_height.last_value / N, 
      total_counter.stable_height.last_value / N,
      # percentage style x7
      total_counter.lifted.percent,
      total_counter.object_contact.percent,
      total_counter.palm_force.percent,
      total_counter.exceed_limits.percent,
      total_counter.exceed_axial.percent,
      total_counter.exceed_lateral.percent,
      total_counter.exceed_palm.percent
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
      self.track.category_target_height = []
      self.track.category_stable_height = []

    self.track.object_categories = list(category_dict.keys())

    category_table += table_header.format("Overall averages by category")

    num_per_obj = []
    reward_per_obj = []
    step_num_per_obj = []
    palm_force_per_obj = []
    finger_force_per_obj = []
    lifted_per_obj = []
    stable_per_obj = []
    oob_per_obj = []
    target_height_per_obj = []
    stable_height_per_obj = []
    lifted_percentage_per_obj = []
    contact_percentage_per_obj = []
    palm_force_percentage_per_obj = []
    exceed_limits_percentage_per_obj = []
    exceed_axial_percentage_per_obj = []
    exceed_lateral_percentage_per_obj = []
    exceed_palm_percentage_per_obj = []

    for i, cat in enumerate(self.track.object_categories):

      category_dict[cat]["counter"].calculate_percentage()
      category_dict[cat]["reward"] /= category_dict[cat]["num"]

      num_per_obj.append(category_dict[cat]["num"])
      reward_per_obj.append(category_dict[cat]["reward"])

      step_num_per_obj.append(category_dict[cat]["counter"].step_num.abs / category_dict[cat]["num"])
      palm_force_per_obj.append(category_dict[cat]["counter"].palm_force.last_value / category_dict[cat]["num"])
      finger_force_per_obj.append(category_dict[cat]["counter"].finger_force.last_value / category_dict[cat]["num"])

      lifted_per_obj.append(category_dict[cat]["counter"].lifted.last_value / category_dict[cat]["num"])
      stable_per_obj.append(category_dict[cat]["counter"].object_stable.last_value / category_dict[cat]["num"])
      oob_per_obj.append(category_dict[cat]["counter"].oob.last_value / category_dict[cat]["num"])
      target_height_per_obj.append(category_dict[cat]["counter"].target_height.last_value / category_dict[cat]["num"])
      stable_height_per_obj.append(category_dict[cat]["counter"].stable_height.last_value / category_dict[cat]["num"])

      lifted_percentage_per_obj.append(category_dict[cat]["counter"].lifted.percent)
      contact_percentage_per_obj.append(category_dict[cat]["counter"].object_contact.percent)
      palm_force_percentage_per_obj.append(category_dict[cat]["counter"].palm_force.percent)
      exceed_limits_percentage_per_obj.append(category_dict[cat]["counter"].exceed_limits.percent)
      exceed_axial_percentage_per_obj.append(category_dict[cat]["counter"].exceed_axial.percent)
      exceed_lateral_percentage_per_obj.append(category_dict[cat]["counter"].exceed_lateral.percent)
      exceed_palm_percentage_per_obj.append(category_dict[cat]["counter"].exceed_palm.percent)

    for c in range(len(self.track.object_categories)):

      cat_row = avg_row_str.format(
        # name x1
        self.track.object_categories[c], 
        # number x1
        int(category_dict[self.track.object_categories[c]]["num"]),
        # float style x4
        reward_per_obj[c], 
        step_num_per_obj[c],
        palm_force_per_obj[c],
        finger_force_per_obj[c],
        # end state style x5
        lifted_per_obj[c], 
        stable_per_obj[c], 
        oob_per_obj[c], 
        target_height_per_obj[c], 
        stable_height_per_obj[c],
        # perentage style x7
        lifted_percentage_per_obj[c],
        contact_percentage_per_obj[c],
        palm_force_percentage_per_obj[c],
        exceed_limits_percentage_per_obj[c],
        exceed_axial_percentage_per_obj[c],
        exceed_lateral_percentage_per_obj[c],
        exceed_palm_percentage_per_obj[c]
      )

      category_table += cat_row

    try:
      self.track.avg_p_lifted = np.append(self.track.avg_p_lifted, total_counter.lifted.percent)
    except AttributeError as e:
      numpy_float = np.float32
      self.track.avg_p_lifted = np.array([], dtype=numpy_float)
      self.track.avg_p_contact = np.array([], dtype=numpy_float)
      self.track.avg_p_palm_force = np.array([], dtype=numpy_float)
      self.track.avg_p_exceed_limits = np.array([], dtype=numpy_float)
      self.track.avg_p_exceed_axial = np.array([], dtype=numpy_float)
      self.track.avg_p_exceed_lateral = np.array([], dtype=numpy_float)
      self.track.avg_p_exceed_palm = np.array([], dtype=numpy_float)
      self.track.avg_lifted = np.array([], dtype=numpy_float)
      self.track.avg_stable = np.array([], dtype=numpy_float)
      self.track.avg_oob = np.array([], dtype=numpy_float)
      self.track.avg_target_height = np.array([], dtype=numpy_float)
      self.track.avg_stable_height = np.array([], dtype=numpy_float)

    # save test results if we are mid-training
    if i_episode != None:

      # overall results
      self.track.test_episodes = np.append(self.track.test_episodes, i_episode)
      self.track.test_durations = np.append(self.track.test_durations, total_counter.step_num.abs / N)
      self.track.test_rewards = np.append(self.track.test_rewards, mean_reward)
      self.track.avg_p_lifted = np.append(self.track.avg_p_lifted, total_counter.lifted.percent)
      self.track.avg_p_contact = np.append(self.track.avg_p_contact, total_counter.object_contact.percent)
      self.track.avg_p_palm_force = np.append(self.track.avg_p_palm_force, total_counter.palm_force.percent)
      self.track.avg_p_exceed_limits = np.append(self.track.avg_p_exceed_limits, total_counter.exceed_limits.percent)
      self.track.avg_p_exceed_axial = np.append(self.track.avg_p_exceed_axial, total_counter.exceed_axial.percent)
      self.track.avg_p_exceed_lateral = np.append(self.track.avg_p_exceed_lateral, total_counter.exceed_lateral.percent)
      self.track.avg_p_exceed_palm = np.append(self.track.avg_p_exceed_palm, total_counter.exceed_palm.percent)
      self.track.avg_lifted = np.append(self.track.avg_lifted, total_counter.lifted.last_value / N)
      self.track.avg_stable = np.append(self.track.avg_stable, total_counter.object_stable.last_value / N)
      self.track.avg_oob = np.append(self.track.avg_oob, total_counter.oob.last_value / N)
      self.track.avg_target_height = np.append(self.track.avg_target_height, total_counter.target_height.last_value / N)
      self.track.avg_stable_height = np.append(self.track.avg_stable_height, total_counter.stable_height.last_value / N)

      # save only select category data
      self.track.category_num.append(num_per_obj)
      self.track.category_lifted.append(lifted_per_obj)
      self.track.category_stable.append(stable_per_obj)
      self.track.category_target_height.append(target_height_per_obj)
      self.track.category_stable_height.append(stable_height_per_obj)

    # finally, assembly the output string
    output_str += start_str + "\n"
    output_str += "\n" + object_table
    output_str += "\n" + category_table
    output_str += "\n" + overall_avg_table

    # print out information based on flags at top of function
    print_out = bool(print_objects + print_categories + print_overall)
    if print_out: print(start_str + "\n")
    if print_objects: print(object_table)
    if print_categories: print(category_table)
    if print_overall: print(overall_avg_table)

    # save a flag for final success rate
    self.last_test_success_rate = total_counter.object_stable.last_value / N

    return output_str

  def finish_training(self):
    """
    Save the best performance from the training
    """

    # get some key details to save in a text file now training is finished
    best_sr, best_ep = self.calc_best_performance()
    finish_txt = f"Training finished after {self.track.episodes_done} episodes"
    finish_txt += f"\n\nBest performance was {best_sr} at episode {best_ep}"
    if self.enable_saving:
      self.modelsaver.save("training_finished", txtonly=True, txtstr=finish_txt)

  def calc_best_performance(self, from_episode=None, return_id=None):
    """
    Find the best success rate by the model, and what episode number it occured
    """

    if from_episode is None: from_episode = 0

    success_rate_vector = self.track.avg_stable_height

    best_sr = 0
    best_ep = 0
    best_id = 0

    # loop through, this is slower than numpy but lets us check for 'from_episode'
    for i, sr in enumerate(success_rate_vector):

      # get info
      this_ep = self.track.test_episodes[i]

      # check if this episode is past our minimum
      if this_ep < from_episode: continue

      # see if this is best
      if sr > best_sr:
        best_sr = sr
        best_ep = this_ep
        best_id = i

    if return_id: return best_sr, best_ep, best_id

    return best_sr, best_ep

  def read_best_performance_from_text(self, silence=False, fulltest=False, heuristic=False):
    """
    Read a text file to get the best model performance. This function contains
    hardcoding
    """

    readroot = self.savedir + self.group_name + "/"

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
          datestr = "%d-%m-%y-%H:%M"
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
            datestr = "%d-%m-%Y-%H:%M"
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

      else: readname = self.best_performance_txt_file_name + '.txt'

      if self.log_level > 0: print(f"Reading text file: {readpath + readname}")
      with open(readpath + readname, 'r') as openfile:
        file_txt = openfile.read()
    except FileNotFoundError as e:
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

if __name__ == "__main__":

  # master seed, torch seed must be set before network creation (random initialisation)
  rngseed = None
  strict_seed = False
  if strict_seed:
    if rngseed is None: rngseed = random.randint(0, 2_147_483_647)
    torch.manual_seed(rngseed)
  
  # create the environment
  env = MjEnv(object_set="set8_fullset_1500", log_level=1)
  env.params.test_objects = 1
  env.params.test_trials_per_object = 1
  env.params.max_episode_steps = 20

  # training device
  device = "cpu"

  # make the agent
  layers = [env.n_obs, 64, 64, env.n_actions]
  network = networks.VariableNetwork(layers, device=device)
  agent = Agent_DQN(device=device)
  agent.init(network)

  # train the agent on the environment
  trainer = MujocoTrainer(agent, env, rngseed=rngseed, device=device, plot=False, save=True,
                    strict_seed=strict_seed, episode_log_rate=1)
  
  # set training parameters then train
  trainer.params.num_episodes = 5
  trainer.params.save_freq = 5
  trainer.params.test_freq = 5
  # trainer.load("run_14-26", path_to_run_folder="/home/luke/mujoco-devel/models/21-09-23", id=2)
  trainer.train()