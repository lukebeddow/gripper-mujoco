#!/usr/bin/env python3

import math
import random
import os
from struct import calcsize
import time
from datetime import datetime
import numpy as np
from collections import namedtuple, deque
from itertools import count
from dataclasses import dataclass, asdict
from copy import deepcopy
import cProfile

import wandb
import torch
import torch.nn as nn
import torch.optim as optim

# confine pytorch to one thread only
torch.set_num_threads(1)

import networks
from env.MjEnv import MjEnv
from ModelSaver import ModelSaver

# from guppy import hpy; guph = hpy()
# from pympler import asizeof

try:
  import matplotlib.pyplot as plt
  plt.ion() # needed for plotting multiple graphs
except ModuleNotFoundError as e:
  print("matplotlib module not found, any attempt to plot will fail")
  print(e)

class TrainDQN():
  """
  This class handles testing and training of a deep q-network
  """

  @dataclass
  class Parameters:

    # key learning hyperparameters
    object_set: str = "set4_fullset_795"
    batch_size: int = 128  
    learning_rate: float = 5e-5
    gamma: float = 0.999 
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay: int = 4000
    target_update: int = 100
    num_episodes: int = 60000
    optimiser: str = "adam" # or "rmsprop"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    # memory replay
    memory_replay: int = 50_000
    min_memory_replay: int = 5000

    # HER settings
    use_HER: bool = False
    HER_mode: str = "final" # 'final' or 'future' or 'episode'
    HER_k: int = 4

    # curriculum learning
    use_curriculum: bool = False
    curriculum_ep_num: int = 8000
    curriculum_object_set: str = "set4_fullset_795"

    # data logging settings
    save_freq: int = 2000
    test_freq: int = 2000
    plot_freq_s: int = 900
    wandb_freq_s: int = 900

  Transition = namedtuple('Transition',
                          ('state', 'action', 'next_state', 'reward'))

  HER_Transition = namedtuple('Transition',
                    ('obs', 'action', 'next_obs', 'reward', 'goal', 'state'))                        

  Save_Tuple = namedtuple('Save_Tuple',
    ('policy_net', 'params', 'memory', 'env', 'track', 'modelsaver', 'extra'))

  class ReplayMemory(object):

    def __init__(self, capacity, device, HER=None, HERMethod="final", k=4):
      self.memory = deque([], maxlen=capacity)
      self.device = device
      self.HER = True if HER is True else False
      if self.HER:
        self.temp_memory = []
        self.HER_method = HERMethod
        if self.HER_method not in ["final", "episode", "random", "future"]:
          print(f"The chosen HER method of '{HERMethod}' is not supported. Setting "
            "to default of 'final'")
          self.HER_method = "final"
        self.k = k

    def push(self, *args):
      """Save a transition"""
      if self.HER:
        self.temp_memory.append(TrainDQN.HER_Transition(*args))
      else:
        self.memory.append(TrainDQN.Transition(*args))

    def to_torch(self, data, dtype=None):
      """
      Convert some data to a torch tensor
      """
      if dtype == None: dtype = torch.float32

      return torch.tensor(np.array([data]), device=self.device, dtype=dtype)

    def sample(self, batch_size):
      if self.HER:
        # first get a random sample
        batch = random.sample(self.memory, batch_size)
        # now we want to transform the sample
        HER_sample = []
        # turn every transition into a regular sample
        for item in batch:
          HER_sample.append(self.HER_to_standard(item))
        return HER_sample
      else: return random.sample(self.memory, batch_size)

    def __len__(self):
      return len(self.memory)

    def to_device(self, tuple_of_tensors):
      """Moves a transition to a device"""
      lst = []
      for item in tuple_of_tensors:
        lst.append(item.to(self.device))
      if self.HER: return TrainDQN.HER_Transition(*tuple(lst))
      else: return TrainDQN.Transition(*tuple(lst))

    def all_to(self, device):
      """Move entire replay memory to a device"""
      self.device = device
      for i, item in enumerate(self.memory):
        self.memory[i] = self.to_device(item)

    def end_HER_episode(self, goal_reward_fcn):
      """Move samples from temp_memory to proper memory and add HER goals"""

      # transfer transitions from temporary buffer to proper memory
      for i, item in enumerate(self.temp_memory):

        # first save the transition with the desired goal (as is)
        self.memory.append(item)

        # ensure this tuple is now on the cpu
        item = self.to_device(item)

        # prepare to get the goals
        goals = []

        # 1 final: use the goal state at the end of the episode
        # 2 future: k states after this one in the episode
        # 3 episode: k states from the episode full stop
        # 4 random: k states from any episode

        if self.HER_method == "final":
          goals.append(self.temp_memory[-1].goal)

        elif self.HER_method == "future":
          if len(self.temp_memory) - i < self.k:
            k = len(self.temp_memory) - i
          else: k = self.k
          for _ in range(k):
            rand = random.randint(i, len(self.temp_memory) - 1)
            goals.append(self.temp_memory[rand].goal)

        elif self.HER_method == "episode":
          for _ in range(self.k):
            rand = random.randint(0, len(self.temp_memory))
            goals.append(self.temp_memory[rand].goal)

        elif self.HER_method == "random":
          raise RuntimeError("random HER replay is not supported currently")

        for new_goal in goals:

          # convert to python lists (moves to cpu as well)
          listgoal = new_goal[0].tolist()
          full_state = item.state[0].tolist()

          # change to the new goals and recalculate the reward
          new_reward = goal_reward_fcn(listgoal, full_state)
          new_reward = self.to_torch(new_reward)

          new_sample = self.to_device(
            TrainDQN.HER_Transition(
              item.obs, item.action, item.next_obs,
              new_reward, new_goal, item.state
            )
          )

          # save the new transition
          self.memory.append(new_sample)
      
      # wipe the temporary buffer
      self.temp_memory = []

    def HER_to_standard(self, HER_trans):
      """Change a HER transition to a normal one"""
      dim = 1
      state = torch.cat((HER_trans.obs, HER_trans.goal), dim)
      action = HER_trans.action
      next_state = torch.cat((HER_trans.next_obs, HER_trans.goal), dim)
      reward = HER_trans.reward
      return TrainDQN.Transition(state, action, next_state, reward)

  class Tracker:

    def __init__(self):
      """
      Class which tracks key data during training and logs to wandb
      """
      # parameters to set
      numpy_float = np.float32
      self.moving_avg_num = 100
      self.static_avg_num = self.moving_avg_num
      self.plot_raw = False
      self.plot_moving_avg = False
      self.plot_static_avg = True
      self.plot_test_raw = True
      self.plot_test_metrics = True
      self.plot_success_rate = True
      self.plot_bar_chart = True
      self.success_rate_metric = "stable height"
      self.plot_time_taken = True
      # general
      self.actions_done = 0
      self.episodes_done = 0
      self.last_log = 0
      self.last_plot = 0
      # utility data
      self.raw_time_taken = np.array([], dtype=numpy_float)
      self.avg_time_taken = np.array([], dtype=numpy_float)
      # training data
      self.train_episodes = np.array([], dtype=np.int32)
      self.train_rewards = np.array([], dtype=numpy_float)
      self.train_durations = np.array([], dtype=np.int32)
      self.avgR_episodes = np.array([], dtype=np.int32)
      self.avgR_rewards = np.array([], dtype=numpy_float)
      self.avgR_durations = np.array([], dtype=numpy_float)
      self.avgS_episodes = np.array([], dtype=np.int32)
      self.avgS_rewards = np.array([], dtype=numpy_float)
      self.avgS_durations = np.array([], dtype=numpy_float)
      # object category data
      self.object_categories = []
      self.category_num = []
      self.category_stable = []
      self.category_lifted = []
      self.category_target_height = []
      self.category_stable_height = []
      # test data
      self.test_episodes = np.array([], dtype=np.int32)
      self.test_rewards = np.array([], dtype=numpy_float)
      self.test_durations = np.array([], dtype=np.int32)
      self.avg_p_lifted = np.array([], dtype=numpy_float)
      self.avg_p_contact = np.array([], dtype=numpy_float)
      self.avg_p_palm_force = np.array([], dtype=numpy_float)
      self.avg_p_exceed_limits = np.array([], dtype=numpy_float)
      self.avg_p_exceed_axial = np.array([], dtype=numpy_float)
      self.avg_p_exceed_lateral = np.array([], dtype=numpy_float)
      self.avg_p_exceed_palm = np.array([], dtype=numpy_float)
      self.avg_lifted = np.array([], dtype=numpy_float)
      self.avg_stable = np.array([], dtype=numpy_float)
      self.avg_oob = np.array([], dtype=numpy_float)
      self.avg_target_height = np.array([], dtype=numpy_float)
      self.avg_stable_height = np.array([], dtype=numpy_float)
      # misc
      self.fig = None
      self.axs = None
      
    def log_episode(self, reward, duration):
      """
      Log data from the last episode
      """

      self.train_episodes = np.append(self.train_episodes, self.episodes_done)
      self.train_durations = np.append(self.train_durations, duration)
      self.train_rewards = np.append(self.train_rewards, reward)
      self.episodes_done += 1

      # update average information
      self.calc_moving_average()
      self.calc_static_average()

    def calc_moving_average(self):
      """
      Save the rewards and durations moving averages
      """
      if len(self.train_episodes) > self.moving_avg_num:
        self.avgR_durations = np.convolve(self.train_durations, np.ones(self.moving_avg_num), 'valid') / self.moving_avg_num
        self.avgR_rewards = np.convolve(self.train_rewards, np.ones(self.moving_avg_num), 'valid') / self.moving_avg_num
        x = int(self.moving_avg_num / 2)
        self.avgR_episodes = self.train_episodes[x - 1:-x]

    def calc_static_average(self):
      """Average rewards and durations to reduce data points"""

      # find number of points we can average
      num_avg_points = len(self.avgS_rewards) * self.static_avg_num

      # if we points which have not been averaged yet
      if num_avg_points + self.static_avg_num < len(self.train_episodes):

        # prepare to average rewards, durations, time taken
        unaveraged_r = self.train_rewards[num_avg_points:]
        unaveraged_d = self.train_durations[num_avg_points:]
        unaveraged_t = self.raw_time_taken[num_avg_points:]

        num_points_to_avg = len(unaveraged_r) // self.static_avg_num

        for i in range(num_points_to_avg):
          # find average values
          avg_e = self.train_episodes[
            num_avg_points + (i * self.static_avg_num) + (self.static_avg_num // 2)]
          avg_r = np.mean(unaveraged_r[i * self.static_avg_num : (i + 1) * self.static_avg_num])
          avg_d = np.mean(unaveraged_d[i * self.static_avg_num : (i + 1) * self.static_avg_num])
          avg_t = np.mean(unaveraged_t[i * self.static_avg_num : (i + 1) * self.static_avg_num])
          # append to average lists
          self.avgS_episodes = np.append(self.avgS_episodes, avg_e)
          self.avgS_rewards = np.append(self.avgS_rewards, avg_r)
          self.avgS_durations = np.append(self.avgS_durations, avg_d)
          self.avg_time_taken = np.append(self.avg_time_taken, avg_t)

    def calc_best_performance(self, from_episode=None):
      """
      Find the best success rate by the model, and what episode number it occured
      """

      if from_episode is None: from_episode = 0

      if self.success_rate_metric == "stable height":
        success_rate_vector = self.avg_stable_height
      elif self.success_rate_metric == "target height":
        success_rate_vector = self.avg_target_height
      elif self.success_rate_metric == "lifted":
        success_rate_vector = self.avg_lifted
      elif self.success_rate_metric == "stable":
        success_rate_vector = self.avg_stable
      else:
        print(f"{self.success_rate_metric} is not valid, Track used 'stable height' instead")
        success_rate_vector = self.avg_stable_height

      best_sr = 0
      best_ep = 0

      # loop through, this is slower than numpy but lets us check for 'from_episode'
      for i, sr in enumerate(success_rate_vector):

        # get info
        this_ep = self.test_episodes[i]

        # check if this episode is past our minimum
        if this_ep < from_episode: continue

        # see if this is best
        if sr > best_sr:
          best_sr = sr
          best_ep = this_ep

      return best_sr, best_ep

    def log_performance_scatter(self, best_ep, best_sr):
      """
      Log scatter graph of best model performance
      """

      X = "Training episode"
      Y = "Best success rate"

      xdata = [best_ep]
      ydata = [best_sr]
      data = [[x, y] for (x, y) in zip(xdata, ydata)]

      table = wandb.Table(data=data, columns = [X, Y])

      wandb.log({"Best success rate scatter" : wandb.plot.scatter(table, X, Y,
                                                 title="Best success rate and occurance episode")})

    def plot_wandb(self, xdata, ydata, xlabel, ylabel, title):
      # plot data to weights and biases
      data = [[x, y] for (x, y) in zip(xdata, ydata)]
      table = wandb.Table(data=data, columns=[xlabel, ylabel])
      wandb.log({title + " plot" : wandb.plot.line(table, xlabel, ylabel, title=title)})

    def plot_matplotlib(self, xdata, ydata, ylabel, title, axs, label=None):
      """Plot a matplotlib 2x1 subplot"""
      axs.plot(xdata, ydata, label=label)
      axs.set_title(title, fontstyle="italic")
      axs.set(ylabel=ylabel)

    def plot(self, plttitle=None, plt_frequency=0):
      """
      Plot training results figures, pass a frequency to plot only if enough
      time has elapsed
      """

      # if not enough time has elapsed since the last plot
      if (self.last_plot + plt_frequency > time.time()):
        return

      self.plot_bar_chart = True

      if self.fig is None:
        # multiple figures
        self.fig = []
        self.axs = []
        if self.plot_raw: 
          fig1, axs1 = plt.subplots(2, 1)
          self.fig.append(fig1)
          self.axs.append(axs1)
        if self.plot_moving_avg:
          fig2, axs2 = plt.subplots(2, 1)
          self.fig.append(fig2)
          self.axs.append(axs2)
        if self.plot_static_avg:
          fig3, axs3 = plt.subplots(2, 1)
          self.fig.append(fig3)
          self.axs.append(axs3)
        if self.plot_test_raw:
          fig4, axs4 = plt.subplots(2, 1)
          self.fig.append(fig4)
          self.axs.append(axs4)
        if self.plot_test_metrics:
          fig5, axs5 = plt.subplots(2, 1)
          self.fig.append(fig5)
          self.axs.append(axs5)
        if self.plot_success_rate:
          fig6, axs6 = plt.subplots(1, 1)
          self.fig.append(fig6)
          self.axs.append([axs6, axs6]) # add paired to hold the pattern
        if self.plot_time_taken:
          fig7, axs7 = plt.subplots(1, 1)
          self.fig.append(fig7)
          self.axs.append([axs7, axs7]) # add paired to hold the pattern
        if self.plot_bar_chart:
          fig8, axs8 = plt.subplots(1, 1)
          self.fig.append(fig8)
          self.axs.append([axs8, axs8]) # add paired to hold the pattern

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

      if self.plot_raw:
        self.plot_matplotlib(self.train_episodes, self.train_durations, D,
                             "Raw durations", self.axs[ind][0])
        self.plot_matplotlib(self.train_episodes, self.train_rewards, R,
                             "Raw rewards", self.axs[ind][1])
        self.fig[ind].subplots_adjust(hspace=0.4)
        ind += 1

      if self.plot_moving_avg:
        self.plot_matplotlib(self.train_episodes, self.avgR_durations, D,
                             f"Durations moving average ({self.moving_avg_num} samples)", 
                             self.axs[ind][0])
        self.plot_matplotlib(self.train_episodes, self.avgR_rewards, R,
                             f"Rewards moving average ({self.moving_avg_num} samples)", 
                             self.axs[ind][1])
        self.fig[ind].subplots_adjust(hspace=0.4)
        ind += 1

      if self.plot_static_avg:
        self.plot_matplotlib(self.avgS_episodes, self.avgS_durations, D,
                             f"Durations static average ({self.static_avg_num} samples)", 
                             self.axs[ind][0])
        self.plot_matplotlib(self.avgS_episodes, self.avgS_rewards, R,
                             f"Rewards static average ({self.static_avg_num} samples)", 
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

      if self.plot_test_metrics:
        # good metrics
        self.plot_matplotlib(self.test_episodes, self.avg_p_lifted, "% steps",
                             "Test good metrics", self.axs[ind][0], label="Lifted")
        self.plot_matplotlib(self.test_episodes, self.avg_p_contact, "% steps",
                             "Test good metrics", self.axs[ind][0], label="Any contact")
        self.plot_matplotlib(self.test_episodes, self.avg_p_palm_force, "% steps",
                             "Test good metrics", self.axs[ind][0], label="Palm contact")
        self.axs[ind][0].legend()
        # bad metrics
        self.plot_matplotlib(self.test_episodes, self.avg_p_exceed_axial, "% steps",
                             "Test bad metrics", self.axs[ind][1], label="Exceed axial force")
        self.plot_matplotlib(self.test_episodes, self.avg_p_exceed_lateral, "% steps",
                             "Test bad metrics", self.axs[ind][1], label="Exceed bending")
        self.plot_matplotlib(self.test_episodes, self.avg_p_exceed_palm, "% steps",
                             "Test bad metrics", self.axs[ind][1], label="Exceed palm force")
        self.axs[ind][1].legend()
        self.fig[ind].subplots_adjust(hspace=0.4)
        ind += 1

      if self.plot_success_rate:
        # what metric are we using to determine success rate
        if self.success_rate_metric == "stable height":
          success_rate_vector = self.avg_stable_height
        elif self.success_rate_metric == "target height":
          success_rate_vector = self.avg_target_height
        elif self.success_rate_metric == "lifted":
          success_rate_vector = self.avg_lifted
        elif self.success_rate_metric == "stable":
          success_rate_vector = self.avg_stable
        else:
          print(f"{self.success_rate_metric} is not valid, Track used 'stable height' instead")
          success_rate_vector = self.avg_stable_height
        # plot
        self.plot_matplotlib(self.test_episodes, success_rate_vector, "Success rate",
                      f"Success rate (metric: {self.success_rate_metric})", self.axs[ind][0])
        ind += 1   

      # create plots for static average of time taken per step
      if self.plot_time_taken:
        self.plot_matplotlib(self.avgS_episodes, self.avg_time_taken, "Time per action / s",
          f"Time per action static average ({self.static_avg_num} samples)", self.axs[ind][0])
        ind += 1 

      if self.plot_bar_chart:

        labels = [*self.object_categories, "all"]
        x = np.arange(len(labels))
        width = 0.35

        rects1 = self.axs[ind][0].bar(x - width/2, [*self.category_target_height[-1], self.avg_target_height[-1]], width, label="target height")
        rects2 = self.axs[ind][0].bar(x + width/2, [*self.category_stable_height[-1], self.avg_stable_height[-1]], width, label="stable height")

        self.axs[ind][0].set_ylabel("Success rate")
        # self.axs[ind][0].set_xlabel("Object category")
        self.axs[ind][0].set_xticks(x)
        self.axs[ind][0].set_xticklabels(labels)
        self.axs[ind][0].set_title("Grasping by category")
        self.axs[ind][0].legend()

        # this gives an error, its intended to put the bar values above them eg 0.43
        # self.axs[ind][0].bar_label(rects1, padding=3)
        # self.axs[ind][0].bar_label(rects2, padding=3)
        
        ind += 1

      plt.pause(0.001)

      # save that we plotted
      self.last_plot = time.time()

      return

    def log_wandb(self, log_frequency=0):
      """
      Log data to weights and biases. Pass a frequency to only log if enough
      time has elapsed since the last log
      """

      # if not enough time has elapsed since the last data upload
      if (self.last_log + log_frequency > time.time()):
        return

      E = "Episode"
      R = "Reward"
      D = "Duration"

      # create plots of raw reward and duration data
      if self.plot_raw:
        self.plot_wandb(self.train_episodes, self.train_rewards, E, R, "Raw rewards")
        self.plot_wandb(self.train_episodes, self.train_durations, E, D, "Raw durations")

      # create plots for a moving average of rewards and durations
      if self.plot_moving_avg:
        self.plot_wandb(self.avgR_episodes, self.avgR_rewards, E, R, 
                        f"Rewards moving average ({self.moving_avg_num} samples)")
        self.plot_wandb(self.avgR_episodes, self.avgR_durations, E, D, 
                        f"Durations moving average ({self.moving_avg_num} samples)")

      # create plots for a static average of rewards and durations
      if self.plot_static_avg:
        self.plot_wandb(self.avgS_episodes, self.avgS_rewards, E, R,
                        f"Rewards static average ({self.static_avg_num} samples)")
        self.plot_wandb(self.avgS_episodes, self.avgS_durations, E, D,
                        f"Durations static average ({self.static_avg_num} samples)")

      # plot the test time reward
      if self.plot_test_raw:
        self.plot_wandb(self.test_episodes, self.test_rewards, E, R, "Test rewards")
        self.plot_wandb(self.test_episodes, self.test_durations, E, D, "Test durations")
        
      if self.plot_test_metrics:
        # define performance metrics to examine
        good_metrics = [
          [self.avg_p_lifted, "% Lifted"],
          [self.avg_p_contact, "% Contact"],
          [self.avg_p_palm_force, "% Palm contact"]
        ]
        bad_metrics = [
          [self.avg_p_exceed_axial, "% Exceed axial force"],
          [self.avg_p_exceed_lateral, "% Exceed bending"],
          [self.avg_p_exceed_palm, "% Exceed palm force"]
        ]

        # create test results plots
        wandb.log({"Test good performance metrics" : wandb.plot.line_series(
          xs=[self.test_episodes for i in range(len(good_metrics))],
          ys=[x[0] for x in good_metrics],
          keys=[x[1] for x in good_metrics],
          title="Test good performance metrics", xname="Training episodes"
        )})
        wandb.log({"Test bad performance metrics" : wandb.plot.line_series(
          xs=[self.test_episodes for i in range(len(bad_metrics))],
          ys=[x[0] for x in bad_metrics],
          keys=[x[1] for x in bad_metrics],
          title="Test bad performance metrics", xname="Training episodes"
        )})

      # create plots of success rate
      if self.plot_success_rate:
        # what metric are we using to determine success rate
        if self.success_rate_metric == "stable height":
          success_rate_vector = self.avg_stable_height
        elif self.success_rate_metric == "target height":
          success_rate_vector = self.avg_target_height
        elif self.success_rate_metric == "lifted":
          success_rate_vector = self.avg_lifted
        elif self.success_rate_metric == "stable":
          success_rate_vector = self.avg_stable
        else:
          print(f"{self.success_rate_metric} is not valid, Track used 'stable height' instead")
          success_rate_vector = self.avg_stable_height
        # plot
        self.plot_wandb(self.test_episodes, success_rate_vector, E, "Success rate", 
                        f"Success rate (metric: {self.success_rate_metric})")

      # create plots for static average of time taken per step
      if self.plot_time_taken:
        self.plot_wandb(self.avgS_episodes, self.avg_time_taken, E, "Time per action / s",
                        f"Time per action static average ({self.static_avg_num} samples)")

      # OLD CODE COMPATIBLE: delete later
      try:
        test = self.plot_bar_chart
      except Exception as e:
        print("Tracker() does not have variable self.plot_bar_chart")
        print(e)
        self.plot_bar_chart = False

      # create bar chart to visualise performance on different categories
      if self.plot_bar_chart and len(self.avg_stable_height) > 0:

        data = [[label, var] for (label, var) in zip(
          [*self.object_categories, "all"],
          [*self.category_stable_height[-1], self.avg_stable_height[-1]]
        )]
        table = wandb.Table(data=data, columns=["categories", "success rate"])
        wandb.log({"stable_height_bar_chart" : wandb.plot.bar(table, "categories", "success rate",
                                                   title="Stable height success rate by category")})

        data = [[label, var] for (label, var) in zip(
          [*self.object_categories, "all"],
          [*self.category_target_height[-1], self.avg_target_height[-1]]
        )]
        table = wandb.Table(data=data, columns=["categories", "success rate"])
        wandb.log({"target_height_bar_chart" : wandb.plot.bar(table, "categories", "success rate",
                                                   title="Target height success rate by category")})

      # finish by recording the last log time
      self.last_log = time.time()

      return

  def __init__(self, run_name=None, group_name=None, device=None, use_wandb=None, 
               no_plot=None, log_level=None, object_set=None, use_curriculum=None,
               num_segments=None, finger_thickness=None, finger_width=None):

    # define key training parameters
    self.params = TrainDQN.Parameters()
    self.track = TrainDQN.Tracker()

    # define some class variables
    self.log_rate_for_episodes = 25
    self.last_test_data = None

    # prepare environment, but don't load a model xml file yet
    self.env = MjEnv(noload=True, num_segments=num_segments)
    self.num_segments = num_segments
    self.finger_thickness = finger_thickness
    self.finger_width = finger_width

    # what machine are we on
    self.machine = self.env._get_machine()

    # set run name and group name if not specified
    if run_name is None:
      run_name = self.machine + "_" + datetime.now().strftime("%H:%M")
    if group_name is None:
      group_name = datetime.now().strftime("%d-%m-%y")

    # configure input options
    self.run_name = run_name
    self.group_name = group_name
    self.savedir = "models/dqn/"
    if device is None: device = "cuda"
    self.set_device(device)
    self.log_level = 1 if log_level is None else log_level
    self.params.use_curriculum = True if use_curriculum is True else False
    if object_set is not None: self.params.object_set = object_set

    # wandb options
    self.wandb_init_flag = False
    self.use_wandb = use_wandb if use_wandb is not None else True
    self.wandb_note = ""
    self.wandb_project = "luke-gripper-mujoco"
    self.wandb_entity = "lbeddow"

    # HER option defaults
    self.HER_mode = None
    self.HER_k = None

    # curriculum defaults
    self.curriculum_applied = None

    # if we are plotting graphs during this training
    if no_plot == True:
      self.no_plot = True
    else:
      global plt
      import matplotlib.pyplot as plt
      self.fig, self.axs = plt.subplots(2, 1)
      self.no_plot = False

  def init(self, network):
    """
    Create the networks
    """

    # are we using HER (python OVERRIDES cpp)
    self.env.mj.set.use_HER = self.params.use_HER

    # apply any thickness changes and then load the object set from params
    # if self.finger_thickness is not None: 
    #   self.env.params.finger_thickness = self.finger_thickness
    self.load_object_set(num_segments=self.num_segments, finger_thickness=self.finger_thickness,
                         finger_width=self.finger_width)

    # now update the environment with correct numbers of actions and observations
    self.env._update_n_actions_obs()

    # create networks
    if network == None:
      raise RuntimeError("TrainDQN network must be specified")
    elif network == "loaded":
      # no need to load networks
      pass
    elif isinstance(network, list):

      # we have been given a list of hidden layer sizes
      layers = [self.env.n_obs, *network, self.env.n_actions]
      self.policy_net = networks.DQN_variable(layers, self.device).to(self.device)
      self.target_net = networks.DQN_variable(layers, self.device).to(self.device)
      self.target_net.load_state_dict(self.policy_net.state_dict())
      self.memory = TrainDQN.ReplayMemory(self.params.memory_replay, self.device,
                      HER=self.params.use_HER, HERMethod=self.params.HER_mode,
                      k=self.params.HER_k)
      # prepare for saving and loading
      self.modelsaver = ModelSaver(self.savedir + self.group_name)

    else:
      # use the network passed as input
      self.policy_net = network(self.env.n_obs, self.env.n_actions,
                                self.device).to(self.device)
      self.target_net = network(self.env.n_obs, self.env.n_actions,
                                self.device).to(self.device)


      self.target_net.load_state_dict(self.policy_net.state_dict())

      self.memory = TrainDQN.ReplayMemory(self.params.memory_replay, self.device,
                      HER=self.params.use_HER, HERMethod=self.params.HER_mode,
                      k=self.params.HER_k)

      # prepare for saving and loading
      self.modelsaver = ModelSaver(self.savedir + self.group_name)

    # configure optimiser
    if self.params.optimiser.lower() == "rmsprop":
      self.optimiser = optim.RMSprop(self.policy_net.parameters(), 
                                     lr=self.params.learning_rate)
    elif self.params.optimiser.lower() == "adam":
      self.optimiser = optim.Adam(self.policy_net.parameters(),
                                  lr=self.params.learning_rate,
                                  betas=(self.params.adam_beta1,
                                         self.params.adam_beta2))
    else:
      print("No valid optimiser name defined! Using RMSProp")
      self.optimiser = optim.RMSprop(self.policy_net.parameters(), 
                                     lr=self.params.learning_rate)

    # save weights and biases
    if self.use_wandb:
      wandb.init(project=self.wandb_project, entity=self.wandb_entity, 
                 name=self.run_name, config=self.get_params_dictionary(),
                 notes=self.wandb_note, group=self.group_name)
      self.wandb_init_flag = True

    # print important info about class settings
    if self.log_level > 0:
      print("TrainDQN init settings:")
      print(" -> Using machine:", self.machine)
      print(" -> Using device:", self.device.type)
      print(" -> Using object set:", self.params.object_set)
      print(" -> Network inputs (n_obs):", self.env.n_obs)
      print(" -> Network outputs (n_actions):", self.env.n_actions)
      print(" -> Network name:", self.policy_net.name)
      print(" -> Using HER:", self.params.use_HER)
      print(" -> Using wandb:", self.use_wandb)
      print(" -> Using curriculum:", self.params.use_curriculum)
      print(" -> Run name:", self.run_name)
      print(" -> Group name:", self.group_name)

  def set_device(self, device):
    """
    Set the memory device to use
    """
    if device == "cuda":
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cpu":
      self.device = torch.device("cpu")
    else:
      raise RuntimeError("device should be 'cuda' or 'cpu'")

  def load_object_set(self, object_set=None, num_segments=None, finger_thickness=None,
                      finger_width=None):
    """
    Load an object set into the simulation
    """

    if object_set is None: object_set = self.params.object_set[:]

    # load the object set and the first xml file
    self.env.load(object_set_name=object_set, num_segments=num_segments,
                  finger_thickness=finger_thickness, finger_width=finger_width)

    # save the changes
    self.params.object_set = object_set
    self.num_segments = num_segments

  def to_torch(self, data, dtype=None):
    """
    Convert some data to a torch tensor
    """
    if dtype == None: dtype = torch.float32

    return torch.tensor(np.array([data]), device=self.device, dtype=dtype)

  def select_action(self, state, decay_num, test=None):

    sample = random.random()

    eps_threshold = (self.params.eps_end 
      + (self.params.eps_start - self.params.eps_end)
      * (math.exp(-1. * decay_num / float(self.params.eps_decay))))

    # if we will not choose randomly
    if sample > eps_threshold or test:
      with torch.no_grad():
        # t.max(1) returns largest column value of each row
        # [1] is second column of max result, the index of max element
        # view(1, 1) selects this element which has max expected reward
        return self.policy_net(state).max(1)[1].view(1, 1)
    # else choose randomly
    else:
      rand_action = random.randrange(self.env.n_actions)
      return torch.tensor([[rand_action]], device=self.device,
                          dtype=torch.long)

  def plot(self, force=None, hang=None, end=None):
    """
    Create a plot to track the training data
    """

    if self.no_plot:
      return

    freq = self.params.plot_freq_s if force is not True else 0

    self.track.plot(plttitle=self.group_name + " / " + self.run_name, plt_frequency=freq)

    # if we are at the end, log the final best performance
    if end is True:

      # if using a curriculum, get the best performance only after the end of the curriculum
      if self.params.use_curriculum:
        if self.curriculum_applied is not None and self.curriculum_applied > 1:
          from_episode = self.curriculum_applied
          print_details = f"(curriculum applied at {self.curriculum_applied})"
        else:
          print_details = "(curriculum not applied)"
      else: 
        from_episode = None
        print_details = ""

      # get best success rate and best episode
      best_sr, best_ep = self.track.calc_best_performance(from_episode=from_episode)
      self.track.log_performance_scatter(best_ep, best_sr)

      print(f"Run best performance is {best_sr} at episode {best_ep} " + print_details)

    if hang == True:
      plt.ioff()
      plt.show() # halts all program execution
      plt.ion()

  def log_wandb(self, force=None, end=None):
    """
    Log to weights and biases
    """

    if not self.use_wandb:
      return

    if not self.wandb_init_flag:
      if self.use_wandb:

        wandb.init(project=self.wandb_project, entity=self.wandb_entity, 
                  name=self.run_name, config=self.get_params_dictionary(),
                  notes=self.wandb_note, group=self.group_name)
        self.wandb_init_flag = True

    freq = self.params.wandb_freq_s if force is not True else 0

    try:
      self.track.log_wandb(log_frequency=freq)
    except Exception as e:
      print("LOG WANDB ERROR:", e)
      print("Taking no action and continuing with program, tracking data may have issues")

    # if we are at the end, log the final best performance
    if end is True:

      # # if using a curriculum, get the best performance only after the end of the curriculum
      # if self.params.use_curriculum:
      #   if self.curriculum_applied is not None and self.curriculum_applied > 1:
      #     from_episode = self.curriculum_applied
      #     print_details = f"curriculum applied at {self.curriculum_applied}"
      #   else:
      #     print_details = "curriculum not applied"
      # else: 
      #   from_episode = None
      #   print_details = ""

      from_episode = None
      print_details = ""

      # get best success rate and best episode
      best_sr, best_ep = self.track.calc_best_performance(from_episode=from_episode)
      self.track.log_performance_scatter(best_ep, best_sr)

      print(f"Run best performance is {best_sr} at episode {best_ep} " + print_details)

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
    normal_row_str = col_str.format("{:<36}", "{:<3}", "{:<6.2f}", "{:<4}",    "{:<3.0f}")
    avg_row_str = col_str.format   ("{:<36}", "{:<3}", "{:<6.2f}", "{:<4.2f}", "{:<3.0f}")

    # insert the names into the top of each column - notice the grouping of styles
    table_header = header_str.format(
      "{:<36}",
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

    # save test results if we are mid-training
    if True or i_episode != None:

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

    return output_str

  def optimise_model(self):
    """
    Optimise the policy
    """

    # only proceed if we have enough memory for a batch
    if len(self.memory) < self.params.batch_size: return

    # only begin to optimise when enough memory is built up
    if (len(self.memory)) < self.params.min_memory_replay: # self.params.batch_size
      return

    transitions = self.memory.sample(self.params.batch_size)

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = TrainDQN.Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device,
                                          dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
                                                
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = self.policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(self.params.batch_size, device=self.device)
    next_state_values[non_final_mask] = self.target_net(
        non_final_next_states).max(1)[0].detach()
    
    # Compute the expected Q values
    expected_state_action_values = (next_state_values 
                                        * self.params.gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    self.optimiser.zero_grad()
    loss.backward()
    for param in self.policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    self.optimiser.step()

    return

  def curriculum_fcn(self, i_episode):
    """
    Implement a learning curriculum
    """

    if not self.params.use_curriculum: return

    # if the curriculum has been applied, return (this disallows multi-stage curriculums)
    if self.curriculum_applied is not None and self.curriculum_applied > 0: return

    # define threshold for changing curriculum
    threshold = 25000

    # define curriculum style (currently either "learning rate" or "object set")
    style = "learning rate"

    # define the curriculum metric
    metric = "episode number"

    # extract the metric to use for comparison
    if metric == "episode number":
      metric_value = i_episode

    elif metric == "success rate":
      # if we have no success rate data, return
      if len(self.track.avg_stable_height) > 0:
        metric_value = self.track.avg_stable_height[-1]
      else: return

    # see if our threshold is EXCEEDED, if so apply the curriculum
    if metric_value > threshold:

      if style == "object set":
        self.load_object_set(object_set=self.params.curriculum_object_set, num_segments=self.num_segments)

      elif style == "learning rate":
        # reduce the learning rate by 75%, eg 50e-6 goes to 12.5e-6
        new_learning_rate = self.params.learning_rate * 0.25
        for param_group in self.optimiser.param_groups:
          param_group['lr'] = new_learning_rate
        self.params.learning_rate = new_learning_rate

      # we have now applied the curriculum change
      self.curriculum_applied = i_episode

      # now save a text file to reflect the changes
      labelstr = f"Hyperparameters after curriculum change which occured at episode {i_episode}\n"
      labelstr += f"The curriculum style is {style} and the metric is {metric}\n"
      labelstr += f"The metric value is {metric_value} and the threshold is {threshold}\n"
      name = "hyperparameters_from_curriculum_change"
      self.save_hyperparameters(labelstr, name)

    # if the threshold is not exceeded
    else: return

  def run_episode(self, i_episode, test=None, heuristic=None):
    """
    Perform one episode of training or testing
    """

    # # for debugging, show memory usage (heapy does not seem to be accurate)
    # if i_episode % 1000 == 1 and not test:
    #   theheap = guph.heap()
    #   print("Heap total size is", theheap.size, "(", theheap.size / 1e6, "MB)")
    #   print("The replay memory size is", asizeof.asizeof(self.memory) / 1e3, "kB",
    #     "with length", len(self.memory))
    #   print("The environment size is", asizeof.asizeof(self.env) / 1e3, "kB")

    # initialise environment and state
    obs = self.env.reset()
    obs = self.to_torch(obs)

    if self.params.use_HER:
      goal = self.env._get_desired_goal()
      goal = self.to_torch(goal)

    ep_start = time.time()

    # # for testing: is noise certainly in use?
    # def print_noise_details(sensor, sensor_name):
    #   in_use = sensor.in_use
    #   use_noise = sensor.use_noise
    #   noise_std = sensor.noise_std
    #   print(f"Sensor: {sensor_name}; in_use = {in_use}; use_noise = {use_noise}; noise_std = {noise_std}")
    # print_noise_details(self.env.mj.set.bending_gauge, "bending_gauge")
    # print_noise_details(self.env.mj.set.palm_sensor, "palm_sensor")

    # count up through actions
    for t in count():

      if self.log_level > 2: print("Episode", i_episode, "action", t)

      # select and perform an action
      if heuristic:
        # human written action selection function
        action = self.env.get_heuristic_action()
        action = torch.tensor([action]) # to fit with select_action(...)
      # else we are using RL q-networks
      elif self.params.use_HER:
        HER_obs = torch.cat((obs, goal), dim=1)
        action = self.select_action(HER_obs, decay_num=i_episode, test=test)
      else:
        action = self.select_action(obs, decay_num=i_episode, test=test)

      # step with this action and receive output data
      step_data = self.env.step(action.item())

      step_data_torch = []
      for i in range(len(step_data)):
        step_data_torch.append(self.to_torch(step_data[i]))

      # step with this action and receive sample data for this transition
      if self.params.use_HER:
        (new_obs, reward, done, state, goal) = step_data_torch
        transition_sample = (obs, action, new_obs, reward, goal, state)
      else:
        (new_obs, reward, done) = step_data_torch
        transition_sample = (obs, action, new_obs, reward)

      # render the new environment
      self.env.render()

      # perform one step of the optimisation on the policy network
      if test != True:
        self.track.actions_done += 1
        self.memory.push(*transition_sample) # the * unpacks tuple to func args
        self.optimise_model()

      obs = new_obs

      # check if this episode is over and log if we aren't testing
      if done:

        # finish timing
        ep_end = time.time()
        time_per_step = (ep_end - ep_start) / float(t + 1)

        if self.log_level > 2:
          print(f"Time for episode was {ep_end - ep_start:.3f}s"
            f", time per action was {time_per_step * 1e3:.3f} ms")

        # if we are testing, no data is logged
        if test: break

        # save training data
        self.track.raw_time_taken = np.append(self.track.raw_time_taken, time_per_step)
        self.track.log_episode(self.env.track.cumulative_reward, t + 1)

        # plot to the screen
        if self.no_plot == False:
          self.plot()

        # save to wandb
        if self.use_wandb:
          self.log_wandb()

        # if using HER, wrap up the episode
        if self.params.use_HER:
          self.memory.end_HER_episode(self.env._goal_reward)

        break

  def train(self, network=None, i_start=None):
    """
    Train the model for the desired number of episodes
    """

    # if we have been given a network to train
    if network != None: self.init(network)

    # if this is a fresh, new training
    if i_start == None or i_start == 0:
      i_start = 0
      # create a new folder to save training results in
      self.modelsaver.new_folder(name=self.run_name, notimestamp=True)
      # save record of the training time hyperparameters and important files
      self.save_hyperparameters()
      self.save_important_files()
      self.save() # save starting network parameters
    else:
      # save a record of the training restart
      continue_label = f"Training is continuing from episode {i_start} with these hyperparameters\n"
      hypername = f"hyperparameters_from_ep_{i_start}"
      self.save_hyperparameters(labelstr=continue_label, name=hypername)

    if self.log_level > 0:
      print(f"\nBEGIN TRAINING, target is {self.params.num_episodes} episodes\n", flush=True)

    # begin training episodes
    for i_episode in range(i_start + 1, self.params.num_episodes + 1):

      if self.log_level == 1 and i_episode % self.log_rate_for_episodes == 1:
        print("Begin training episode", i_episode, flush=True)
      elif self.log_level > 1:
        print("Begin training episode", i_episode, flush=True)

      self.run_episode(i_episode)

      # check if time to change curriculum
      self.curriculum_fcn(i_episode)

      # update the target network every target_update episodes
      if i_episode % self.params.target_update == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())

      # test the target network and then save it
      if i_episode % self.params.test_freq == 0 and i_episode != 0:
        test_data = self.test()
        # process test data
        test_report = self.create_test_report(test_data, i_episode=i_episode)
        additional_data = (test_data)
        # save the result
        self.save(txtstring=test_report, txtlabel="test_results", 
                  tupledata=additional_data)

      # or only save the network
      elif i_episode % self.params.save_freq == 0:
        self.save()

    # update the target network at the end
    self.target_net.load_state_dict(self.policy_net.state_dict())

    # save, log and plot now we are finished
    if self.log_level > 0:
      print("\nTRAINING COMPLETE, finished", i_episode, "episodes\n")
    self.save(txtstring=f"Training finished after {i_episode} episodes",
              txtlabel="training_finished")
    self.env.render()
    self.log_wandb(force=True, end=True)
    self.plot(force=True, end=True, hang=True) # leave plots on screen if we are plotting

    # end of training
    self.env.close()

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

      if heuristic: self.env.start_heuristic_grasping()

      if pause_each_episode: input("Press enter to continue")

      self.run_episode(i_episode, test=True, heuristic=heuristic)

    # get the test data out
    test_data = self.env.test_trial_data

    # plot to the screen
    if self.no_plot == False:
      self.plot(force=True) # guarantees data will be plotted

    # save to wandb
    if self.use_wandb:
      self.log_wandb(force=True) # guarantees data will be logged

    # save internally this test data
    self.last_test_data = test_data

    if self.log_level > 0: print("Testing complete, finished", i_episode, "episodes")

    return test_data

  def test_heuristic_baseline(self, pause_each_episode=None, label=None):
    """
    Test the grasping performance of a heurisitc baseline
    """

    print_out = True

    if label is not None: label = "heuristic_test_" + label
    else: label = "heuristic_test"

    if self.log_level > 0: print("Testing heuristic baseline")

    # run the test chooseing heuristic actions
    test_data = self.test(heuristic=True, pause_each_episode=pause_each_episode)

    # save results
    test_report = self.create_test_report(test_data)
    self.modelsaver.save(label, txtstr=test_report, txtonly=True)

    if self.log_level > 0:
      print("Finished testing heuristic baseline")
      if print_out: print(test_report)

  def get_params_dictionary(self):
    """
    Return a dictionary of parameters, this function is partly incompatible with old
    code as MjEnv() did not used to have a parameter dictionary
    """

    params_dict = {}
    params_dict.update(asdict(self.params))

    # OLD CODE COMPATIBLE: try to get mjenv parameters
    try:
      params_dict.update(self.env.get_parameters())
    except AttributeError as e:
      print("TrainDQN.get_params_dictionary() error with env params:", e)

    manual_params = {
      "Network name" : self.policy_net.name,
      "Finger stiffness setting" : self.env.mj.set.finger_stiffness,
      "Mujoco timestep" : self.env.mj.set.mujoco_timestep,
      "Sim steps per action" : self.env.mj.set.sim_steps_per_action,
      "Bend gauge normalise" : self.env.mj.set.bending_gauge.normalise,
      "n_obs" : self.env.n_obs,
      "n_actions" : self.env.n_actions
    }

    params_dict.update(manual_params)

    return params_dict

  def save_hyperparameters(self, labelstr=None, name=None, printonly=None):
    """
    Save a text file with the current hyperparameters
    """

    if self.log_level > 0:
      print_out = True
    else:
      print_out = False

    param_str = ""
    time_stamp = datetime.now().strftime("%d-%m-%Y-%H:%M")

    if labelstr != None:
      param_str += labelstr + '\n'

    if name == None:
      name = "hyperparameters"

    # capture parameters info
    param_str += f"""\nHyperparameters at training time:\n\n"""

    if self.wandb_note != "":
      param_str += "Wandb note:\n" + self.wandb_note + "\n"

    # add in some important information
    param_str += "Network name: " + self.policy_net.name + "\n"
    param_str += "Save time and date: " + time_stamp + "\n"
    param_str += "Running on machine: " + self.machine + "\n"
    param_str += "Running with device: " + self.device.type + "\n"
    param_str += "Object set in use: " + self.env.mj.object_set_name + "\n"
    param_str += "Random seed: " + str(self.env.myseed) + "\n"

    # convert parameters to a string
    param_str += "\nParameters dictionaries:\n"
    param_str += str(self.get_params_dictionary())

    # swap commas for newlines for prettier printing
    param_str = param_str.replace(',', '\n')

    # add the c++ settings to the string
    param_str += "\n\n" + self.env._get_cpp_settings()

    # if we are printing, put this info also into the terminal
    if print_out: print(param_str)

    # if we are only printing
    if printonly is True:
      print(param_str)
      return

    savepath = self.modelsaver.save(name, txtstr=param_str,
                                    txtonly=True)

    return savepath

  def save_important_files(self):
    """
    Hardcoded function to save some important files in case we need to continue training
    """

    path = os.path.dirname(os.path.abspath(__file__)) + "/"
    self.modelsaver.copy_files(path + "env/mjpy/", "bind.so")
    self.modelsaver.copy_files(path + "env/", "MjEnv.py")
    self.modelsaver.copy_files(path, "TrainDQN.py")
    self.modelsaver.copy_files(path, "array_training_DQN.py")

    return

  def save(self, txtstring=None, txtlabel=None, tupledata=None):
    """
    Save the model policy network, return save path
    """

    save_data = TrainDQN.Save_Tuple(
      policy_net = self.policy_net,
      params = self.params,
      memory = self.memory,
      env = self.env,
      track = self.track,
      modelsaver = self.modelsaver,
      extra = tupledata
    )

    savepath = self.modelsaver.save(self.policy_net.name, pyobj=save_data, 
                                    txtstr=txtstring, txtlabel=txtlabel)

    return savepath

  def load(self, id=None, folderpath=None, foldername=None):
    """
    Load the most recent model, overwrite current networks
    """

    # check if modelsaver is defined
    if not hasattr(self, "modelsaver"):
      if folderpath is not None:
        print(f"load not given a modelsaver, making one from folderpath: {folderpath}")
        self.modelsaver = ModelSaver(folderpath)
      else:
        raise RuntimeError("load not given a folderpath or a modelsaver")

    # load the model
    load_data = self.modelsaver.load(id=id, folderpath=folderpath, 
                                     foldername=foldername)

    self.policy_net = load_data.policy_net
    self.params = load_data.params
    self.memory = load_data.memory
    self.env = load_data.env
    self.track = load_data.track
      
    if load_data.extra != None:
      self.last_test_data = load_data.extra

    # reseed and reload environment
    self.env.seed()      # reseeds with same seed as before (but not contiguous!)
    self.env._load_xml() # segfault without this

    # delete this later: needed until wrist_z_offset is saved in bind.cpp
    self.env.reset(hard=True)

    # move to the current device
    self.memory.all_to(self.device)
    self.policy_net.to(self.device)
    self.policy_net.device = self.device

    # reinitialise to prepare for further training
    self.target_net = deepcopy(self.policy_net)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.to(self.device)

    # re-initialise the class
    self.init(network="loaded")

    # return the path of the loaded model
    return self.modelsaver.last_loadpath
  
  def continue_training(self, foldername, folderpath, new_endpoint=None,
                        extra_episodes=None, object_set=None):
    """
    Load a model and then continue training it
    """

    self.run_name = foldername + "_continued"
    self.modelsaver = ModelSaver(folderpath)

    # load the most recent model in the given folder
    self.load(foldername=foldername, folderpath=folderpath)
    self.modelsaver.enter_folder(foldername, folderpath=folderpath)

    # update the new training episode target
    if new_endpoint != None:
      self.params.num_episodes = new_endpoint

    # or add extra episodes on to what has already been done
    elif extra_episodes != None:
      self.params.num_episodes = self.params.num_episodes + extra_episodes

    # or if we have done exactly our target episodes, double the target
    elif self.params.num_episodes == self.track.episodes_done:
      # if no target is set but we have finished, double the initial target
      self.params.num_episodes = 2 * self.track.episodes_done
      print("New training endpoint set as:", self.params.num_episodes)

    # check if our training endpoint is already satisfied
    if self.params.num_episodes < self.track.episodes_done:
      raise RuntimeError("episodes done exceeds the params.num_episodes target")

    # load a new object set if we are told to
    if object_set is not None:
      self.env._load_object_set(name=object_set, num_segments=self.num_segments)

    # begin the training at the given starting point (always uses most recent pickle)
    self.train(i_start=self.track.episodes_done)

if __name__ == "__main__":
  
  # ----- prepare ----- #

  use_wandb = False
  force_device = "cpu"
  no_plot = True

  model = TrainDQN(device=force_device, use_wandb=use_wandb, no_plot=no_plot)

  # if we want to adjust parameters
  # model.log_level = 2
  # model.params.num_episodes = 11
  # model.env.max_episode_steps = 20
  # model.params.wandb_freq_s = 5
  # model.env.mj.set.action_motor_steps = 350
  # model.env.disable_rendering = False
  # model.params.test_freq = 10
  # model.env.params.test_trials_per_object = 1
  # model.env.test_obj_limit = 10

  # # plotting options
  # model.no_plot = False
  # model.params.plot_freq_s = 10
  # model.track.moving_avg_num = 10
  # model.track.static_avg_num = model.track.moving_avg_num
  # model.track.plot_raw = True
  # model.track.plot_moving_avg = False
  # model.track.plot_static_avg = True
  # model.track.plot_test_raw = False
  # model.track.plot_test_metrics = False
  # model.track.plot_success_rate = False
  # model.track.success_rate_metric = "stable height"
  # model.track.plot_time_taken = True

  # # if we want to configure HER
  # model.params.use_HER = True
  # model.env.mj.goal.step_num.involved = True
  # model.env.mj.goal.lifted.involved = True
  # model.env.mj.goal.object_contact.involved = True
  # model.env.mj.goal.ground_force.involved = True
  # model.env.mj.goal.palm_force.involved = True

  # ----- heuristic test ----- #

  # net = networks.DQN_2L60
  # model.init(network=net)
  # model.log_level = 2
  # model.env.mj.set.debug = False
  # model.env.disable_rendering = False
  # # model.env.params.test_trials_per_object = 1
  # model.env.params.test_objects = 20
  # model.env.params.max_episode_steps = 250
  # # model.env.log_level = 2

  # model.env.mj.set.bending_gauge.in_use = False
  # model.env.mj.set.palm_sensor.in_use = False
  # model.env.mj.set.wrist_sensor_Z.in_use = False

  # test_data = model.test(pause_each_episode=False, heuristic=True)
  # test_report = model.create_test_report(test_data)
  # exit()

  # ----- load ----- #

  # load
  # folder = "mymujoco"
  # group = "05-12-22"
  # run = "luke-PC_11:39_A1"
  folder = "mymujoco"
  group = "02-12-22"
  run = "luke-PC_16:55_A3"
  folderpath = f"/home/luke/{folder}/rl/models/dqn/{group}/"
  model.set_device("cuda")
  model.load(id=None, folderpath=folderpath, foldername=run)

  # ----- train ----- #

  # train
  # net = networks.DQN_3L60
  # model.env.disable_rendering = True
  # model.env.mj.set.debug = False
  # model.num_segments = 8
  # model.finger_thickness = 0.8e-3
  # model.train(network=net)

  # # continue training
  # folderpath = "/home/luke/mymujoco/rl/models/dqn/DQN_3L60/"# + model.policy_net.name + "/"
  # foldername = "luke-PC_A3_24-05-22-18:19"
  # model.continue_training(foldername, folderpath)

  # ----- profile ----- #
  # net = networks.DQN_3L60
  # model.env.disable_rendering = True
  # model.env.mj.set.debug = False
  # model.params.num_episodes = 10
  # cProfile.run("model.train(network=net)", "/home/luke/mymujoco/python_profile_results.xyz")
  # # in order to read profile results, run: $ python3 -m pstats /path/to/results.xyz
  # # do: $ sort cumtime OR $ sort tottime AND THEN $ stats
  # exit()

  # ----- visualise ----- #

  # # visualise training performance
  # plt.ioff()
  # model.track.plot()
  # plt.show()

  # ----- apply reward configuration ----- #

  # import array_training_DQN
  # model = array_training_DQN.apply_to_all_models(model)
  # model.env.max_episode_steps = 300
  # model.env.mj.set.scale_rewards(100 / model.env.max_episode_steps)
  # model.env.mj.set.motor_state_sensor.read_rate = -2
  # model.env.mj.set.axial_gauge.in_use = True
  # model.env.mj.set.wrist_sensor_Z.in_use = True
  # model = array_training_DQN.new_rewards(model)

  # ----- test ----- #

  # test
  model.log_level = 2
  model.env.mj.set.debug = False
  model.env.disable_rendering = False
  # model.env.params.test_trials_per_object = 1
  model.env.params.test_objects = 20
  # model.env.params.test_obj_per_file = 5
  # model.env.params.max_episode_steps = 20

  # model.env.mj.set.step_num.set          (0,      70,     1)
  # model.env.mj.set.exceed_limits.set     (-0.005, True,   10)
  # model.env.mj.set.exceed_axial.set      (-0.005, True,   10,    3.0,  6.0,  -1)
  # model.env.mj.set.exceed_lateral.set    (-0.005, True,   10,    4.0,  6.0,  -1)
  input("Press enter to begin")
  test_data = model.test(pause_each_episode=False)

  # save results
  test_report = model.create_test_report(test_data)
  # model.modelsaver.new_folder(label="DQN_testing")
  # model.save_hyperparameters(labelstr=f"Loaded model path: {model.modelsaver.last_loadpath}\n")
  # model.save(txtstring=test_report, txtlabel="test_results_demo")
  


 
