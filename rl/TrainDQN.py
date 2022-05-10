#!/usr/bin/env python3

import math
import random
import time
from datetime import datetime
import numpy as np
from collections import namedtuple, deque
from itertools import count
from dataclasses import dataclass, asdict
from copy import deepcopy

import wandb
import torch
import torch.nn as nn
import torch.optim as optim

import networks
from env.MjEnv import MjEnv
from ModelSaver import ModelSaver

from guppy import hpy; guph = hpy()

class TrainDQN():
  """
  This class handles testing and training of a deep q-network
  """

  class Tracker:

    def __init__(self):
      """
      Class which tracks key data during training and logs to wandb
      """
      # parameters to set
      self.moving_avg_num = 50
      self.plot_raw = False
      self.plot_avg = True
      self.plot_test_reward = True
      self.plot_test_duration = True
      self.plot_test_metrics = True
      # general
      self.actions_done = 0
      self.episodes_done = 0
      self.last_log = 0
      # training data
      self.train_episodes = np.array([], dtype=np.int32)
      self.train_rewards = np.array([], dtype=np.float64)
      self.train_durations = np.array([], dtype=np.int32)
      self.episodes_avg = np.array([], dtype=np.int32)
      self.rewards_avg = np.array([], dtype=np.float64)
      self.durations_avg = np.array([], dtype=np.float64)
      # test data
      self.test_episodes = np.array([], dtype=np.int32)
      self.test_rewards = np.array([], dtype=np.float64)
      self.test_durations = np.array([], dtype=np.int32)
      self.avg_p_lifted = np.array([], dtype=np.float64)
      self.avg_p_contact = np.array([], dtype=np.float64)
      self.avg_p_palm_force = np.array([], dtype=np.float64)
      self.avg_p_exceed_limits = np.array([], dtype=np.float64)
      self.avg_p_exceed_axial = np.array([], dtype=np.float64)
      self.avg_p_exceed_lateral = np.array([], dtype=np.float64)
      self.avg_p_exceed_palm = np.array([], dtype=np.float64)
      
    def calc_moving_average(self):
      # save the rewards and durations moving averages
      if len(self.train_episodes) > self.moving_avg_num:
        self.durations_avg = np.convolve(self.train_durations, np.ones(self.moving_avg_num), 'valid') / self.moving_avg_num
        self.rewards_avg = np.convolve(self.train_rewards, np.ones(self.moving_avg_num), 'valid') / self.moving_avg_num
        x = int(self.moving_avg_num / 2)
        self.episodes_avg = self.train_episodes[x - 1:-x]

    def plot_wandb(self, xdata, ydata, xlabel, ylabel, title):
      # plot data to weights and biases
      data = [[x, y] for (x, y) in zip(xdata, ydata)]
      table = wandb.Table(data=data, columns=[xlabel, ylabel])
      wandb.log({title + " plot" : wandb.plot.line(table, xlabel, ylabel, title=title)})

    def log_wandb(self, log_frequency):

      # if enough time has elapsed for another data upload
      if (self.last_log + log_frequency < time.time()):

        E = "Episode"
        R = "Reward"
        D = "Duration"

        # create plots of raw reward and duration data
        if self.plot_raw:
          self.plot_wandb(self.train_episodes, self.train_rewards, E, R, "Raw rewards")
          self.plot_wandb(self.train_episodes, self.train_durations, E, D, "Raw durations")

        # create plots for average rewards and durations
        if self.plot_avg:
          self.plot_wandb(self.episodes_avg, self.rewards_avg, E, R, 
                          f"Average rewards ({self.moving_avg_num} samples)")
          self.plot_wandb(self.episodes_avg, self.durations_avg, E, D, 
                          f"Average durations ({self.moving_avg_num} samples)")

        # plot the test time reward
        if self.plot_test_reward:
          self.plot_wandb(self.test_episodes, self.test_rewards, E, R, "Test rewards")

        # plot the test time duration
        if self.plot_test_duration:
          self.plot_wandb(self.test_episodes, self.test_durations, E, D, "Test durations")
          
        if self.plot_test_metrics:
          # define performance metrics to examine
          good_metrics = [
            [self.avg_p_lifted, "% Lifted"],
            [self.avg_p_contact, "% Contact"],
            [self.avg_p_palm_force, "% Palm contact"]
          ]
          bad_metrics = [
            [self.avg_p_exceed_limits, "% Exceed limits"],
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

        # finish by recording the last log time
        self.last_log = time.time()

      return

  @dataclass
  class Parameters:
    batch_size: int = 128           # initial 128
    learning_rate: float = 0.01     # initial 0.01
    gamma: float = 0.999            # initial 0.999
    eps_start: float = 0.9          # initial 0.9
    eps_end: float = 0.05           # initial 0.05
    eps_decay: int = 1000           # initial 200 (currently !episodes! to get exp(-1)*eps)
    target_update: int = 100        # initial 10
    num_episodes: int = 10000       # initial 40
    memory_replay: int = 10000      # initial 10000
    min_memory_replay: int = 5000   # initial 5000

    save_freq: int = 1000
    test_freq: int = 1000
    wandb_freq_s: int = 30

  Transition = namedtuple('Transition',
                          ('state', 'action', 'next_state', 'reward'))

  class ReplayMemory(object):

    def __init__(self, capacity):
      self.memory = deque([], maxlen=capacity)

    def push(self, *args):
      """Save a transition"""
      self.memory.append(TrainDQN.Transition(*args))

    def sample(self, batch_size):
      return random.sample(self.memory, batch_size)

    def __len__(self):
      return len(self.memory)

    def to(self, device):
      """Move to a device"""
      for (s1, a, s2, r) in self.memory:
        s1.to(device)
        a.to(device)
        s2.to(device)
        r.to(device)

  def __init__(self, save_suffix=None, device=None, notimestamp=None, 
               use_wandb=None, wandb_name=None, no_plot=None, log_level=None):

    # define key training parameters
    self.params = TrainDQN.Parameters()
    self.track = TrainDQN.Tracker()

    # prepare environment
    self.env = MjEnv()

    # what machine are we on
    self.machine = self.env._get_machine()

    # configure options
    if device != None: self.device = device
    else: self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.save_suffix = save_suffix
    self.notimestamp = notimestamp
    self.log_level = 1 if log_level is None else log_level

    # wandb options
    self.use_wandb = use_wandb if use_wandb is not None else True
    self.wandb_name = wandb_name
    self.wandb_note = ""

    # if we are plotting graphs during this training
    if no_plot == True:
      self.no_plot = True
    else:
      global plt
      import matplotlib.pyplot as plt
      self.fig, self.axs = plt.subplots(2, 1)
      
    # print important info
    if self.log_level > 0:
      print("Using machine:", self.machine)
      print("Using device:", self.device)
      print("Using wandb:", self.use_wandb)

  def init(self, network):
    """
    Create the networks
    """

    # update the environment with correct numbers of actions and observations
    self.env._update_n_actions_obs()

    # create networks
    if network == None:
      raise RuntimeError("TrainDQN network must be specified")
    else:
      # use the network passed as input
      self.policy_net = network(self.env.n_obs, self.env.n_actions,
                                self.device).to(self.device)
      self.target_net = network(self.env.n_obs, self.env.n_actions,
                                self.device).to(self.device)


    self.target_net.load_state_dict(self.policy_net.state_dict())

    # configure optimiser and memory replay
    self.optimiser = optim.RMSprop(self.policy_net.parameters(), 
                                   lr=self.params.learning_rate)
    self.memory = TrainDQN.ReplayMemory(self.params.memory_replay)

    # prepare for saving and loading
    self.modelsaver = ModelSaver('models/dqn/' + self.policy_net.name())

    # save weights and biases
    if self.use_wandb:
      wandb.init(project="luke-gripper-mujoco", entity="lbeddow", 
                 name=self.wandb_name, config=asdict(self.params),
                 notes=self.wandb_note + "\n\n" + self.env._get_cpp_settings())

    # print important info
    print("Using model:", self.policy_net.name())

  def to_torch(self, data, dtype=None):
    """
    Convert some data to a torch tensor
    """
    if dtype == None: dtype = torch.float32

    return torch.tensor(np.array([data]), device=self.device, dtype=dtype)

  def select_action(self, state, decay_num):

    sample = random.random()

    eps_threshold = (self.params.eps_end 
      + (self.params.eps_start - self.params.eps_end)
      * (math.exp(-1. * decay_num / self.params.eps_decay)))

    # if we will not choose randomly
    if sample > eps_threshold:
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

  def plot(self, pltname=None):
    """
    Create a plot to track the training data
    """

    self.track.calc_moving_average()

    if self.no_plot:
      return

    # clear figure
    self.axs[0].clear()
    self.axs[1].clear()

    # plot
    self.axs[0].plot(self.track.train_episodes, self.track.train_durations, label="Raw")
    self.axs[1].plot(self.track.train_episodes, self.track.train_rewards, label="Raw")
    self.axs[0].plot(self.track.test_episodes, self.track.test_durations, "r*", label="Test")
    self.axs[1].plot(self.track.test_episodes, self.track.test_rewards, "r*", label="Test")

    # # plot moving average
    # if len(self.track.train_durations) > self.moving_avg_num:
    #   x = int(self.moving_avg_num / 2)
    #   self.axs[0].plot(self.track.train_episodes[x - 1:-x], durations_avg, label="Average")
    #   self.axs[1].plot(self.track.train_episodes[x - 1:-x], rewards_avg, label="Average")

    # plot moving average
    self.axs[0].plot(self.track.episodes_avg, self.track.durations_avg, label="Average")
    self.axs[1].plot(self.track.episodes_avg, self.track.rewards_avg, label="Average")

    # label
    # self.fig.tight_layout(rect=[0, 0.03, 0, 0.9]) # warning: not applied
    self.fig.subplots_adjust(hspace=0.4)
    self.axs[0].set_title("Episode durations", fontstyle="italic")
    self.axs[1].set_title("Episode rewards", fontstyle="italic")
    self.axs[0].set(ylabel="Duration")
    self.axs[1].set(ylabel="Reward")
    self.axs[0].legend(loc="lower left")
    self.axs[1].legend(loc="upper left")

    if pltname != None:
      self.fig.suptitle(pltname)

    # show on screen
    # self.fig.tight_layout(rect=[0, 0, 0, 0.9])
    # self.fig.show()
    plt.pause(0.001)

  def create_test_report(self, test_data, i_episode=None):
    """
    Process the test data from a finished test
    """

    print_out = False

    len_data = len(test_data)
    num_trials = self.env.test_trials_per_obj
    num_obj = int(len_data / num_trials)

    # safety check
    if len_data % num_trials != 0:
      raise RuntimeError("incorrect test_data length")

    # create and initialise
    names = []
    avg_rewards = []
    avg_steps = []
    avg_palm_force = []
    avg_finger_force = []
    num_stable = 0
    num_lifted = 0
    num_oob = 0
    num_target_height = 0
    num_stable_height = 0

    lifted_vec = []
    contact_vec = []
    palm_force_vec = []
    exceed_limits_vec = []
    exceed_axial_vec = []
    exceed_lateral_vec = []
    exceed_palm_vec = []

    # save all outputs in one place
    output_str = ""

    # define the printing format
    #              name    reward  steps   palm f  fing.f   Lft     Stb     oob     t.h     s.h     pLft   pCon   pPlmFrc  pXLim  pXAxial  pXlaT.  pXPalm
    header_str = "{:<36} | {:<6} | {:<6} | {:<6} | {:<6} | {:<4} | {:<4} | {:<4} | {:<4} | {:<4} | {:<3} | {:<3} | {:<3} | {:<3} | {:<3} | {:<3} | {:<3}\n"
    #              name     reward     steps      palm f     fing.f     Lft     Stb     oob     t.h     s.h     pLft       pCon       pPlmFrc    pXLim      pXAxial    pXlaT.     pXPalm
    row_str =    "{:<36} | {:<6.3f} | {:<6.1f} | {:<6.3f} | {:<6.3f} | {:<4} | {:<4} | {:<4} | {:<4} | {:<4} | {:<3.0f} | {:<3.0f} | {:<3.0f} | {:<3.0f} | {:<3.0f} | {:<3.0f} | {:<3.0f}\n"
    #              name     reward     steps      palm f     fing.f     Lft        Stb        oob        t.h        s.h        pLft       pCon       pPlmFrc    pXLim      pXAxial    pXlaT.     pXPalm
    res_str =    "{:<36} | {:<6.3f} | {:<6.1f} | {:<6.3f} | {:<6.3f} | {:<4.2f} | {:<4.2f} | {:<4.2f} | {:<4.2f} | {:<4.2f} | {:<3.0f} | {:<3.0f} | {:<3.0f} | {:<3.0f} | {:<3.0f} | {:<3.0f} | {:<3.0f}\n"
    first_row = header_str.format(
      "Object name", "Reward", "Steps", "Palm f", "Fing.f", "lft", "stb", "oob", "t.h", "s.h", "%Lt", "%Cn", "%PF", "%XL", "%XA", "%XT", "%XP"
    )

    start_str = f"Starting test on {num_obj} objects, with {num_trials} trials each"
    if i_episode != None: start_str += f", after {i_episode} training steps"
    output_str += start_str + "\n"
    output_str += "\n" + first_row

    if print_out: print(start_str)

    for i in range(num_obj):

      j = i * num_trials

      names.append(test_data[j].object_name)
      total_rewards = 0
      total_steps = 0
      total_lifted = 0
      total_stable = 0
      total_oob = 0
      total_target_height = 0
      total_stable_height = 0
      total_palm_force = 0
      total_finger_force = 0

      cnt_lifted = 0
      cnt_object_contact = 0
      cnt_palm_force = 0
      cnt_exceed_limits = 0
      cnt_exceed_axial = 0
      cnt_exceed_lateral = 0
      cnt_exceed_palm = 0

      for k in range(num_trials):

        # sum end of episode totals for this set of trials
        total_rewards += test_data[j + k].reward
        total_steps += test_data[j + k].steps
        total_lifted += test_data[j + k].lifted
        total_stable += test_data[j + k].stable
        total_oob += test_data[j + k].oob
        total_target_height += test_data[j + k].target_height
        total_stable_height += test_data[j + k].stable_height
        total_palm_force += test_data[j + k].palm_force
        total_finger_force += test_data[j + k].finger_force

        # sum during episode step events for this set of trials
        cnt_lifted += test_data[j + k].cnt_lifted
        cnt_object_contact += test_data[j + k].cnt_object_contact
        cnt_palm_force += test_data[j + k].cnt_palm_force
        cnt_exceed_limits += test_data[j + k].cnt_exceed_limits
        cnt_exceed_axial += test_data[j + k].cnt_exceed_axial
        cnt_exceed_lateral += test_data[j + k].cnt_exceed_lateral
        cnt_exceed_palm += test_data[j + k].cnt_exceed_palm

      # calculate averages for the set of trials
      avg_rewards.append(total_rewards / float(num_trials))
      avg_steps.append(total_steps / float(num_trials))
      avg_palm_force.append(total_palm_force / float(num_trials))
      avg_finger_force.append(total_finger_force / float(num_trials))

      # keep running totals of which events were active when episode ended
      num_stable += total_stable
      num_lifted += total_lifted
      num_oob += total_oob
      num_target_height += total_target_height
      num_stable_height += total_stable_height

      # calculate the percentage of steps that events occured for these trials
      p_lifted = 100 * (cnt_lifted / float(total_steps))
      p_object_contact = 100 * (cnt_object_contact / float(total_steps))
      p_palm_force = 100 * (cnt_palm_force / float(total_steps))
      p_exceed_limits = 100 * (cnt_exceed_limits / float(total_steps))
      p_exceed_axial = 100 * (cnt_exceed_axial / float(total_steps))
      p_exceed_lateral = 100 * (cnt_exceed_lateral / float(total_steps))
      p_exceed_palm = 100 * (cnt_exceed_palm / float(total_steps))

      # save these percentages in vectors
      lifted_vec.append(p_lifted)
      contact_vec.append(p_object_contact)
      palm_force_vec.append(p_palm_force)
      exceed_limits_vec.append(p_exceed_limits)
      exceed_axial_vec.append(p_exceed_axial)
      exceed_lateral_vec.append(p_exceed_lateral)
      exceed_palm_vec.append(p_exceed_palm)

      # save all data in a string to output to a test summary text file
      obj_row = row_str.format(
        names[-1], avg_rewards[-1], avg_steps[-1], avg_palm_force[-1], avg_finger_force[-1],
        total_lifted, total_stable, total_oob, total_target_height, total_stable_height,
        p_lifted, p_object_contact, p_palm_force, p_exceed_limits,
        p_exceed_axial, p_exceed_lateral, p_exceed_palm
      )

      output_str += obj_row

      if print_out: print(obj_row)

    # now get the overall averages for float/integer values
    mean_reward = np.mean(np.array(avg_rewards))
    mean_steps = np.mean(np.array(avg_steps))
    mean_palm_force = np.mean(np.array(avg_palm_force))
    mean_finger_force = np.mean(np.array(avg_finger_force))
    avg_lifted = num_lifted / float(num_obj)
    avg_stable = num_stable / float(num_obj)
    avg_oob = num_oob / float(num_obj)
    avg_target_height = num_target_height / float(num_obj)
    avg_stable_height = num_stable_height / float(num_obj)

    # get the overall averages for percentage of step values
    avg_p_lifted = np.mean(np.array(lifted_vec))
    avg_p_contact = np.mean(np.array(contact_vec))
    avg_p_palm_force = np.mean(np.array(palm_force_vec))
    avg_p_exceed_limits = np.mean(np.array(exceed_limits_vec))
    avg_p_exceed_axial = np.mean(np.array(exceed_axial_vec))
    avg_p_exceed_lateral = np.mean(np.array(exceed_lateral_vec))
    avg_p_exceed_palm = np.mean(np.array(exceed_palm_vec))

    # add the overall averages to the test report string
    end_str = res_str.format(
      "\nOverall averages per object: ", mean_reward, mean_steps, mean_palm_force,
      mean_finger_force, avg_lifted, avg_stable, avg_oob, avg_target_height,
      avg_stable_height, avg_p_lifted, avg_p_contact, avg_p_palm_force,
      avg_p_exceed_limits, avg_p_exceed_axial, avg_p_exceed_lateral,
      avg_p_exceed_palm
    )

    if i_episode != None:
      # save outputs if we are mid-training
      self.track.test_episodes = np.append(self.track.test_episodes, i_episode)
      self.track.test_durations = np.append(self.track.test_durations, mean_steps)
      self.track.test_rewards = np.append(self.track.test_rewards, mean_reward)
      self.track.avg_p_lifted = np.append(self.track.avg_p_lifted, avg_p_lifted)
      self.track.avg_p_contact = np.append(self.track.avg_p_contact, avg_p_contact)
      self.track.avg_p_palm_force = np.append(self.track.avg_p_palm_force, avg_p_palm_force)
      self.track.avg_p_exceed_limits = np.append(self.track.avg_p_exceed_limits, avg_p_exceed_limits)
      self.track.avg_p_exceed_axial = np.append(self.track.avg_p_exceed_axial, avg_p_exceed_axial)
      self.track.avg_p_exceed_lateral = np.append(self.track.avg_p_exceed_lateral, avg_p_exceed_lateral)
      self.track.avg_p_exceed_palm = np.append(self.track.avg_p_exceed_palm, avg_p_exceed_palm)

    output_str += end_str

    if print_out: print(end_str)

    return output_str

  def optimise_model(self):
    """
    Optimise the policy
    """

    # only optimise when enough memory is built up
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

  def train(self, network=None, i_start=None):
    """
    Train the model for the desired number of episodes
    """

    # if we have been given a network to train
    if network != None:
      self.init(network)

    # if this is a fresh, new training
    if i_start == None or i_start == 0:
      i_start = 0
      # create a new folder to save training results in
      self.modelsaver.new_folder(label=self.machine, suffix=self.save_suffix, 
                                 notimestamp=self.notimestamp)
      # save record of the training time hyperparameters
      self.save_hyperparameters()
    else:
      # save a record of the training restart
      continue_label = f"Training is continuing from episode {i_start} with these hyperparameters\n"
      hypername = f"hyperparameters_from_ep_{i_start}"
      self.save_hyperparameters(labelstr=continue_label, name=hypername)

    # begin training episodes
    for i_episode in range(i_start, self.params.num_episodes):
      
      if self.log_level > 0: print("Begin training episode", i_episode)

      # for debugging, show memory usage
      if i_episode % 100 == 0:
        theheap = guph.heap()
        print("Heap total size is", theheap.size, "(", theheap.size / 10e6, "GB)")

      # initialise environment and state
      obs = self.env.reset()
      obs = self.to_torch(obs)

      # count up through actions
      for t in count():

        t_step_start = time.time()

        if self.log_level > 1: print("Episode", i_episode, "action", t)

        # select and perform an action

        # QUESTION: how to decay epsilon
        # action = self.select_action(obs, decay_num=self.track.actions_done)
        action = self.select_action(obs, decay_num=i_episode)

        new_obs, reward, done, _ = self.env.step(action.item())
        new_obs = self.to_torch(new_obs)
        reward = self.to_torch(reward)

        self.track.actions_done += 1
        
        # render the new environment
        self.env.render()

        # store this new transition in memory
        self.memory.push(obs, action, new_obs, reward)
        obs = new_obs

        # perform one step of the optimisation on the policy network
        self.optimise_model()

        t_step_end = time.time()

        if self.log_level > 1: print("Time for action in TrainDQN", t_step_end - t_step_start, 
                                     "seconds")

        # check if this episode is over
        if done:

          # save training data and plot it
          self.track.episodes_done = i_episode + 1
          self.track.train_episodes = np.append(self.track.train_episodes, i_episode)
          self.track.train_durations = np.append(self.track.train_durations, t + 1)
          self.track.train_rewards = np.append(self.track.train_rewards, self.env.track.cumulative_reward)
          self.plot()

          # save to wandb
          if self.use_wandb:
            self.track.log_wandb(self.params.wandb_freq_s)

          break

      # update the target network every target_update episodes
      if i_episode % self.params.target_update == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())

      # test the target network and then save it
      if i_episode % self.params.test_freq == 0 and i_episode != 0:
        test_data = self.test()
        # process test data
        test_report = self.create_test_report(test_data, i_episode=i_episode)
        additional_data = (self.track, test_data)
        # save the result
        self.save(txtstring=test_report, txtlabel="test_results", 
                  tupledata=additional_data)

      # or only save the network
      elif i_episode % self.params.save_freq == 0:
        self.save()

    # update the target network at the end
    self.target_net.load_state_dict(self.policy_net.state_dict())

    # end of training
    if self.log_level > 0:
      print("Training complete, finished", i_episode + 1, "episodes")
    self.env.render()
    self.env.close()

  def test(self):
    """
    Test the target net performance, return a test report
    """

    # begin test mode
    self.env.start_test()

    for i_episode in count():

      # check whether the test has completed
      if self.env.test_completed:
        break

      if self.log_level > 0: print("Begin test episode", i_episode)

      # initialise environment and state
      obs = self.env.reset()
      obs = self.to_torch(obs)

      # count up through actions
      for t in count():

        if self.log_level > 1: ("Test episode", i_episode, "action", t)

        # choose the best action
        with torch.no_grad():
          # t.max(1) returns largest column value of each row
          # [1] is second column of max result, the index of max element
          # view(1, 1) selects this element which has max expected reward
          action = self.target_net(obs).max(1)[1].view(1, 1)

        # select and perform an action
        obs, reward, done, _ = self.env.step(action.item())
        obs = self.to_torch(obs)
        reward = self.to_torch(reward)
        
        # render the new environment
        self.env.render()

        # check if this episode is over, if so plot and break
        if done:
          # # don't capture this data here for testing, it is done seperately
          # self.episode_durations.append(t + 1)
          # self.episode_rewards.append(self.env.track.cumulative_reward)
          # self.plot()
          break

    # end of testing

    # get the test data out
    test_data = self.env.test_trials

    if self.log_level > 0: print("Testing complete, finished", i_episode, "episodes")

    return test_data

  def save_hyperparameters(self, labelstr=None, name=None):
    """
    Save a text file with the current hyperparameters
    """

    param_str = ""
    time_stamp = datetime.now().strftime("%d-%m-%Y-%H:%M")

    if labelstr != None:
      param_str += labelstr + '\n'

    if name == None:
      name = "hyperparameters"

    # capture parameters info
    param_str += f"""Hyperparameters at training time:\n\n"""

    # add in some important information
    param_str += "Network name: " + self.policy_net.name() + "\n"
    param_str += "Save time and date: " + time_stamp + "\n"
    param_str += "Running on machine: " + self.machine + "\n"

    # convert parameters to a string
    param_str += "Parameters dataclass:\n"
    param_str += str(asdict(self.params))

    # swap commas for newlines for prettier printing
    param_str = param_str.replace(',', '\n')

    # add the c++ settings to the string
    param_str += "\n\n" + self.env._get_cpp_settings()

    savepath = self.modelsaver.save(name, txtstr=param_str,
                                    txtonly=True)

    return savepath

  def save(self, txtstring=None, txtlabel=None, tupledata=None):
    """
    Save the model policy network, return save path
    """

    # save all needed internal data to continue training with
    core_data = (self.policy_net, self.params, self.memory, self.env) # add self.env

    # data structure we will save
    to_save = (core_data, tupledata)

    savepath = self.modelsaver.save(self.policy_net.name(), pyobj=to_save, 
                                    txtstr=txtstring, txtlabel=txtlabel)


    return savepath

  def load(self, id=None, folderpath=None, foldername=None):
    """
    Load the most recent model, overwrite current networks
    """

    # load the model
    (core, tupledata) = self.modelsaver.load(id=id, folderpath=folderpath, 
                                             foldername=foldername)

    # extract core data
    self.policy_net = core[0]
    self.params = core[1]
    self.memory = core[2]
    self.env = core[3]

    # extract additional data
    self.track = tupledata[0]
    self.loaded_test_data = tupledata[1]

    # reload environment
    self.env._load_xml # segfault without this

    # reinitialise to prepare for further training
    self.target_net = deepcopy(core[0])
    self.target_net.load_state_dict(self.policy_net.state_dict())

    # move to the current device
    self.memory.to(self.device)
    self.policy_net.to(self.device)
    self.target_net.to(self.device)

    # re-initialise optimiser
    self.optimiser = optim.RMSprop(self.policy_net.parameters(), 
                                   lr=self.params.learning_rate)

    # return the path of the loaded model
    return self.modelsaver.last_loadpath

  def continue_training(self, foldername, folderpath=None, new_endpoint=None):
    """
    Load a model and then continue training it
    """

    # load the most recent model in the given folder
    self.load(foldername=foldername, folderpath=folderpath)
    self.modelsaver.enter_folder(foldername, folderpath=folderpath)

    # update the new training episode target
    if new_endpoint != None:
      self.params.num_episodes = new_endpoint
    elif self.params.num_episodes == self.track.episodes_done:
      # if no target is set but we have finished, double the initial target
      self.params.num_episodes = 2 * self.track.episodes_done
      print("New training endpoint set as:", self.params.num_episodes)

    # check if our training endpoint is already satisfied
    if self.params.num_episodes < self.track.episodes_done:
      raise RuntimeError("episodes done exceeds the params.num_episodes target")

    # begin the training at the given starting point (always uses most recent pickle)
    self.train(i_start=self.track.episodes_done)


if __name__ == "__main__":
  
  # ----- prepare ----- #

  use_wandb = False
  force_device = "cpu"

  model = TrainDQN(device=force_device, use_wandb=use_wandb,
                   wandb_name=None)

  # if we want to adjust parameters
  # model.params.num_episodes = 11
  # model.params.test_freq = 10
  # model.env.test_trials_per_obj = 1
  # model.env.max_episode_steps = 20
  # model.params.wandb_freq_s = 5
  # model.env.mj.set.action_motor_steps = 350

  # now set up the network, ready for training
  net = networks.DQN_2L60
  model.init(net)

  # ----- load ----- #

  # load
  folderpath = "/home/luke/cluster/rl/models/dqn/" + model.policy_net.name() + "/"
  foldername = "train_cluster_29-04-2022-12:51_array_4"
  model.load(id=None, folderpath=folderpath, foldername=foldername)

  # ----- train ----- #

  # # train
  # model.env.disable_rendering = False
  # model.env.mj.set.debug = True
  # model.train()

  # # continue training
  # folderpath = "/home/luke/cluster/rl/models/dqn/" + model.policy_net.name() + "/"
  # foldername = "train_cluster_29-04-2022-12:51_array_4"
  # model.continue_training(foldername, folderpath=folderpath)

  # ----- visualise ----- #

  # visualise training performance
  # plt.ion()
  # model.plot()
  # plt.show()

  import array_training_DQN
  model = array_training_DQN.apply_to_all_models(model)
  model = array_training_DQN.make_rewards_negative(model)
  model.env.max_episode_steps = 200
  model.env.mj.set.motor_state_sensor.read_rate = -2
  model.env.mj.set.axial_gauge.in_use = True
  model.env.mj.set.wrist_sensor_Z.in_use = True

  # # test
  print(model.env._get_cpp_settings())
  model.env.mj.set.debug = True
  model.env.disable_rendering = False
  model.env.test_trials_per_obj = 1
  # model.env.mj.set.step_num.set          (0,      70,   1)
  # model.env.mj.set.exceed_limits.set     (-0.005, True,   10)
  # model.env.mj.set.exceed_axial.set      (-0.005, True,   10,    3.0,  6.0,  -1)
  # model.env.mj.set.exceed_lateral.set    (-0.005, True,   10,    4.0,  6.0,  -1)
  test_data = model.test()

  # # save results
  # test_report = model.create_test_report(test_data)
  # model.modelsaver.new_folder(label="DQN_testing")
  # model.save_hyperparameters(labelstr=f"Loaded model path: {model.modelsaver.last_loadpath}\n")
  # model.save(txtstring=test_report, txtlabel="test_results_demo")


 
