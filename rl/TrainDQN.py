#!/usr/bin/env python3

import gym
import math
import random
import time
import numpy as np
from collections import namedtuple, deque
from itertools import count
from dataclasses import dataclass
from copy import deepcopy

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
      self.actions_done = 0
      self.episodes_done = 0
      self.train_rewards = np.array([], dtype=np.float64)
      self.train_durations = np.array([], dtype=np.int32)
      self.train_episodes = np.array([], dtype=np.int32)
      self.test_rewards = np.array([], dtype=np.float64)
      self.test_durations = np.array([], dtype=np.int32)
      self.test_episodes = np.array([], dtype=np.int32)

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

    save_freq: int = 1000
    test_freq: int = 1000

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

  def __init__(self, cluster=True, save_suffix=None, device=None):

    # define key training parameters
    self.params = TrainDQN.Parameters()
    self.track = TrainDQN.Tracker()

    # prepare environment
    self.env = MjEnv()
    if device != None: self.device = device
    else: self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.actions_taken = 0
    self.log_level = 1
    self.cluster = cluster
    self.save_suffix = save_suffix

    # cluster specific
    if not self.cluster:
      # import into global variable
      global plt
      import matplotlib.pyplot as plt

    # prepare for plotting
    self.episode_durations = []
    self.episode_rewards = []
    self.test_durations = []
    self.test_rewards = []
    self.test_episodes = []
    self.no_plot = self.cluster
    self.plot_limit = 2000
    if not self.no_plot:
      self.fig, self.axs = plt.subplots(2, 1)

    # print important info
    print("On cluster:", "TRUE" if self.cluster else "FALSE")
    print("Using device:", self.device)

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

    # print important info
    print("Using model:", self.policy_net.name())

  def to_torch(self, data, dtype=None):
    """
    Convert some data to a torch tensor
    """
    if dtype == None: dtype = torch.float32

    return torch.tensor([data], device=self.device, dtype=dtype)

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
      print("rand action is ", rand_action)
      print("env.n_actions is ", self.env.n_actions)
      print("mj n_actions is ", self.env.mj.get_n_actions())
      return torch.tensor([[rand_action]], device=self.device,
                          dtype=torch.long)

  def plot(self, pltname=None):
    """
    Create a plot to track the training data
    """

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

    # get moving average
    avg_num = 50 # must be even number
    if len(self.track.train_durations) > avg_num:
      durations_avg = np.convolve(self.track.train_durations, np.ones(avg_num), 'valid') / avg_num
      rewards_avg = np.convolve(self.track.train_rewards, np.ones(avg_num), 'valid') / avg_num
      # plot
      x = int(avg_num / 2)
      self.axs[0].plot(self.track.train_episodes[x - 1:-x], durations_avg, label="Average")
      self.axs[1].plot(self.track.train_episodes[x - 1:-x], rewards_avg, label="Average")

    # label
    self.fig.tight_layout(rect=[0, 0.03, 0, 0.9])
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

        total_rewards += test_data[j + k].reward
        total_steps += test_data[j + k].steps
        total_lifted += test_data[j + k].lifted
        total_stable += test_data[j + k].stable
        total_oob += test_data[j + k].oob
        total_target_height += test_data[j + k].target_height
        total_stable_height += test_data[j + k].stable_height
        total_palm_force += test_data[j + k].palm_force
        total_finger_force += test_data[j + k].finger_force

        cnt_lifted += test_data[j + k].cnt_lifted
        cnt_object_contact += test_data[j + k].cnt_object_contact
        cnt_palm_force += test_data[j + k].cnt_palm_force
        cnt_exceed_limits += test_data[j + k].cnt_exceed_limits
        cnt_exceed_axial += test_data[j + k].cnt_exceed_axial
        cnt_exceed_lateral += test_data[j + k].cnt_exceed_lateral
        cnt_exceed_palm += test_data[j + k].cnt_exceed_palm

      avg_rewards.append(total_rewards / float(num_trials))
      avg_steps.append(total_steps / float(num_trials))
      avg_palm_force.append(total_palm_force / float(num_trials))
      avg_finger_force.append(total_finger_force / float(num_trials))
      num_stable += total_stable
      num_lifted += total_lifted
      num_oob += total_oob
      num_target_height += total_target_height
      num_stable_height += total_stable_height

      p_lifted = 100 * (cnt_lifted / float(total_steps))
      p_object_contact = 100 * (cnt_object_contact / float(total_steps))
      p_palm_force = 100 * (cnt_palm_force / float(total_steps))
      p_exceed_limits = 100 * (cnt_exceed_limits / float(total_steps))
      p_exceed_axial = 100 * (cnt_exceed_axial / float(total_steps))
      p_exceed_lateral = 100 * (cnt_exceed_lateral / float(total_steps))
      p_exceed_palm = 100 * (cnt_exceed_palm / float(total_steps))

      lifted_vec.append(p_lifted)
      contact_vec.append(p_object_contact)
      palm_force_vec.append(p_palm_force)
      exceed_limits_vec.append(p_exceed_limits)
      exceed_axial_vec.append(p_exceed_axial)
      exceed_lateral_vec.append(p_exceed_lateral)
      exceed_palm_vec.append(p_exceed_palm)

      per_obj_str = f"Object: {names[-1]}, "\
        f"avg reward {avg_rewards[-1]:.3f}, "\
        f"avg steps {avg_steps[-1]:.1f}, "\
        f"avg palm force {avg_palm_force[-1]:.1f}, "\
        f"lifted {total_lifted}, "\
        f"stable {total_stable}, "\
        f"target height {total_target_height}, "\
        f"stable height {total_stable_height}"

      obj_row = row_str.format(
        names[-1], avg_rewards[-1], avg_steps[-1], avg_palm_force[-1], avg_finger_force[-1],
        total_lifted, total_stable, total_oob, total_target_height, total_stable_height,
        p_lifted, p_object_contact, p_palm_force, p_exceed_limits,
        p_exceed_axial, p_exceed_lateral, p_exceed_palm
      )

      # output_str += per_obj_str + "\n"
      output_str += obj_row

      if print_out: print(per_obj_str)

    # now get the overall averages
    mean_reward = np.mean(np.array(avg_rewards))
    mean_steps = np.mean(np.array(avg_steps))
    mean_palm_force = np.mean(np.array(avg_palm_force))
    mean_finger_force = np.mean(np.array(avg_finger_force))
    avg_lifted = num_lifted / float(num_obj)
    avg_stable = num_stable / float(num_obj)
    avg_oob = num_oob / float(num_obj)
    avg_target_height = num_target_height / float(num_obj)
    avg_stable_height = num_stable_height / float(num_obj)

    lifted_vec.append(p_lifted)
    contact_vec.append(p_object_contact)
    palm_force_vec.append(p_palm_force)
    exceed_limits_vec.append(p_exceed_limits)
    exceed_axial_vec.append(p_exceed_axial)
    exceed_lateral_vec.append(p_exceed_lateral)
    exceed_palm_vec.append(p_exceed_palm)

    end_str = res_str.format(
      "Overall averages per object: ", mean_reward, mean_steps, mean_palm_force,
      mean_finger_force, avg_lifted, avg_stable, avg_oob, avg_target_height,
      avg_stable_height,
      np.mean(np.array(lifted_vec)), np.mean(np.array(contact_vec)),
      np.mean(np.array(palm_force_vec)), np.mean(np.array(exceed_limits_vec)),
      np.mean(np.array(exceed_axial_vec)), np.mean(np.array(exceed_lateral_vec)),
      np.mean(np.array(exceed_palm_vec))
    )

    # save outputs if we are mid-training
    if i_episode != None:
      self.track.test_episodes = np.append(self.track.test_episodes, i_episode)
      self.track.test_durations = np.append(self.track.test_durations, mean_steps)
      self.track.test_rewards = np.append(self.track.test_rewards, mean_reward)

    # output_str += "\n" + end_str + "\n\n"
    output_str += "\n" + end_str

    if print_out: print(end_str)

    return output_str

  def optimise_model(self):
    """
    Optimise the policy
    """

    # only optimise when enough memory is built up
    if (len(self.memory)) < self.params.batch_size:
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

  def train(self, i_start=None):
    """
    Train the model for the desired number of episodes
    """

    # if this is a fresh, new training
    if i_start == None:
      i_start = 0
      # create a new folder to save training results in
      self.modelsaver.new_folder(label="cluster" if self.cluster else "laptop",
                                 suffix=self.save_suffix)
      # save record of the training time hyperparameters
      self.save_hyperparameters()
    else:
      # save a record of the training restart
      continue_label = f"Training is continuing from episode {i_start} with these hyperparameters\n"
      hypername = f"hyperparameters_from_ep_{i_start}"
      self.save_hyperparameters(labelstr=continue_label, name=hypername)


    for i_episode in range(i_start, self.params.num_episodes):

      if self.log_level > 0: print("Begin training episode", i_episode)

      # for debugging, show memory usage
      if i_episode % 10 == 0:
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

        # check if this episode is over, if so plot and break
        if done:

          self.track.episodes_done = i_episode + 1
          self.track.train_episodes = np.append(self.track.train_episodes, i_episode)
          self.track.train_durations = np.append(self.track.train_durations, t + 1)
          self.track.train_rewards = np.append(self.track.train_rewards, self.env.track.cumulative_reward)
          self.plot()

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
    Save a text file with the hyperparameters
    """

    param_str = ""
    network_name = self.policy_net.name()

    if labelstr != None:
      param_str += labelstr + '\n'

    if name == None:
      name = "hyperparameters"

    # capture parameters info
    param_str += f"""Hyperparameters at training time:

    network_name = {network_name}
    batch_size = {self.params.batch_size}
    gamma = {self.params.gamma}
    eps_start = {self.params.eps_start}
    eps_end = {self.params.eps_end}
    eps_decay = {self.params.eps_decay}
    target_update = {self.params.target_update}
    num_episodes = {self.params.num_episodes}
    max_episode_steps = {self.env.max_episode_steps}
    memory_replay = {self.params.memory_replay}
    test_freq = {self.params.test_freq}
    save_freq = {self.params.save_freq}
    """

    param_str += "\n" + self.env._get_cpp_settings()

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

    if len(core) == 4:
      self.env = core[3]
      self.env._load_xml(random=True) # segfault without this

    # extract additional data
    self.track = tupledata[0]
    self.loaded_test_data = tupledata[1]

    # reinitialise to prepare for further training
    self.target_net = deepcopy(core[0])
    self.target_net.load_state_dict(self.policy_net.state_dict())

    # move to the device
    self.policy_net.to(self.device)
    self.target_net.to(self.device)

    # re-initialise optimiser
    self.optimiser = optim.RMSprop(self.policy_net.parameters(), 
                                   lr=self.params.learning_rate)

    # return the path of the loaded model
    return self.modelsaver.last_loadpath

  def continue_training(self, foldername, folderpath=None, new_endpoint=None):
    """
    Continue training of a model
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

    # begin the training at the given starting point (always uses most recent pickle)
    self.train(i_start=self.track.episodes_done)


if __name__ == "__main__":
  
  # ----- prepare ----- #

  cluster = True
  force_device = "cpu"
  model = TrainDQN(cluster=cluster, device=force_device)

  # if we want to adjust parameters
  # model.params.num_episodes = 11
  # model.params.test_freq = 250
  # model.env.max_episode_steps = 1
  # model.env.mj.set.max_action_steps = 1000
  # model.env._override_binary(model.env.mj.set.target_height, 1.0, 1, 1)

  # now set up the network, ready for training
  net = networks.DQN_2L60
  model.init(network=net)

  # ----- load ----- #

  # # load
  # folderpath = "/home/luke/cluster/rl/models/dqn/DQN_2L60/"
  # folderpath = "/home/luke/mymujoco/rl/models/dqn/DQN_3L60/"
  # foldername = "train_cluster_28-03-2022_16:55_array_17"
  # model.load(id=13, folderpath=folderpath, foldername=foldername)

  # ----- train ----- #

  # train
  model.env.disable_rendering = True
  model.env.mj.set.debug = False
  model.train()

  # continue training
  # model.continue_training('train_cluster_24-02-2022_12:43_array_6', folderpath=folderpath)

  # ----- visualise ----- #

  # visualise training performance
  # plt.ion()
  # model.plot()
  # plt.show()

  # # test
  # model.env.disable_rendering = False
  # model.env.test_trials_per_obj = 1
  # test_data = model.test()

  # # save results
  # test_report = model.create_test_report(test_data)
  # model.modelsaver.new_folder(label="DQN_testing")
  # model.save_hyperparameters(labelstr=f"Loaded model path: {model.modelsaver.last_loadpath}\n")
  # model.save(txtstring=test_report, txtlabel="test_results_demo")


 
