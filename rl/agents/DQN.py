#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, asdict
from collections import namedtuple, deque
from copy import deepcopy
import random

class ReplayMemory(object):

  def __init__(self, capacity, device, transition, rngseed=None):
    self.memory = deque([], maxlen=capacity)
    self.device = device
    self.transition = transition
    self.capacity = capacity
    self.seed(rngseed)

  def push(self, *args):
    """Save a transition"""
    self.memory.append(self.transition(*args))

  def to_torch(self, data, dtype=None):
    """
    Convert some data to a torch tensor
    """
    if dtype == None: dtype = torch.float32
    return torch.tensor(data, device=self.device, dtype=dtype).unsqueeze(0)

  def sample(self, batch_size):
    return self.rng.sample(self.memory, batch_size)
  
  def seed(self, rngseed=None):
    self.rngseed = rngseed
    # create an instance of the Python random class with the given seed
    # the random module functions just call an instance of this class
    # we can't use numpy random as we want to support torch tensors and deque()
    self.rng = random.Random(rngseed)

  def batch(self, batch_size):
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    transitions = self.sample(batch_size)
    return self.transition(*zip(*transitions))

  def __len__(self):
    return len(self.memory)

  def to_device(self, tuple_of_tensors):
    """Moves a transition to a device"""
    lst = []
    for item in tuple_of_tensors:
      lst.append(item.to(self.device))
    return self.transition(*tuple(lst))

  def all_to(self, device):
      """Move entire replay memory to a device"""
      self.device = device
      for i, item in enumerate(self.memory):
        self.memory[i] = self.to_device(item)

class Agent_DQN:

  name = "Agent_DQN"

  @dataclass
  class Parameters:

    # key learning hyperparameters
    learning_rate: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 128 
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay: int = 1000

    # memory replay
    min_memory_replay: int = 250
    memory_replay: int = 10_000

    # options
    optimiser: str = "adamW" # adam/adamW/RMSProp
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    target_update: int = 250
    soft_target_update: bool = True
    soft_target_tau: float = 0.05
    grad_clamp_value: bool = 100 # None to disable
    loss_criterion: str = "smoothL1Loss" # smoothL1Loss/MSELoss/Huber

    def update(self, newdict):
      for key, value in newdict.items():
        if hasattr(self, key):
          setattr(self, key, value)
        else: raise RuntimeError(f"incorrect key: {key}")

  Transition = namedtuple('Transition',
                          ('state', 'action', 'next_state', 'reward', 'terminal'))
  
  def __init__(self, device="cpu", rngseed=None, mode="train"):
    """
    Deep Q-network agent for a given environment
    """

    self.params = Agent_DQN.Parameters()
    self.device = torch.device(device)
    self.rngseed = rngseed
    self.mode = mode

  def init(self, network):
    """
    Main initialisation of the agent, applies settings and creates network. This
    function can be called with network='loaded' to initialise settings but not
    reinitialise the networks
    """

    self.memory = ReplayMemory(self.params.memory_replay, self.device, Agent_DQN.Transition)

    if network != "loaded":
      self.policy_net = network
      self.target_net = deepcopy(self.policy_net)
      self.n_actions = network.n_output # network must provide this member variable

    self.policy_net.to(self.device)
    self.target_net.to(self.device)

    if self.params.optimiser.lower() == "rmsprop":
      self.optimiser = torch.optim.RMSprop(self.policy_net.parameters(), 
                                           lr=self.params.learning_rate)
    elif self.params.optimiser.lower() == "adam":
      self.optimiser = torch.optim.Adam(self.policy_net.parameters(),
                                        lr=self.params.learning_rate,
                                        betas=(self.params.adam_beta1,
                                               self.params.adam_beta2))
    elif self.params.optimiser.lower() == "adamw":
      self.optimiser = torch.optim.AdamW(self.policy_net.parameters(),
                                        lr=self.params.learning_rate,
                                        betas=(self.params.adam_beta1,
                                               self.params.adam_beta2),
                                        amsgrad=True)
    else: raise RuntimeError(f"Invalid optimiser choice '{self.params.optimiser}', options are 'adam' or 'rmsprop'")

    if self.params.loss_criterion.lower() == "smoothl1loss":
      self.loss_criterion = nn.SmoothL1Loss()
    elif self.params.loss_criterion.lower() == "mseloss":
      self.loss_criterion = nn.MSELoss()
    elif self.params.loss_criterion.lower() == "huber":
      self.loss_criterion = nn.HuberLoss()

    self.seed()

    if self.mode == "train": self.training_mode()
    elif self.mode == "test": self.testing_mode()
    else: raise RuntimeError(f"Agent_DQN given mode={self.mode} but valid options are 'test' and 'train'")
  
  def set_device(self, device):
    """
    Set whether on cpu or gpu, and move all the memory replay buffer to that device
    """

    if device == "cuda" and not torch.cuda.is_available(): 
      print("Agent_DQN.set_device() received request for 'cuda', but torch.cuda.is_available() == False, hence using cpu")
      device = "cpu"

    self.device = torch.device(device)
    self.memory.all_to(device)
    self.policy_net.to(device)
    self.target_net.to(device)
  
  def seed(self, rngseed=None):
    """
    Set a random seed for the entire environment
    """
    if rngseed is None:
      if self.rngseed is not None: rngseed = self.rngseed
      else: rngseed = np.random.randint(0, 2_147_483_647)
    self.rngseed = rngseed
    self.rng = np.random.default_rng(rngseed)
    self.memory.seed(rngseed)

  def select_action(self, state, decay_num, test=False):
    """
    Choose an action using the policy network, or randomly depending on decay.

    This function should return a tensor([x])
    """

    # determine if we will act randomly
    random_action = False
    if not test:
      rand = self.rng.random()
      eps_threshold = (self.params.eps_end 
        + (self.params.eps_start - self.params.eps_end)
        * (np.exp(-1. * decay_num / float(self.params.eps_decay))))
      if rand < eps_threshold: random_action = True

    # take an action
    if random_action: 
      return torch.tensor([self.rng.integers(0, self.n_actions)], dtype=torch.long)
    else:
      with torch.no_grad():
        # t.max(1) returns largest column value of each row
        # [1] is second column of max result, the index of max element
        # view(1, 1) selects this element which has max expected reward
        # [0][0] changes Tensor([[x]]) -> Tensor(x)
        return self.policy_net(state).max(1)[1].view(1, 1)[0]
      
  def get_save_state(self):
    """
    Return the state of the model that should be saved
    """

    to_save = {
      "name" : self.name,
      "parameters" : self.params,
      "memory" : self.memory.memory,
      "network" : self.policy_net, # do I want to save only the state dict?
      "optimiser_state_dict" : self.optimiser.state_dict()
    }

    return to_save
  
  def load_save_state(self, saved_dict):
    """
    Load the agent with a given saved state
    """

    # check we are compatible
    if self.name != saved_dict["name"]:
      raise RuntimeError(f"{self.name}.load_save_state() failed as given save state has agent name: {saved_dict['name']}")
    
    # input the saved data
    self.params = saved_dict["parameters"]
    self.init(saved_dict["network"])
    self.memory.memory = saved_dict["memory"]
    self.memory.all_to(self.device)
    self.optimiser.load_state_dict(saved_dict["optimiser_state_dict"])

  def get_params_dict(self):
    """
    Return a dictionary of hyperparameters
    """
    param_dict = asdict(self.params)
    param_dict.update({
      "network_name" : self.policy_net.name
    })
    return param_dict

  def optimise_model(self):

    # only proceed if we have enough memory for a batch
    if len(self.memory) < self.params.batch_size: 
      return

    # only begin to optimise when enough memory is built up
    if (len(self.memory)) < self.params.min_memory_replay:
      return
    
    # transitions = self.memory.sample(self.params.batch_size)
    # # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # # detailed explanation). This converts batch-array of Transitions
    # # to Transition of batch-arrays.
    # batch = Agent_DQN.Transition(*zip(*transitions))

    # sample and transpose a batch
    batch = self.memory.batch(self.params.batch_size)

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

    # print("state", state_batch.shape)
    # print("action", action_batch.shape)
    # print("reward", reward_batch.shape)

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
    with torch.no_grad():
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * self.params.gamma) + reward_batch

    # Compute Huber loss
    # criterion = nn.SmoothL1Loss()
    loss = self.loss_criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    self.optimiser.zero_grad()
    loss.backward()
    # In-place gradient clipping
    if self.params.grad_clamp_value is not None:
      torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), self.params.grad_clamp_value)
    self.optimiser.step()

  def update_step(self, state, action, next_state, reward, terminal, truncated):
    """
    Run this every training action step to update the model
    """

    if self.mode != "train": return

    # print("state", state.shape)
    # print("action", action.shape)
    # print("next_state", next_state.shape)
    # print("reward", reward.unsqueeze(0).shape)
    # print("terminal", terminal.unsqueeze(0).shape)

    self.memory.push(
      state,
      action,
      next_state,
      reward,
      terminal
    )

    self.optimise_model()

    if self.params.soft_target_update:
      # Soft update of the target network's weights
      # θ′ ← τ θ + (1 −τ )θ′
      tau = self.params.soft_target_tau
      target_net_state_dict = self.target_net.state_dict()
      policy_net_state_dict = self.policy_net.state_dict()
      for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
      self.target_net.load_state_dict(target_net_state_dict)

  def update_episode(self, i_episode, finished=False):
    """
    Run this at the end of every training episode
    """

    # if we are finished training
    if finished:
      # update the target network
      self.target_net.load_state_dict(self.policy_net.state_dict())

    if self.mode != "train": return

    if not self.params.soft_target_update:
      if i_episode % self.params.target_update == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())

  def training_mode(self):
    """
    Put the agent into training mode
    """
    self.mode = "train"
    self.policy_net.train()
    self.target_net.train()

  def testing_mode(self):
    """
    Put the agent in testing mode
    """
    self.mode = "test"
    self.policy_net.eval()
    self.target_net.eval()
