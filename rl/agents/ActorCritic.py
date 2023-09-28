#!/usr/bin/env python3

import numpy as np
from dataclasses import dataclass, asdict
from collections import namedtuple, deque
from copy import deepcopy
import random
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

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

def mlp(sizes, activation, output_activation=nn.Identity):
  layers = []
  for j in range(len(sizes)-1):
    act = activation if j < len(sizes)-2 else output_activation
    layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
  return nn.Sequential(*layers)

def count_vars(module):
  return sum([np.prod(p.shape) for p in module.parameters()])

class SquashedGaussianMLPActor(nn.Module):

  def __init__(self, n_obs, n_actions, hidden_sizes, activation, action_limit,
                log_std_max=2, log_std_min=-20):
    super().__init__()
    self.net = mlp([n_obs] + list(hidden_sizes), activation, activation)
    self.mu_layer = nn.Linear(hidden_sizes[-1], n_actions)
    self.log_std_layer = nn.Linear(hidden_sizes[-1], n_actions)
    self.action_limit = action_limit
    self.log_std_max = log_std_max
    self.log_std_min = log_std_min

  def forward(self, obs, deterministic=False, with_logprob=True):
    net_out = self.net(obs)
    mu = self.mu_layer(net_out)
    log_std = self.log_std_layer(net_out)
    log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
    std = torch.exp(log_std)

    # Pre-squash distribution and sample
    pi_distribution = Normal(mu, std)
    if deterministic:
      # Only used for evaluating policy at test time.
      pi_action = mu
    else:
      pi_action = pi_distribution.rsample()

    if with_logprob:
      # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
      # NOTE: The correction formula is a little bit magic. To get an understanding 
      # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
      # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
      # Try deriving it yourself as a (very difficult) exercise. :)
      logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
      logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
    else:
      logp_pi = None

    pi_action = torch.tanh(pi_action)
    pi_action = self.action_limit * pi_action

    return pi_action, logp_pi

class MLPQFunction(nn.Module):

  def __init__(self, n_obs, n_actions, hidden_sizes, activation):
    super().__init__()
    self.q = mlp([n_obs + n_actions] + list(hidden_sizes) + [1], activation)

  def forward(self, obs, act):
    q = self.q(torch.cat([obs, act], dim=-1))
    return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):
    
  name = "MLPActorCritic_"

  def __init__(self, n_obs, n_actions, hidden_sizes=(256,256),
              activation=nn.ReLU, action_limit=1.0, mode="train"):
    super().__init__()

    # build policy and value functions
    self.pi = SquashedGaussianMLPActor(n_obs, n_actions, hidden_sizes, activation, action_limit)
    self.q1 = MLPQFunction(n_obs, n_actions, hidden_sizes, activation)
    self.q2 = MLPQFunction(n_obs, n_actions, hidden_sizes, activation)
    self.n_obs = n_obs
    self.n_actions = n_actions
    self.mode = mode

    if self.mode not in ["test", "train"]:
      raise RuntimeError(f"MLPActorCritic given mode={mode}, should be 'test' or 'train'")

    # add hidden layer size into the name
    for i in range(len(hidden_sizes)):
      if i == 0: self.name += f"{hidden_sizes[i]}"
      if i > 0: self.name += f"x{hidden_sizes[i]}"

  def act(self, obs, deterministic=False):
    with torch.no_grad():
      actions, _ = self.pi(obs, deterministic, False)
      return actions.squeeze(0)
      
  def set_device(self, device):
    self.pi.to(device)
    self.q1.to(device)
    self.q2.to(device)

  def training_mode(self):
    """
    Put the agent into training mode
    """
    self.mode = "train"
    self.pi.train()
    self.q1.train()
    self.q2.train()

  def testing_mode(self):
    """
    Put the agent in testing mode
    """
    self.mode = "test"
    self.pi.eval()
    self.q1.eval()
    self.q2.eval()

class Agent_SAC:

  name = "Agent_SAC"

  @dataclass
  class Parameters:

    # key learning hyperparameters
    learning_rate: float = 1e-4
    gamma: float = 0.99
    alpha: float = 0.2 # tradeoff temperature
    batch_size: int = 128 
    update_after_steps: int = 1000
    update_every_steps: int = 50
    random_start_episodes: int = 1000

    # memory replay
    min_memory_replay: int = 250
    memory_replay: int = 10_000

    # options
    optimiser: str = "adam" # adam/adamW/RMSProp
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    soft_target_tau: float = 0.05
    # loss_criterion: str = "MSELoss" # smoothL1Loss/MSELoss/Huber

    def update(self, newdict):
      for key, value in newdict.items():
        if hasattr(self, key):
          setattr(self, key, value)
        else: raise RuntimeError(f"incorrect key: {key}")
      
  Transition = namedtuple('Transition',
                          ('state', 'action', 'next_state', 'reward', 'terminal'))

  def __init__(self, device="cpu", rngseed=None, mode="train"):
    """
    Soft actor-critic agent for a given environment
    """

    self.params = Agent_SAC.Parameters()
    self.device = torch.device(device)
    self.rngseed = rngseed
    self.mode = mode
    self.steps_done = 0

  def init(self, network):
    """
    Main initialisation of the agent, applies settings and creates network. This
    function can be called with network='loaded' to initialise settings but not
    reinitialise the networks
    """

    self.memory = ReplayMemory(self.params.memory_replay, self.device, Agent_SAC.Transition)

    if network != "loaded":

      self.mlp_ac = network
      self.mlp_ac_targ = deepcopy(network)

      self.n_actions = self.mlp_ac.n_actions # network must provide this member variable

    self.set_device(self.device)

    self.q_network_parameters = itertools.chain(self.mlp_ac.q1.parameters(),
                                           self.mlp_ac.q2.parameters())

    if self.params.optimiser.lower() == "rmsprop":
      self.q_optimiser = torch.optim.RMSprop(self.q_network_parameters, 
                                             lr=self.params.learning_rate)
      self.pi_optimiser = torch.optim.RMSprop(self.mlp_ac.pi.parameters(),
                                              lr=self.params.learning_rate)
    elif self.params.optimiser.lower() == "adam":
      self.q_optimiser = torch.optim.Adam(self.q_network_parameters,
                                          lr=self.params.learning_rate,
                                          betas=(self.params.adam_beta1,
                                                 self.params.adam_beta2))
      self.pi_optimiser = torch.optim.Adam(self.mlp_ac.pi.parameters(),
                                           lr=self.params.learning_rate,
                                           betas=(self.params.adam_beta1,
                                                  self.params.adam_beta2))
    elif self.params.optimiser.lower() == "adamw":
      self.q_optimiser = torch.optim.AdamW(self.q_network_parameters,
                                           lr=self.params.learning_rate,
                                           betas=(self.params.adam_beta1,
                                                  self.params.adam_beta2),
                                           amsgrad=True)
      self.pi_optimiser = torch.optim.AdamW(self.mlp_ac.parameters(),
                                            lr=self.params.learning_rate,
                                            betas=(self.params.adam_beta1,
                                                   self.params.adam_beta2),
                                            amsgrad=True)
      
    else: raise RuntimeError(f"Invalid optimiser choice '{self.params.optimiser}', options are 'adam' or 'rmsprop'")

    # if self.params.loss_criterion.lower() == "smoothl1loss":
    #   self.loss_criterion = nn.SmoothL1Loss()
    # elif self.params.loss_criterion.lower() == "mseloss":
    #   self.loss_criterion = nn.MSELoss()
    # elif self.params.loss_criterion.lower() == "huber":
    #   self.loss_criterion = nn.HuberLoss()

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
    self.mlp_ac.set_device(device)
    self.mlp_ac_targ.set_device(device)
  
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
    Choose action values, a vector of values which should be [-1, +1]

    This function should return a tensor([x, y, z, ...])
    """

    # # determine if we will act randomly
    # random_action = False
    # if not test:
    #   rand = self.rng.random()
    #   eps_threshold = (self.params.eps_end 
    #     + (self.params.eps_start - self.params.eps_end)
    #     * (np.exp(-1. * decay_num / float(self.params.eps_decay))))
    #   if rand < eps_threshold: random_action = True

    if decay_num < self.params.random_start_episodes:
      return torch.tensor([2*self.rng.random() - 1 for x in range(self.n_actions)], dtype=torch.float32)

    return self.mlp_ac.act(state, test) # test=True means determinstic=True
      
  def get_save_state(self):
    """
    Return the state of the model that should be saved
    """

    to_save = {
      "name" : self.name,
      "parameters" : self.params,
      "memory" : self.memory.memory,
      "network" : self.mlp_ac, # do I want to save only the state dict?
      "optimiser_state_dict" : {
         "q_optimiser" : self.q_optimiser.state_dict(),
         "pi_optimiser" : self.pi_optimiser.state_dict()
      }
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
    self.q_optimiser.load_state_dict(saved_dict["optimiser_state_dict"]["q_optimiser"])
    self.pi_optimiser.load_state_dict(saved_dict["optimiser_state_dict"]["pi_optimiser"])

  def get_params_dict(self):
    """
    Return a dictionary of hyperparameters
    """
    param_dict = asdict(self.params)
    param_dict.update({
      "network_name" : self.mlp_ac.name
    })
    return param_dict

  def compute_loss_q(self, batch):
    """
    compute the SAC Q-losses from batch of transitions
    """

    # extract the observations etc from the batch
    o = torch.cat(batch.state)
    a = torch.cat(batch.action)
    o2 = torch.cat(batch.next_state)
    r = torch.cat(batch.reward)
    d = torch.cat(batch.terminal)

    q1 = self.mlp_ac.q1(o,a)
    q2 = self.mlp_ac.q2(o,a)

    # Bellman backup for Q functions
    with torch.no_grad():
      # Target actions come from *current* policy
      a2, logp_a2 = self.mlp_ac.pi(o2)

      # Target Q-values
      q1_pi_targ = self.mlp_ac_targ.q1(o2, a2)
      q2_pi_targ = self.mlp_ac_targ.q2(o2, a2)
      q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
      # backup = r + self.params.gamma * (1 - d) * (q_pi_targ - self.params.alpha * logp_a2)
      backup = r + self.params.gamma * (d.logical_not()) * (q_pi_targ - self.params.alpha * logp_a2)

    # MSE loss against Bellman backup
    loss_q1 = ((q1 - backup)**2).mean()
    loss_q2 = ((q2 - backup)**2).mean()
    loss_q = loss_q1 + loss_q2

    # # Useful info for logging
    # q_info = dict(Q1Vals=q1.detach().numpy(),
    #               Q2Vals=q2.detach().numpy())

    return loss_q#, q_info
  
  def compute_loss_pi(self, batch):
    """
    compute the SAC policy pi loss
    """
    o = torch.cat(batch.state)
    pi, logp_pi = self.mlp_ac.pi(o)
    q1_pi = self.mlp_ac.q1(o, pi)
    q2_pi = self.mlp_ac.q2(o, pi)
    q_pi = torch.min(q1_pi, q2_pi)

    # Entropy-regularized policy loss
    loss_pi = (self.params.alpha * logp_pi - q_pi).mean()

    # # Useful info for logging
    # pi_info = dict(LogPi=logp_pi.detach().numpy())

    return loss_pi#, pi_info

  def optimise_model(self):

    # only proceed if we have enough memory for a batch
    if len(self.memory) < self.params.batch_size: 
      return

    # only begin to optimise when enough memory is built up
    if (len(self.memory)) < self.params.min_memory_replay:
      return

    # sample and transpose a batch
    batch = self.memory.batch(self.params.batch_size)
    
    # First run one gradient descent step for Q1 and Q2
    self.q_optimiser.zero_grad()
    loss_q = self.compute_loss_q(batch)
    loss_q.backward()
    self.q_optimiser.step()

    # # Record things
    # logger.store(LossQ=loss_q.item(), **q_info)

    # Freeze Q-networks so you don't waste computational effort 
    # computing gradients for them during the policy learning step.
    for p in self.q_network_parameters:
      p.requires_grad = False

    # Next run one gradient descent step for pi.
    self.pi_optimiser.zero_grad()
    loss_pi = self.compute_loss_pi(batch)
    loss_pi.backward()
    self.pi_optimiser.step()

    # Unfreeze Q-networks so you can optimize it at next DDPG step.
    for p in self.q_network_parameters:
      p.requires_grad = True

    # # Record things
    # logger.store(LossPi=loss_pi.item(), **pi_info)

    # Finally, update target networks by polyak averaging.
    with torch.no_grad():
      for p, p_targ in zip(self.mlp_ac.parameters(), self.mlp_ac_targ.parameters()):
        # NB: We use an in-place operations "mul_", "add_" to update target
        # params, as opposed to "mul" and "add", which would make new tensors.
        p_targ.data.mul_(1 - self.params.soft_target_tau)
        p_targ.data.add_(self.params.soft_target_tau * p.data)

  def update_step(self, state, action, next_state, reward, terminal):
    """
    Run this every training action step to update the model
    """

    if self.mode != "train": return

    # print("state", state.shape)
    # print("action", action.shape)
    # print("next_state", next_state.shape)
    # print("reward", reward.shape)
    # print("terminal", terminal.shape)

    self.memory.push(
      state,
      action,
      next_state,
      reward,
      terminal
    )

    self.steps_done += 1

    if self.steps_done > self.params.update_after_steps:
      if self.steps_done % self.params.update_every_steps == 0:
        for _ in range(self.params.update_every_steps):
          # do one update for every step
          self.optimise_model()

    # avoid excessive values
    if self.steps_done > 100_000_000: self.steps_done = self.params.update_after_steps + 1

  def update_episode(self, i_episode, finished=False):
    """
    Run this at the end of every training episode
    """

    # if we are finished training
    if finished: pass

    if self.mode != "train": return

  def training_mode(self):
    """
    Put the agent into training mode
    """
    self.mode = "train"
    self.mlp_ac.training_mode()
    self.mlp_ac_targ.training_mode()

  def testing_mode(self):
    """
    Put the agent in testing mode
    """
    self.mode = "test"
    self.mlp_ac.testing_mode()
    self.mlp_ac_targ.testing_mode()


if __name__ == "__main__":

  pass