#!/usr/bin/env python3

import numpy as np
from dataclasses import dataclass, asdict
from collections import namedtuple, deque
from copy import deepcopy
import random
import itertools
from math import floor, ceil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torchaudio.functional import lfilter

# ----- helper functions, networks, data buffers ----- #

def combined_shape(length, shape=None):
  if shape is None:
    return (length,)
  return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount, device):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    b_coeff = torch.tensor([1, 0], device=device)
    a_coeff = torch.tensor([1, float(-discount)], device=device)
    waveform = x.flip(dims=(-1,))
    return lfilter(waveform, a_coeff, b_coeff).flip(dims=(-1,))

def mlp(sizes, activation, output_activation=nn.Identity):
  layers = []
  for j in range(len(sizes)-1):
    act = activation if j < len(sizes)-2 else output_activation
    layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
  return nn.Sequential(*layers)

def calc_conv_layer_size(W, H, C, kernel_num, kernel_size, stride, padding, prnt=False):

  new_W = floor(((W - kernel_size + 2*padding) / (stride)) + 1)
  new_H = floor(((H - kernel_size + 2*padding) / (stride)) + 1)

  if prnt:
    print(f"Convolution transforms ({C}x{W}x{H}) to ({kernel_num}x{new_W}x{new_H})")

  return new_W, new_H, kernel_num

def calc_max_pool_size(W, H, C, pool_size, stride, prnt=False):

  new_W = floor(((W - pool_size) / stride) + 1)
  new_H = floor(((H - pool_size) / stride) + 1)

  if prnt:
    print(f"Max pool transforms ({C}x{W}x{H}) to ({C}x{new_W}x{new_H})")

  return new_W, new_H, C

def calc_adaptive_avg_size(W, H, C, output_size, prnt=False):

  if prnt:
    print(f"Adaptive pool transforms ({C}x{W}x{H}) to ({C}x{output_size[0]}x{output_size[1]})")

  return output_size[0], output_size[1], C

def calc_FC_layer_size(W, H, C, prnt=False):

  new_W = 1
  new_H = 1
  new_C = W * H * C

  if prnt:
    print(f"The first FC layer should take size ({C}x{W}x{H}) as ({new_C}x{new_W}x{new_H})")

  return new_W, new_H, new_C

class MixedNetwork(nn.Module):

  name = "MixedNetwork"

  def __init__(self, numeric_inputs, image_size, outputs):

    super(MixedNetwork, self).__init__()
    self.image_size = image_size
    self.numeric_inputs = numeric_inputs
    self.num_outputs = outputs

    (channel, width, height) = image_size
    self.name += f"_{width}x{height}"

    # calculate the size of the first fully connected layer (ensure settings match image_features_ below)
    prnt = False
    w, h, c = calc_conv_layer_size(width, height, channel, kernel_num=16, kernel_size=5, stride=2, padding=2, prnt=prnt)
    w, h, c = calc_max_pool_size(w, h, c, pool_size=3, stride=2, prnt=prnt)
    w, h, c = calc_conv_layer_size(w, h, c, kernel_num=64, kernel_size=5, stride=2, padding=2, prnt=prnt)
    w, h, c = calc_max_pool_size(w, h, c, pool_size=3, stride=2, prnt=prnt)
    w, h, c = calc_FC_layer_size(w, h, c, prnt=prnt)
    fc_layer_num = c

    # define the CNN to handle the images
    self.image_features_ = nn.Sequential(

      # input CxWxH, output 16xWxH
      nn.Conv2d(channel, 16, kernel_size=5, stride=2, padding=2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2),
      # nn.Dropout(),
      nn.Conv2d(16, 64, kernel_size=5, stride=2, padding=2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2),
      # nn.Dropout(),
      nn.Flatten(),
      nn.Linear(fc_layer_num, 128),
      nn.ReLU(),
      # nn.Linear(64*16, 64),
      # nn.ReLU(),
    )

    # define the MLP to handle the sensor data
    self.numeric_features_ = nn.Sequential(
      nn.Linear(numeric_inputs, 128),
      nn.ReLU(),
      # nn.Linear(150, 100),
      # nn.ReLU(),
      # nn.Linear(100, 50),
      # nn.ReLU(),
    )

    # combine the image and MLP features
    self.combined_features_ = nn.Sequential(
      nn.Linear(128 + 128, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, outputs),
    )

  def split_obs(self, obs):
    """
    Split the incoming observation into a tuple:
      (image_size, sensor_obs)
    """

    # print("obs size is", obs.shape)
    # print("img size is", self.img_size)
  
    # split up the observation vector from the image
    (img, sensors) = torch.split(obs, [self.image_size[0], 1], dim=1)
    sensors = torch.flatten(sensors, start_dim=2)

    # remove padded zeros and redundant channel dimension
    sensors = sensors[:, :, :self.numeric_inputs]
    sensors = torch.squeeze(sensors, dim=1)

    # print("img shape after split", img.shape)
    # print("sensor shape after split", sensors.shape)

    return (img, sensors)

  def forward(self, img_and_sensor_matrix):
    """
    Receives input matrix which contains both the image and the sensor value
    vector together. So for rgb images size (3x25x25) and a sensor vector of
    length 100, we would get an input (with batch_size=B):
      input.shape = (B, 4, 25, 25)
      The first 3 channels are rgb
      The last channel is the reshaped sensor vector, padded with zeros
    """
    tuple_img_sensors = self.split_obs(img_and_sensor_matrix)
    image = tuple_img_sensors[0]#.to(self.device)
    sensors = tuple_img_sensors[1]#.to(self.device)
    x = self.image_features_(image)
    # x = x.view(-1, 64*64)
    y = self.numeric_features_(sensors)
    z = torch.cat((x, y), 1)
    z = self.combined_features_(z)
    return z
  
  def to_device(self, device=None):
    """
    Set a pytorch device for the network
    """
    if device is not None:
      device = torch.device(device)
    self.image_features_.to(device)
    self.numeric_features_.to(device)
    self.combined_features_.to(device)
    
class MixedNetworkFromEncoder(nn.Module):

  name = "MixedNetworkFromEncoder"

  def __init__(self, numeric_inputs, image_size, outputs, device):

    super(MixedNetworkFromEncoder, self).__init__()
    self.device = device
    self.image_size = image_size
    self.numeric_inputs = numeric_inputs
    self.num_outputs = outputs

    (channel, width, height) = image_size
    self.name += f"_{width}x{height}"

    # calculate the size of the first fully connected layer (ensure settings match image_features_ below
    prnt = False
    w, h, c = calc_conv_layer_size(width, height, channel, kernel_num=16, kernel_size=5, stride=2, padding=2, prnt=prnt)
    w, h, c = calc_max_pool_size(w, h, c, pool_size=3, stride=2, prnt=prnt)
    w, h, c = calc_conv_layer_size(w, h, c, kernel_num=64, kernel_size=5, stride=2, padding=2, prnt=prnt)
    w, h, c = calc_max_pool_size(w, h, c, pool_size=3, stride=2, prnt=prnt)
    w, h, c = calc_FC_layer_size(w, h, c, prnt=prnt)
    fc_layer_num = c

    # define the CNN to handle the images
    self.image_features_ = nn.Sequential(

      nn.Conv2d(channel, 196, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.AdaptiveAvgPool2d((1, 1)),

    )

    # define the MLP to handle the sensor data
    self.numeric_features_ = nn.Sequential(
      nn.Linear(numeric_inputs, 128),
      nn.ReLU(inplace=True),
    )

    # combine the image and MLP features
    self.combined_features_ = nn.Sequential(
      nn.Linear(128 + 128, 128 + 128),
      nn.ReLU(inplace=True),
      nn.Linear(256, 128),
      nn.ReLU(inplace=True),
      nn.Linear(128, outputs),
    )

  def split_obs(self, obs):
    """
    Split the incoming observation into a tuple:
      (image_size, sensor_obs)
    """

    # print("obs size is", obs.shape)
    # print("img size is", self.img_size)
  
    # split up the observation vector from the image
    (img, sensors) = torch.split(obs, [self.image_size[0], 1], dim=1)
    sensors = torch.flatten(sensors, start_dim=2)

    # remove padded zeros and redundant channel dimension
    sensors = sensors[:, :, :self.numeric_inputs]
    sensors = torch.squeeze(sensors, dim=1)

    # print("img shape after split", img.shape)
    # print("sensor shape after split", sensors.shape)

    return (img, sensors)

  def forward(self, img_and_sensor_matrix):
    """
    Receives input matrix which contains both the image and the sensor value
    vector together. So for rgb images size (3x25x25) and a sensor vector of
    length 100, we would get an input (with batch_size=B):
      input.shape = (B, 4, 25, 25)
      The first 3 channels are rgb
      The last channel is the reshaped sensor vector, padded with zeros
    """
    # split the observation into image and sensor parts
    tuple_img_sensors = self.split_obs(img_and_sensor_matrix)
    image = tuple_img_sensors[0].to(self.device)
    sensors = tuple_img_sensors[1].to(self.device)

    # feed each part through a seperate head
    x = self.image_features_(image)
    y = self.numeric_features_(sensors)

    # concatenate the parts and feed into the last head of the network
    x = x.view(x.shape[0], 128) # from shape [B, 128, 1, 1] -> [B, 128]
    z = torch.cat((x, y), 1)
    z = self.combined_features_(z)

    return z

class PPOBuffer:

  def __init__(self, capacity, obs_dim, action_dim, device, gamma, lam):
    self.device = device
    self.capacity = capacity
    self.states = torch.zeros(combined_shape(capacity, obs_dim), dtype=torch.float32, device=device)
    self.actions = torch.zeros(combined_shape(capacity, action_dim), dtype=torch.float32, device=device)
    self.advantages = torch.zeros(capacity, dtype=torch.float32, device=device)
    self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
    self.returns = torch.zeros(capacity, dtype=torch.float32, device=device)
    self.values = torch.zeros(capacity, dtype=torch.float32, device=device)
    self.logprobs = torch.zeros(capacity, dtype=torch.float32, device=device)
    self.gamma = gamma
    self.lam = lam
    self.index = 0
    self.trajectory_start_idx = 0
    self.capacity = capacity

  def push(self, state, action, reward, value, logprob):
    """
    Save a transition, assume that all incoming argument are tensor([x])
    """

    assert self.index < self.capacity
    self.states[self.index] = state[0]
    self.actions[self.index] = action[0]
    self.rewards[self.index] = reward[0]
    self.values[self.index] = value[0]
    self.logprobs[self.index] = logprob[0]
    self.index += 1

  def to_torch(self, data, dtype=None):
    """
    Convert some data to a torch tensor([x])
    """
    if dtype == None: dtype = torch.float32
    return torch.tensor(data, device=self.device, dtype=dtype).unsqueeze(0)

  def __len__(self):
    return len(self.capacity)

  def all_to(self, device):
    """
    Move entire buffer to a device
    """
    self.device = device
    self.states = self.states.to(torch.device(device))
    self.actions = self.actions.to(torch.device(device))
    self.advantages = self.advantages.to(torch.device(device))
    self.rewards = self.rewards.to(torch.device(device))
    self.returns = self.returns.to(torch.device(device))
    self.values = self.values.to(torch.device(device))
    self.logprobs = self.logprobs.to(torch.device(device))

  def finish_trajectory(self, last_val=0):
    """
    Call this at the end of a trajectory, or when one gets cut off
    by an epoch ending. This looks back in the buffer to where the
    trajectory started, and uses rewards and value estimates from
    the whole trajectory to compute advantage estimates with GAE-Lambda,
    as well as compute the rewards-to-go for each state, to use as
    the targets for the value function.

    The "last_val" argument should be 0 if the trajectory ended
    because the agent reached a terminal state (died), and otherwise
    should be V(s_T), the value function estimated for the last state.
    This allows us to bootstrap the reward-to-go calculation to account
    for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
    """

    trajectory_slice = slice(self.trajectory_start_idx, self.index)
    rewards = torch.cat((self.rewards[trajectory_slice], torch.tensor([last_val], device=self.device)))
    values = torch.cat((self.values[trajectory_slice], torch.tensor([last_val], device=self.device)))

    # the next two lines implement GAE-Lambda advantage calculation
    deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
    self.advantages[trajectory_slice] = discount_cumsum(deltas, self.gamma * self.lam, self.device)

    # the next line computes rewards-to-go, to be targets for the value function
    self.returns[trajectory_slice] = discount_cumsum(rewards, self.gamma, self.device)[:-1]
    
    self.path_start_idx = self.index

  def get(self):
    """
    Call this at the end of an epoch to get all of the data from
    the buffer, with advantages appropriately normalized (shifted to have
    mean zero and std one). Also, resets some pointers in the buffer.
    """
    assert self.index == self.capacity    # buffer has to be full before you can get
    
    # reset before the next trajectory
    self.index = 0
    self.path_start_idx = 0

    # print("Buffer devices")
    # print("obs.device", self.states[0].device, self.states[-1].device)
    # print("act.device", self.actions[0].device, self.actions[-1].device)
    # print("ret.device", self.returns[0].device, self.returns[-1].device)
    # print("adv.device", self.advantages[0].device, self.advantages[-1].device)

    # the next two lines implement the advantage normalization trick
    adv_std, adv_mean = torch.std_mean(self.advantages)
    self.advantages = (self.advantages - adv_mean) / adv_std
    data = dict(obs=self.states, act=self.actions, ret=self.returns,
                adv=self.advantages, logp=self.logprobs)
    return data

# ----- base actor critic network building blocks ----- #

class Actor(nn.Module):

  def _distribution(self, obs):
    raise NotImplementedError

  def _log_prob_from_distribution(self, pi, act):
    raise NotImplementedError

  def forward(self, obs, act=None):
    # Produce action distributions for given observations, and 
    # optionally compute the log likelihood of given actions under
    # those distributions.
    pi = self._distribution(obs)
    logp_a = None
    if act is not None:
      logp_a = self._log_prob_from_distribution(pi, act)
    return pi, logp_a

class MLPCategoricalActor(Actor):
    
  def __init__(self, obs_dim, act_dim, hidden_sizes, activation, device="cpu"):
    super().__init__()
    self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
    self.to_device(device)

  def _distribution(self, obs):
    logits = self.logits_net(obs)
    return Categorical(logits=logits)

  def _log_prob_from_distribution(self, pi, act):
    return pi.log_prob(act)

  def to_device(self, device=None):
    if device is None:
      device = self.device
    self.logits_net.to(device)
    self.device = device

class MLPGaussianActor(Actor):

  def __init__(self, obs_dim, act_dim, hidden_sizes, activation, device="cpu"):
    super().__init__()
    log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
    self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
    self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
    self.to_device(device)

  def _distribution(self, obs):
    mu = self.mu_net(obs)
    # log_std stays on cpu so we need to move this to our device
    std = torch.exp(self.log_std).to(self.device)
    return Normal(mu, std)

  def _log_prob_from_distribution(self, pi, act):
    return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

  def to_device(self, device=None):
    if device is None:
      device = self.device
    self.mu_net.to(device)
    self.device = device

class MLPCritic(nn.Module):

  def __init__(self, obs_dim, hidden_sizes, activation, device="cpu"):
    super().__init__()
    self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
    self.to_device(device)

  def forward(self, obs):
    return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

  def to_device(self, device=None):
    if device is None:
      device = self.device
    self.v_net.to(device)
    self.device = device

class MLPActorCriticPG(nn.Module):

  name = "MLPActorCriticPG_"

  def __init__(self, n_obs, action_dim, continous_actions=True,
                hidden_sizes=(64,64), activation=nn.Tanh, mode="train",
                device="cpu"):
    super().__init__()

    self.n_obs = n_obs
    self.n_actions = action_dim
    self.mode = mode
    self.device = device

    # policy builder depends on action space
    if continous_actions:
      self.action_dim = action_dim
      self.pi = MLPGaussianActor(n_obs, action_dim, hidden_sizes, activation, device=device)
    else:
      self.action_dim = 1 # discrete so only one action
      self.pi = MLPCategoricalActor(n_obs, action_dim, hidden_sizes, activation, device=device)

    # build value function
    self.vf  = MLPCritic(n_obs, hidden_sizes, activation)

    if self.mode not in ["test", "train"]:
      raise RuntimeError(f"MLPActorCriticPG given mode={mode}, should be 'test' or 'train'")

    # add hidden layer size into the name
    for i in range(len(hidden_sizes)):
      if i == 0: self.name += f"{hidden_sizes[i]}"
      if i > 0: self.name += f"x{hidden_sizes[i]}"

  def step(self, obs):
    with torch.no_grad():
      pi = self.pi._distribution(obs)
      a = pi.sample()
      logp_a = self.pi._log_prob_from_distribution(pi, a)
      val = self.vf(obs)
    return a, val, logp_a

  def act(self, obs):
    return self.step(obs)[0]

  def set_device(self, device):
    self.pi.to_device(device)
    self.vf.to_device(device)
    self.device = device

  def training_mode(self):
    """
    Put the agent into training mode
    """
    self.mode = "train"
    self.pi.train()
    self.vf.train()

  def testing_mode(self):
    """
    Put the agent in testing mode
    """
    self.mode = "test"
    self.pi.eval()
    self.vf.eval()

# ----- CNN and multi-input actor critics ----- #

class CNNCategoricalActor(Actor):
    
  def __init__(self, img_dim, sensor_dim, act_dim, device):
    super().__init__()
    self.logits_net = MixedNetwork(sensor_dim, img_dim, act_dim)
    self.name = self.logits_net.name
    self.device = device
    self.logits_net.to_device(device)

  def _distribution(self, obs):
    logits = self.logits_net(obs)
    return Categorical(logits=logits)

  def _log_prob_from_distribution(self, pi, act):
    return pi.log_prob(act)
  
  def to_device(self, device=None):
    if device is None:
      device = self.device
    self.logits_net.to(device)
    self.device = device

class CNNGaussianActor(Actor):

  def __init__(self, img_dim, sensor_dim, act_dim, device):
    super().__init__()
    log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
    self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
    self.mu_net = MixedNetwork(sensor_dim, img_dim, act_dim)
    self.name = self.mu_net.name
    self.device = device
    self.mu_net.to_device(device)

  def _distribution(self, obs):
    mu = self.mu_net(obs)
    # log_std stays on cpu so we need to move this to our device
    std = torch.exp(self.log_std).to(self.device)
    return Normal(mu, std)

  def _log_prob_from_distribution(self, pi, act):
    return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

  def to_device(self, device=None):
    if device is None:
      device = self.device
    self.mu_net.to_device(device)
    self.mu_net.to(device)
    self.device = device

class CNNCritic(nn.Module):

  def __init__(self, img_dim, sensor_dim, device):
    super().__init__()
    self.v_net = MixedNetwork(sensor_dim, img_dim, 1) # act dim = 1
    self.device = device
    self.v_net.to_device(device)

  def forward(self, obs):
    return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

  def to_device(self, device=None):
    if device is None:
      device = self.device
    self.v_net.to_device(device)
    self.v_net.to(device)
    self.device = device

class CNNActorCriticPG(nn.Module):

  name = "CNNActorCriticPG_"

  def __init__(self, img_size, sensor_dim, action_dim, continous_actions=True,
               mode="train", device="cuda"):
    super().__init__()

    self.img_size = img_size
    self.sensor_dim = sensor_dim
    self.n_actions = action_dim
    self.mode = mode
    self.device = device

    self.n_obs = [img_size[0] + 1, img_size[1], img_size[2]]

    # policy builder depends on action space
    if continous_actions:
      self.action_dim = action_dim
      self.pi = CNNGaussianActor(img_size, sensor_dim, action_dim, device=device)
    else:
      self.action_dim = 1 # discrete so only one action
      self.pi = CNNCategoricalActor(img_size, sensor_dim, action_dim, device=device)

    # build value function
    self.vf  = CNNCritic(img_size, sensor_dim, device=device)

    if self.mode not in ["test", "train"]:
      raise RuntimeError(f"CNNActorCriticPG given mode={mode}, should be 'test' or 'train'")

    # # add hidden layer size into the name
    # for i in range(len(hidden_sizes)):
    #   if i == 0: self.name += f"{hidden_sizes[i]}"
    #   if i > 0: self.name += f"x{hidden_sizes[i]}"

    self.name += self.pi.name

  def step(self, obs):
    with torch.no_grad():
      pi = self.pi._distribution(obs)
      a = pi.sample()
      logp_a = self.pi._log_prob_from_distribution(pi, a)
      val = self.vf(obs)
    return a, val, logp_a

  def act(self, obs):
    return self.step(obs)[0]

  def set_device(self, device):
    self.pi.to_device(device)
    self.vf.to_device(device)
    self.device = device

  def training_mode(self):
    """
    Put the agent into training mode
    """
    self.mode = "train"
    self.pi.train()
    self.vf.train()

  def testing_mode(self):
    """
    Put the agent in testing mode
    """
    self.mode = "test"
    self.pi.eval()
    self.vf.eval()

class NetCategoricalActor(Actor):
    
  def __init__(self, network, img_dim, sensor_dim, act_dim, device, netargs={}):
    super().__init__()
    self.logits_net = network(sensor_dim, img_dim, act_dim, **netargs)
    self.name = self.logits_net.name
    self.device = device
    self.logits_net.to_device(device)

  def _distribution(self, obs):
    logits = self.logits_net(obs)
    return Categorical(logits=logits)

  def _log_prob_from_distribution(self, pi, act):
    return pi.log_prob(act)
  
  def to_device(self, device=None):
    if device is None:
      device = self.device
    self.logits_net.to(device)
    self.device = device

class NetGaussianActor(Actor):

  def __init__(self, network, img_dim, sensor_dim, act_dim, device, netargs={}):
    super().__init__()
    log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
    self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
    self.mu_net = network(sensor_dim, img_dim, act_dim, **netargs)
    self.name = self.mu_net.name
    self.device = device
    self.mu_net.to_device(device)

  def _distribution(self, obs):
    mu = self.mu_net(obs)
    # log_std stays on cpu so we need to move this to our device
    std = torch.exp(self.log_std).to(self.device)
    return Normal(mu, std)

  def _log_prob_from_distribution(self, pi, act):
    return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

  def to_device(self, device=None):
    if device is None:
      device = self.device
    self.mu_net.to(device)
    self.device = device

class NetCritic(nn.Module):

  def __init__(self, network, img_dim, sensor_dim, device, netargs={}):
    super().__init__()
    self.v_net = network(sensor_dim, img_dim, 1, **netargs) # act dim = 1
    self.device = device
    self.v_net.to_device(device)

  def forward(self, obs):
    return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

  def to_device(self, device=None):
    if device is None:
      device = self.device
    self.v_net.to(device)
    self.device = device

class NetActorCriticPG(nn.Module):

  name = "NetActorCriticPG"

  def __init__(self, network, img_size, sensor_dim, action_dim, continous_actions=True,
               mode="train", device="cuda", netargs={}):
    super().__init__()

    self.img_size = img_size
    self.sensor_dim = sensor_dim
    self.n_actions = action_dim
    self.mode = mode
    self.device = device

    self.n_obs = [img_size[0] + 1, img_size[1], img_size[2]]

    # policy builder depends on action space
    if continous_actions:
      self.action_dim = action_dim
      self.pi = NetGaussianActor(network, img_size, sensor_dim, action_dim, device=device,
                                 netargs=netargs)
    else:
      self.action_dim = 1 # discrete so only one action
      self.pi = NetCategoricalActor(network, img_size, sensor_dim, action_dim, device=device,
                                    netargs=netargs)

    # build value function
    self.vf  = NetCritic(network, img_size, sensor_dim, device=device, netargs=netargs)

    if self.mode not in ["test", "train"]:
      raise RuntimeError(f"NetActorCriticPG given mode={mode}, should be 'test' or 'train'")

    # # add hidden layer size into the name
    # for i in range(len(hidden_sizes)):
    #   if i == 0: self.name += f"{hidden_sizes[i]}"
    #   if i > 0: self.name += f"x{hidden_sizes[i]}"

    self.name += self.pi.name

  def step(self, obs):
    with torch.no_grad():
      pi = self.pi._distribution(obs)
      a = pi.sample()
      logp_a = self.pi._log_prob_from_distribution(pi, a)
      val = self.vf(obs)
    return a, val, logp_a

  def act(self, obs):
    return self.step(obs)[0]

  def set_device(self, device):
    self.pi.to_device(device)
    self.vf.to_device(device)
    self.device = device

  def training_mode(self):
    """
    Put the agent into training mode
    """
    self.mode = "train"
    self.pi.train()
    self.vf.train()

  def testing_mode(self):
    """
    Put the agent in testing mode
    """
    self.mode = "test"
    self.pi.eval()
    self.vf.eval()

# ----- MAT specific ----- #

class MATNet(nn.Module):

  name = "MATNet"

  def __init__(self, act_dim, n=20, tanh_features=10, use_extra=False, use_Z=False):

    super(MATNet, self).__init__()

    self.n = n
    self.con_n = 4 + use_Z # wrist sensor added
    self.jnt_n = 3 + use_Z # wrist motion added
    self.xyz_n = 12

    self.contacts_binary = nn.Sequential(
      nn.Linear((n+1) * self.con_n, 128), nn.ReLU(),
      nn.Linear(128, 128), nn.ReLU(),
      nn.Linear(128, 128), nn.ReLU(),
      nn.Linear(128, tanh_features), nn.Tanh()
    )

    self.delta_contacts = nn.Sequential(
      nn.Linear(n * self.con_n, 128), nn.ReLU(),
      nn.Linear(128, 128), nn.ReLU(),
      nn.Linear(128, 128), nn.ReLU(),
      nn.Linear(128, tanh_features), nn.Tanh()
    )

    self.joint_angles = nn.Sequential(
      nn.Linear((n+1) * self.jnt_n, 128), nn.ReLU(),
      nn.Linear(128, 128), nn.ReLU(),
      nn.Linear(128, 128), nn.ReLU(),
      nn.Linear(128, tanh_features), nn.Tanh()
    )

    self.delta_joints = nn.Sequential(
      nn.Linear(n * self.jnt_n, 128), nn.ReLU(),
      nn.Linear(128, 128), nn.ReLU(),
      nn.Linear(128, 128), nn.ReLU(),
      nn.Linear(128, tanh_features), nn.Tanh()
    )

    self.contacts_xyz = nn.Sequential(
      nn.Linear((n+1) * self.xyz_n, 128), nn.ReLU(),
      nn.Linear(128, 128), nn.ReLU(),
      nn.Linear(128, 128), nn.ReLU(),
      nn.Linear(128, tanh_features), nn.Tanh()
    )

    self.delta_xyz = nn.Sequential(
      nn.Linear(n * self.xyz_n, 128), nn.ReLU(),
      nn.Linear(128, 128), nn.ReLU(),
      nn.Linear(128, 128), nn.ReLU(),
      nn.Linear(128, tanh_features), nn.Tanh()
    )

    self.main_net = nn.Sequential(
      nn.Linear(tanh_features*6, 128), nn.ReLU(),
      nn.Linear(128, 128), nn.ReLU(),
      nn.Linear(128, 128), nn.ReLU(),
      nn.Linear(128, act_dim)
    )

    # determine how to divide the deltas from main data
    n2 = self.n * 2 + 1
    self.con_ind = sum([(list(range(n2*i, n2*(i+1), 2))) for i in range(self.con_n)], [])
    self.dcon_ind = sum([(list(range(1 + n2*i, n2*(i+1) - 1, 2))) for i in range(self.con_n)], [])
    self.jnt_ind = sum([(list(range(n2*i, n2*(i+1), 2))) for i in range(self.jnt_n)], [])
    self.djnt_ind = sum([(list(range(1 + n2*i, n2*(i+1) - 1, 2))) for i in range(self.jnt_n)], [])
    self.xyz_ind = sum([(list(range(n2*i, n2*(i+1), 2))) for i in range(self.xyz_n)], [])
    self.dxyz_ind = sum([(list(range(1 + n2*i, n2*(i+1) - 1, 2))) for i in range(self.xyz_n)], [])

  def split_obs(self, obs):
    """
    Split the incoming observation into parts given that the structure is that
    for each category (contacts, joints, xyz) the regular and deltas are
    alternating order
    """

    # print("obs size is", obs.shape)
    # print("estimated size is", [self.jnt_n * (2*self.n + 1), 
    #    self.con_n * (2*self.n + 1), 
    #    self.xyz_n * (2*self.n + 1)],
    #    "with sum", [self.jnt_n * (2*self.n + 1) + 
    #    self.con_n * (2*self.n + 1) + 
    #    self.xyz_n * (2*self.n + 1)])

    # order is important since some are composed of more values
    (all_contacts, all_joints, all_xyz) = torch.split(obs,
      [self.con_n * (2*self.n + 1), 
       self.jnt_n * (2*self.n + 1), 
       self.xyz_n * (2*self.n + 1)], 
       dim=1
    )

    # seperate out the alternating values
    contacts = all_contacts[:, self.con_ind]
    d_contacts = all_contacts[:, self.dcon_ind]
    joints = all_joints[:, self.jnt_ind]
    d_joints = all_joints[:, self.djnt_ind]
    xyz = all_xyz[:, self.xyz_ind]
    d_xyz = all_xyz[:, self.dxyz_ind]

    # print("contacts shape is", contacts.shape, "values are", contacts[0])
    # print("d_contacts shape is", d_contacts.shape, "values are", d_contacts[0])
    # print("joints shape is", joints.shape, "values are", joints[0])
    # print("d_joints shape is", d_joints.shape, "values are", d_joints[0])
    # print("xyz shape is", xyz.shape, "values are", xyz[0])
    # print("d_xyz shape is", d_xyz.shape, "values are", d_xyz[0])

    return (contacts, d_contacts, joints, d_joints, xyz, d_xyz)

  def forward(self, obs):
    """
    Takes the input vector, splits it up, and runs it through
    """

    # split the observation into the constituent parts
    (contacts, d_contacts, joints, d_joints, xyz, d_xyz) = self.split_obs(obs)

    # run all the inputs through networks into tanh features
    f_con = self.contacts_binary(contacts)
    f_dcon = self.delta_contacts(d_contacts)
    f_jnt = self.joint_angles(joints)
    f_djnt = self.delta_joints(d_joints)
    f_xyz = self.contacts_xyz(xyz)
    f_dxyz = self.delta_xyz(d_xyz)

    # concatenate all the features
    feature_vec = torch.concat(
      (f_con, f_dcon, f_jnt, f_djnt, f_xyz, f_dxyz), dim=1)
    
    # run through the final network
    output = self.main_net(feature_vec)

    return output
  
  def to_device(self, device=None):
    """
    Set a pytorch device for the network
    """
    if device is not None:
      device = torch.device(device)
    self.contacts_binary.to(device)
    self.delta_contacts.to(device)
    self.joint_angles.to(device)
    self.delta_joints.to(device)
    self.contacts_xyz.to(device)
    self.delta_xyz.to(device)
    self.main_net.to(device)
   
class MATActor(Actor):

  def __init__(self, obs_dim, act_dim, hidden_sizes, activation, device="cpu",
               use_extra_actions=True, use_Z=False):
    super().__init__()
    self.act_dim = act_dim
    log_std = -0.5 * np.ones(1, dtype=np.float32) # only for wrist action
    self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
    if hidden_sizes == "paper": 
      self.net = MATNet(act_dim, use_extra=use_extra_actions, use_Z=use_Z)
    else: self.net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
    self.sigmoid = nn.Sigmoid()
    self.use_extra_actions = use_extra_actions
    self.use_Z = use_Z
    self.disable_MAT_logprob = False # can override to do normal logprob
    self.use_MAT_sampling = True # can override to have regular action sampling
    self.to_device(device)

  def disable_MAT_sampling(self):
    """
    Turn off MAT action sampling
    """
    log_std = -0.5 * np.ones(self.act_dim, dtype=np.float32) # for all actions
    self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
    self.use_MAT_sampling = False
    self.disable_MAT_logprob = True

  def _distribution(self, obs):
    act = self.net(obs) # run observation through network
    if self.use_MAT_sampling:
      if self.use_extra_actions:
        sigact = self.sigmoid(act[:, :-1]) # exclude wrist
        wrist_mu = act[:,-1] # exctract wrist action mean
        # log_std stays on cpu so we need to move this to our device
        std = torch.exp(self.log_std).to(self.device) # convert from log->regular
        return (Bernoulli(probs=sigact), Normal(wrist_mu, std))
      else:
        sigact = self.sigmoid(act)
        return Bernoulli(probs=sigact)
    else:
      std = torch.exp(self.log_std).to(self.device)
      return Normal(act, std)
    
  def _bernoulli_log_prob(self, act, prob):
    """
    Calculate the log probability of a bernoulli sampled action.

    The formula is: x log(p) + (1 - x) log(1-p)

    where x is the sampled value (act) and p is the probability
    """
    return act * torch.log(prob) + (1 - act) * torch.log(1 - prob)

  def _log_prob_from_distribution(self, pi, act):
    """Calculate the log prob using the equation from the MAT paper"""
    
    # special case for debugging, disable parts of MAT
    if self.disable_MAT_logprob:
      if self.use_MAT_sampling and self.use_extra_actions:
        return pi[0].log_prob(act[:,:-1]).sum(axis=-1) + pi[1].log_prob(act[:,-1]).sum(axis=-1)
      else:
        return pi.log_prob(act).sum(axis=-1)
      
    # regular case: use termination action
    if not self.use_Z:
      if self.use_extra_actions:
        finger_probs = pi[0].probs[:,:-2] # exclude lift and reopen (pi[0] has no wrist)
        lift_probs = pi[0].probs[:,-2]
        reopen_probs = pi[0].probs[:,-1]
        fing_act = act[:,:-3]
        lift_act = act[:,-3]
        reopen_act = act[:,-2]
        wrist_act = act[:,-1] # excluded from logprob calculation
        logprob = (
          self._bernoulli_log_prob(reopen_act, reopen_probs) 
          + (1-reopen_act) * self._bernoulli_log_prob(lift_act, lift_probs)
          + (1-reopen_act) * (1-lift_act) * (
              torch.sum(self._bernoulli_log_prob(fing_act, finger_probs), axis=1)
            )
        )
      else:
        finger_probs = pi.probs[:,:-1] # exclude lift
        lift_probs = pi.probs[:,-1]
        fing_act = act[:,:-1]
        lift_act = act[:,-1]
        logprob = (
          self._bernoulli_log_prob(lift_act, lift_probs)
          + (1-lift_act) * (
            torch.sum(self._bernoulli_log_prob(fing_act, finger_probs), axis=1)
          )
        )

    # modified case: use Z height instead of termination action
    else:
      if self.use_extra_actions:
        finger_probs = pi[0].probs[:,:-1] # exclude reopen (pi[0] has no wrist)
        reopen_probs = pi[0].probs[:,-1]
        fing_act = act[:,:-2]
        reopen_act = act[:,-2]
        wrist_act = act[:,-1] # excluded from logprob calculation
        logprob = (
          self._bernoulli_log_prob(reopen_act, reopen_probs)
          + (1-reopen_act) * (
            torch.sum(self._bernoulli_log_prob(fing_act, finger_probs), axis=1)
          )
        )

        finger_probs = pi[0].probs[:,:-2] # exclude lift and reopen (pi[0] has no wrist)
        lift_probs = pi[0].probs[:,-2]
        reopen_probs = pi[0].probs[:,-1]
        fing_act = act[:,:-3]
        lift_act = act[:,-3]
        reopen_act = act[:,-2]
        wrist_act = act[:,-1] # excluded from logprob calculation
        logprob = (
          self._bernoulli_log_prob(reopen_act, reopen_probs) 
          + (1-reopen_act) * self._bernoulli_log_prob(lift_act, lift_probs)
          + (1-reopen_act) * (1-lift_act) * (
              torch.sum(self._bernoulli_log_prob(fing_act, finger_probs), axis=1)
            )
        )
      else:
        finger_probs = pi.probs[:,:] # no lift termination, so include all
        fing_act = act[:,:]
        logprob = (
          torch.sum(self._bernoulli_log_prob(fing_act, finger_probs), axis=1)
        )
        
    return logprob
  
  def to_device(self, device=None):
    if device is None:
      device = self.device
    self.net.to(device)
    self.device = device

class MATActorCriticPG(nn.Module):

  name = "MLPActorCriticPG_"

  def __init__(self, n_obs, action_dim, hidden_sizes=(128, 128, 128, 128, 128, 128),
                activation=nn.ReLU, mode="train", device="cpu", use_extra_actions=True,
                use_Z=False):
    super().__init__()

    self.n_obs = n_obs
    self.n_actions = action_dim
    self.mode = mode
    self.device = device

    self.action_dim = action_dim
    self.pi = MATActor(n_obs, action_dim, hidden_sizes, activation, device=device,
                            use_extra_actions=use_extra_actions, use_Z=use_Z)

    # build value function - use paper specified hidden sizes
    self.vf  = MLPCritic(n_obs, (128, 128, 128), activation)

    if self.mode not in ["test", "train"]:
      raise RuntimeError(f"MLPActorCriticPG given mode={mode}, should be 'test' or 'train'")

    # add hidden layer size into the name
    if hidden_sizes == "paper":
      self.name += "paper_architecture"
    else:
      for i in range(len(hidden_sizes)):
        if i == 0: self.name += f"{hidden_sizes[i]}"
        if i > 0: self.name += f"x{hidden_sizes[i]}"

  def step(self, obs):
    with torch.no_grad():
      pi = self.pi._distribution(obs)
      if self.pi.use_extra_actions:
        a_1 = pi[0].sample()
        a_2 = pi[1].sample()
        # if reopen action is triggered, lift must be zero
        reopen_triggered = a_1[:,-1] > 0.5
        a_1[reopen_triggered, -2] = 0.0
        # concatenate resultant action tensor
        a = torch.concatenate((a_1, a_2.unsqueeze(0)), dim=1)
      else:
        a = pi.sample()
      logp_a = self.pi._log_prob_from_distribution(pi, a)
      val = self.vf(obs)
    return a, val, logp_a

  def act(self, obs):
    return self.step(obs)[0]

  def set_device(self, device):
    self.pi.to_device(device)
    self.vf.to_device(device)
    self.device = device

  def training_mode(self):
    """
    Put the agent into training mode
    """
    self.mode = "train"
    self.pi.train()
    self.vf.train()

  def testing_mode(self):
    """
    Put the agent in testing mode
    """
    self.mode = "test"
    self.pi.eval()
    self.vf.eval()

# ----- proper agents ----- #

class Agent_PPO:

  name = "Agent_PPO"

  @dataclass
  class Parameters:

    # key learning hyperparameters
    learning_rate_pi: float = 3e-4
    learning_rate_vf: float = 1e-3
    gamma: float = 0.99
    steps_per_epoch: int = 4000
    clip_ratio: float = 0.2
    train_pi_iters: int = 80
    train_vf_iters: int = 80
    lam: float = 0.97
    target_kl: float = 0.01
    max_kl_ratio: float = 1.5
    use_random_action_noise: bool = True
    random_action_noise_size: float = 0.2

    # options
    optimiser: str = "adam" # adam/adamW/RMSProp
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    grad_clamp_value: float = None
    # loss_criterion: str = "MSELoss" # smoothL1Loss/MSELoss/Huber

    def update(self, newdict, hard=True):
      for key, value in newdict.items():
        if hasattr(self, key):
          setattr(self, key, value)
        elif hard: raise RuntimeError(f"incorrect key: {key}")
      
  Transition = namedtuple('Transition',
                          ('state', 'action', 'reward', 'advantage', 'log_prob', 'terminal'))

  def __init__(self, device="cpu", rngseed=None, mode="train", steps=None, debug=False):
    """
    Soft actor-critic agent for a given environment
    """

    self.params = Agent_PPO.Parameters()
    self.device = torch.device(device)
    self.rngseed = rngseed
    self.rng = np.random.default_rng(rngseed)
    self.mode = mode
    self.steps_done = 0
    self.debug = debug
    if steps is not None:
      self.params.steps_per_epoch = steps

  def init(self, network, obs_dim=None, action_dim=None, device=None):
    """
    Main initialisation of the agent, applies settings and creates network. This
    function can be called with network='loaded' to initialise settings but not
    reinitialise the networks
    """

    if network != "loaded":

      if obs_dim is None: obs_dim = network.n_obs
      if action_dim is None: action_dim = network.action_dim

      self.mlp_ac = network
      self.action_dim = action_dim
      self.n_actions = network.n_actions
      self.n_obs = obs_dim

    # create a fresh buffer for storing trajectories, rewards, actions etc
    self.buffer = PPOBuffer(self.params.steps_per_epoch, self.n_obs, self.action_dim, 
                            self.device, self.params.gamma, self.params.lam)

    # set the network to the given device
    if device is not None: self.device = device
    self.set_device(self.device)

    if self.params.optimiser.lower() == "rmsprop":
      self.vf_optimiser = torch.optim.RMSprop(self.mlp_ac.vf.parameters(), 
                                             lr=self.params.learning_rate_vf)
      self.pi_optimiser = torch.optim.RMSprop(self.mlp_ac.pi.parameters(),
                                              lr=self.params.learning_rate_pi)
    elif self.params.optimiser.lower() == "adam":
      self.vf_optimiser = torch.optim.Adam(self.mlp_ac.vf.parameters(),
                                          lr=self.params.learning_rate_vf,
                                          betas=(self.params.adam_beta1,
                                                 self.params.adam_beta2))
      self.pi_optimiser = torch.optim.Adam(self.mlp_ac.pi.parameters(),
                                           lr=self.params.learning_rate_pi,
                                           betas=(self.params.adam_beta1,
                                                  self.params.adam_beta2))
    elif self.params.optimiser.lower() == "adamw":
      self.vf_optimiser = torch.optim.AdamW(self.mlp_ac.vf.parameters(),
                                           lr=self.params.learning_rate_vf,
                                           betas=(self.params.adam_beta1,
                                                  self.params.adam_beta2),
                                           amsgrad=True)
      self.pi_optimiser = torch.optim.AdamW(self.mlp_ac.pi.parameters(),
                                            lr=self.params.learning_rate_pi,
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
      print("Agent_PPO.set_device() received request for 'cuda', but torch.cuda.is_available() == False, hence using cpu")
      device = "cpu"

    self.device = torch.device(device)

    self.buffer.all_to(self.device)
    self.mlp_ac.set_device(self.device)

    # if hasattr(self, "pi_optimiser"):
    #   # extra safety: https://github.com/pytorch/pytorch/issues/2830#issuecomment-336194949
    #   for state in self.vf_optimiser.state.values():
    #     for k, v in state.items():
    #       if isinstance(v, torch.Tensor):
    #         state[k] = v.to(self.device)
    #   for state in self.pi_optimiser.state.values():
    #     for k, v in state.items():
    #       if isinstance(v, torch.Tensor):
    #         state[k] = v.to(self.device)

  def seed(self, rngseed=None):
    """
    Set a random seed for the entire environment
    """
    if rngseed is None:
      if self.rngseed is not None: rngseed = self.rngseed
      else: rngseed = np.random.randint(0, 2_147_483_647)
    self.rngseed = rngseed
    self.rng = np.random.default_rng(rngseed)

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

    # if decay_num < self.params.random_start_episodes:
    #   return torch.tensor([2*self.rng.random() - 1 for x in range(self.action_dim)], dtype=torch.float32)

    if self.mode == "test": test = True

    action, value, logprob = self.mlp_ac.step(state)

    if not test and self.params.use_random_action_noise:
      a = self.params.random_action_noise_size
      noise = 2 * a * self.rng.random((1, len(action[0]))) - a
      # print("action is", action)
      # print("noise is", noise)
      action += torch.tensor(noise, device=action.device)
      # print("result is", action)

    # print("action", action.shape)
    # print("logprob", logprob.shape)
    # print("value", value.shape)

    # store values from before the environment steps to put into buffer
    self.last_state = state
    self.last_value = value
    self.last_logprob = logprob

    return action.squeeze(0) # from tensor([[x]]) to tensor([x])
      
  def get_save_state(self):
    """
    Return the state of the model that should be saved
    """

    to_save = {
      "name" : self.name,
      "parameters" : self.params,
      "buffer" : self.buffer,
      "network" : self.mlp_ac, # do I want to save only the state dict?
      "optimiser_state_dict" : {
         "vf_optimiser" : self.vf_optimiser.state_dict(),
         "pi_optimiser" : self.pi_optimiser.state_dict()
      }
    }

    return to_save
  
  def load_save_state(self, saved_dict, device="cpu"):
    """
    Load the agent with a given saved state
    """

    # check we are compatible
    if self.name != saved_dict["name"]:
      raise RuntimeError(f"{self.name}.load_save_state() failed as given save state has agent name: {saved_dict['name']}")
    
    # input the saved data
    self.params = saved_dict["parameters"]
    self.init(saved_dict["network"], device=device)
    self.buffer = saved_dict["buffer"]
    self.vf_optimiser.load_state_dict(saved_dict["optimiser_state_dict"]["vf_optimiser"])
    self.pi_optimiser.load_state_dict(saved_dict["optimiser_state_dict"]["pi_optimiser"])

    # move to the correct device
    self.set_device(device)

    # prepare class variables
    self.steps_done = self.buffer.index

  def get_params_dict(self):
    """
    Return a dictionary of hyperparameters
    """
    param_dict = asdict(self.params)
    param_dict.update({
      "network_name" : self.mlp_ac.name
    })
    return param_dict

  def compute_loss_v(self, data):
    """
    Compute the value function loss
    """
    obs, ret = data['obs'], data['ret']
    return ((self.mlp_ac.vf(obs) - ret)**2).mean()
  
  def compute_loss_pi(self, data):
    """
    Compute PPO policy loss
    """
    obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

    # print("act is", act)
    # print("act shape is", act.shape)
    # if act.shape == (10, 8): act = act[:,0]

    # Policy loss
    pi, logp = self.mlp_ac.pi(obs, act)
    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1-self.params.clip_ratio, 1+self.params.clip_ratio) * adv
    loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

    # TEMPORARY FIX FOR MAT, DELETE LATER
    if isinstance(pi, tuple):
      pi = pi[0]

    # Useful extra info
    approx_kl = (logp_old - logp).mean().item()
    ent = pi.entropy().mean().item()
    clipped = ratio.gt(1+self.params.clip_ratio) | ratio.lt(1-self.params.clip_ratio)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

    return loss_pi, pi_info

  def optimise_model(self):

    if self.debug:
      t1 = time.perf_counter()
      print("Agent.optimise_model() now running ...", end="", flush=True)

    data = self.buffer.get()

    pi_l_old, pi_info_old = self.compute_loss_pi(data)
    pi_l_old = pi_l_old.item()
    v_l_old = self.compute_loss_v(data).item()

    # Train policy with multiple steps of gradient descent
    for i in range(self.params.train_pi_iters):
      self.pi_optimiser.zero_grad()
      loss_pi, pi_info = self.compute_loss_pi(data)
      kl = pi_info['kl']
      if kl > self.params.max_kl_ratio * self.params.target_kl:
        # print('Early stopping at step %d due to reaching max kl.'%i)
        break
      loss_pi.backward()
      if self.params.grad_clamp_value is not None:
        torch.nn.utils.clip_grad_value_(self.mlp_ac.pi.parameters(), self.params.grad_clamp_value)
      self.pi_optimiser.step()

    # logger.store(StopIter=i)

    # Value function learning
    for i in range(self.params.train_vf_iters):
      self.vf_optimiser.zero_grad()
      loss_v = self.compute_loss_v(data)
      loss_v.backward()
      if self.params.grad_clamp_value is not None:
        torch.nn.utils.clip_grad_value_(self.mlp_ac.vf.parameters(), self.params.grad_clamp_value)
      self.vf_optimiser.step()

    # Log changes from update
    # kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
    # logger.store(LossPi=pi_l_old, LossV=v_l_old,
    #               KL=kl, Entropy=ent, ClipFrac=cf,
    #               DeltaLossPi=(loss_pi.item() - pi_l_old),
    #               DeltaLossV=(loss_v.item() - v_l_old))
      
    if self.debug:
      t2 = time.perf_counter()
      print(f"finished after {(t2 - t1):.3f} seconds")

  def update_step(self, state, action, next_state, reward, terminal, truncated):
    """
    Run this every training action step to update the model
    """

    if self.mode != "train": return

    # print("state", state.shape)
    # print("action", action.shape)
    # print("next_state", next_state.shape)
    # print("reward", reward.shape)
    # print("terminal", terminal.shape)

    # save buffer, but we use the state/value/logprob from before the action was
    # applied in the environment, we will get this 'state' next step
    self.buffer.push(
      self.last_state,
      action,
      reward,
      self.last_value,
      self.last_logprob
    )

    self.steps_done += 1

    epoch_ended = (self.steps_done >= self.params.steps_per_epoch)
    terminal = bool(terminal.item())

    # if trajectory did not reach terminal state, bootstrap value target
    if truncated or epoch_ended:
      _, end_value, _ = self.mlp_ac.step(self.last_state)
    else:
      end_value = 0

    # add the trajectory to the buffer
    self.buffer.finish_trajectory(end_value)

    # update the model and reset the epoch step counter
    if epoch_ended:
      self.optimise_model()
      self.steps_done = 0

  def update_episode(self, i_episode, finished=False):
    """
    Run this at the end of every training episode
    """
    pass

  def training_mode(self):
    """
    Put the agent into training mode
    """
    self.mode = "train"
    self.mlp_ac.training_mode()

  def testing_mode(self):
    """
    Put the agent in testing mode
    """
    self.mode = "test"
    self.mlp_ac.testing_mode()

class Agent_PPO_MAT:

  name = "Agent_PPO_MAT"

  @dataclass
  class Parameters:

    # key learning hyperparameters
    learning_rate_pi: float = 1e-4 # paper specified
    learning_rate_vf: float = 1e-4 # paper specified
    gamma: float = 0.999 # paper specified
    steps_per_epoch: int = 300 # paper specified (based on my understanding)
    clip_ratio: float = 0.2 # paper specified
    train_pi_iters: int = 10 # paper specified (based on my understanding)
    train_vf_iters: int = 10 # paper specified (based on my understanding)
    lam: float = 0.95 # paper specified
    temperature_alpha: float = 5e-4 # paper specified
    optimiser: str = "adam" # paper specified
    adam_beta1: float = 0.9 # paper implied
    adam_beta2: float = 0.999 # paper implied
    grad_clamp_value: float = 200 # paper specified

    # options
    use_extra_actions: bool = True

    # options not in use for exact paper implementation
    use_random_action_noise: bool = False # paper implied
    use_KL_early_stop: bool = False # paper implied
    target_kl: float = 0.01
    max_kl_ratio: float = 1.5
    random_action_noise_size: float = 0.05

    def update(self, newdict, hard=True):
      for key, value in newdict.items():
        if hasattr(self, key):
          setattr(self, key, value)
        elif hard: raise RuntimeError(f"incorrect key: {key}")
      
  Transition = namedtuple('Transition',
                          ('state', 'action', 'reward', 'advantage', 'log_prob', 'terminal'))

  def __init__(self, device="cpu", rngseed=None, mode="train", steps=None, debug=False):
    """
    Soft actor-critic agent for a given environment
    """

    self.params = Agent_PPO_MAT.Parameters()
    self.device = torch.device(device)
    self.rngseed = rngseed
    self.rng = np.random.default_rng(rngseed)
    self.mode = mode
    self.steps_done = 0
    self.debug = debug
    if steps is not None:
      self.params.steps_per_epoch = steps

  def init(self, network, obs_dim=None, action_dim=None, device=None):
    """
    Main initialisation of the agent, applies settings and creates network. This
    function can be called with network='loaded' to initialise settings but not
    reinitialise the networks
    """

    if network != "loaded":

      if obs_dim is None: obs_dim = network.n_obs
      if action_dim is None: action_dim = network.action_dim

      self.mlp_ac = network
      self.action_dim = action_dim
      self.n_actions = network.n_actions
      self.n_obs = obs_dim

    # create a fresh buffer for storing trajectories, rewards, actions etc
    self.buffer = PPOBuffer(self.params.steps_per_epoch, self.n_obs, self.action_dim, 
                            self.device, self.params.gamma, self.params.lam)

    # set the network to the given device
    if device is not None: self.device = device
    self.set_device(self.device)

    if self.params.optimiser.lower() == "rmsprop":
      self.vf_optimiser = torch.optim.RMSprop(self.mlp_ac.vf.parameters(), 
                                             lr=self.params.learning_rate_vf)
      self.pi_optimiser = torch.optim.RMSprop(self.mlp_ac.pi.parameters(),
                                              lr=self.params.learning_rate_pi)
    elif self.params.optimiser.lower() == "adam":
      self.vf_optimiser = torch.optim.Adam(self.mlp_ac.vf.parameters(),
                                          lr=self.params.learning_rate_vf,
                                          betas=(self.params.adam_beta1,
                                                 self.params.adam_beta2))
      self.pi_optimiser = torch.optim.Adam(self.mlp_ac.pi.parameters(),
                                           lr=self.params.learning_rate_pi,
                                           betas=(self.params.adam_beta1,
                                                  self.params.adam_beta2))
    elif self.params.optimiser.lower() == "adamw":
      self.vf_optimiser = torch.optim.AdamW(self.mlp_ac.vf.parameters(),
                                           lr=self.params.learning_rate_vf,
                                           betas=(self.params.adam_beta1,
                                                  self.params.adam_beta2),
                                           amsgrad=True)
      self.pi_optimiser = torch.optim.AdamW(self.mlp_ac.pi.parameters(),
                                            lr=self.params.learning_rate_pi,
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
      print("Agent_PPO.set_device() received request for 'cuda', but torch.cuda.is_available() == False, hence using cpu")
      device = "cpu"

    self.device = torch.device(device)

    self.buffer.all_to(self.device)
    self.mlp_ac.set_device(self.device)

    # if hasattr(self, "pi_optimiser"):
    #   # extra safety: https://github.com/pytorch/pytorch/issues/2830#issuecomment-336194949
    #   for state in self.vf_optimiser.state.values():
    #     for k, v in state.items():
    #       if isinstance(v, torch.Tensor):
    #         state[k] = v.to(self.device)
    #   for state in self.pi_optimiser.state.values():
    #     for k, v in state.items():
    #       if isinstance(v, torch.Tensor):
    #         state[k] = v.to(self.device)

  def seed(self, rngseed=None):
    """
    Set a random seed for the entire environment
    """
    if rngseed is None:
      if self.rngseed is not None: rngseed = self.rngseed
      else: rngseed = np.random.randint(0, 2_147_483_647)
    self.rngseed = rngseed
    self.rng = np.random.default_rng(rngseed)

  def select_action(self, state, decay_num, test=False):
    """
    Choose action values, a vector of values which should be [-1, +1]

    This function should return a tensor([x, y, z, ...])
    """

    if self.mode == "test": test = True

    action, value, logprob = self.mlp_ac.step(state)

    if not test and self.params.use_random_action_noise:
      a = self.params.random_action_noise_size
      noise = 2 * a * self.rng.random((1, len(action[0]))) - a
      action += torch.tensor(noise, device=action.device)

    # store values from before the environment steps to put into buffer
    self.last_state = state
    self.last_value = value
    self.last_logprob = logprob 

    return action.squeeze(0) # from tensor([[x]]) to tensor([x])
      
  def get_save_state(self):
    """
    Return the state of the model that should be saved
    """

    to_save = {
      "name" : self.name,
      "parameters" : self.params,
      "buffer" : self.buffer,
      "network" : self.mlp_ac, # do I want to save only the state dict?
      "optimiser_state_dict" : {
         "vf_optimiser" : self.vf_optimiser.state_dict(),
         "pi_optimiser" : self.pi_optimiser.state_dict()
      }
    }

    return to_save
  
  def load_save_state(self, saved_dict, device="cpu"):
    """
    Load the agent with a given saved state
    """

    # check we are compatible
    if self.name != saved_dict["name"]:
      raise RuntimeError(f"{self.name}.load_save_state() failed as given save state has agent name: {saved_dict['name']}")
    
    # input the saved data
    self.params = saved_dict["parameters"]
    self.init(saved_dict["network"], device=device)
    self.buffer = saved_dict["buffer"]
    self.vf_optimiser.load_state_dict(saved_dict["optimiser_state_dict"]["vf_optimiser"])
    self.pi_optimiser.load_state_dict(saved_dict["optimiser_state_dict"]["pi_optimiser"])

    # move to the correct device
    self.set_device(device)

    # prepare class variables
    self.steps_done = self.buffer.index

    # make MATActor forward compatible
    if not hasattr(self.mlp_ac.pi, "disable_MAT_logprob"):
      self.mlp_ac.pi.disable_MAT_logprob = False
    if not hasattr(self.mlp_ac.pi, "use_Z"):
      self.mlp_ac.pi.use_Z = False

  def get_params_dict(self):
    """
    Return a dictionary of hyperparameters
    """
    param_dict = asdict(self.params)
    param_dict.update({
      "network_name" : self.mlp_ac.name
    })
    return param_dict

  def compute_loss_v(self, data):
    """
    Compute the value function loss
    """
    obs, ret = data['obs'], data['ret']
    return ((self.mlp_ac.vf(obs) - ret)**2).mean()
  
  def compute_loss_pi(self, data):
    """
    Compute PPO policy loss for MAT with soft term
    """
    obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

    # Policy loss
    pi, logp = self.mlp_ac.pi(obs, act)
    ratio = torch.exp(logp - logp_old)
    soft_advantage = adv - self.params.temperature_alpha * logp # MAT addition, to make soft
    clip_ratio = torch.clamp(ratio, 1-self.params.clip_ratio, 1+self.params.clip_ratio)
    min_ratio = torch.min(ratio, clip_ratio)
    loss_pi = -(min_ratio * soft_advantage).mean()

    # Useful extra info - should not be used with MAT!
    if self.mlp_ac.pi.use_extra_actions: pi = pi[0]
    approx_kl = (logp_old - logp).mean().item()
    ent = pi.entropy().mean().item()
    clipped = ratio.gt(1+self.params.clip_ratio) | ratio.lt(1-self.params.clip_ratio)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

    return loss_pi, pi_info

  def optimise_model(self):

    if self.debug:
      t1 = time.perf_counter()
      print("Agent.optimise_model() now running ...", end="", flush=True)

    data = self.buffer.get()

    pi_l_old, pi_info_old = self.compute_loss_pi(data)
    pi_l_old = pi_l_old.item()
    v_l_old = self.compute_loss_v(data).item()

    # Train policy with multiple steps of gradient descent
    for i in range(self.params.train_pi_iters):
      self.pi_optimiser.zero_grad()
      loss_pi, pi_info = self.compute_loss_pi(data)
      kl = pi_info['kl']
      if self.params.use_KL_early_stop:
        if kl > self.params.max_kl_ratio * self.params.target_kl:
          # print('Early stopping at step %d due to reaching max kl.'%i)
          break
      loss_pi.backward()
      if self.params.grad_clamp_value is not None:
        torch.nn.utils.clip_grad_value_(self.mlp_ac.pi.parameters(), self.params.grad_clamp_value)
      self.pi_optimiser.step()

    # Value function learning
    for i in range(self.params.train_vf_iters):
      self.vf_optimiser.zero_grad()
      loss_v = self.compute_loss_v(data)
      loss_v.backward()
      if self.params.grad_clamp_value is not None:
        torch.nn.utils.clip_grad_value_(self.mlp_ac.vf.parameters(), self.params.grad_clamp_value)
      self.vf_optimiser.step()
      
    if self.debug:
      t2 = time.perf_counter()
      print(f"finished after {(t2 - t1):.3f} seconds")

  def update_step(self, state, action, next_state, reward, terminal, truncated):
    """
    Run this every training action step to update the model
    """

    if self.mode != "train": return

    # print("state", state.shape)
    # print("action", action.shape)
    # print("next_state", next_state.shape)
    # print("reward", reward.shape)
    # print("terminal", terminal.shape)

    # save buffer, but we use the state/value/logprob from before the action was
    # applied in the environment, we will get this 'state' next step
    self.buffer.push(
      self.last_state,
      action,
      reward,
      self.last_value,
      self.last_logprob * (not truncated) # no prob if max_step_num exceeded
    )

    self.steps_done += 1

    epoch_ended = (self.steps_done >= self.params.steps_per_epoch)
    terminal = bool(terminal.item())

    # if trajectory did not reach terminal state, bootstrap value target
    if truncated or epoch_ended:
      _, end_value, _ = self.mlp_ac.step(self.last_state)
    else:
      end_value = 0

    # add the trajectory to the buffer
    self.buffer.finish_trajectory(end_value)

    # update the model and reset the epoch step counter
    if epoch_ended:
      self.optimise_model()
      self.steps_done = 0

  def update_episode(self, i_episode, finished=False):
    """
    Run this at the end of every training episode
    """
    pass

  def training_mode(self):
    """
    Put the agent into training mode
    """
    self.mode = "train"
    self.mlp_ac.training_mode()

  def testing_mode(self):
    """
    Put the agent in testing mode
    """
    self.mode = "test"
    self.mlp_ac.testing_mode()

if __name__ == "__main__":

  network = CNNActorCriticPG((3, 40, 40), 
                               64, 6, continous_actions=True, 
                               device="cpu")
