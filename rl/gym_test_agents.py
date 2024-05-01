#!/usr/bin/env python3

import gymnasium as gym
import torch
import random
from Trainer import Trainer
from agents.DQN import Agent_DQN
from agents.ActorCritic import MLPActorCriticAC, Agent_SAC
from agents.PolicyGradient import MLPActorCriticPG, Agent_PPO
import networks

# wrapper to use gym environments
class GymHandler():
  def __init__(self, gymenv, rngseed=None):
    self.env = gymenv
    obs, info = self.env.reset()
    self.n_actions = self.env.action_space.shape[0]
    self.n_obs = len(obs)
    self.rngseed = rngseed
    self.seed()
    self.continous_actions = False
  def step(self, action):
    return self.env.step(action)
  def reset(self, rngseed=None):
    obs, info = self.env.reset(seed=rngseed)
    return obs
  def seed(self, rngseed=None):
    if rngseed is None:
      if self.rngseed is not None: rngseed = self.rngseed
      else: rngseed = random.randint(0, 2_147_483_647)
    self.rngseed = rngseed
    self.env.action_space.seed(self.rngseed)
    self.env.reset(seed=self.rngseed)
  def close(self):
    self.env.close()
  def render(self):
    self.env.render()
  def get_params_dict(self):
    return {
      "n_actions" : self.n_actions,
      "n_observations" : self.n_obs,
      "env_type" : "gym env"
    }
  
# agent hyperparameters
hypers = {
  "Agent_DQN" : {
    "learning_rate" : 5e-5,
    "gamma" : 0.999,
    "batch_size" : 128,
    "eps_start" : 0.9,
    "eps_end" : 0.05,
    "eps_decay" : 4000,
    "target_update" : 50,
    "optimiser" : "adam",
    "adam_beta1" : 0.9,
    "adam_beta2" : 0.999,
    "min_memory_replay" : 5000,
    "memory_replay" : 75_000,
    "soft_target_update" : False,
    "soft_target_tau" : 0.05,
    "grad_clamp_value" : 1.0,
    "loss_criterion" : "smoothL1Loss"
  },

  "Agent_SAC" : {
    "learning_rate" : 5e-3,
    "gamma" : 0.999,
    "alpha" : 0.2,
    "batch_size" : 128,
    "update_after_steps" : 10000,
    "update_every_steps" : 1000,
    "random_start_episodes" : 10,
    "optimiser" : "adam",
    "adam_beta1" : 0.9,
    "adam_beta2" : 0.999,
    "min_memory_replay" : 5000,
    "memory_replay" : 25_000,
    "soft_target_tau" : 0.05,
  },

  "Agent_PPO" : {
    "learning_rate_pi" : 5e-4,
    "learning_rate_vf" : 5e-4,
    "gamma" : 0.99,
    "steps_per_epoch" : 6000,
    "clip_ratio" : 0.2,
    "train_pi_iters" : 80,
    "train_vf_iters" : 80,
    "lam" : 0.97,
    "target_kl" : 0.01,
    "max_kl_ratio" : 1.5,
    "use_random_action_noise" : True,
    "random_action_noise_size" : 0.05,
    "optimiser" : "adam",
    "adam_beta1" : 0.9,
    "adam_beta2" : 0.999,
    "grad_clamp_value" : None,
  },
}

if __name__ == "__main__":

  # master seed, torch seed must be set before network creation (random initialisation)
  rngseed = None
  strict_seed = False
  if strict_seed:
    if rngseed is None: rngseed = random.randint(0, 2_147_483_647)
    torch.manual_seed(rngseed)

  # training device
  device = "cpu"
  continous = True
  log_level = 2

  # make the environment
  # env = gym.make("LunarLander-v2") #, render_mode="human")
  if continous:
    env = gym.make('MountainCarContinuous-v0')
  else:
    env = gym.make('MountainCar-v0')
  env = GymHandler(env)

  # make the agent
  agent = Agent_SAC
  layers = [128 for i in range(2)]
  if agent.name == "Agent_DQN":
    if continous: raise RuntimeError("DQN is for discrete")
    network = networks.VariableNetwork([env.n_obs, *layers, env.n_actions], device=device)
  elif agent.name == "Agent_SAC":
    if not continous: raise RuntimeError("SAC is for continous")
    network = MLPActorCriticAC(env.n_obs, env.n_actions, hidden_sizes=layers)
  elif agent.name == "Agent_PPO":
    network = MLPActorCriticPG(env.n_obs, env.n_actions, hidden_sizes=layers,
                               continous_actions=continous)

  agent = agent(device=device, debug=True if log_level>1 else False)
  agent.params.update(hypers[agent.name])
  agent.init(network)

  # create the trainer
  trainer = Trainer(agent, env, rngseed=rngseed, device=device, plot=False, save=False,
                    strict_seed=strict_seed, print_avg_return=True)
  
  # prepare settings and proceed with training
  trainer.params.num_episodes = 10000
  trainer.log_level = log_level
  trainer.log_rate_for_episodes = 1
  trainer.track.avg_num = 5
  # trainer.load("run_16-28", group_name="19-09-23")
  trainer.train()