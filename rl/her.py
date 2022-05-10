#!/usr/bin/env python3

"""
What do we need to achieve with HER? In TrainDQN.py we need our memory replay
buffer to:
  - save transition data as it happens in train()
  - sample a given batch size of data in optimise_model()

In addition to this, a HER replay buffer needs to:
  - save transitions with the desired goal
  - assign the actual goal to transitions from the episode
  - assign adjusted rewards based on the actual goal
  - extract samples which combine actual goals and desired goals

Therefore, the HER replay buffer must known when an episode ends. It also
needs reward information if it is to compute the new reward, however it would
likely be better to achieve this in a different place to make this class
data-type agnostic.
"""

from collections import namedtuple, deque
import random

HER_Transition = namedtuple('HERTransition',
      ('state', 'action', 'next_state', 'reward', 'goal'))


class HER_Memory(object):

  def __init__(self, capacity):
    self.memory = deque([], maxlen=capacity)
    self.temp = deque()
    self.k = k

  def __len__(self):
    return len(self.memory)

  def end_episode(self, goals, individual_goals=None):
    """
    Ends an episode and applies the given end_goal to all transitions. The
    parameter k is the ratio of HER data to regular data, and it is set by
    the length of end_goal, eg len(goals) = 4 then each transition will be
    saved once with the desired goal and 4 times with the alternative goals.

    So the reward function in the paper is binary:
        r(s,a,g) = 0 if g - s_object < threshold
                 = -1 otherwise

    And they state s_object is simply the position of the object. So in cases
    were the final state is used as the goal, then reward will be = 0.

    The other strategies are:
        - future, replay with k random states which come from the same episode
          being replayed and were observed after it (best, recommend k=4-8)
        - episode, replay with k random states coming form the same episode
          as the transition being replayed (2nd best, recommend k=4)
        - random, replay with k random states encountered so far in the whole
          training procedure (poorest performance)

    If the HER_Transition contains both the desired goal and the current goal/
    state (assuming each state maps to a goal) then we can implement all of these
    strategies in this function - with the exception of being apply to calculate
    the new reward after the goal is changed.

    So how can we cope with the new reward? Well, the reward function in use is
    not simple or binary, and it is in the c++ code not python. But, the reward
    function uses the cnt variable essentially (and entirely if only binary rewards).
    So this is essentially a state vector from which the reward for every state
    is calculated. So we could try to use this somehow.

    For example, a goal could be a certain force vector on the object. We input
    the finger, palm, and ground forces and perhaps we take the L2 norm. Next we
    calculate the distance from these to the desired force vector and we have
    our 1 dimensional goal. Or we keep the forces seperate and consider the goal
    5 dimensional.
    
    Or even simpler, we use the cnt variable for a series of boolean options, eg
      (lifted, finger1force, finger2force, finger3force, palmforce)
    which may give a vector like (0, 0, 1, 1, 0). So this is our goal, and we
    get from our state (cnt variable) this information. Now we compare the two
    and for every match (0=0 or 1=1) we give 1/5th of the maximum reward. Or even
    simpler we only give reward=1 if every single one matches.

    Crazy idea - could we combine this multi-dimensional reward with the learned
    weights/automatic method, so we have our 5dim vectors:
      current state = (a1, a2, a3, a4, a5)
      goal = (g1, g2, g3, g4, g5)
    and the reward is currently:
      r = 0.2 * (a1==g1) + 0.2 * (a2==g2) + ... + 0.2 * (a5==g5)

    But what if instead of equal slices of 0.2 we have weights w1, w2, w3, w4, w5.
    Then we learn or optimse these weights in the style of Hu2020

    NB get rid of 'individual_goals' and implement the idea above
    """
    # transfer transitions from temporary buffer to proper memory
    for i, item in enumerate(self.temp):

      # first save the transition with the desired goal (as is)
      self.memory.append(item)

      # if we have individualised goals for each transition
      if individual_goals:
        for g in goals[i]:
          item[4] = g
          self.memory.append(item)
      else:
        for g in goals:
          item[4] = g
          self.memory.append(item)
    
    # wipe the temporary buffer
    self.temp = deque()
    
  def push(self, *args):
    """Save a transition to the temporary buffer"""
    self.temp.append(HER_Transition(*args))

  def sample(self, batch_size):
    """Get a batch of regular replay memory and HER samples"""


    return random.sample(self.memory, batch_size)

  def to(self, device):
    """Move to a device"""
    for (s1, a, s2, r, dg, ag) in self.memory:
      s1.to(device)
      a.to(device)
      s2.to(device)
      r.to(device)
      dg.to(device)
      ag.to(device)