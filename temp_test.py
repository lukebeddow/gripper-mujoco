#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def get_reward(force):

  min = 1.0
  max = 10.0
  target = 5.0
  reward = 1.0

  if force > min and force < max:

    # overshoot
    if force > target:
      scale = max - force
      rise = max - target

    # undershoot
    else:
      scale = force - min
      rise = target - min

    # linearly scale
    return reward * (scale / rise)

  return 0.0


forces = []
rewards = []

for i in range(110):

  forces.append(i * 0.1)
  rewards.append(get_reward(i * 0.1))

plt.figure()
plt.plot(forces, rewards)
plt.show()

