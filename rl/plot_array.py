#!/usr/bin/env python3

import os
import networks
from TrainDQN import TrainDQN
from matplotlib import pyplot as plt

if __name__ == "__main__":

  # ----- set parameters ----- #

  # what network
  net = networks.DQN_4L60

  # what training session
  training = "27-05-22-11:49"

  # ----- plot ----- #

  model = TrainDQN(use_wandb=False)
  model.init(net)

  plt.ion()

  folderpath = "/home/luke/cluster/rl/models/dqn/DQN_3L60/"
  dirs = [x for x in os.listdir(folderpath) if x.endswith(training)]
  print("dirs to check in are:", dirs)

  for d in dirs:

    # plt.figure()
    model.fig, model.axs = plt.subplots(2, 1)

    print("Load the folder:", d)
    try:
      model.load(folderpath=folderpath, foldername=d)
      model.plot(pltname=f"{d}")
      plt.show()
    except Exception as e:
      model.fig.suptitle(d)
      print("failed:", e)

  plt.ioff()
  plt.show()

