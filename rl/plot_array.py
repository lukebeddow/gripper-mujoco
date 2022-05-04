#!/usr/bin/env python3

import os
import networks
from TrainDQN import TrainDQN
from matplotlib import pyplot as plt

if __name__ == "__main__":

  # ----- set parameters ----- #

  # what network
  net = networks.DQN_3L60

  # what training session
  training = "train_cluster_29-04"

  # ----- plot ----- #

  cluster = False
  model = TrainDQN(cluster=cluster)
  model.init(net)

  plt.ion()

  folderpath = "/home/luke/cluster/rl/models/dqn/" + model.policy_net.name() + "/"
  dirs = [x for x in os.listdir(folderpath) if x.startswith(training)]
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

