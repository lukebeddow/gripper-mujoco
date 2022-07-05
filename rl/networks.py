import torch
import torch.nn as nn

class DQN_60x60(nn.Module):

  def __init__(self, inputs, outputs, device):
    super(DQN_60x60, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, 60)
    self.linear2 = torch.nn.Linear(60, outputs)
    self.activation = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1) # we use dim=1 because our data is like this: [data]
                                           # ie: [[0, 0.1, 0.2, 0.3]], so dim=1 means peel off
                                           # the outer layer. If our data was [0, 0.1,...] we
                                           # would use dim=0

  def name(self):
    # needed for the 'save' and 'load' functions
    return "DQN_60x60"

  def forward(self, x):
    x = x.to(self.device)
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    x = self.softmax(x)
    return x

class DQN_2L60(nn.Module):

  name = "DQN_2L60"

  def __init__(self, inputs, outputs, device):
    super(DQN_2L60, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, 60)
    self.linear2 = torch.nn.Linear(60, 60)
    self.linear3 = torch.nn.Linear(60, outputs)
    self.activation = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, x):
    x = x.to(self.device)
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    x = self.activation(x)
    x = self.linear3(x)
    x = self.softmax(x)

    return x

class DQN_3L60(nn.Module):

  name = "DQN_3L60"

  def __init__(self, inputs, outputs, device):
    super(DQN_3L60, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, 60)
    self.linear2 = torch.nn.Linear(60, 60)
    self.linear3 = torch.nn.Linear(60, 60)
    self.linear4 = torch.nn.Linear(60, outputs)
    self.activation = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, x):
    x = x.to(self.device)
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    x = self.activation(x)
    x = self.linear3(x)
    x = self.activation(x)
    x = self.linear4(x)
    x = self.softmax(x)

    return x

class DQN_4L60(nn.Module):

  name = "DQN_4L60"

  def __init__(self, inputs, outputs, device):
    super(DQN_4L60, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, 60)
    self.linear2 = torch.nn.Linear(60, 60)
    self.linear3 = torch.nn.Linear(60, 60)
    self.linear4 = torch.nn.Linear(60, 60)
    self.linear5 = torch.nn.Linear(60, outputs)
    self.activation = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, x):
    x = x.to(self.device)
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    x = self.activation(x)
    x = self.linear3(x)
    x = self.activation(x)
    x = self.linear4(x)
    x = self.activation(x)
    x = self.linear5(x)
    x = self.softmax(x)

    return x


class DQN_5L60(nn.Module):

  name = "DQN_5L60"

  def __init__(self, inputs, outputs, device):
    super(DQN_5L60, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, 60)
    self.linear2 = torch.nn.Linear(60, 60)
    self.linear3 = torch.nn.Linear(60, 60)
    self.linear4 = torch.nn.Linear(60, 60)
    self.linear5 = torch.nn.Linear(60, 60)
    self.linear6 = torch.nn.Linear(60, outputs)
    self.activation = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, x):
    x = x.to(self.device)
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    x = self.activation(x)
    x = self.linear3(x)
    x = self.activation(x)
    x = self.linear4(x)
    x = self.activation(x)
    x = self.linear5(x)
    x = self.activation(x)
    x = self.linear6(x)
    x = self.softmax(x)

    return x


# ----- for saving and loading models ----- #

# import os
# import pickle

# filepath = "/home/luke/mymujoco/rl/models/dqn/"
# filename = "{0}_{1:03d}.pickle"
# log_level = 0

# def set_filepath(path):
#   global filepath
#   filepath = path
#   if log_level > 1: print("The new save filepath is", filepath)

# def save(net):

#     global filepath, filename
#     i = 1

#     folderpath = filepath + net.name + "/"
#     if not os.path.exists(folderpath):
#       os.makedirs(folderpath)
#       if log_level > 1: print(f"Created a new directory called {folderpath}")

#     file = folderpath + filename.format(net.name, i)

#     while os.path.exists(file):
#         i += 1
#         file = folderpath + filename.format(net.name, i)

#     if log_level > 0:
#       print("Saving file {} with pickle".format(filename.format(net.name, i)))

#     # with open(file, 'wb') as f:
#     #     pickle.dump(net, f)
#     #     print("Finished saving")

#     torch.save(net, file)
#     if log_level > 1: print("Finished saving")

#     return file

# def load(netname, id=None):

#     global filepath, filename

#     folderpath = filepath + netname + "/"
#     if not os.path.exists(folderpath):
#       if log_level > 0:
#         print(f"The folder path {folderpath} does not exist - nothing loaded")
#       return None
    
#     if id == None:
#       i = 1
#       while os.path.exists(folderpath + filename.format(netname, i)):
#           i += 1
#       id = i - 1
        
#     file = folderpath + filename.format(netname, id)

#     if log_level > 0:
#       print("Loading file {} with pickle".format(filename.format(netname, id)))

#     # with open(file, 'rb') as f:
#     #     loaded_model = pickle.load(f)
#     #     print("Finished loading")

#     loaded_model = torch.load(file)
#     if log_level > 1: print("Finished loading")

#     return loaded_model