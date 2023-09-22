import torch
import torch.nn as nn
from math import floor, ceil

# --- variable architecture --- #

class VariableNetwork(nn.Module):

  name = "VariableNetwork_"

  def __init__(self, layers, device):

    super(VariableNetwork, self).__init__()
    self.device = device
    self.n_input = layers[0]
    self.n_output = layers[-1]

    self.linear = []
    for i in range(len(layers) - 1):
      self.linear.append(nn.Linear(layers[i], layers[i + 1]))
      if i == 1: self.name += f"{layers[i]}"
      if i > 1: self.name += f"x{layers[i]}"

    self.linear = nn.ModuleList(self.linear)
    self.activation = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):

    x = x.to(self.device)

    for i in range(len(self.linear) - 1):
      x = self.linear[i](x)
      x = self.activation(x)

    x = self.linear[len(self.linear) - 1](x)
    x = self.softmax(x)

    return x

class DQN_variable(nn.Module):

  name = "DQN_"

  def __init__(self, layers, device):

    super(DQN_variable, self).__init__()
    self.device = device

    self.linear = []
    for i in range(len(layers) - 1):
      self.linear.append(torch.nn.Linear(layers[i], layers[i + 1]))
      if i == 1: self.name += f"{layers[i]}"
      if i > 1: self.name += f"x{layers[i]}"

    self.linear = nn.ModuleList(self.linear)
    self.activation = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, x):

    x = x.to(self.device)

    for i in range(len(self.linear) - 1):
      x = self.linear[i](x)
      x = self.activation(x)

    x = self.linear[i + 1](x)
    x = self.softmax(x)

    return x

# --- CNN and sensor data --- #

def calc_conv_layer_size(W, H, C, kernel_num, kernel_size, stride, padding, print=True):

  new_W = floor(((W - kernel_size + 2*padding) / (stride)) + 1)
  new_H = floor(((H - kernel_size + 2*padding) / (stride)) + 1)

  if print:
    print(f"Convolution transforms ({C}x{W}x{H}) to ({kernel_num}x{new_W}x{new_H})")

  return new_W, new_H, kernel_num

def calc_max_pool_size(W, H, C, pool_size, stride, print=True):

  new_W = floor(((W - pool_size) / stride) + 1)
  new_H = floor(((H - pool_size) / stride) + 1)

  if print:
    print(f"Max pool transforms ({C}x{W}x{H}) to ({C}x{new_W}x{new_H})")

  return new_W, new_H, C

def calc_FC_layer_size(W, H, C, print=True):

  new_W = 1
  new_H = 1
  new_C = W * H * C

  if print:
    print(f"The first FC layer should take size ({C}x{W}x{H}) as ({new_C}x{new_W}x{new_H})")

  return new_W, new_H, new_C


class MixedNetwork(nn.Module):

  name = "MixedNetwork"

  def __init__(self, numeric_inputs, image_channels, outputs, device, fcn_size):

    super(MixedNetwork, self).__init__()
    self.device = device

    # define the CNN to handle the images
    self.image_features_ = nn.Sequential(

      # input CxWxH, output 16xWxH
      nn.Conv2d(image_channels, 16, kernel_size=5, stride=2, padding=2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2),
      # nn.Dropout(),
      nn.Conv2d(16, 64, kernel_size=5, stride=1, padding=2),
      nn.ReLU(),
      # output 64*64
      nn.MaxPool2d(kernel_size=3, stride=2),
      # nn.Dropout(),
      nn.Flatten(),
      nn.Linear(fcn_size, 150),
      nn.ReLU(),
      # nn.Linear(64*16, 64),
      # nn.ReLU(),
    )

    # define the MLP to handle the sensor data
    self.numeric_features_ = nn.Sequential(
      nn.Linear(numeric_inputs, 150),
      nn.ReLU(),
      # nn.Linear(150, 100),
      # nn.ReLU(),
      # nn.Linear(100, 50),
      # nn.ReLU(),
    )

    # combine the image and MLP features
    self.combined_features_ = nn.Sequential(
      nn.Linear(150 + 150, 300),
      nn.ReLU(),
      nn.Linear(300, 100),
      nn.ReLU(),
      nn.Linear(100, outputs),
      nn.Softmax(dim=1),
    )

  def forward(self, tuple_img_sensors):
    image = tuple_img_sensors[0].to(self.device)
    sensors = tuple_img_sensors[1].to(self.device)
    x = self.image_features_(image)
    # x = x.view(-1, 64*64)
    y = self.numeric_features_(sensors)
    z = torch.cat((x, y), 1)
    z = self.combined_features_(z)
    return z

class MixedNetwork2(nn.Module):

  name = "MixedNetwork2"

  def __init__(self, numeric_inputs, image_size, outputs, device):

    super(MixedNetwork2, self).__init__()
    self.device = device

    (channel, width, height) = image_size
    self.name += f"_{width}x{height}"

    # calculate the size of the first fully connected layer (ensure settings match image_features_ below)
    w, h, c = calc_conv_layer_size(width, height, channel, kernel_num=16, kernel_size=5, stride=2, padding=2, print=False)
    w, h, c = calc_max_pool_size(w, h, c, pool_size=3, stride=2, print=False)
    w, h, c = calc_conv_layer_size(w, h, c, kernel_num=64, kernel_size=5, stride=2, padding=2, print=False)
    w, h, c = calc_max_pool_size(w, h, c, pool_size=3, stride=2, print=False)
    w, h, c = calc_FC_layer_size(w, h, c, print=False)
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
      nn.Linear(128 + 128, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, outputs),
      nn.Softmax(dim=1),
    )

  def forward(self, tuple_img_sensors):
    image = tuple_img_sensors[0].to(self.device)
    sensors = tuple_img_sensors[1].to(self.device)
    x = self.image_features_(image)
    # x = x.view(-1, 64*64)
    y = self.numeric_features_(sensors)
    z = torch.cat((x, y), 1)
    z = self.combined_features_(z)
    return z
  
class CNN_only(nn.Module):

  name = "CNN_only"

  def __init__(self, image_size, outputs, device):

    super(CNN_only, self).__init__()
    self.device = device

    (channel, width, height) = image_size
    self.name += f"_{width}x{height}"

    # calculate the size of the first fully connected layer (ensure settings match image_features_ below)
    w, h, c = calc_conv_layer_size(width, height, channel, kernel_num=16, kernel_size=5, stride=2, padding=2, print=False)
    w, h, c = calc_max_pool_size(w, h, c, pool_size=3, stride=2, print=False)
    w, h, c = calc_conv_layer_size(w, h, c, kernel_num=64, kernel_size=5, stride=2, padding=2, print=False)
    w, h, c = calc_max_pool_size(w, h, c, pool_size=3, stride=2, print=False)
    w, h, c = calc_FC_layer_size(w, h, c, print=False)
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

    # # define the MLP to handle the sensor data
    # self.numeric_features_ = nn.Sequential(
    #   nn.Linear(numeric_inputs, 128),
    #   nn.ReLU(),
    #   # nn.Linear(150, 100),
    #   # nn.ReLU(),
    #   # nn.Linear(100, 50),
    #   # nn.ReLU(),
    # )

    # combine the image and MLP features
    self.combined_features_ = nn.Sequential(
      nn.Linear(128 + 0, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, outputs),
      nn.Softmax(dim=1),
    )

  def forward(self, img):
    image = img.to(self.device)
    # sensors = tuple_img_sensors[1].to(self.device)
    x = self.image_features_(image)
    # x = x.view(-1, 64*64)
    # y = self.numeric_features_(sensors)
    # z = torch.cat((x, y), 1)
    z = self.combined_features_(x)
    return x

class LeNet(nn.Module):

  def __init__(self, input_channels, outputs):
    # call the parent constructor
    super(LeNet, self).__init__()
    # initialize first set of CONV => RELU => POOL layers
    self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=20,
      kernel_size=(5, 5))
    self.relu1 = nn.ReLU()
    self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
    # initialize second set of CONV => RELU => POOL layers
    self.conv2 = nn.Conv2d(in_channels=20, out_channels=50,
      kernel_size=(5, 5))
    self.relu2 = nn.ReLU()
    self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
    # initialize first (and only) set of FC => RELU layers
    self.fc1 = nn.Linear(in_features=800, out_features=500)
    self.relu3 = nn.ReLU()
    # initialize our softmax classifier
    self.fc2 = nn.Linear(in_features=500, out_features=outputs)
    self.logSoftmax = nn.LogSoftmax(dim=1)

  def forward(self, x):
    # pass the input through our first set of CONV => RELU =>
    # POOL layers
    x = self.conv1(x)
    x = self.relu1(x)
    x = self.maxpool1(x)
    # pass the output from the previous layer through the second
    # set of CONV => RELU => POOL layers
    x = self.conv2(x)
    x = self.relu2(x)
    x = self.maxpool2(x)
    # flatten the output from the previous layer and pass it
    # through our only set of FC => RELU layers
    x = nn.flatten(x, 1)
    x = self.fc1(x)
    x = self.relu3(x)
    # pass the output to our softmax classifier to get our output
    # predictions
    x = self.fc2(x)
    output = self.logSoftmax(x)
    # return the output predictions
    return output

class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
 
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
 
        self.flat = nn.Flatten()
 
        self.fc3 = nn.Linear(8192, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
 
        self.fc4 = nn.Linear(512, 10)
 
    def forward(self, x):
        # input 3x32x32, output 32x32x32
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32x32x32, output 32x32x32
        x = self.act2(self.conv2(x))
        # input 32x32x32, output 32x16x16
        x = self.pool2(x)
        # input 32x16x16, output 8192
        x = self.flat(x)
        # input 8192, output 512
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        # input 512, output 10
        x = self.fc4(x)
        return x
    
# --- old --- #

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

class DQN_3L100(nn.Module):

  name = "DQN_3L100"

  def __init__(self, inputs, outputs, device):
    super(DQN_3L100, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, 100)
    self.linear2 = torch.nn.Linear(100, 100)
    self.linear3 = torch.nn.Linear(100, 100)
    self.linear4 = torch.nn.Linear(100, outputs)
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

class DQN_4L100(nn.Module):

  name = "DQN_4L100"

  def __init__(self, inputs, outputs, device):
    super(DQN_4L100, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, 100)
    self.linear2 = torch.nn.Linear(100, 100)
    self.linear3 = torch.nn.Linear(100, 100)
    self.linear4 = torch.nn.Linear(100, 100)
    self.linear5 = torch.nn.Linear(100, outputs)
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

class DQN_5L100(nn.Module):

  name = "DQN_5L100"

  def __init__(self, inputs, outputs, device):
    super(DQN_5L100, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, 100)
    self.linear2 = torch.nn.Linear(100, 100)
    self.linear3 = torch.nn.Linear(100, 100)
    self.linear4 = torch.nn.Linear(100, 100)
    self.linear5 = torch.nn.Linear(100, 100)
    self.linear6 = torch.nn.Linear(100, outputs)
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

class DQN_6L60(nn.Module):

  name = "DQN_6L60"

  def __init__(self, inputs, outputs, device):
    super(DQN_6L60, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, 60)
    self.linear2 = torch.nn.Linear(60, 60)
    self.linear3 = torch.nn.Linear(60, 60)
    self.linear4 = torch.nn.Linear(60, 60)
    self.linear5 = torch.nn.Linear(60, 60)
    self.linear6 = torch.nn.Linear(60, 60)
    self.linear7 = torch.nn.Linear(60, outputs)
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
    x = self.activation(x)
    x = self.linear7(x)
    x = self.softmax(x)

    return x

class DQN_7L60(nn.Module):

  name = "DQN_7L60"

  def __init__(self, inputs, outputs, device):
    super(DQN_7L60, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, 60)
    self.linear2 = torch.nn.Linear(60, 60)
    self.linear3 = torch.nn.Linear(60, 60)
    self.linear4 = torch.nn.Linear(60, 60)
    self.linear5 = torch.nn.Linear(60, 60)
    self.linear6 = torch.nn.Linear(60, 60)
    self.linear7 = torch.nn.Linear(60, 60)
    self.linear8 = torch.nn.Linear(60, outputs)
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
    x = self.activation(x)
    x = self.linear7(x)
    x = self.activation(x)
    x = self.linear8(x)
    x = self.softmax(x)

    return x

class DQN_7L100(nn.Module):

  name = "DQN_7L100"

  def __init__(self, inputs, outputs, device):
    super(DQN_7L100, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, 100)
    self.linear2 = torch.nn.Linear(100, 100)
    self.linear3 = torch.nn.Linear(100, 100)
    self.linear4 = torch.nn.Linear(100, 100)
    self.linear5 = torch.nn.Linear(100, 100)
    self.linear6 = torch.nn.Linear(100, 100)
    self.linear7 = torch.nn.Linear(100, 100)
    self.linear8 = torch.nn.Linear(100, outputs)
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
    x = self.activation(x)
    x = self.linear7(x)
    x = self.activation(x)
    x = self.linear8(x)
    x = self.softmax(x)

    return x

# --- ideas --- #

class DQN_4L120_60(nn.Module):

  name = "DQN_4L120_60"

  def __init__(self, inputs, outputs, device):
    super(DQN_4L120_60, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, 120)
    self.linear2 = torch.nn.Linear(120, 60)
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

class DQN_2LX(nn.Module):

  name = "DQN_2L"

  def __init__(self, inputs, outputs, device, width):
    super(DQN_2LX, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, width)
    self.linear2 = torch.nn.Linear(width, width)
    self.linear3 = torch.nn.Linear(width, outputs)
    self.activation = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)
    self.name += str(width)

  def forward(self, x):
    x = x.to(self.device)
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    x = self.activation(x)
    x = self.linear3(x)
    x = self.softmax(x)

    return x

class DQN_3LX(nn.Module):

  name = "DQN_3L"

  def __init__(self, inputs, outputs, device, width):
    super(DQN_3LX, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, width)
    self.linear2 = torch.nn.Linear(width, width)
    self.linear3 = torch.nn.Linear(width, width)
    self.linear4 = torch.nn.Linear(width, outputs)
    self.activation = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)
    self.name += str(width)

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
  
if __name__ == "__main__":

  # self.image_features_ = nn.Sequential(

  #     # input CxWxH, output 16xWxH
  #     nn.Conv2d(image_channels, 16, kernel_size=5, stride=2, padding=2),
  #     nn.ReLU(),
  #     nn.MaxPool2d(kernel_size=3, stride=2),
  #     # nn.Dropout(),
  #     nn.Conv2d(16, 64, kernel_size=5, padding=2),
  #     nn.ReLU(),
  #     # output 64*64
  #     nn.MaxPool2d(kernel_size=3, stride=2),
  #     # nn.Dropout()
  #     # nn.Linear(64*64, 64*8),
  #     # nn.ReLU(),
  #     # nn.Linear(64*16, 64),
  #     # nn.ReLU(),
  #   )
  
  w = 25
  h = 25
  c = 3

  sizes = [25, 50, 75, 100]
  roundup_str = ""

  for s in sizes:

    w = s
    h = s
    c = 3

    w, h, c = calc_conv_layer_size(w, h, c, kernel_num=16, kernel_size=5, stride=2, padding=2)
    w, h, c = calc_max_pool_size(w, h, c, pool_size=3, stride=2)
    w, h, c = calc_conv_layer_size(w, h, c, kernel_num=64, kernel_size=5, stride=1, padding=2)
    w, h, c = calc_max_pool_size(w, h, c, pool_size=3, stride=2)
    w, h, c = calc_FC_layer_size(w, h, c)

    roundup_str += f"size ({s} x {s}) gives fully connected layer width {c}\n"

  roundup_str += "\nOr using stride=2 gives\n"

  for s in sizes:

    w = s
    h = s
    c = 3

    w, h, c = calc_conv_layer_size(w, h, c, kernel_num=16, kernel_size=5, stride=2, padding=2)
    w, h, c = calc_max_pool_size(w, h, c, pool_size=3, stride=2)
    w, h, c = calc_conv_layer_size(w, h, c, kernel_num=64, kernel_size=5, stride=2, padding=2)
    w, h, c = calc_max_pool_size(w, h, c, pool_size=3, stride=2)
    w, h, c = calc_FC_layer_size(w, h, c)

    roundup_str += f"size ({s} x {s}) gives fully connected layer width {c}\n"

  print("\n" + roundup_str)

  # self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=5, stride=2, padding=2)
  # self.relu1 = nn.ReLU()
  # self.maxp1 = nn.MaxPool2d(kernel_size=3, stride=2)
  
  # self.conv2 = nn.Conv2d(16, 64, kernel_size=5, padding=2)
  # self.relu2 = nn.ReLU()
  # self.maxp2 = nn.MaxPool2d(kernel_size=3, stride=2)

  # # alex net: correct answer for FC layer is 9216
  # w = 227
  # h = 227
  # c = 3
  # w, h, c = calc_conv_layer_size(w, h, c, 96, 11, 4, 0)
  # w, h, c = calc_max_pool_size(w, h, c, 3, 2)
  # w, h, c = calc_conv_layer_size(w, h, c, 256, 5, 1, 2)
  # w, h, c = calc_max_pool_size(w, h, c, 3, 2)
  # w, h, c = calc_conv_layer_size(w, h, c, 384, 3, 1, 1)
  # w, h, c = calc_conv_layer_size(w, h, c, 384, 3, 1, 1)
  # w, h, c = calc_conv_layer_size(w, h, c, 256, 3, 1, 1)
  # w, h, c = calc_max_pool_size(w, h, c, 3, 2)
  # w, h, c = calc_FC_layer_size(w, h, c)