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

# --- CNN and sensor data --- #

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

  def __init__(self, numeric_inputs, image_size, outputs):

    super(MixedNetworkFromEncoder, self).__init__()
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
    image = tuple_img_sensors[0]#.to(self.device)
    sensors = tuple_img_sensors[1]#.to(self.device)

    # feed each part through a seperate head
    x = self.image_features_(image)
    y = self.numeric_features_(sensors)

    # concatenate the parts and feed into the last head of the network
    x = x.view(x.shape[0], 128) # from shape [B, 128, 1, 1] -> [B, 128]
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

class MixedNetworkFromEncoder2(nn.Module):

  name = "MixedNetworkFromEncoder2"

  def __init__(self, numeric_inputs, image_size, outputs):

    super(MixedNetworkFromEncoder2, self).__init__()

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

      # nn.Conv2d(channel, channel * 0.5, kernel_size=3, stride=1, padding=1),
      # nn.ReLU(inplace=True),
      # nn.MaxPool2d(kernel_size=3, stride=1),
      # nn.Conv2d(channel * 0.5, channel * 0.25, kernel_size=3, stride=1, padding=1),
      # nn.ReLU(inplace=True),
      # nn.MaxPool2d(kernel_size=3, stride=1),
      nn.Flatten(),
      nn.LazyLinear(256),
      nn.ReLU(True),
      nn.Linear(256, 128),
      nn.ReLU(True),
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
      nn.Linear(256, 256),
      nn.ReLU(inplace=True),
      nn.Linear(256, outputs),
    )

  def split_obs(self, obs):
    """
    Split the incoming observation into a tuple:
      (image_size, sensor_obs)
    """

    # print("obs size is", obs.shape)
    # print("img size is", self.image_size)
  
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
    image = tuple_img_sensors[0]
    sensors = tuple_img_sensors[1]

    # feed each part through a seperate head
    x = self.image_features_(image)
    y = self.numeric_features_(sensors)

    # concatenate the parts and feed into the last head of the network
    x = x.view(x.shape[0], 128) # from shape [B, 128, 1, 1] -> [B, 128]
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

class MxNetFeedforward(nn.Module):

  name = "MxNetFeedforward"

  def __init__(self, numeric_inputs, image_size, outputs, feedforwardsize=0):

    super(MxNetFeedforward, self).__init__()
    self.image_size = image_size
    self.numeric_inputs = numeric_inputs
    self.num_outputs = outputs
    self.ffsize = feedforwardsize

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
      # nn.Linear(128, outputs),
    )

    self.ff_layers_ = nn.Sequential(
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

    if self.ffsize > 0:
      if self.ffsize > sensors.shape[1]:
        raise RuntimeError(f"MxNetFeedforward.split_obs() error: self.ffsize = {self.ffsize} too large for sensor obs shape = {sensors.shape}")
      ff = sensors[:, -self.ffsize:]
    else:
      ff = None

    return (img, sensors, ff)

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

    # add the feedforward values
    if tuple_img_sensors[2] is not None:
      z[:,-self.ffsize:] += tuple_img_sensors[2]

    z = self.ff_layers_(z)

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
    self.ff_layers_.to(device)


# --- old --- #

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