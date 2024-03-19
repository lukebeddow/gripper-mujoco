import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2lab, rgb2hsv
from skimage import img_as_ubyte
from PIL import Image
import os
import time

import sys
sys.path.insert(0, "/home/luke/luke-gripper-ros/rl/gripper_dqn/scripts")
from grasp_test_data import GraspTestData, set_palm_frc_threshold

from ModelSaver import ModelSaver

def visualise_sim_image(datapoint, rgb=True, depth=False, seg=False):
  """
  Show a simulated image
  """

  num = rgb + depth + seg

  fig, axs = plt.subplots(num, 1, sharex=True)
  k = 0

  if num == 1: axs = [axs]

  # numpy likes image arrays like this: height x width x channels (ie rows x columns x dim)
  # torch likes image arrays like this: channels x width x height
  # hence convert from torch style back to numpy style for plotting

  if rgb:
    axs[k].imshow(np.einsum("ijk->kji", datapoint["rgb"])) # swap to numpy style rows/cols (eg 3x640x480 -> 480x640x3)
    k += 1

  if depth:
    axs[k].imshow(np.transpose(datapoint["depth"])) # remove the depth 'channel' then swap (eg 1x640x480 -> 480x640)
    k += 1

  if seg:
    max_num = np.max(datapoint["rgb_mask"])
    scale = 255 // max_num

    # apply scalings to give each mask integers (1-7 usually) a different colour
    red_scaled = scale * (datapoint["rgb_mask"])
    green_scaled = scale * 2 * (max_num - datapoint["rgb_mask"]) * (datapoint["rgb_mask"] % 3)
    blue_scaled = scale * 2 * (max_num - datapoint["rgb_mask"]) * (datapoint["rgb_mask"] + 2 % 3)

    # reconstruct a three-channel image
    scaled_rgb = np.array([red_scaled, green_scaled, blue_scaled], dtype=np.uint8)

    # plot the visualisation of the mask
    axs[k].imshow(np.einsum("ijk->kji", scaled_rgb)) # swap to numpy style rows/cols (eg 3x640x480 -> 480x640x3)
    k += 1

  plt.show()

def visualise_real_image(realimage, rgb=True, depth=False):
  """
  Show a simulated image
  """

  num = rgb + depth

  fig, axs = plt.subplots(num, 1, sharex=True)
  k = 0

  if num == 1: axs = [axs]

  # numpy likes image arrays like this: height x width x channels (ie rows x columns x dim)
  # torch likes image arrays like this: channels x width x height
  # hence convert from torch style back to numpy style for plotting

  if rgb:
    axs[k].imshow(realimage.rgb) # swap to numpy style rows/cols (eg 3x640x480 -> 480x640x3)
    k += 1

  if depth:
    # saturate any values above this (ignore outliers)
    depth_max = 5000

    # find minimum of minima & maximum of depth data
    dmin = 1e10
    dmax = -1e10
    this_min = np.min(realimage.depth[realimage.depth < depth_max])
    this_max = np.max(realimage.depth[realimage.depth < depth_max])
    if this_min < dmin: dmin = this_min
    if this_max > dmax: dmax = this_max
    axs[k].imshow(realimage.depth, vmin=dmin, vmax=dmax, cmap="viridis")
    k += 1


  plt.show()

def resize_and_save_sim_image(datapoint, saveas="test_img.jpg", width=100, height=100):
  """
  Change the size of a simulated image
  """
  
  # print(datapoint["rgb"].shape)
  # print(datapoint["depth"].shape)
  # print(np.expand_dims(datapoint["rgb_mask"], axis=0).shape)

  if datapoint["rgb_mask"] is not None:
    new_img_array = np.concatenate((datapoint["rgb"], 
                                    datapoint["depth"], 
                                    np.expand_dims(datapoint["rgb_mask"], axis=0)), 
                                    axis=0, dtype=np.float64)
    img_resized = resize(new_img_array, (5, width, height))
  else:
    img_resized = resize(np.array(datapoint["rgb"], dtype=np.float64), (3, width, height))

  # Convert the NumPy array to a PIL Image
  image = Image.fromarray(np.einsum("ijk->kji", img_resized[:3]).astype(np.uint8))

  # Save the image as a JPEG file
  image.save(saveas, 'JPEG')

def resize_and_save_real_image(realimage, saveas="test_img.jpg", width=100, height=100):
  """
  Change the size of a real image
  """

  # print(realimage.rgb.shape)
  # print(np.expand_dims(realimage.depth, axis=2).shape)

  global add_green_noise, noise_method
  if add_green_noise:
    if noise_method.lower() == "lab":
      rgbpart = replace_green_region_with_noise_lab(realimage.rgb)
    elif noise_method.lower() == "hsv":
      rgbpart = replace_green_region_with_noise_hsv(realimage.rgb)
  else:
    rgbpart = realimage.rgb

  new_img_array = np.concatenate((rgbpart, 
                                  np.expand_dims(realimage.depth, axis=2)), 
                                  axis=2, dtype=np.float64)
  new_img_array = np.einsum("ijk->kji", new_img_array)
  
  # print(new_img_array.shape)

  img_resized = resize(new_img_array, (4, width, height))

  # Convert the NumPy array to a PIL Image
  image = Image.fromarray(np.einsum("ijk->kji", img_resized[:3]).astype(np.uint8))
  # image = Image.fromarray(img_resized[:,:,:3].astype(np.uint8))

  # Save the image as a JPEG file
  image.save(saveas, 'JPEG')

def replace_green_region_with_noise_lab(image):
    
    # Convert the image to LAB color space
    lab_image = rgb2lab(image)

    # Define the lower and upper bounds for green color in LAB
    lower_green = np.array([20, -40, 10])  # Adjust as needed
    upper_green = np.array([60, 20, 20])    # Adjust as needed

    # Create a mask for the green region
    green_mask = np.all((lab_image >= lower_green) & (lab_image <= upper_green), axis=-1)

    # Generate uniform noise of the same size as the image
    noise = np.random.randint(0, 256, size=image.shape, dtype=np.uint8)

    # Replace the green region with uniform noise
    image_with_noise = np.where(green_mask[:, :, np.newaxis], noise, image)

    return image_with_noise

def replace_green_region_with_noise_hsv(image):

    # Convert the image to HSV color space
    hsv_image = rgb2hsv(image)

    # Define the lower and upper bounds for green color in HSV
    lower_green = np.array([60 / 360, 0.2, 0.1])  # Adjust as needed
    upper_green = np.array([180 / 360, 0.9, 0.7])  # Adjust as needed

    # Create a mask for the green region
    green_mask = np.all((hsv_image >= lower_green) & (hsv_image <= upper_green), axis=-1)

    # Generate uniform noise of the same size as the image
    block_noise = True
    if block_noise:
      # main colour which is disturbed by some noise
      noise_value = 30
      noise = np.random.randint(-noise_value, noise_value, size=image.shape, dtype=np.int8)
      block_colour = np.random.randint(0, 256, size=(1, 1, 3), dtype=np.uint8)
      block_colour[:, :, 0] = np.random.randint(0, 256)
      block_colour[:, :, 1] = np.random.randint(0, 256)
      block_colour[:, :, 2] = np.random.randint(0, 256)
      noise = np.array(noise + block_colour, dtype=np.uint8)
    else:
      # uniform white noise
      noise = np.random.randint(0, 256, size=image.shape, dtype=np.uint8)

    # Replace the green region with uniform noise
    image_with_noise = np.where(green_mask[:, :, np.newaxis], noise, image)

    return image_with_noise

# ----- setup variables ----- #
  
generate_sim = True
generate_real = False

# beware: exact numbers will not result out of this script
num_sim_images = 4000
num_real_images = 4000

add_green_noise = True
noise_method = "hsv"

width = 200
height = 100

save_sim_folder = "image_datasets/multi_object_200x100/trainA"
save_real_folder = "image_datasets/multi_object_200x100/trainB"

# load_sim_folder = "models/sim_images_3"
load_sim_folder = "models/sim_images_clutter/05-02-24"
load_real_folder = "models/pb1_test_data"

# simruns = [
#   "run_12-53_A1",
#   "run_12-53_A2",
#   "run_12-53_A3",
#   "run_12-53_A4",
#   "run_13-47_A1",
#   # "run_13-47_A2",
#   # "run_13-47_A3",
#   # "run_13-47_A4",
# ]
simruns = [
  "run_12-16_A1",
  "run_12-16_A2",
  "run_12-16_A3",
  "run_12-16_A4",
]
real_tests = [
  # "pb1_E1_S0", 
  "pb1_E1_S1", # multi-object
  "pb1_E1_S2", # multi-object   
  # "pb1_E1_S3", 
  # "pb1_E2_S3",
  # "pb1_E3_S3", 
  # "pb1_E2_S3_YCB",
  # "pb1_E2_S3_real",
  "YCB_heuristic", # multi-object
  # "real_heuristic", 
  # "high_noise_single_object",
]

# ----- begin generation code ----- #

t0 = time.time()

num_sim = 0
num_real = 0

sim_names = np.random.permutation(np.array(list(range(num_sim_images + 1))))
real_names = np.random.permutation(np.array(list(range(num_real_images + 1))))

# create directories if they don't already exist
if not os.path.exists(save_sim_folder):
  os.makedirs(save_sim_folder)
if not os.path.exists(save_real_folder):
  os.makedirs(save_real_folder)

if generate_sim:
  for simrun in simruns:

    simloader = ModelSaver(f"{load_sim_folder}/{simrun}")
    num_files = simloader.get_recent_file(name="image_collection", return_int=True)

    rand_files = np.random.permutation(np.array(list(range(1, num_files + 1))))

    # # can clip loaded files to be in a particular range
    # min_sim = 1
    # max_sim = num_files + 1
    # rand_files = np.random.permutation(np.array(list(range(min_sim, max_sim))))

    num_files = len(rand_files)

    run_target = int(num_sim_images / len(simruns))
    file_target = int((num_sim_images / len(simruns)) / num_files)

    target_num = int((num_sim_images / len(simruns)) / num_files)
    num_done = 0

    if target_num < 1: target_num = 1
    else:
      if np.random.random() > 0.5:
        target_num += 1

    for i in rand_files:

      if num_done >= run_target: break

      simdata = simloader.load("image_collection", id=i)

      rand_walk = np.random.permutation(np.array(list(range(len(simdata)))))

      if len(simdata) < target_num:
        target_num = len(simdata)

      for j in range(target_num):

        resize_and_save_sim_image(simdata[rand_walk[j]],
                                  saveas=f"{save_sim_folder}/img_{sim_names[num_sim]}.jpg",
                                  width=width,
                                  height=height)
        num_done += 1
        num_sim += 1

    print(f"Number of simulated images completed {num_sim} / {num_sim_images}")

  print(f"Finished {num_sim} simulated images")

t1 = time.time()

if generate_real:
  for real_test in real_tests:

    initial_green_noise = add_green_noise

    if "YCB" in real_test or "real" in real_test:
      add_green_noise = False
      noise_changed = True
    else:
      noise_changed = False

    realloader = ModelSaver(f"{load_real_folder}/{real_test}")
    num_files = realloader.get_recent_file(name="trial_image_batch", return_int=True)
    
    if num_files is None:
      raise RuntimeError(f"no files found for test: {load_real_folder}/{real_test}, check you are running from the right location")

    run_target = int(num_real_images / len(real_tests))
    file_target = int((num_real_images / len(real_tests)) / num_files)

    target_num = int((num_real_images / len(real_tests)) / num_files)
    num_done = 0

    rand_files = np.random.permutation(np.array(list(range(1, num_files + 1))))

    if target_num < 1: target_num = 1
    else:
      if np.random.random() > 0.5:
        target_num += 1

    for i in rand_files:

      if num_done >= run_target: break

      realdata = realloader.load("trial_image_batch", id=i)

      rand_walk = np.random.permutation(np.array(list(range(len(realdata.trials[0].images)))))

      if len(realdata.trials[0].images) < target_num:
        target_num = len(realdata.trials[0].images)

      for j in range(target_num):

        resize_and_save_real_image(realdata.trials[0].images[rand_walk[j]],
                                  saveas=f"{save_real_folder}/img_{real_names[num_real]}.jpg",
                                  width=width,
                                  height=height)
        num_done += 1
        num_real += 1

    print(f"Number of real images completed {num_real} / {num_real_images}")

    if noise_changed:
      add_green_noise = initial_green_noise

  print(f"Finished {num_real} real images")

t2 = time.time()

sim_time = t1 - t0
real_time = t2 - t1

if sim_time < 60:
  sim_str = f"{sim_time:.1f} seconds"
elif sim_time < 3600:
  sim_str = f"{sim_time / 60:.1f} minutes"
else:
  sim_str = f"{sim_time / 3600:.1f} hours"

if real_time < 60:
  real_str = f"{real_time:.1f} seconds"
elif real_time < 3600:
  real_str = f"{real_time / 60:.1f} minutes"
else:
  real_str = f"{real_time / 3600:.1f} hours"

print(f"\nGenerated {num_sim} simulated images in {sim_str}")
print(f"Generated {num_real} real images in {real_str}")
print(f"These images have width={width} and height={height}")
  