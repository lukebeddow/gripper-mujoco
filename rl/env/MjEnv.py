#!/usr/bin/env python3

import os
import sys
import yaml
import subprocess
from copy import deepcopy
import numpy as np
from dataclasses import dataclass, asdict
import time
import torch

# get the path to this file and insert it to python path (for mjpy.bind)
pathhere = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, pathhere)

# with env in path, we can now import the shared cpp library
from mjpy.bind import MjClass, EventTrack

import image_transform

# random generators for training time and test time
random_train = np.random.default_rng()
random_test = np.random.default_rng() 

def get_yaml_hash(filepath):
  """
  Get a simple hash of the text of the yaml file, stripping all whitespace
  """

  # simple string hash function which returns same hash each run (unlike Python hash())
  def myHash(text:str):
    hash=0
    for ch in text:
      hash = ( hash*281  ^ ord(ch)*997) & 0xFFFFFFFF
    # return hash
    return hex(hash)[2:].upper().zfill(8)

  # read the yaml file as a string for hashing
  with open(filepath, "r") as yamlfile:
    yaml_string = yamlfile.read()
    yaml_string = "".join(yaml_string.split())

  # hash the yaml string for the task folder name
  return myHash(yaml_string)

class MjEnv():

  @dataclass
  class Track:
    # tracking variables that are reset after each episode
    current_step: int = 0
    last_action: int = -1
    is_done: bool = False
    last_reward: float = 0
    cumulative_reward: float = 0

  class Test:
    # saved after each test trial ends
    obj_idx: int = 0
    obj_trial: int = 0
    obj_counter: int = 0
    xml_file: int = 0
    reward: float = 0
    steps: int = 0
    object_name: str = ""
    object_category: str = ""
    cnt = None

  @dataclass
  class Parameters:

    # note all XYZ distances are symettric
    # eg noise_mm = 10 means +-10mm noise from central point
    # eg base_lim_X_mm = 100 means +-100mm, so 200mm working area

    # training parameters
    max_episode_steps: int = 250
    object_position_noise_mm: int = 10
    object_rotation_noise_deg: int = 5 # depreciated, now spawns in any available orientation
    base_lim_X_mm: int = 300
    base_lim_Y_mm: int = 200
    base_lim_Z_mm: int = 30
    base_lim_yaw_rad: float = np.pi / 4

    # camera grasping settings
    use_rgb_in_observation: bool = False
    use_depth_in_observation: bool = False
    use_rgb_rendering: bool = False
    rgb_rendering_method: str = "CUT"
    image_width: int = 50
    image_height: int = 50

    # image preprocessing settings
    use_standard_transform: bool = False
    transform_resize_square: int = 144
    transform_crop_size: int = 128

    # grasping scene parameters
    use_scene_settings: bool = False
    num_objects_in_scene: int = 1
    scene_grasp_target: int = 1
    origin_noise_X_mm: int = 150
    origin_noise_Y_mm: int = 50
    scene_X_dimension_mm: int = 300
    scene_Y_dimension_mm: int = 200

    # file and testing parameters
    test_obj_per_file: int = 20
    task_reload_chance: float = 1.0 / float(test_obj_per_file)
    test_trials_per_object: int = 3
    test_objects: int = 100

    # model parameters (for loading xml files)
    object_set_name: str = "set9_fullset"
    num_segments: int = 8
    finger_thickness: float = 0.9e-3
    finger_length: float = 235e-3
    finger_width: float = 28e-3
    finger_modulus: float = 193e9
    depth_camera: bool = False
    XY_base_actions: bool = False
    Z_base_rotation: bool = False
    fixed_finger_hook: bool = True
    finger_hook_angle_degrees: float = 90.0
    finger_hook_length: float = 35e-3
    segment_inertia_scaling: float = 50.0
    fingertip_clearance: float = 10e-3

    def update(self, newdict):
      for key, value in newdict.items():
        if hasattr(self, key):
          setattr(self, key, value)
        else: raise RuntimeError(f"incorrect key: {key}")

  def __init__(self, object_set=None, seed=None, num_segments=None, finger_width=None, 
               depth_camera=None, finger_thickness=None, finger_modulus=None,
               log_level=0, render=False, continous_actions=False, use_torch=True,
               device="cpu"):
    """
    A mujoco environment, optionally set the random seed or prevent loading a
    model, in which case the user should call load() before using the class
    """

    self.params = MjEnv.Parameters()
    self.load_next = MjEnv.Parameters() # for xml loading, other params do nothing

    # define file structure
    self.task_xml_folder = "task"
    self.task_folder_template = "gripper_N{}"
    self.task_xml_template = "gripper_task_{}.xml"

    # general class settings
    self.log_level = log_level
    self.render_window = render
    self.prevent_reload = False
    self.use_yaml_hashing = False
    self.torch = use_torch # do we return and expect torch tensors
    self.torch_device = torch.device(device)
    self.randomise_colours_every_step = False
    self.render_net = None

    # initialise class variables
    self.test_in_progress = False
    self.test_completed = False
    self.scene_recursion_counter = 0

    # if not given values, set to defaults from the dataclass
    if num_segments is not None: self.load_next.num_segments = num_segments
    if finger_width is not None: self.load_next.finger_width = finger_width
    if finger_thickness is not None: self.load_next.finger_thickness = finger_thickness
    if finger_modulus is not None: self.load_next.finger_modulus = finger_modulus
    if depth_camera is not None: self.load_next.depth_camera = depth_camera

    # calculate how many files we need to reserve for testing
    self.testing_xmls = int(np.ceil(self.params.test_objects / float(self.params.test_obj_per_file)))
    
    # create mujoco instance
    self.mj = MjClass()
    self.mj.set.continous_actions = continous_actions
    if self.log_level <= 3: self.mj.set.debug = False
    elif self.log_level >= 4: self.mj.set.debug = True

    # seed the environment
    self.myseed = None
    self.seed(seed)

    # load the mujoco models, if not then load() must be run by the user
    if object_set is not None: self.load(object_set)

    # initialise tracking variables
    self.track = MjEnv.Track()
    self.test = MjEnv.Test()
    self.prev_test = MjEnv.Test()

    # initialise heuristic parameters
    self._initialise_heuristic_parameters()

    # save default rgbd camera size
    self.default_rgbd_width = 848
    self.default_rgbd_height = 480

    # set image collection defaults
    self.collect_images = False
    self.new_image_collected = False
    self.image_collection_data = None
    self.image_collection_chance = 0.01

    return

  # ----- semi-private functions, advanced use ----- #

  def _auto_generate_xml_file(self, object_set, use_hashes=False, silent=True, force=False):
    """
    Automatically generate the xml file that we need. Note this function
    overrides the currently set mjcf path to the path that leads to the
    autogenerated file.
    """

    # first we need to set the gripper.yaml file details
    repo_path = os.path.dirname(os.path.dirname(pathhere))
    description_path = repo_path + "/description"
    config_folder = "config"
    config_file = "gripper.yaml"
    mjcf_folder = "mjcf"
    yaml_path = f"{description_path}/{config_folder}/{config_file}"
    with open(yaml_path) as file:
      gripper_details = yaml.safe_load(file)

    original_details = deepcopy(gripper_details)

    # override with our settings
    c = "gripper_config"
    p = "gripper_params"
    gripper_details[c]["fixed_hook_segment"] = self.load_next.fixed_finger_hook
    gripper_details[c]["num_segments"] = self.load_next.num_segments
    gripper_details[c]["xy_base_joint"] = self.load_next.XY_base_actions
    gripper_details[c]["z_base_rotation"] = self.load_next.Z_base_rotation
    # gripper_details[c]["xy_base_rotation"] = self.load_next.  not added yet

    # ignore finger thickness setting, do manual setting. Note that myfunctions.cpp
    # also ignores this setting and requires a manual override
    # gripper_details[p]["finger_thickness"] = self.load_next.finger_thickness

    gripper_details[p]["finger_width"] = self.load_next.finger_width
    gripper_details[p]["finger_E"] = self.load_next.finger_modulus
    gripper_details[p]["hook_angle_degrees"] = self.load_next.finger_hook_angle_degrees
    gripper_details[p]["segment_inertia_scaling"] = self.load_next.segment_inertia_scaling
    gripper_details[p]["fingertip_clearance"] = self.load_next.fingertip_clearance
    gripper_details[p]["finger_length"] = self.load_next.finger_length
    gripper_details[p]["hook_length"] = self.load_next.finger_hook_length

    # check for special case with no segments
    if self.load_next.num_segments == 0:
      gripper_details[c]["is_segmented"] = False
    elif self.load_next.num_segments == 1:
      gripper_details[c]["is_segmented"] = True
      gripper_details[c]["fixed_first_segment"] = True
    elif self.load_next.num_segments == 2:
      raise RuntimeError("MjEnv._auto_generate_xml_file() error: num_segments=2 does not work")
    else:
      gripper_details[c]["is_segmented"] = True
      gripper_details[c]["fixed_first_segment"] = False

    # now override the existing file with our new changes
    with open(yaml_path, "w") as outfile:
      yaml.dump(gripper_details, outfile, default_flow_style=False)

    # determine the task name
    N = self.load_next.num_segments
    W = self.load_next.finger_width
    if use_hashes:
      yaml_hash = get_yaml_hash(yaml_path)
      taskname = f"gripper_N{N}_H{yaml_hash}"
    else:
      taskname = f"gripper_N{N}_{W*1e3:.0f}"

    # generate if the task folder does not already exist
    if not os.path.exists(f"{repo_path}/{mjcf_folder}/{object_set}/{taskname}") or force:

      if self.log_level > 1:
        print(f"Target not found, generating: {repo_path}/{mjcf_folder}/{object_set}/{taskname}")

      # determine what machine we are running on to adjust the make command
      machine = self._get_machine()
      if machine == "luke-laptop":
        m = "luke"
      elif machine == "luke-PC":
        m = "lab"
      elif machine == "operator-PC":
        m = "lab-op"
      elif machine == "zotac-PC":
        m = "zotac"
      else:
        raise RuntimeError(f"MjEnv()._auto_generate_xml_file() does not recognise this machine: {machine}")
      
      if self.log_level > 1:
        print(f"MjEnv()._auto_generate_xml_file() found machine '{machine}'")

      # now run make in order to generate the files for this object set
      silence = "-s" if silent else ""
      hash_str = "yes" if use_hashes else "no"
      make = f"make {silence} {m} sets SET={object_set} EXTRA_COPY_TO_MERGE_SETS=yes USE_HASHES={hash_str}"
      subprocess.run([make], shell=True, cwd=repo_path)

    else:
      if self.log_level > 0:
        print(f"MjEnv._auto_generate_xml_file() found that '{repo_path}/{mjcf_folder}/{object_set}/{taskname}' already exists. Nothing generated - use force=True to force generation.")

    # override the current mjcf path with the new path
    self.mj.model_folder_path = f"{repo_path}/{mjcf_folder}"

    # restore the gripper.yaml file to its original state
    with open(yaml_path, "w") as outfile:
      yaml.dump(original_details, outfile, default_flow_style=False)

    return taskname

  def _set_finger_variables(self, num_segments=None, width=None):
    """
    Set the number of segments in use and also the finger width. The available
    options will depend on the object set chosen, None means use the value in
    params. This function is ignored if auto_generate=True in _load_object_set
    """

    debug_fcn = False

    if num_segments is None: num_segments = self.load_next.num_segments
    if width is None: width = self.load_next.finger_width

    # ensure width is in integer millimeters
    if width < 1:
      width = int(width * 1000)
    if isinstance(width, float):
      width = int(width)

    if debug_fcn:
      print(f"_set_finger_variables fcn input: num_segments={num_segments}, width={width}")

    # determine if object set has multiple widths available
    task_path = self.mj.model_folder_path + "/" + self.mj.object_set_name
    task_folders = [x for x in os.listdir(task_path) if os.path.isdir(task_path + "/" + x) is True
                                                      and x.startswith("gripper")]
    width_options = []
    for folder in task_folders:
      namesplit = folder.split("_")
      if len(namesplit) == 3:
        if int(namesplit[2]) not in width_options:
          width_options.append(int(namesplit[2]))

    if debug_fcn:
      print("task folders are", task_folders)
      print("width options are", width_options)

    # apply the chosen width option
    if width_options == []:
      self.load_next.finger_width = 28e-3 # hardcoded default
      self.task_xml_folder = self.task_folder_template.format(num_segments)
      if width != 28:
        print(f"MjEnv warning: selected finger width of {width} is not available from this object set")
    else:
      if width in width_options:
        self.load_next.finger_width = width * 1e-3 # convert from mm to m
        self.task_xml_folder = self.task_folder_template.format("{0}_{1}".format(num_segments, width))
      else:
        raise RuntimeError(f"chosen width of {width} not found amoung width options: {width_options}")

    # apply the selected finger width in mujoco (EI change requires reset to finalise)
    self.mj.set_finger_width(self.load_next.finger_width)

    if debug_fcn:
      print("the width which will be loaded is:", self.load_next.finger_width)
      print("final task_xml_folder is:", self.task_xml_folder)

  def _load_xml(self, test=None, index=None):
    """
    Load the mujoco instance with the given mjcf xml file name
    """

    global random_train

    if index:
      filename = self.task_xml_template.format(index)
    elif test is not None:
      # load the specified test xml
      filename = self.task_xml_template.format(test)
    else:
      # get a random task xml file
      r = random_train.integers(self.testing_xmls, self.testing_xmls + self.training_xmls)
      filename = self.task_xml_template.format(r)

    if self.log_level >= 3: 
      print("Load path: ", self.mj.model_folder_path
            + self.mj.object_set_name + "/" + self.task_xml_folder)
    if self.log_level >= 2: print("Loading xml: ", filename)

    # load the new task xml (old model/data are deleted)
    self.mj.load_relative(self.task_xml_folder + '/' + filename)
    self.num_objects = self.mj.get_number_of_objects()

    self.reload_flag = False

    # get the parameters from the newly loaded file
    try:
      self.params.object_set_name = self.mj.object_set_name
      self.params.num_segments = self.mj.get_N()
      self.params.finger_width = self.mj.get_finger_width()
      self.params.finger_thickness = self.mj.get_finger_thickness()
      self.params.finger_modulus = self.mj.get_finger_modulus()
      self.params.finger_length = self.mj.get_finger_length()
      self.params.finger_hook_angle_degrees = self.mj.get_finger_hook_angle_degrees()
      self.params.fixed_finger_hook = self.mj.is_finger_hook_fixed()
      self.params.fingertip_clearance = self.mj.get_fingertip_clearance()
      self.params.XY_base_actions = self.mj.using_xyz_base_actions()
      self.params.finger_hook_length = self.mj.get_finger_hook_length()
    except AttributeError as e:
      print("WARNING: MjEnv.load_xml() failed to update parameters for gripper dimensions, you may be running an old policy")
      print("Error is:", e)
      print("Execution is continuing, beware saving this model will have incorrect params")
      self.params = self.load_next

    # assume loading is correct and directly copy
    self.params.segment_inertia_scaling = self.load_next.segment_inertia_scaling

    # update base limits
    self.mj.set_base_XYZ_limits(self.params.base_lim_X_mm * 1e-3, 
                                self.params.base_lim_Y_mm * 1e-3,
                                self.params.base_lim_Z_mm * 1e-3)
    self.mj.set_base_yaw_limit(self.params.base_lim_yaw_rad)

  def _load_object_set(self, name=None, mjcf_path=None, num_segments=None, 
                       finger_width=None, auto_generate=False, use_hashes=True):
    """
    Load in an object set and sort out details (like number of xml files).
    This functions does NOT load a new XML file from this object set.
    """

    debug_fcn = False

    # if a mjcf_path is given, override, otherwise we use default
    if mjcf_path != None: self.mj.model_folder_path = mjcf_path

    # if a object set name is given, override, otherwise we use default from load_next
    if name != None: 
      self.mj.object_set_name = name
      self.load_next.object_set_name = name
    else:
      name = self.load_next.object_set_name
      self.mj.object_set_name = self.load_next.object_set_name

    if auto_generate:

      # warn the user that the given path value is going to be overriden
      if mjcf_path is not None:
        print(f"MjEnv() warning: given mjcf_path='{mjcf_path}' is about to be overriden by MjEnv._auto_generate_xml_file()")

      # create the file we need
      self.task_xml_folder = self._auto_generate_xml_file(self.load_next.object_set_name, use_hashes=use_hashes)

      # apply the selected finger width in mujoco (EI change requires reset to finalise)
      self.mj.set_finger_width(self.load_next.finger_width)

    else:

      # manually set the task name we should look for
      self._set_finger_variables(num_segments=num_segments, width=finger_width)

    # check the mjcf_path is correctly formatted
    if self.mj.model_folder_path[-1] != '/':
      self.mj.model_folder_path += '/'

    # now determine the model xml path
    self.xml_path = (self.mj.model_folder_path + self.mj.object_set_name 
                      + '/' + self.task_xml_folder + '/')

    # find out how many xmls are available for training/testing
    self.testing_xmls = int(np.ceil(self.params.test_objects / float(self.params.test_obj_per_file)))
    xml_files = [x for x in os.listdir(self.xml_path) if os.path.isdir(self.xml_path + "/" + x) is False
                                                        and x.endswith(".xml")]
    self.training_xmls = len(xml_files) - self.testing_xmls

    if debug_fcn or self.log_level >= 2:
      print("_load_object_set() gives xml path:", self.xml_path)
      print(f"Training xmls: {self.training_xmls}, testing xmls: {self.testing_xmls}")

    if self.training_xmls < 1:
      raise RuntimeError(f"enough training xmls failed to be found in MjEnv at: {self.xml_path}. xml files = {len(xml_files)}, testing xmls = {self.testing_xmls}")

  def _update_n_actions_obs(self):
    """
    Get an updated number of actions and observations
    """

    self.n_actions = self.mj.get_n_actions()
    self.n_obs = self.mj.get_n_obs()

  def _make_event_track(self):
    """
    Create an EventTrack object
    """
    return EventTrack()

  def _add_events(self, e1, e2):
    """
    Add two events and return the sum
    """
    return self.mj.add_events(e1, e2)

  def _calc_rewards(self, event):
    """
    Calculate a reward from a set of events
    """
    return self.mj.reward(event)

  def _get_machine(self):
    """
    Get the machine that the MjClass library is compiled for, and so presumably
    the machine which the code is currently running on.
    Current options: "luke-laptop", "luke-PC", "cluster"
    """
    return self.mj.machine

  def _get_cpp_settings(self):
    """
    Return a string of all the cpp simulation settings
    """

    cpp_settings = self.mj.set.get_settings()

    if self.mj.set.use_HER:
      her_settings = self.mj.goal.get_goal_info()
      cpp_settings += "\n" + her_settings

    return cpp_settings

  def _set_action(self, action):
    """
    Set the action, either for simulation or real world. If using simulation
    however, it is better to use the MjEnv._take_action() function
    """

    # for continous actions set them all, mag should be [-1, +1] and is clipped internally
    if self.mj.set.continous_actions:
      for i in range(len(action)):
        new_state = self.mj.set_continous_action(i, action[i])

    # for discrete actions, input the action to perform
    else: new_state = self.mj.set_discrete_action(action)

    return new_state

  def _take_action(self, action):
    """
    Take an action in the simulation
    """

    self._set_action(action)

    # step the simulation for the given time (mj.set.time_for_action)
    self.mj.action_step()

    return

  def _is_done(self):
    """
    Determine if the epsiode should finish.

    Terminated - episode reached termination condition
    Trucnated - episode exceeded max number of steps
    """

    terminated = False
    truncated = False

    # if we have exceeded our time limit
    if self.track.current_step >= self.params.max_episode_steps:
      if self.log_level >= 3 or self.mj.set.debug: 
        print("is_done() = true (in python) as max step number exceeded")
      truncated = True
      return terminated, truncated

    # check the cpp side
    terminated = self.mj.is_done()

    return terminated, truncated
    
  def _init_rgbd(self, width=None, height=None):
    """
    Initialise an rgbd camera in the simulation. Default size matches realsense.
    """
    
    self.rgbd_enabled = self.mj.init_rgbd()
    if self.rgbd_enabled:
      self.params.depth_camera = True
      self._set_rgbd_size(width, height)
    else:
      if self.log_level > 0:
        print("MjEnv() failed to initialise an RGBD camera, not enabled in compilation")
      self.params.depth_camera = False

    # load the standard image transform
    if self.params.use_standard_transform:
      self.std_img_transform = image_transform.standard_transform(self.params.transform_resize_square,
                                                                  self.params.transform_crop_size)
    else:
      self.std_img_transform = None

    # return if the camera is running or not
    return self.params.depth_camera

  def _set_rgbd_size(self, width=None, height=None, default=False):
    """
    Set the size of simulated RGBD images
    """

    if width is None: width = self.params.image_width
    if height is None: height = self.params.image_height
    if default and width is None and height is None:
      width = self.default_rgbd_width
      height = self.default_rgbd_height

    self.rgbd_enabled = self.mj.rendering_enabled()
    self.params.image_width = width
    self.params.image_height = height

    if self.rgbd_enabled:
      self.mj.set_RGBD_size(width, height)
    else:
      if self.log_level > 0:
        print("MjClass rendering is disabled in compilation settings, no RGBD images possible")
      self.rgbd_enabled = False

  def _get_rgbd_image(self, mask=False, transform=False):
    """
    Return rgbd data from the current state of the simulation, rescaled to match
    the size of the rgbd data from real life.

    Returns two numpy arrays ready for conversion into torch:
      - rgb    (with shape 3 x width x height)
      - depth  (with shape 1 x width x height)

    If mask=True returns a segmentation mask
    """

    if not self.rgbd_enabled:
      if self.log_level > 0:
        print("MjClass rendering not enabled, unabled to get RGBD image")
      if self.params.depth_camera:
        if self.log_level > 0:
          print("self.params.depth_camera is True, running _init_rgbd()")
        self._init_rgbd()
        if not self.rgbd_enabled: 
          print("Warning: MjClass rendering failed, disabled by compilation, _get_rgbd() is returning NONE")
          return
      else: 
        print("Warning: MjClass rendering not enabled and self.params.depth_camera is false, unabled to get RGBD image, _get_rgbd() returning NONE")
        return

    # get rgbd information out of the simulation (unit8, float)
    if mask:
      rgb, depth = self.mj.get_mask_numpy()
    else:
      rgb, depth = self.mj.get_RGBD_numpy()

    # reshape the numpy arrays to the correct aspect ratio
    rgb = rgb.reshape(self.params.image_height, self.params.image_width, 3)
    depth = depth.reshape(self.params.image_height, self.params.image_width, 1)

    # if we are applying the transforms used for image rendering models (for visualisation)
    if transform:

      # convert to torch image
      rgb = np.transpose(rgb, (2, 0, 1))
      img_torch = torch.tensor(rgb, dtype=torch.float)

      if self.params.use_standard_transform:
        img_torch = self.std_img_transform(img_torch)
      else:
        if self.log_level >= 0:
          print("MjEnv._get_rgbd_image() warning: transform=True but self.params.use_standard_transform=False")
        std_transform = image_transform.standard_transform(self.params.transform_resize_square,
                                                              self.params.transform_crop_size)
        img_torch = std_transform(img_torch)
      
      # now revert the image to numpy unit8 from float [0, 1], since we want to visualise the transform
      img_torch = torch.transpose(img_torch, 1, 2) # rotate the image so it comes out properly
      rgb = (img_torch.clamp(-1.0, 1.0).cpu().float().numpy() + 1) / 2.0 * 255.0
      rgb = np.transpose(rgb, (2, 1, 0)).astype(np.uint8)

    # numpy likes image arrays like this: width x height x channels
    # torch likes image arrays like this: channels x width x height
    rgb = np.transpose(rgb, (2, 1, 0))
    depth = np.transpose(depth, (2, 1, 0))

    return rgb, depth # ready for conversion to torch tensors
  
  def _plot_rgbd_image(self, mask=False, transform=False):
    """
    Get and then plot an rgbd image from the simulation. Use this for debugging only
    """

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1)
    
    rgb, depth = self._get_rgbd_image(mask=mask, transform=transform)

    # numpy likes image arrays like this: height x width x channels (ie rows x columns x dim)
    # torch likes image arrays like this: channels x width x height
    # hence convert from torch style back to numpy style for plotting
    # axs[0].imshow(np.einsum("ijk->kji", rgb)) # swap to numpy style rows/cols (eg 3x640x480 -> 480x640x3)
    axs[0].imshow(np.transpose(rgb, (2, 1, 0)))
    axs[1].imshow(np.transpose(depth[0])) # remove the depth 'channel' then swap (eg 1x640x480 -> 480x640)

    plt.show()

  def _plot_scene_mask(self):
    """
    Plot a mask of the scene, with different colours for different parts
    """

    # # make the mask in the simulation
    # if mask == "object":
    #   self.mj.create_ground_mask()
    # elif mask == "gripper":
    #   self.mj.create_gripper_mask()
    # elif mask.startswith("finger"):
    #   if mask.endswith("s"):
    #     for i in range(3):
    #       self.mj.create_finger_mask(i + 1)
    #   elif mask.endswith("1"):
    #     self.mj.create_finger_mask(1)
    #   elif mask.endswith("2"):
    #     self.mj.create_finger_mask(2)
    #   elif mask.endswith("3"):
    #     self.mj.create_finger_mask(3)
    # elif mask == "palm":
    #   self.mj.create_finger_mask(4)
    # elif mask == "ground":
    #   self.mj.create_ground_mask()

    # get an rgbd image of the mask
    rgb, depth = self._get_rgbd_image(mask=True)

    # mask is only in red channel, prepare to post-process
    segmented_image = rgb[0]
    max_num = np.max(segmented_image)
    scale = 255 // max_num

    # apply scalings to give each mask integers (1-7 usually) a different colour
    red_scaled = scale * (segmented_image)
    green_scaled = scale * 2 * (max_num - segmented_image) * (segmented_image % 3)
    blue_scaled = scale * 2 * (max_num - segmented_image) * (segmented_image + 2 % 3)

    # reconstruct a three-channel image
    rgb = np.array([red_scaled, green_scaled, blue_scaled], dtype=np.uint8)

    # plot the visualisation of the mask
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 1)
    axs.imshow(np.einsum("ijk->kji", rgb)) # swap to numpy style rows/cols (eg 3x640x480 -> 480x640x3)
    plt.show()

    return

  def _next_observation(self):
    """
    Returns the next observation from the simuation
    """

    # get an observation as a numpy array
    try:
      obs = self.mj.get_observation_numpy()
    except AttributeError as e:
      print("Error is:", e)
      obs = self.mj.get_observation()

    if self.torch:
      obs = torch.tensor(obs, device=self.torch_device)

    # if we are using a depth camera, combine this into the observation
    if self.params.depth_camera:

      # check if we should save an image for image collection
      do_image_collection = False
      if self.collect_images:
        global random_train
        if self.image_collection_chance > random_train.random():
          do_image_collection = True

      # determine if we are adding this image into our observation of the scene
      image_observation = False
      if self.params.use_rgb_in_observation or self.params.use_depth_in_observation:
        image_observation = True

      if image_observation or do_image_collection:

        # get the rgb and depth from the scene
        rgb, depth = self._get_rgbd_image()

        if self.params.use_rgb_in_observation:

          if not self.params.use_rgb_rendering:

            # normalise to range 0-1 (worth normalising? Changing range [-1, +1]?)
            if self.torch:
              img_obs = torch.tensor(rgb, device=self.torch_device, dtype=torch.float32)
              if self.params.use_standard_transform:
                img_obs = self.std_img_transform(img_obs)
              else:
                img_obs = torch.divide(img_obs, 255)
            else:
              img_obs = np.divide(img_obs, 255, dtype=np.float32)
            
          # we are using rgb observation rendering
          else:

            if self.log_level >= 3:
              print("Preparing to use rgb rendering model on image observation")

            # convert to cpu tensor initially as transform operations are on cpu
            img_obs = torch.tensor(rgb, device="cpu")

            # we need to transpose and end up with a torch tensor
            if self.torch:
              img_obs = torch.transpose(img_obs, 1, 2)
            else:
              img_obs = np.transpose(img_obs, (0, 2, 1))
              img_obs = torch.tensor(img_obs)

            # now transform the image into the networks expected format
            if self.params.use_standard_transform:
              img_torch = self.std_img_transform(img_obs)
            else:
              # use the loaded transform
              img_torch = self.img_transform(img_obs)

            # unsqueeze to simulate batch
            img_torch = img_torch.unsqueeze(0)

            t4 = time.time()

            # now run it through the rendering model (preferably on GPU)
            img_obs.to(self.torch_device)
            with torch.no_grad():
              img_obs = self.render_net(img_torch.to(self.torch_device))
              # img_obs = torch.nn.AdaptiveAvgPool2d((1, 1))

            img_obs = img_obs.squeeze(0)

            if self.torch:
              img_obs = torch.transpose(img_obs, 1, 2)
            else:
              # change back to numpy
              img_obs = img_obs.clamp(-1.0, 1.0).cpu().float().numpy()
              img_obs = np.transpose(img_obs, (0, 2, 1))

        if self.params.use_depth_in_observation:

          if self.torch:
            depth = torch.tensor(depth, device=self.torch_device)

          if self.params.use_rgb_in_observation:
            if self.torch:
              img_obs = torch.concat((img_obs, depth), axis=0)
            else:
              img_obs = np.concatenate((img_obs, depth), axis=0)
          else:
            img_obs = depth

        # add the regular observation, reshaped and padded with zeros, to the image observation
        if image_observation:

          channels, width, height = img_obs.shape

          if self.torch:

            new_obs = torch.zeros(width * height, device=self.torch_device)
            new_obs[:len(obs)] = obs
            obs = new_obs.reshape((1, width, height))
          else:
            obs.resize(width * height)
            obs = obs.reshape((1, width, height))

          check = False
          if check:
            print("img_obs.shape", img_obs.shape)
            print("obs.shape", obs.shape)

          if len(obs) > width * height:
            raise RuntimeError(f"MjEnv()._next_observation() failed as observation length {len(obs)} exceeds image number of pixels {width * height}")

          if self.torch:
            obs = torch.concatenate((img_obs, obs), axis=0)
          else:
            obs = np.concatenate((img_obs, obs), axis=0, dtype=np.float32)

        if do_image_collection:

          # get segmentation mask of the scene
          rgb_mask, _ = self._get_rgbd_image(mask=True)

          # get details of the object
          details_dict = {
            "object_names" : self.mj.get_live_object_names(),
            "object_bounding_boxes" : self.mj.get_object_bounding_boxes(),
            "object_relative_XY" : self.mj.get_object_XY_relative_to_gripper(),
          }

          self.image_collection_data = {
            "rgb" : rgb,
            "depth" : depth,
            "rgb_mask" : rgb_mask[0],
            "obs" : obs,
            "details" : details_dict,
          }

          self.new_image_collected = True

    return obs

  def _event_state(self):
    """
    Get the state of the simluation, including the observation
    """
    return self.mj.get_event_state()

  def _get_desired_goal(self):
    """
    Get the current desired goal in the simulation
    """
    return self.mj.get_goal()

  def _assess_goal(self, state):
    """
    Assess goal performance given a state vector
    """
    return self.mj.assess_goal(state)

  def _reward(self):
    """
    Calculate the reward on this step
    """

    return self.mj.reward()

  def _object_discrimination_target(self):
    """
    Get the reward from predicting the object from a vector:
      [a, b] where:
        a = [x, y, z] envelope of the object normalised -1 to +1
        b = [b1, b2, b3, ... , bn] from -1 to +1 for n categories
    """

    [x, y, z] = self.mj.get_object_xyz_bounding_box()
    name = self.mj.get_current_object_name()

    cat = np.zeros(5, dtype=np.float64)

    if name.startswith("sphere"):
      cat[0] = 1
    elif name.startswith("cube"):
      cat[1] = 1
    elif name.startswith("cuboid"):
      cat[2] = 1
    elif name.startswith("cylinder"):
      cat[3] = 1
    elif name.startswith("ellipse"):
      cat[4] = 1

    # scale the bounding box to the range [0, 1]
    box_min = 0e-3
    box_max = 200e-3

    x = max(box_min, min(box_max, x)) * (1.0/(box_max - box_min))
    y = max(box_min, min(box_max, y)) * (1.0/(box_max - box_min))
    z = max(box_min, min(box_max, z)) * (1.0/(box_max - box_min))

    box = np.array([x, y, z], dtype=np.float32) # float32 for torch
    target = np.concatenate((box, cat))

    # convert target from [0,1] to [-1,1]
    target *= 2
    target -= 1

    return target

  def _goal_reward(self, goal, state):
    """
    Calculate the reward based on a state and goal
    """
    return self.mj.reward(goal, state)

  def _spawn_scene(self):
    """
    Spawn objects into a scene
    """

    global random_test
    global random_train

    # are we using train or test time random generator
    if self.test_in_progress:
      generator = random_test
    else:
      generator = random_train

    # # if we are doing a test, chose a specific object
    # if self.test_in_progress:
    #   obj_idx = self.current_test_trial.obj_idx
    # else:
    #   # otherwise choose a random object
    #   obj_idx = generator.integers(0, self.num_objects)

    if self.params.num_objects_in_scene > self.num_objects:
      raise RuntimeError(f"MjEnv._spawn_scene() error: num_objects_in_scene ({self.params.num_objects_in_scene}) > num_objects {self.num_objects}")

    # determine origin noise for XY and set the gripper there
    origin_noise_X_mm = generator.integers(-self.params.origin_noise_X_mm, self.params.origin_noise_X_mm + 1)
    origin_noise_Y_mm = generator.integers(-self.params.origin_noise_Y_mm, self.params.origin_noise_Y_mm + 1)
    self.mj.set_new_base_XY(origin_noise_X_mm * 1e-3, origin_noise_Y_mm * 1e-3)

    # # determine origin noise for gripper yaw rotation and set it to the gripper
    # if self.params.Z_base_rotation:
    #   origin_noise_Z_yaw = ((generator.random() * 2) - 1) * self.params.base_lim_yaw_rad
    #   self.mj.set_new_base_yaw(origin_noise_Z_yaw)

    # shuffle possible objects ensuring no duplicates
    obj_idx = np.array(list(range(self.num_objects)))
    generator.shuffle(obj_idx)

    # apply spawning parameters
    self.mj.default_spawn_params.x = origin_noise_X_mm * 1e-3
    self.mj.default_spawn_params.y = origin_noise_Y_mm * 1e-3
    self.mj.default_spawn_params.xrange = self.params.object_position_noise_mm * 1e-3
    self.mj.default_spawn_params.yrange = self.params.object_position_noise_mm * 1e-3
    self.mj.default_spawn_params.xmin = -self.params.scene_X_dimension_mm * 1e-3
    self.mj.default_spawn_params.xmax = self.params.scene_X_dimension_mm * 1e-3
    self.mj.default_spawn_params.ymin = -self.params.scene_Y_dimension_mm * 1e-3
    self.mj.default_spawn_params.ymax = self.params.scene_Y_dimension_mm * 1e-3
    self.mj.default_spawn_params.rotrange = np.pi / 2.0
    self.mj.default_spawn_params.smallest_gap = 5e-3 # avoid collisions after 1st action

    num_spawned = 0

    for i in range(self.params.num_objects_in_scene):

      spawned = False
      max_count = 3
      count = 0

      while not spawned and count < max_count:

        spawned = self.mj.spawn_into_scene(obj_idx[i])
        count += 1
        if not spawned and self.log_level >= 3:
          print(f"Object '{self.mj.get_object_name(obj_idx[i])}' failed to spawn, try {count}")

      if spawned:
        num_spawned += 1
      else:
        if self.log_level >= 2:
          print(f"Warning: object '{self.mj.get_object_name(obj_idx[i])}' failed to spawn")

    # did we fail to spawn any objects
    if num_spawned == 0:
      if self.log_level >= 1:
        print(f"MjEnv._spawn_scene() warning: no objects were able to be spawned in this scene, recursively trying again, this was attempt {self.scene_recursion_counter + 1}")
      self.scene_recursion_counter += 1
      if self.scene_recursion_counter > 10:
        raise RuntimeError("MjEnv._spawn_scene() failed after 10 attempts")
      self._spawn_scene()
      return
    else:
      self.scene_recursion_counter = 0

    # episode done when number of target objects is grasped
    if self.params.scene_grasp_target > num_spawned:
      self.mj.set_scene_grasp_target(num_spawned)
    else:
      self.mj.set_scene_grasp_target(self.params.scene_grasp_target)

  def _spawn_object(self):
    """
    Spawn an object into the simulation randomly
    """

    if self.params.use_scene_settings:
      self._spawn_scene()
      return

    global random_test
    global random_train

    new = True

    # are we using train or test time random generator
    if self.test_in_progress:
      generator = random_test
    else:
      generator = random_train

    # if we are doing a test, chose a specific object
    if self.test_in_progress:
      obj_idx = self.current_test_trial.obj_idx
    else:
      # otherwise choose a random object
      obj_idx = generator.integers(0, self.num_objects)

    # # determine noise for gripper yaw rotation and set it to the gripper
    # if self.params.Z_base_rotation:
    #   noise_Z_yaw = ((generator.random() * 2) - 1) * (np.pi / 6)
    #   self.mj.set_new_base_yaw(noise_Z_yaw)

    if new:

      # apply spawning parameters
      self.mj.default_spawn_params.xrange = self.params.object_position_noise_mm * 1e-3
      self.mj.default_spawn_params.yrange = self.params.object_position_noise_mm * 1e-3
      self.mj.default_spawn_params.rotrange = np.pi / 2.0

      spawned = False
      max_count = 3
      count = 0

      while not spawned and count < max_count:

        spawned = self.mj.spawn_into_scene(obj_idx)
        count += 1
        if not spawned and self.log_level >= 3:
          print(f"Object '{self.mj.get_object_name(obj_idx)}' failed to spawn, try {count}")

      if not spawned:
        if self.log_level >= 2:
          print(f"Warning: object '{self.mj.get_object_name(obj_idx)}' failed to spawn, resorting to old method")

        # use old method: randomly generate the object (x, y) position
        xy_noise = self.params.object_position_noise_mm
        x_pos_mm = generator.integers(-xy_noise, xy_noise + 1)
        y_pos_mm = generator.integers(-xy_noise, xy_noise + 1)

        # randomly choose a z rotation
        angle_options = [0, 60, 120]
        th_noise = self.params.object_rotation_noise_deg
        angle_noise_deg = generator.integers(-th_noise, th_noise + 1)
        rand_index = generator.integers(0, len(angle_options))
        z_rot_rad = (angle_options[rand_index] + angle_noise_deg) * (np.pi / 180.0)

        # spawn in the object
        self.mj.spawn_object(obj_idx, x_pos_mm * 1e-3, y_pos_mm * 1e-3, z_rot_rad)

        # for observing failure cases
        # self.render()
        # input("enter to continue")
        
    else:

      # randomly generate the object (x, y) position
      xy_noise = self.params.object_position_noise_mm
      x_pos_mm = generator.integers(-xy_noise, xy_noise + 1)
      y_pos_mm = generator.integers(-xy_noise, xy_noise + 1)

      # randomly choose a z rotation
      angle_options = [0, 60, 120]
      th_noise = self.params.object_rotation_noise_deg
      angle_noise_deg = generator.integers(-th_noise, th_noise + 1)
      rand_index = generator.integers(0, len(angle_options))
      z_rot_rad = (angle_options[rand_index] + angle_noise_deg) * (np.pi / 180.0)

      # spawn in the object
      self.mj.spawn_object(obj_idx, x_pos_mm * 1e-3, y_pos_mm * 1e-3, z_rot_rad)

    return

  def _end_test(self):
    """
    End test mode, called automatically and sets the test_completed flag
    """

    self.test_in_progress = False
    self.test_completed = True
    self.current_test_trial = MjEnv.Test()
    self.reload_flag = True

  def _monitor_test(self):
    """
    Monitor the current test, called automatically
    """

    # safety checks
    if not self.test_in_progress:
      return
    if not self.track.is_done:
      return

    # get test report
    test_report = self.mj.get_test_report()

    trial_data = MjEnv.Test()

    # save information about this trial
    trial_data.obj_idx = self.current_test_trial.obj_idx
    trial_data.obj_trial = self.current_test_trial.obj_trial
    trial_data.object_name = test_report.object_name
    trial_data.reward = self.track.cumulative_reward
    trial_data.cnt = test_report.cnt

    # get the object category by assuming its the first word of _ seperated string
    trial_data.object_category = trial_data.object_name.split("_")[0]

    # insert information into stored data list
    self.test_trial_data.append(trial_data)

    # increment object trial
    self.current_test_trial.obj_trial += 1

    # if trials done, move to the next object, reset trial counter
    if self.current_test_trial.obj_trial >= self.params.test_trials_per_object:

      self.current_test_trial.obj_idx += 1
      self.current_test_trial.obj_trial = 0
      self.current_test_trial.obj_counter += 1

      # check if we have finished
      if self.current_test_trial.obj_counter >= self.params.test_objects:
        self._end_test()

      # check if we need to load the next test object set
      elif self.current_test_trial.obj_idx >= self.params.test_obj_per_file:
        self.current_test_trial.xml_file += 1
        self._load_xml(test=self.current_test_trial.xml_file)
        self.current_test_trial.obj_idx = 0

  def _initialise_heuristic_parameters(self):
    """
    Get initial values for heuristic parameters
    """

    self.heuristic_params = {
      "target_angle_deg" : 15,
      "target_wrist_force_N" : 2,
      "min_x_value_m" : 80e-3,
      "target_x_constrict_m" : 110e-3,
      "target_palm_bend_increase_percentage" : 10,
      "target_z_position_m" : 80e-3,
      "bend_history_length" : 6, # how many historical bending values to save
      "bend_update_length" : 3,  # how many values in history do we consider 'new'
      "initial_z_height_target_m" : 5e-3,
      "final_z_height_target_m" : -20e-3,
      "initial_bend_target_N" : 1.5,
      "initial_palm_target_N" : 1.5,
      "limit_bend_target_N" : 3, #self.mj.set.stable_finger_force_lim,
      "limit_palm_target_N" : 3, # self.mj.set.stable_palm_force_lim,
      "final_bend_target_N" : 1.5,
      "final_palm_target_N" : 1.5,
      "final_squeeze" : True,
      "fixed_angle" : True,
      # "palm_first" : True,
    }

  # ----- utilities for loading CUT/cycleGAN models ----- #

  def _load_image_rendering_options(self, filepath, filename="train_opt.txt", test=True):
    """
    Loads the model options from a given options filename (should include full path)
    """

    log_num = 2
    if self.log_level >= log_num:
      print(f"MjEnv._load_image_rendering_options() loading from: {filename}")

    from options.test_options import TestOptions
    from options.train_options import TrainOptions

    # initially load test options
    if test:
      opt = TestOptions(cmd_line="").parse()
    else:
      opt = TrainOptions(cmd_line="").parse()

    # now open the file
    with open(f"{filepath}/{filename}", "r") as f:
      lines = f.readlines()

    # now we check for any options which are not set to default
    for l in lines:
      if "[default:" in l:
        # replace key tokens so we can split our line easily
        l = l.replace("[default:", ":")
        splits = l.split(":")
        option_name = "".join(splits[0].split()) # remove whitespace
        option_val = "".join(splits[1].split())  # remove whitespace
        option_default = "".join((splits[2].split("]")[0]).split())

        if self.log_level >= log_num:
          print(f" -> Found option={option_name} with new value={option_val}, whereas default={option_default}")

        # if we don't have a number, add in string quotes
        if not option_val.isnumeric():
          option_val = "'" + option_val + "'"

        # now apply this option change to opt
        to_exec = f"opt.{option_name} = {option_val}"
        exec(to_exec)

    opt.checkpoints_dir = f"{filepath}/checkpoints"

    if test:
      opt.num_threads = 0   # test code only supports num_threads = 1
      opt.batch_size = 1    # test code only supports batch_size = 1
      opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
      opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
      opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    # loading doesn't work correctly without reseting this back to None
    opt.isTrain = None

    return opt

  def _load_image_rendering_model(self, device="cuda", initial_load_cpu=True, loadA=True):
    """
    Load a model to perform image rendering on images extracted from the
    simulator. Can initially load on the cpu which is slower but reduces the
    surge of GPU RAM usage
    """

    self.set_device(device)

    global to_pil_image
    from torchvision.transforms.functional import to_pil_image

    # CUT repo path
    cut_path = "/home/luke/repo/CUT"
    sys.path.insert(0, f"{cut_path}")

    from models import create_model

    if self.log_level >= 1:
        print(f"MjEnv._load_image_rendering_model() is preparing to load a {self.params.rgb_rendering_method}")

    if self.params.rgb_rendering_method.lower() == "cut":

      # load a specific, hardcoded CUT model
      epoch = 245
      experiment = "single_object_128_1"
      path_to_load = f"image_render_models/CUT/{experiment}"
      file_to_load = f"{epoch}_net_G.pth"
      full_path = f"{path_to_load}/{file_to_load}"

      # load the model options and then create the model
      opt = self._load_image_rendering_options(path_to_load)
      model = create_model(opt)
      model.opt.isTrain = False # put into test mode (ie evaluation mode)
      if initial_load_cpu: model.opt.gpu_ids = [] # force model to be created on cpu

      # extract the generator network, alongside the image size, and image transform
      self.render_net = model.netG
      self.render_net_output_size = (model.opt.crop_size, model.opt.crop_size)
      self.img_transform = image_transform.get_transform_2(opt)

      # load the model weights and move to the desired device
      weights = torch.load(full_path, map_location="cpu" if initial_load_cpu else None)
      self.render_net.load_state_dict(weights)
      self.render_net.to(self.torch_device)

    elif self.params.rgb_rendering_method.lower() == "cyclegan":

      # load a specific, hardcoded cycleGAN model
      load_GA = True # else load GB
      epoch = 115
      experiment = "single_object_128_CG"
      path_to_load = f"image_render_models/cycleGAN/{experiment}"
      file_to_load_GA = f"{epoch}_net_G_A.pth"
      file_to_load_GB = f"{epoch}_net_G_B.pth"
      if load_GA:
        full_path = f"{path_to_load}/{file_to_load_GA}"
      else:
        full_path = f"{path_to_load}/{file_to_load_GB}"

      # load the model options and then create the model
      opt = self._load_image_rendering_options(path_to_load)
      model = create_model(opt)
      model.opt.isTrain = False # put into test mode (ie evaluation mode)
      if initial_load_cpu: model.opt.gpu_ids = [] # force model to be created on cpu

      # extract the generator network, alongside the image size, and image transform
      self.render_net = model.netG_A
      self.render_net_output_size = (model.opt.crop_size, model.opt.crop_size)
      self.img_transform = image_transform.get_transform_2(opt)

      # load the model weights and move to the desired device
      weights = torch.load(full_path, map_location="cpu" if initial_load_cpu else None)
      self.render_net.load_state_dict(weights)
      self.render_net.to(self.torch_device)

    elif self.params.rgb_rendering_method.lower() == "cyclegan_encoder":

      # load a specific, hardcoded cycleGAN model
      epoch = 400
      experiment = "gan_52"
      path_to_load = f"/home/luke/mujoco-devel/rl/env/image_render_models/cycleGAN/{experiment}"
      file_to_load_GA = f"{epoch}_net_G_A.pth"
      file_to_load_GB = f"{epoch}_net_G_B.pth"
      if loadA:
        full_path = f"{path_to_load}/{file_to_load_GA}"
      else:
        full_path = f"{path_to_load}/{file_to_load_GB}"

      # load the model options and then create the model
      opt = self._load_image_rendering_options(path_to_load)
      model = create_model(opt)
      model.opt.isTrain = False # put into test mode (ie evaluation mode)
      if initial_load_cpu: model.opt.gpu_ids = [] # force model to be created on cpu

      # special added feature: only load GA
      if loadA:
        opt.GA_only = True

      # extract the generator network, alongside the image size, and image transform
      self.render_net = model.netG_A
      self.render_net_output_size = (model.opt.crop_size, model.opt.crop_size)
      self.img_transform = image_transform.get_transform_2(opt)

      # load the model weights and move to the desired device
      weights = torch.load(full_path, map_location="cpu" if initial_load_cpu else None)
      self.render_net.load_state_dict(weights)
      self.render_net.to(self.torch_device)

      # # trim the generator network to use as an encoder
      # num_resnet_blocks = 9
      # self.render_net.model = self.render_net.model[:12 + num_resnet_blocks]

    else:
      raise RuntimeError(f"MjEnv._load_image_rendering_model() error: unrecognised rgb_rendering_method = {self.params.rgb_rendering_method}")

    if self.log_level >= 1:
        print(f"{self.params.rgb_rendering_method} model now loaded and moved onto device = {self.torch_device}")

    # we can now use rgb rendering
    self.params.use_rgb_rendering = True

  def _preprocess_real_image(self, rgb):
    """
    Prepares a real image as an image observation, expected as a numpy array in
    standard numpy format (height x width x channels). The output will be on
    self.torch_device, set in the class
    """

    # convert to torch image
    rgb = np.transpose(rgb, (2, 0, 1))
    img_torch = torch.tensor(rgb, dtype=torch.float, device=self.torch_device)

    if self.params.use_standard_transform:
      img_torch = self.std_img_transform(img_torch)

    if self.params.use_rgb_rendering and self.render_net is not None:
      img_torch = img_torch.unsqueeze(0) # from [C, W, H] -> [B, C, W, H]
      with torch.no_grad():
        img_torch = self.render_net(img_torch)
      img_torch = img_torch.squeeze(0) # back to [C, W, H]

    return img_torch
  
  def _make_img_obs(self, rgb, obs):
    """
    Make an observation that combines an image and regular sensor data
    """

    # move both to the current device
    rgb = rgb.to(self.torch_device)
    if not torch.is_tensor(obs):
      obs = torch.tensor(obs, device=self.torch_device)
    else:
      obs.to(self.torch.device)

    channels, width, height = rgb.shape

    new_obs = torch.zeros(width * height, device=self.torch_device)
    new_obs[:len(obs)] = obs
    obs = new_obs.reshape((1, width, height))

    check = False
    if check:
      print("img_obs.shape", rgb.shape)
      print("obs.shape", obs.shape)

    if len(obs) > width * height:
      raise RuntimeError(f"MjEnv()._next_observation() failed as observation length {len(obs)} exceeds image number of pixels {width * height}")

    obs = torch.concatenate((rgb, obs), axis=0)
    obs = obs.unsqueeze(0) # from [C, W, H] -> [B, C, W, H] 

    return obs

  # ----- public functions ----- #

  def set_device(self, device):
    """
    Set the torch device to use with this class. Also sets torch=True, so all
    input and output tensors will be torch.tensors rather than numpy.array
    """

    if device == "cuda" and not torch.cuda.is_available():
      if self.log_level >= 1:
        print("MjEnv.set_device() received request for 'cuda', but torch.cuda.is_available() == False, hence using cpu")
      device = "cpu"

    self.torch_device = torch.device(device)
    self.torch = True

  def yield_load(self):
    """
    Get the yield load of the current fingers, even if we have not fully loaded
    them in yet
    """

    debug_fcn = False

    loaded_yield = self.mj.yield_load()
    if debug_fcn: print("The loaded yield is", loaded_yield)

    to_load_yield = self.mj.yield_load(self.params.finger_thickness,
                                       self.load_next.finger_width)
    if debug_fcn: print("The to load yield is", to_load_yield)

    return to_load_yield

  def start_heuristic_grasping(self, realworld=False):
    """
    Prepare to begin a heuristic grasping procedure.
    """

    self.grasp_phase = 0
    self.palm_done = False
    self.gauge_read_history = np.array([])
    self.heuristic_real_world = realworld

    # for help debugging
    print_on = False
    if print_on:
      bending = self.mj.set.bending_gauge.in_use
      palm = self.mj.set.palm_sensor.in_use
      wrist = self.mj.set.wrist_sensor_Z.in_use
      print(f"Sensors (bend, palm, wrist) are ({bending}, {palm}, {wrist})")

  def get_heuristic_action(self):
    """
    Return an action based on a simplistic grasping strategy. The grasp has
    the following phases:
      - 1. Lower fingers to table height
      - 2. Angle fingers at 20deg
      - 3. Constrict fingers to threshold force
      - 4. Advance palm until threshold force
      - 5. Lift to target height
      - 6. Loop squeezing actions to aim for specific grasp force
    """

    def bend_to_force(target_force_N, possible_action=None, limit_force_N=10):
      """
      Try to bend the fingers to a certain force
      """

      if possible_action is None:
        possible_action = X_close

      action = None

      # if we have access to bending sensors
      if bending:
        # wait for a certain bending
        if print_on:
          print(f"target bending is {target_force_N}, actual is {avg_bend}")
        if avg_bend < target_force_N:
          action = possible_action
          # if we have closed as much as we can
          if state_readings[0] < min_x_value_m:
            # bend a bit more
            action = Y_close
            # halt as we have reached gripper limits
            if print_on:
              print(f"minimum x is {min_x_value_m}, actual is {state_readings[0]}")
        elif avg_bend > limit_force_N:
          action = possible_action + 1

      else:
        # simply constrict to a predetermined point
        if print_on:
          print(f"x constrict target: {target_x_constrict_m}, actual: {state_readings[0]}")
        if state_readings[0] > target_x_constrict_m:
          action = possible_action

      return action

    def palm_push_to_force(target_force_N, limit_force_N=10):
      """
      Try to push with the palm to a specific force
      """

      time.sleep(0.2)

      action = None

      # if we have palm sensing
      if True or palm:
        if print_on:
          print(f"target palm force is {target_force_N}, actual is {palm_reading}")
        if palm_reading < target_force_N:
          action = Z_plus
        elif palm_reading > limit_force_N:
          action = Z_minus

      # if we don't have palm sensing, but we do have bend sensing
      elif bending:
        # start saving previous gauge readings
        if len(self.gauge_read_history) < bend_history_length:
          self.gauge_read_history = np.append(self.gauge_read_history, avg_bend)
        else:
          self.gauge_read_history[:-1] = self.gauge_read_history[1:]
          self.gauge_read_history[-1] = avg_bend
        if len(self.gauge_read_history) < bend_update_length + 1:
          if print_on:
            print("shortcircuit by giving Z_plus")
          return Z_plus # shortcut to ensure our buffer is full before logic
        # if we have finger bending data, try to infer palm contact
        old_bend_avg = np.mean(self.gauge_read_history[:-bend_update_length])
        new_bend_avg = np.mean(self.gauge_read_history[bend_history_length - bend_update_length:])
        # see if bending values have increased by a certain percentage
        if print_on:
          print(f"old bend avg: {old_bend_avg}, new bend avg: {new_bend_avg}, factor: {new_bend_avg / old_bend_avg}, target: {1 + (target_palm_bend_increase_percentage / 100.0)}")
        if new_bend_avg / old_bend_avg < 1 + (target_palm_bend_increase_percentage / 100.0):
          action = Z_plus
      
      # we have neither palm nor bend sensing
      else:
        # simply aim for a target palm position
        if state_readings[2] < target_z_position_m:
          action = Z_plus

      return action

    print_on = False

    global random_train

    # first, determine what sensors are available
    bending = self.mj.set.bending_gauge.in_use
    palm = self.mj.set.palm_sensor.in_use
    wrist = self.mj.set.wrist_sensor_Z.in_use

    bending = True
    palm = True
    wrist = True

    # extract parameters from dictionary
    target_angle_deg = self.heuristic_params["target_angle_deg"]
    target_wrist_force_N = self.heuristic_params["target_wrist_force_N"]
    min_x_value_m = self.heuristic_params["min_x_value_m"]
    target_x_constrict_m = self.heuristic_params["target_x_constrict_m"]
    target_palm_bend_increase_percentage = self.heuristic_params["target_palm_bend_increase_percentage"]
    target_z_position_m = self.heuristic_params["target_z_position_m"]
    bend_history_length = self.heuristic_params["bend_history_length"]
    bend_update_length = self.heuristic_params["bend_update_length"]
    initial_z_height_target_m = self.heuristic_params["initial_z_height_target_m"]
    final_z_height_target_m = self.heuristic_params["final_z_height_target_m"]
    initial_bend_target_N = self.heuristic_params["initial_bend_target_N"]
    initial_palm_target_N = self.heuristic_params["initial_palm_target_N"]
    limit_bend_target_N = self.heuristic_params["limit_bend_target_N"]
    limit_palm_target_N = self.heuristic_params["limit_palm_target_N"]
    final_bend_target_N = self.heuristic_params["final_bend_target_N"]
    final_palm_target_N = self.heuristic_params["final_palm_target_N"]
    final_squeeze = self.heuristic_params["final_squeeze"]
    fixed_angle = self.heuristic_params["fixed_angle"]
    # palm_first = self.heuristic_params["palm_first"]

    # hardcode action values
    X_close = 0
    X_open = 1
    Y_close = 2
    Y_open = 3
    Z_plus = 4
    Z_minus = 5
    H_down = 6
    H_up = 7

    # get sensor output if we can
    state_readings = self.mj.get_state_metres(self.heuristic_real_world)
    if bending:
      bending_readings = self.mj.get_finger_forces(self.heuristic_real_world)
      avg_bend = (bending_readings[0] + bending_readings[1] + bending_readings[2]) / 3.
      # max_bend = max(bending_readings)
      # avg_bend = max_bend
    if palm:
      palm_reading = self.mj.get_palm_force(self.heuristic_real_world)
      print("palm reading is", palm_reading)
    if wrist:
      wrist_reading = self.mj.get_wrist_force(self.heuristic_real_world)

    action = None

    # lower fingers to table height
    if self.grasp_phase == 0:

      if wrist:
        # detect the ground with the wrist sensor
        if print_on:
          print(f"target wrist force is {target_wrist_force_N}, actual is {wrist_reading}")
        if wrist_reading < target_wrist_force_N:
          action = H_down
        else:
          if print_on:
            print("Grasp phase 0 completed")
          self.grasp_phase = 1
          # new: lift a bit to avoid hitting the ground
          action = H_up
          return action

      else:
        # we aim for a z certain height
        if print_on:
          print(f"target z height is {initial_z_height_target_m}, actual is {state_readings[3]}")
        if state_readings[3] < initial_z_height_target_m:
          action = H_down
        else:
          if print_on:
            print("Grasp phase 0 completed")
          self.grasp_phase = 1

    # if self.grasp_phase == 1 and not self.palm_done and palm_first:
    #   self.grasp_phase = 3

    # angle the fingers
    if self.grasp_phase == 1:

      if fixed_angle:

        # we aim for a certain angle
        if print_on:
          print(f"target angle is {-target_angle_deg * (3.14159 / 180.0)}, actual is {self.mj.get_finger_angle()}")
        if -1 * self.mj.get_finger_angle() < target_angle_deg * (3.14159 / 180.0):
          action = Y_close
        else:
          if print_on:
            print("Grasp phase 1 completed")
          self.grasp_phase = 2

      else:
        if print_on:
          print("Grasp phase 1 skipped, fixed angle is False")
        self.grasp_phase = 2

    # constrict until we feel the squeeze on the object
    if self.grasp_phase == 2:

      # constrict the fingers either with X (fixed angle) or Y (changing angle)
      if fixed_angle:
        action = bend_to_force(initial_bend_target_N, possible_action=X_close)
      else:
        action = bend_to_force(initial_bend_target_N, possible_action=Y_close)

      if action is None:
        if print_on:
          print("Grasp phase 2 completed")
        self.grasp_phase = 3

    # if self.palm_done and self.heuristic_params["palm_first"]:
    #   self.grasp_phase = 4

    # advance palm to contact object
    if self.grasp_phase == 3:

      action = palm_push_to_force(initial_palm_target_N)

      if action is None:
        if print_on:
          print("Grasp phase 3 completed")
        self.grasp_phase = 4

        # if palm_first:
        #   self.grasp_phase = 1
        #   self.palm_done = True
        #   action = X_close

        self.gauge_read_history = np.array([])

    # lift up object
    if self.grasp_phase == 4:

      if state_readings[3] > final_z_height_target_m:
          if print_on:
            print(f"target height is {final_z_height_target_m}, actual is {state_readings[3]}")
          action = H_up
      else:
        if print_on:
          print("Grasp phase 4 completed")
        self.grasp_phase = 5

    # squeeze grip further
    if self.grasp_phase == 5:

      if final_squeeze: 

        # squeeze fingers
        if fixed_angle:
          action = bend_to_force(final_bend_target_N, possible_action=X_close, limit_force_N=limit_bend_target_N)
          action_name = "finger Y close"
        else:
          action = bend_to_force(final_bend_target_N, possible_action=Y_close, limit_force_N=limit_bend_target_N)
          action_name = "finger X close"

        # squeeze palm
        if action is None:
          action = palm_push_to_force(final_palm_target_N, limit_force_N=limit_palm_target_N)
          action_name = "palm forward"

      # or just try to lift higher
      if action is None:
        action = H_up
        action_name = "height up"

      # if no sensors, choose random action
      if not bending and not palm and not wrist:
        # choice = random_train.integers(0, 3)
        # options = [
        #   (X_close, "finger X close"),
        #   (Z_plus, "palm forward"),
        #   (H_up, "height up")
        # ]
        # action = options[choice][0]
        # action_name = options[choice][1]

        action = H_up
        action_name = "height up"

      if print_on:
        print("Grasp phase 5: action is", action_name)

    # check for dangerous behaviour?
    # - lifted object too high
    # - panda dangerously low
    # - dangerous bending
    # - dangerous palm force

    return action

  def using_continous_actions(self):
    """
    Return if action space is continous or discrete
    """
    return bool(self.mj.set.continous_actions)

  def seed(self, seed=None):
    """
    Set the seed for the environment, applied upon call to reset
    """

    global random_train # reseed only the training generator

    # if we have not been given a seed
    if seed is None:
      # if we have previously had a seed, reuse the same one (eg reloading from pickle)
      if self.myseed is not None:
        seed = self.myseed
      else:
        # otherwise, get a random seed from [0, maxint]
        seed = random_train.integers(0, 2_147_483_647)

    # create a new generator with the given seed
    random_train = np.random.default_rng(seed)

    # set the same cpp random seed (reseeded upon call to reset())
    self.mj.set.random_seed = seed

    # save the seed
    self.myseed = seed

    # tell the simulation to reload a new xml (upon call to reset())
    self.reload_flag = True

  def start_test(self):
    """
    Begin test mode, should be called by class user
    """

    # reset the test seed to ensure reproducable object noise every test
    global random_test
    test_seed = 13337419
    random_test = np.random.default_rng(test_seed)

    self.current_test_trial = MjEnv.Test()
    self.test_trial_data = []
    self.test_in_progress = True
    self.test_completed = False
    self._load_xml(test=0) # load first test set xml, always index 0

  def get_parameters(self):
    """
    Return a dictionary of the class parameters. This code is needed for backwards
    compatibility since the class does not currently use the parameter variables.
    """

    return asdict(self.params)
  
  def get_params_dict(self):
    """
    Return a dictionary of class parameters
    """
    param_dict = asdict(self.params)
    param_dict.update({
      "n_actions" : self.n_actions,
      "n_observation" : self.n_obs
    })
    # cpp_str = self._get_cpp_settings()
    # param_dict.update({"cpp_settings" : cpp_str})
    return param_dict

  def get_save_state(self):
    """
    Return a saveable state of the environment
    """
    save_dict = {
      "parameters" : self.params,
      "mjcpp" : self.mj,
    }
    return save_dict
  
  def load_save_state(self, state_dict, device="cpu"):
    """
    Load the environment from a saved state
    """
    self.params = state_dict["parameters"]
    self.load_next = deepcopy(state_dict["parameters"])
    self.mj = state_dict["mjcpp"]
    self.set_device(device)
    self.load()

  def load(self, object_set_name=None, object_set_path=None, index=None, 
           num_segments=None, finger_width=None, finger_thickness=None,
           finger_modulus=None, depth_camera=None, auto_generate=True,
           use_hashes=True):
    """
    Load and prepare the mujoco environment, uses defaults if arguments are not given.
    This function sets the 'params' for the class as well.
    """

    # put inputs into the 'load_next' datastructure
    if object_set_name is not None: self.load_next.object_set_name = object_set_name
    if num_segments is not None: self.load_next.num_segments = num_segments
    if finger_width is not None: self.load_next.finger_width = finger_width
    if finger_thickness is not None: self.load_next.finger_thickness = finger_thickness
    if finger_modulus is not None: self.load_next.finger_modulus = finger_modulus
    if depth_camera is not None: self.load_next.depth_camera = depth_camera

    # set the thickness/modulus (changes only applied upon reset(), causes hard_reset() if changed)
    try:
      self.mj.set_finger_thickness(self.load_next.finger_thickness) # required as xml thickness ignored
      self.mj.set_finger_modulus(self.load_next.finger_modulus) # duplicate xml setting
    except AttributeError as e:
      print("WARNING: MjEnv.load_xml() failed to update parameters for gripper dimensions, you may be running an old policy")
      print("Error is:", e)
      print("Execution is continuing, beware saving this model will have incorrect params")
      self.params = self.load_next

    self._load_object_set(name=self.load_next.object_set_name, mjcf_path=object_set_path,
                          auto_generate=auto_generate, use_hashes=use_hashes)
    self._load_xml(index=index)  

    # auto generated parameters
    self._update_n_actions_obs()

    # check if the depth camera is included in the object set chosen
    if self.load_next.depth_camera:
      self.load_next.depth_camera = self._init_rgbd()
    else:
      self.params.depth_camera = False
      self.rgbd_enabled = False

    # determine if we are loading a rendering network
    if self.params.use_rgb_rendering and self.render_net is None:
      self._load_image_rendering_model(device=self.torch_device)
      # try to clear away large memory required to load the model
      import gc
      torch.cuda.empty_cache()
      gc.collect()
    else:
      self.render_net = None

    # reset any lingering goal defaults
    self.mj.reset_goal()

  def step(self, action):
    """
    Perform an action and step the simulation until it is resolved
    """

    # safety check: if step is called when done=true
    if self.track.is_done:
      raise RuntimeError("step has been called with done=true, use reset()")

    self.track.current_step += 1

    # have we been given a torch tensor, in which case move to cpu and convert
    if self.torch:
      if self.mj.set.continous_actions:
        action = (action.cpu()).numpy()
      else:
        action = (action.cpu()).item()
    
    self._take_action(action)
    obs = self._next_observation()
    terminated, truncated = self._is_done()
    info = {}

    done = bool(terminated + truncated)

    # what method are we using
    if self.mj.set.use_HER:
      state = self._event_state()
      goal = self._assess_goal(state)
      reward = 0.0
      if done or not self.mj.set.reward_on_end_only:
        # do we only award a reward when the episode ends
        reward = self._goal_reward(goal, state)
      to_return = (obs, reward, terminated, state, goal, info)
    else:
      reward = self._reward()
      to_return = (obs, reward, terminated, truncated, info)

    self.track.last_action = action
    self.track.last_reward = reward
    self.track.is_done = done
    self.track.cumulative_reward += reward

    if self.randomise_colours_every_step:
      self.mj.randomise_every_colour()

    # track testing if this result has finished
    if done and self.test_in_progress:
      self._monitor_test()

    return to_return

  def reset(self, hard=None, timestep=None, realworld=False, nospawn=False):
    """
    Reset the simulation to the start
    """

    global random_train

    # reset tracking variables
    self.track = MjEnv.Track()

    # there is a small chance we reload a new random task
    if not self.test_in_progress and not realworld and not self.prevent_reload:
      if (random_train.random() < self.params.task_reload_chance
          or self.reload_flag):
        self._load_xml()

    # reset the simulation
    if hard is True: 
      self.mj.hard_reset()
      # base limits do not persist through a hard reset, so set them again
      self.mj.set_base_XYZ_limits(self.params.base_lim_X_mm * 1e-3, 
                                  self.params.base_lim_Y_mm * 1e-3,
                                  self.params.base_lim_Z_mm * 1e-3)
      self.mj.set_base_yaw_limit(self.params.base_lim_yaw_rad)
    elif timestep is True: 
      self.mj.reset_timestep() # recalibrate timestep
    else: 
      self.mj.reset()

    # if real world, recalibrate the sensors
    if realworld is True: 
      self.mj.calibrate_real_sensors() # re-zero sensors
      return None # no need to return an actual observation
    
    # spawn a new random object
    if not realworld or nospawn:
      self._spawn_object()

    # if we are using a camera, randomise colours
    if self.params.depth_camera: self.mj.randomise_every_colour()

    return self._next_observation()

  def render(self):
    """
    Render the simulation to a window
    """

    if self.render_window:
      self.mj.render()

    if self.log_level >= 3:
      print('MjEnv render update:')
      print(f'\tStep: {self.track.current_step}')
      print(f'\tLast action: {self.track.last_action}')
      print(f'\tLast reward: {self.track.last_reward:.3f}')
      print(f'\tCumulative reward: {self.track.cumulative_reward:.3f}')
      print(f'\tDone: {self.track.is_done}')
      print(f'\tTesting: {self.test_in_progress}')
      print()

  def close(self):
    """
    Tidy up and finish everything
    """
    if self.log_level > 0: print("Environment closed")

if __name__ == "__main__":

  # import pickle

  mj = MjEnv(depth_camera=True, log_level=2, seed=None)
  mj.render_window = False
  mj.mj.set.mujoco_timestep = 3.187e-3
  mj.mj.set.auto_set_timestep = False

  mj.load("set8_fullset_1500", num_segments=8, finger_width=28e-3, finger_thickness=0.9e-3)
  
  # ----- try out rgb rendering with CUT models ----- #

  test_CUT_model = False
  if test_CUT_model:

    mj.params.use_rgb_in_observation = True
    mj.params.use_depth_in_observation = False
    mj.params.use_rgb_rendering = True
    mj.params.rgb_rendering_method = "cycleGAN_encoder"
    mj.params.use_standard_transform = True
    mj.params.transform_resize_square = 144
    mj.params.transform_crop_size = 128

    mj._spawn_object()
    mj._set_rgbd_size(200, 100)

    # try to load the CUT
    mj._load_image_rendering_model(device="cuda")

    num = 3
    originals = []
    conversions = []

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(num, 2)
    if num == 1: axs = [axs]

    for i in range(num):

      mj.reset()

      # num_rand_steps = 20
      # for i in range(num_rand_steps):
      #   new_obs, reward, terminated, truncated, info = mj.step(torch.tensor(random_train.integers(0, mj.n_actions)))
      #   if terminated or truncated:
      #     break

      new_obs = mj._next_observation()
      new_obs = (new_obs.cpu()).detach().numpy()

      # now extract and plot the rendered image
      rgb, depth = mj._get_rgbd_image()

      # print(new_obs[:,100:110, 200:210])

      # numpy likes image arrays like this: height x width x channels (ie rows x columns x dim)
      # torch likes image arrays like this: channels x width x height
      # hence convert from torch style back to numpy style for plotting
      axs[i][0].imshow(np.einsum("ijk->kji", rgb)) # swap to numpy style rows/cols (eg 3x640x480 -> 480x640x3)
      axs[i][1].imshow(np.einsum("ijk->kji", np.array((new_obs[:3] + 1) / 2.0 * 255.0, dtype=np.uint8)))

    fig.tight_layout()

    # visualise how simulated images are transformed
    mj._plot_rgbd_image(transform=True)

    # plt.show()

  # ----- visualise segmentation masks ----- #

  examine_scene_mask = False
  if examine_scene_mask:

    mj.load_next.XY_base_actions = True
    mj.params.use_scene_settings = True
    mj.params.scene_X_dimension_mm = 300
    mj.params.scene_Y_dimension_mm = 300
    mj.params.object_position_noise_mm = 1000
    mj.params.num_objects_in_scene = 5

    mj.load("set8_fullset_1500", num_segments=8, finger_width=28e-3, finger_thickness=0.9e-3)

    mj._spawn_object()
    # mj._set_rgbd_size(20, 10)

    mj.reset()
    num = 0
    for i in range(num):
      mj.step(random_train.integers(0, mj.n_actions))
    
    import time
    t1 = time.time()
    rgb, depth = mj._get_rgbd_image()
    t2 = time.time()
    rgb, depth = mj._get_rgbd_image(mask=True)
    t3 = time.time()
    print(f"Time taken for regular RGBD was {(t2 - t1) * 1e3:.3f} ms")
    print(f"Time taken for mask was {(t3 - t2) * 1e3:.3f} ms")

    mj._plot_scene_mask()

  # ---- automatically generate new xml files ----- #

  generate_new_xml = True
  if generate_new_xml:

    # widths = [24e-3, 28e-3]
    # segments = [8]
    # xy_base = [False, True]
    # inertia = [1, 50]

    # mj.load_next.num_segments = 8
    # angles = [90]
    # for a in angles:
    #   mj.load_next.finger_hook_angle_degrees = a
    #   mj.load_next.finger_hook_length = 100e-3
    #   mj.load_next.XY_base_actions = True
    #   mj._auto_generate_xml_file("set_test_large", use_hashes=True, silent=True)

    # for w in widths:
    #   for N in segments:
    #     for xy in xy_base:
    #       for i in inertia:
    #         mj.load_next.finger_width = w
    #         mj.load_next.num_segments = N
    #         mj.load_next.XY_base_actions = xy
    #         mj.load_next.segment_inertia_scaling = i
    #         mj._auto_generate_xml_file("set8_fullset_1500", use_hashes=True)

    mj.params.test_objects = 20
    mj.load_next.finger_hook_angle_degrees = 75
    mj.load_next.finger_width = 28e-3
    mj.load_next.fingertip_clearance = 0.01
    mj.load_next.XY_base_actions = False
    mj.load_next.Z_base_rotation = False
    mj.load_next.num_segments = 3
    mj.load_next.segment_inertia_scaling = 1.0
    # mj.load_next.finger_length = 200e-3
    # mj.load_next.finger_thickness = 1.9e-3

    gen_obj_set = "set9_fullset"
    name = mj._auto_generate_xml_file("set9_fullset", use_hashes=True)
    runstr = f"bin/mysimulate -p /home/luke/mujoco-devel/mjcf -o {gen_obj_set} -g {name}"
    print(runstr)

  # ----- evaluate speed of rgbd function ----- #

  test_rgbd_speed = False
  if test_rgbd_speed:
      
      # mj._spawn_object()
      mj._set_rgbd_size(50, 50)

      num = 1000
      mj.mj.tick()

      for i in range(num):
        mj._get_rgbd_image()

      time_taken = mj.mj.tock()
      print(f"Time taken for {num} fcn calls was {time_taken:.3f} seconds")

      rgb, depth = mj._get_rgbd_image()
      print(f"rgb size is {rgb.shape}")
      print(f"depth size is {depth.shape}")

  # ----- examine size of RGBD images ---- #

  see_rgbd_size = False
  if see_rgbd_size:

    import sys

    # sizes of python lists (rgb, depth) = (25.9, 8.6) MB
    rgbd = mj.mj.get_RGBD()
    print(f"Size of rgb is: {sys.getsizeof(rgbd.rgb) * 1e-6:.3f} MB")
    print(f"Size of depth is: {sys.getsizeof(rgbd.depth) * 1e-6:.3f} MB")

    # sizes of numpy arrays -> (3.2, 4.3) MB, uint8 is the same as uint_fast8
    rgb, depth = mj.mj.get_RGBD_numpy()
    print(f"Size of numpy rgb is: {sys.getsizeof(rgb) * 1e-6:.3f} MB")
    print(f"Size of numpy depth is: {sys.getsizeof(depth) * 1e-6:.3f} MB")
