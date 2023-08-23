#!/usr/bin/env python3

import os
import sys
import yaml
import subprocess
from copy import deepcopy

# get the path to this file and insert it to python path (for mjpy.bind)
pathhere = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, pathhere)

# with env in path, we can now import the shared cpp library
from mjpy.bind import MjClass, EventTrack

import time
import numpy as np
from dataclasses import dataclass, asdict

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

    # training parameters
    max_episode_steps: int = 250
    object_position_noise_mm: int = 10
    object_rotation_noise_deg: int = 5

    # file and testing parameters
    test_obj_per_file: int = 20
    task_reload_chance: float = 1.0 / float(test_obj_per_file)
    test_trials_per_object: int = 3
    test_objects: int = 100

    # model parameters (for loading xml files)
    num_segments: int = 8
    finger_thickness: float = 0.9e-3
    finger_length: float = 235e-3
    finger_width: float = 28e-3
    finger_modulus: float = 193e9
    depth_camera: bool = False
    XY_base_actions: bool = False
    fixed_finger_hook: bool = True
    finger_hook_angle_degrees: float = 90.0
    finger_hook_length: float = 35e-3
    segment_inertia_scaling: float = 50.0
    fingertip_clearance: float = 10e-3

  def __init__(self, seed=None, noload=None, num_segments=None, finger_width=None, 
               depth_camera=None, finger_thickness=None, finger_modulus=None,
               log_level=0):
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
    self.disable_rendering = True
    self.prevent_reload = False
    self.use_yaml_hashing = False

    # initialise class variables
    self.test_in_progress = False
    self.test_completed = False

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
    if self.log_level == 0: self.mj.set.debug = False
    elif self.log_level >= 4: self.mj.set.debug = True

    # seed the environment
    self.myseed = None
    self.seed(seed)

    # load the mujoco models, if not then load() must be run by the user
    if noload is not True: self.load()

    # initialise tracking variables
    self.track = MjEnv.Track()
    self.test = MjEnv.Test()
    self.prev_test = MjEnv.Test()

    # initialise heuristic parameters
    self._initialise_heuristic_parameters()

    # set default rgbd camera size
    self.rgbd_width = 848
    self.rgbd_height = 480

    return

  # ----- semi-private functions, advanced use ----- #

  def _auto_generate_xml_file(self, object_set, use_hashes=False, silent=True, force=False):
    """
    Automatically generate the xml file that we need. Note this function
    overrides the currently set mjcf path to the path that leads to the
    autogenerated file.
    """

    # tempory fix for backwards compatibility
    if object_set.startswith("set7"): use_hashes = False

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
    # gripper_details[c]["xy_base_rotation"] = self.load_next.  not added yet
    # gripper_details[c]["z_base_rotation"] = self.load_next.  not added yet

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

    # assume loading is correct and directly copy
    self.params.segment_inertia_scaling = self.load_next.segment_inertia_scaling

  def _load_object_set(self, name=None, mjcf_path=None, num_segments=None, 
                       finger_width=None, auto_generate=False, use_hashes=True):
    """
    Load in an object set and sort out details (like number of xml files).
    This functions does NOT load a new XML file from this object set.
    """

    debug_fcn = True

    # if a mjcf_path is given, override, otherwise we use default
    if mjcf_path != None: self.mj.model_folder_path = mjcf_path

    # if a object set name is given, override, otherwise we use default
    if name != None: self.mj.object_set_name = name

    if auto_generate:

      # warn the user that the given path value is going to be overriden
      if mjcf_path is not None:
        print(f"MjEnv() warning: given mjcf_path='{mjcf_path}' is about to be overriden by MjEnv._auto_generate_xml_file()")

      # create the file we need
      self.task_xml_folder = self._auto_generate_xml_file(self.mj.object_set_name, use_hashes=use_hashes)

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

    if debug_fcn:
      print("_load_object_set() gives xml path:", self.xml_path)
      print(f"Training xmls: {self.training_xmls}, testing xmls: {self.testing_xmls}")

    if self.training_xmls < 1:
      raise RuntimeError(f"enough training xmls failed to be found in MjEnv at: {self.xml_path}")

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

  def _take_action(self, action):
    """
    Take an action in the simulation
    """

    # set the action and step the simulation
    self.mj.set_action(action)
    self.mj.action_step()

    return

  def _is_done(self):
    """
    Determine if the epsiode should finish
    """

    # if we have exceeded our time limit
    if self.track.current_step >= self.params.max_episode_steps:
      if self.log_level >= 3 or self.mj.set.debug: 
        print("is_done() = true (in python) as max step number exceeded")
      return True

    # check the cpp side
    done = self.mj.is_done()

    return done
    
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

    # return if the camera is running or not
    return self.params.depth_camera

  def _set_rgbd_size(self, width=None, height=None):
    """
    Set the size of simulated RGBD images
    """

    if width is None: width = self.rgbd_width
    if height is None: height = self.rgbd_height

    self.rgbd_enabled = self.mj.rendering_enabled()
    self.rgbd_width = width
    self.rgbd_height = height

    if self.rgbd_enabled:
      self.mj.set_RGBD_size(width, height)
    else:
      if self.log_level > 0:
        print("MjClass rendering is disabled in compilation settings, no RGBD images possible")
      self.rgbd_enabled = False

  def _get_rgbd_image(self):
    """
    Return rgbd data from the current state of the simulation, rescaled to match
    the size of the rgbd data from real life.

    Returns two numpy arrays ready for conversion into torch:
      - rgb    (with shape 3 x width x height)
      - depth  (with shape 1 x width x height)
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
    rgb, depth = self.mj.get_RGBD_numpy()

    # reshape the numpy arrays to the correct aspect ratio
    rgb = rgb.reshape(self.rgbd_height, self.rgbd_width, 3)
    depth = depth.reshape(self.rgbd_height, self.rgbd_width, 1)

    # numpy likes image arrays like this: width x height x channels
    # torch likes image arrays like this: channels x width x height
    rgb = np.einsum("ijk->kji", rgb)
    depth = np.einsum("ijk->kji", depth)

    return rgb, depth # ready for conversion to torch tensors
  
  def _plot_rgbd_image(self):
    """
    Get and then plot an rgbd image from the simulation. Use this for debugging only
    """

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1)
    
    rgb, depth = self._get_rgbd_image()

    # numpy likes image arrays like this: height x width x channels (ie rows x columns x dim)
    # torch likes image arrays like this: channels x width x height
    # hence convert from torch style back to numpy style for plotting
    axs[0].imshow(np.einsum("ijk->kji", rgb)) # swap to numpy style rows/cols (eg 3x640x480 -> 480x640x3)
    axs[1].imshow(np.transpose(depth[0])) # remove the depth 'channel' then swap (eg 1x640x480 -> 480x640)

    plt.show()

  def _next_observation(self):
    """
    Returns the next observation from the simuation
    """

    # get an observation as a numpy array
    obs = self.mj.get_observation_numpy()

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

  def _goal_reward(self, goal, state):
    """
    Calculate the reward based on a state and goal
    """
    return self.mj.reward(goal, state)

  def _spawn_object(self):
    """
    Spawn an object into the simulation randomly
    """

    global random_test
    global random_train

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
      "target_angle_deg" : 10,
      "target_wrist_force_N" : 1,
      "min_x_value_m" : 55e-3,
      "target_x_constrict_m" : 110e-3,
      "target_palm_bend_increase_percentage" : 10,
      "target_z_position_m" : 80e-3,
      "bend_history_length" : 6, # how many historical bending values to save
      "bend_update_length" : 3,  # how many values in history do we consider 'new'
      "initial_z_height_target_m" : 5e-3,
      "final_z_height_target_m" : -20e-3,
      "initial_bend_target_N" : 1.5,
      "initial_palm_target_N" : 1.5,
      "final_bend_target_N" : 1.5,
      "final_palm_target_N" : 1.5,
      "final_squeeze" : True,
      "fixed_angle" : False,
      # "palm_first" : True,
    }

  # ----- public functions ----- #

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

    def bend_to_force(target_force_N, possible_action=None):
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
            # halt as we have reached gripper limits
            if print_on:
              print(f"minimum x is {min_x_value_m}, actual is {state_readings[0]}")

      else:
        # simply constrict to a predetermined point
        if print_on:
          print(f"x constrict target: {target_x_constrict_m}, actual: {state_readings[0]}")
        if state_readings[0] > target_x_constrict_m:
          action = possible_action

      return action

    def palm_push_to_force(target_force_N):
      """
      Try to push with the palm to a specific force
      """

      action = None

      # if we have palm sensing
      if palm:
        if print_on:
          print(f"target palm force is {target_force_N}, actual is {palm_reading}")
        if palm_reading < target_force_N:
          action = Z_plus

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
          action = bend_to_force(final_bend_target_N, possible_action=X_close)
          action_name = "finger Y close"
        else:
          action = bend_to_force(final_bend_target_N, possible_action=Y_close)
          action_name = "finger X close"

        # squeeze palm
        if action is None:
          action = palm_push_to_force(final_palm_target_N)
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

  def seed(self, seed=None):
    """
    Set the seed for the environment
    """

    global random_train # reseed only the training generator

    # if we have not been given a seed
    if seed is None:
      # if we have previously had a seed, reuse the same one (eg reloading from pickle)
      if self.myseed is not None:
        seed = self.myseed
      else:
        # otherwise, get a random seed from [0, maxint]
        seed = random_train.integers(0, 2_147_483_647) #np.random.randint(0, 2_147_483_647)

    # set the python random seed in numpy
    # np.random.seed(seed)

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

  def load(self, object_set_name=None, object_set_path=None, index=None, 
           num_segments=None, finger_width=None, finger_thickness=None,
           finger_modulus=None, depth_camera=None, auto_generate=True,
           use_hashes=True):
    """
    Load and prepare the mujoco environment, uses defaults if arguments are not given.
    This function sets the 'params' for the class as well.
    """

    # put inputs into the 'load_next' datastructure
    if num_segments is not None: self.load_next.num_segments = num_segments
    if finger_width is not None: self.load_next.finger_width = finger_width
    if finger_thickness is not None: self.load_next.finger_thickness = finger_thickness
    if finger_modulus is not None: self.load_next.finger_modulus = finger_modulus
    if depth_camera is not None: self.load_next.depth_camera = depth_camera

    # set the thickness/modulus (changes only applied upon reset(), causes hard_reset() if changed)
    self.mj.set_finger_thickness(self.load_next.finger_thickness) # required as xml thickness ignored
    self.mj.set_finger_modulus(self.load_next.finger_modulus) # duplicate xml setting

    self._load_object_set(name=object_set_name, mjcf_path=object_set_path,
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
    
    self._take_action(action)
    obs = self._next_observation()
    done = self._is_done()

    # what method are we using
    if self.mj.set.use_HER:
      state = self._event_state()
      goal = self._assess_goal(state)
      reward = 0.0
      if done or not self.mj.set.reward_on_end_only:
        # do we only award a reward when the episode ends
        reward = self._goal_reward(goal, state)
      to_return = (obs, reward, done, state, goal)
    else:
      reward = self._reward()
      to_return = (obs, reward, done)

    self.track.last_action = action
    self.track.last_reward = reward
    self.track.is_done = done
    self.track.cumulative_reward += reward

    # track testing if this result has finished
    if done and self.test_in_progress:
      self._monitor_test()

    return to_return

  def reset(self, hard=None, timestep=None, realworld=False):
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
    if hard is True: self.mj.hard_reset()
    elif timestep is True: self.mj.reset_timestep() # recalibrate timestep
    else: self.mj.reset()

    # if real world, recalibrate the sensors
    if realworld is True: self.mj.calibrate_real_sensors() # re-zero sensors
    
    # spawn a new random object
    if not realworld:
      self._spawn_object()

    # if we are using a camera, randomise colours
    if self.params.depth_camera: self.mj.randomise_every_colour()

    return self._next_observation()

  def render(self):
    """
    Render the simulation to a window
    """

    if not self.disable_rendering:

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

  mj = MjEnv(noload=True, depth_camera=True, log_level=2, seed=122)
  mj.disable_rendering = True
  mj.mj.set.mujoco_timestep = 3.187e-3
  mj.mj.set.auto_set_timestep = False

  # mj.load("set7_fullset_1500_50i_updated", num_segments=8, finger_width=28, finger_thickness=0.9e-3)
  # mj._spawn_object()
  # mj._set_rgbd_size(848, 480)
  # mj._set_rgbd_size(1000, 1000)

  # widths = [24e-3, 28e-3]
  # segments = [8]
  # xy_base = [False, True]
  # inertia = [1, 50]

  mj.load_next.num_segments = 8
  angles = [90]
  for a in angles:
    mj.load_next.finger_hook_angle_degrees = a
    mj.load_next.finger_hook_length = 100e-3
    mj.load_next.XY_base_actions = True
    mj._auto_generate_xml_file("set_test_large", use_hashes=True, silent=True)

  # for w in widths:
  #   for N in segments:
  #     for xy in xy_base:
  #       for i in inertia:
  #         mj.load_next.finger_width = w
  #         mj.load_next.num_segments = N
  #         mj.load_next.XY_base_actions = xy
  #         mj.load_next.segment_inertia_scaling = i
  #         mj._auto_generate_xml_file("set8_fullset_1500", use_hashes=True)

  exit()

  mj.params.test_objects = 20
  # mj.load_next.finger_hook_angle_degrees = 45.678
  mj.load_next.finger_width = 28e-3
  # mj.load_next.fingertip_clearance = 0.143e-3
  # mj.load_next.finger_length = 200e-3
  # mj.load_next.finger_thickness = 1.9e-3

  mj.load("set_test", depth_camera=True)
  mj._spawn_object()
  mj.reset()

  print(mj.get_parameters())

  mj._plot_rgbd_image()

  mj.reset()
  mj._plot_rgbd_image()

  # name = mj._auto_generate_xml_file("set_test", use_hashes=True)

  # print(name)

  exit()

  num = 10000
  mj.mj.tick()

  for i in range(num):
    mj._get_rgbd_image()

  time_taken = mj.mj.tock()
  print(f"Time taken for {num} fcn calls was {time_taken:.3f} seconds")

  rgb, depth = mj._get_rgbd_image()
  print(f"rgb size is {rgb.shape}")
  print(f"depth size is {depth.shape}")

  mj._plot_rgbd_image()

  mj._set_rgbd_size(250, 150)
  mj._plot_rgbd_image()

  exit()

  import sys

  # sizes of python lists (rgb, depth) = (25.9, 8.6) MB
  rgbd = mj.mj.get_RGBD()
  print(f"Size of rgb is: {sys.getsizeof(rgbd.rgb) * 1e-6:.3f} MB")
  print(f"Size of depth is: {sys.getsizeof(rgbd.depth) * 1e-6:.3f} MB")

  # sizes of numpy arrays -> (3.2, 4.3) MB, uint8 is the same as uint_fast8
  rgb, depth = mj.mj.get_RGBD_numpy()
  print(f"Size of numpy rgb is: {sys.getsizeof(rgb) * 1e-6:.3f} MB")
  print(f"Size of numpy depth is: {sys.getsizeof(depth) * 1e-6:.3f} MB")

  exit()

  # sizes = [(720, 740), (200, 740), (720, 200),
  #          (1500, 1500)]
  
  # sizes = [(900, 1200)]

  # import matplotlib.pyplot as plt
  # fig, axs = plt.subplots(len(sizes), 1)

  # for i, (h, w) in enumerate(sizes):
  #   rgb, d = mj.mj.get_RGBD_numpy(h, w)
  #   axs.imshow(rgb.reshape(h, w, 3))

  # fig2, axs2 = plt.subplots(3, 1)

  # from torch.nn.functional import interpolate
  # import torch
  # from torchvision import transforms

  # resizes= [0.5, 0.1, 0.01]
  # for i, x in enumerate(resizes):

    # rgbarr = interpolate(torch.tensor([np.transpose(rgb.reshape(h, w, 3))]), scale_factor=x)
    # axs2[i].imshow(np.array(rgbarr[0]).T)
    
    # trans = transforms.Compose([transforms.Resize((100, 100))])
    # img = trans(torch.tensor(np.transpose(rgb.reshape(h, w, 3))))
    # axs2[i].imshow(np.array(img).T)

    # trans = transforms.Compose([transforms.Resize((100, 100))])
    # img = trans(torch.tensor(rgb.reshape(3, h, w)))
    # axs2[i].imshow(np.array(img).T)

  # rgbnp, depthnp = mj.mj.get_RGBD_numpy()
  # print(f"Size of numpy rgb is: {sys.getsizeof(rgbnp) * 1e-6:.3f} MB")
  # print(f"Size of numpy depth is: {sys.getsizeof(depthnp) * 1e-6:.3f} MB")

  # print("The rgb information is", rgbd.rgb[:10])
  # print("The depth information is", rgbd.depth[:10])

  # print(rgbnp)

  # print(isinstance(rgbnp, np.ndarray))
  # print(rgbnp.dtype)

  # import matplotlib.pyplot as plt

  # plt.imshow(rgbnp.reshape(720, 740, 3))
  # plt.show()

  # plt.imshow(depthnp.reshape(720, 740))
  # plt.show()

  # mj.mj.set.set_sensor_prev_steps_to(3)
  mj.mj.set.sensor_n_prev_steps = 1
  mj.mj.set.state_n_prev_steps = 1
  mj.mj.set.sensor_sample_mode = 3
  mj.mj.set.debug = False
  mj.reset()

  for i in range(20):
    mj.step(np.random.randint(0,8))

  print("\n\n\n\n\n\n\n\n\n about to set finger thickness in python")
  mj.params.finger_thickness = 0.8e-3
  mj._load_xml()
  print("SET IN PYTHON, about to run reset()")
  mj.reset()
  print("reset is finished")

  print("\n\n\nSTART")

  for i in range (3):
    print("\nObservation", i)
    # mj.step(np.random.randint(0,8))
    mj.step(2)
    # print(mj.mj.get_observation())
    

  # mj.mj.set.wipe_rewards()
  # mj.mj.set.lifted.set(100, 10, 1)
  # print(mj._get_cpp_settings())

  # with open("test_file.pickle", 'wb') as f:
  #   pickle.dump(mj, f)
  #   print("Pickle saved")

  # with open("test_file.pickle", 'rb') as f:
  #   mj = pickle.load(f)
  #   print("Pickle loaded")

  # mj._load_xml(index=0)
  # mj._load_xml(index=1)
  # mj._load_xml(index=2)

  # mj.step(0)

  # print(mj._get_cpp_settings())

  # obs = mj.mj.get_observation()
  # print(obs)
