#!/usr/bin/env python3

import os
import sys

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
    # key class parameters with default values
    max_episode_steps: int = 250
    object_position_noise_mm: int = 10
    test_obj_per_file: int = 20
    task_reload_chance: float = 1.0 / float(test_obj_per_file)
    test_trials_per_object: int = 1
    test_objects: int = 100
    num_segments: int = 8
    finger_thickness: float = 0.9e-3
    finger_width: float = 28e-3

  def __init__(self, seed=None, noload=None, num_segments=None, finger_width=None):
    """
    A mujoco environment, optionally set the random seed or prevent loading a
    model, in which case the user should call load() before using the class
    """

    self.params = MjEnv.Parameters()

    # define file structure
    self.task_xml_folder = "task"
    self.task_folder_template = "gripper_N{}"
    self.task_xml_template = "gripper_task_{}.xml"

    # general class settings
    self.log_level = 0
    self.disable_rendering = True

    # initialise class variables
    self.test_in_progress = False
    self.test_completed = False

    # how many segments to load next time, default is params.num_segments
    if num_segments == None: self.load_num_segments = self.params.num_segments
    else: self.load_num_segments = num_segments

    # what finger width to load, default is params.finger_width
    if finger_width == None: self.load_finger_width = self.params.finger_width
    else: self.load_finger_width = finger_width

    # calculate how many files we need to reserve for testing
    self.testing_xmls = int(np.ceil(self.params.test_objects / float(self.params.test_obj_per_file)))
    
    # create mujoco instance
    self.mj = MjClass()
    if self.log_level == 0: self.mj.set.debug = False

    # seed the environment
    self.myseed = None
    self.seed(seed)

    # load the mujoco models, if not then load() must be run by the user
    if noload is not True: self.load(num_segments=self.load_num_segments)

    # initialise tracking variables
    self.track = MjEnv.Track()
    self.test = MjEnv.Test()
    self.prev_test = MjEnv.Test()

    return

  # ----- semi-private functions, advanced use ----- #

  def _set_finger_variables(self, num_segments=None, width=None):
    """
    Set the number of segments in use and also the finger width. The available
    options will depend on the object set chosen, None means use the value in
    params
    """

    debug_fcn = False

    if num_segments is None: num_segments = self.params.num_segments
    if width is None: width = self.params.finger_width

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
      self.load_finger_width = 28e-3 # hardcoded default
      self.task_xml_folder = self.task_folder_template.format(num_segments)
      if width != 28:
        print(f"MjEnv warning: selected finger width of {width} is not available from this object set")
    else:
      if width in width_options:
        self.load_finger_width = width * 1e-3 # convert from mm to m
        self.task_xml_folder = self.task_folder_template.format("{0}_{1}".format(num_segments, width))
      else:
        raise RuntimeError(f"chosen width of {width} not found amoung width options: {width_options}")

    # apply the selected finger width in mujoco (EI change requires reset to finalise)
    self.mj.set_finger_width(self.load_finger_width)
    self.params.finger_width = self.load_finger_width

    if debug_fcn:
      print("the width which will be loaded is:", self.load_finger_width)
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

    if self.log_level > 1: 
      print("Load path: ", self.mj.model_folder_path
            + self.mj.object_set_name + "/" + self.task_xml_folder)
    if self.log_level > 0: print("Loading xml: ", filename)

    # load the new task xml (old model/data are deleted)
    self.mj.load_relative(self.task_xml_folder + '/' + filename)
    self.num_objects = self.mj.get_number_of_objects()

    self.reload_flag = False

    # get the number of segments currently in use
    self.params.num_segments = self.mj.get_N()

  def _load_object_set(self, name=None, mjcf_path=None, num_segments=None, finger_width=None):
    """
    Load and determine how many model xml files are in the object set
    """

    debug_fcn = False

    # if a mjcf_path is given, override, otherwise we use default
    if mjcf_path != None: self.mj.model_folder_path = mjcf_path

    # if a object set name is given, override, otherwise we use default
    if name != None: self.mj.object_set_name = name

    # how many segments and what finger width are in use
    self._set_finger_variables(num_segments=num_segments, width=finger_width)

    # check the mjcf_path is correctly formatted
    if self.mj.model_folder_path[-1] != '/':
      self.mj.model_folder_path += '/'

    # now determine the model xml path
    self.xml_path = (self.mj.model_folder_path + self.mj.object_set_name 
                      + '/' + self.task_xml_folder + '/')

    # find out how many xmls are available for training/testing
    xml_files = [x for x in os.listdir(self.xml_path) if os.path.isdir(self.xml_path + "/" + x) is False]
    self.training_xmls = len(xml_files) - self.testing_xmls

    if debug_fcn:
      print("_load_object_set gives xml path:", self.xml_path)
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
      if self.log_level > 1 or self.mj.set.debug: 
        print("is_done() = true (in python) as max step number exceeded")
      return True

    # check the cpp side
    done = self.mj.is_done()

    return done
    
  def _next_observation(self):
    """
    Returns the next observation from the simuation
    """

    obs = self.mj.get_observation()
    return np.array(obs)

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
    noise = self.params.object_position_noise_mm
    x_pos_mm = generator.integers(-noise, noise + 1)
    y_pos_mm = generator.integers(-noise, noise + 1)

    # randomly choose a z rotation
    angle_options = [0, 60, 120]
    rand_index = generator.integers(0, len(angle_options))
    z_rot_rad = angle_options[rand_index] * (np.pi / 180.0)

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

  # ----- public functions ----- #

  def start_heuristic_grasping(self, realworld=False):
    """
    Prepare to begin a heuristic grasping procedure.
    """

    self.grasp_phase = 0
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

    def bend_to_force(target_force_N):
      """
      Try to bend the fingers to a certain force
      """

      action = None

      # if we have access to bending sensors
      if bending:
        # wait for a certain bending
        if print_on:
          print(f"target bending is {target_force_N}, actual is {avg_bend}")
        if avg_bend < target_force_N:
          action = X_close
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
          action = X_close

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

    # hardcoded 'globals' used in sub-functions
    target_angle_deg = 15
    target_wrist_force_N = 1
    min_x_value_m = 55e-3
    target_x_constrict_m = 110e-3
    target_palm_bend_increase_percentage = 10
    target_z_position_m = 80e-3
    bend_history_length = 6 # how many historical bending values to save
    bend_update_length = 3  # how many values in history do we consider 'new'
    initial_z_height_target_m = 5e-3
    final_z_height_target_m = -20e-3
    initial_bend_target_N = 1
    initial_palm_target_N = 2
    final_bend_target_N = 2
    final_palm_target_N = 2

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

    # angle the fingers
    if self.grasp_phase == 1:

      # we aim for a certain angle
      if print_on:
        print(f"target angle is {-target_angle_deg * (3.14159 / 180.0)}, actual is {self.mj.get_finger_angle()}")
      if -1 * self.mj.get_finger_angle() < target_angle_deg * (3.14159 / 180.0):
        action = Y_close
      else:
        if print_on:
          print("Grasp phase 1 completed")
        self.grasp_phase = 2

    # constrict until we feel the squeeze on the object
    if self.grasp_phase == 2:

      action = bend_to_force(initial_bend_target_N)

      if action is None:
        if print_on:
          print("Grasp phase 2 completed")
        self.grasp_phase = 3

    # advance palm to contact object
    if self.grasp_phase == 3:

      action = palm_push_to_force(initial_palm_target_N)

      if action is None:
        if print_on:
          print("Grasp phase 3 completed")
        self.grasp_phase = 4

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

      # squeeze fingers
      action = bend_to_force(final_bend_target_N)
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
           num_segments=None, finger_width=None, finger_thickness=None):
    """
    Load and prepare the mujoco environment, uses defaults if arguments are not given.
    This function sets the 'params' for the class as well.
    """

    # old code compatibility: can delete once all models have finger width options
    try:
      test1 = self.load_finger_width
      test2 = self.params.finger_width
    except AttributeError as e:
      print("MjEnv old code catch, error is:", e)
      self.load_finger_width = 28e-3
      self.params.finger_width = 28e-3

    # if not given an input, use class value
    if num_segments is None: set_N = self.load_num_segments
    else: set_N = num_segments

    if finger_width is None: set_W = self.load_finger_width
    else: set_W = finger_width

    if finger_thickness is None: set_T = self.params.finger_thickness
    else: set_T = finger_thickness

    # set the finger thickness (changes only applied upon reset(), causes hard_reset() if changed)
    self.mj.set_finger_thickness(set_T)
    self.params.finger_thickness = self.mj.get_finger_thickness()

    self._load_object_set(name=object_set_name, mjcf_path=object_set_path,
                          num_segments=set_N, finger_width=set_W)
    self._load_xml(index=index)  

    # auto generated parameters
    self._update_n_actions_obs()

    # reset any lingering goal defaults
    self.mj.reset_goal()

  def step(self, action):
    """
    Perform an action and step the simulation until it is resolved
    """

    t0 = time.time()

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

    t1 = time.time()

    if self.log_level > 2: print("MjEnv step() time was ", t1 - t0)

    return to_return

  def reset(self, hard=None, timestep=None, realworld=False):
    """
    Reset the simulation to the start
    """

    global random_train

    # reset tracking variables
    self.track = MjEnv.Track()

    # there is a small chance we reload a new random task
    if not self.test_in_progress:
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
    self._spawn_object()

    return self._next_observation()

  def render(self):
    """
    Render the simulation to a window
    """

    if not self.disable_rendering:

      self.mj.render()

      if self.log_level > 2:
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

  mj = MjEnv(noload=True)

  mj.load_finger_width = 24e-3

  mj.load("set_fullset_795", num_segments=7, finger_width=None, finger_thickness=1.0e-3)

  exit()


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
