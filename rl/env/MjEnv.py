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
    max_episode_steps: int = 100
    object_position_noise_mm: int = 10
    test_obj_per_file: int = 20
    task_reload_chance: float = 1.0 / float(test_obj_per_file)
    test_trials_per_object: int = 1
    test_objects: int = 100
    num_segments: int = 6
    finger_thickness: float = 0.9e-3

  def __init__(self, seed=None, noload=None, num_segments=None):
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

  def _set_num_segments(self, N):
    """
    Set the number of segments in use, typically should be from 5-30 but depends on object set.
    None means we are using an old object set where we cannot choose the number of segments.
    This functionality can be removed once backwards compatibility is not needed.
    """

    if N is None:
      self.task_xml_folder = "task" # old default folder for gripper files
      self.load_num_segments = None
      return

    self.task_xml_folder = self.task_folder_template.format(N)
    self.load_num_segments = N

  def _load_xml(self, test=None, index=None):
    """
    Load the mujoco instance with the given mjcf xml file name
    """
    if index:
      filename = self.task_xml_template.format(index)
    elif test is not None:
      # load the specified test xml
      filename = self.task_xml_template.format(test)
    else:
      # get a random task xml file
      r = np.random.randint(self.testing_xmls, self.testing_xmls + self.training_xmls)
      filename = self.task_xml_template.format(r)

    if self.log_level > 2: 
      print("Load path: ", self.mj.model_folder_path
            + self.mj.object_set_name + "/" + self.task_xml_folder)
    if self.log_level > 1: print("Loading xml: ", filename)

    # load the new task xml (old model/data are deleted)
    self.mj.load_relative(self.task_xml_folder + '/' + filename)
    self.num_objects = self.mj.get_number_of_objects()

    self.reload_flag = False

    # get the number of segments currently in use
    self.params.num_segments = self.mj.get_N()

  def _load_object_set(self, name=None, mjcf_path=None, num_segments=-1):
    """
    Load and determine how many model xml files are in the object set
    """

    # if a mjcf_path is given, override, otherwise we use default
    if mjcf_path != None: self.mj.model_folder_path = mjcf_path

    # if a object set name is given, override, otherwise we use default
    if name != None: self.mj.object_set_name = name

    # how many segments will we use (None means use 'task' folder)
    if num_segments == -1: load_num_segments = self.load_num_segments
    else: load_num_segments = num_segments
    self._set_num_segments(load_num_segments)

    # check the mjcf_path is correctly formatted
    if self.mj.model_folder_path[-1] != '/':
      self.mj.model_folder_path += '/'

    # now determine the model xml path
    self.xml_path = (self.mj.model_folder_path + self.mj.object_set_name 
                      + '/' + self.task_xml_folder + '/')

    # find out how many xmls are available for training/testing
    xml_files = [x for x in os.listdir(self.xml_path) if os.path.isdir(self.xml_path + "/" + x) is False]
    self.training_xmls = len(xml_files) - self.testing_xmls

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

    # randomly generate the object index and (x, y) position
    noise = self.params.object_position_noise_mm
    obj_idx = np.random.randint(0, self.num_objects)
    x_pos_mm = np.random.randint(-noise, noise + 1)
    y_pos_mm = np.random.randint(-noise, noise + 1)

    # randomly choose a z rotation
    z_rot_rad = np.random.choice([0, 60, 120]) * (np.pi / 180.0)

    # if we are doing a test, chose a specific object
    if self.test_in_progress:
      obj_idx = self.current_test_trial.obj_idx

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

  def start_heuristic_grasping(self):
    """
    Prepare to begin a heuristic grasping procedure.
    """

    self.grasp_phase = 0

  def get_heuristic_action(self):
    """
    Return an action based on a simplistic grasping strategy. The grasp has
    the following phases:
      - 1. Lower fingers to table height
      - 2. Angle fingers at 20deg
      - 3. Constrict fingers until bending limit exceeded
      - 4. Advance palm until contact is made
      - 5. Lift to target height
    """

    # first, determine what sensors are available
    bending = self.mj.set.bending_gauge.in_use
    palm = self.mj.set.palm_sensor.in_use
    wrist = self.mj.set.wrist_sensor_Z.in_use

    # get sensor output if we can
    unnormalise_state = True
    state_readings = self.mj.get_state_readings(unnormalise_state)
    if bending:
      unnormalise_bend = True
      bending_readings = self.mj.get_bend_gauge_readings(unnormalise_bend)
    if palm:
      unnormalise_palm = True
      palm_reading = self.mj.get_palm_reading(unnormalise_palm)
    if wrist:
      unnormalise_wrist = True
      wrist_reading = self.mj.get_wrist_reading(unnormalise_wrist)

    action = None

    target_z_height = 10e-3
    min_x_value = 55e-3
    target_angle_deg = 15
    target_bend_force_N = 1
    target_palm_force_N = 1
    target_wrist_force_N = 1

    # hardcode action values
    X_close = 0
    X_open = 1
    Y_close = 2
    Y_open = 3
    Z_plus = 4
    Z_minus = 5
    H_down = 6
    H_up = 7

    # lower fingers to table height
    if self.grasp_phase == 0:

      if wrist:
        # detect the ground with the wrist sensor
        if wrist_reading < target_wrist_force_N:
          action = H_down
        else:
          self.grasp_phase = 1

      else:
        # we aim for a z certain height
        if state_readings[3] < target_z_height:
          action = H_down
        else:
          self.grasp_phase = 1

    # angle the fingers
    if self.grasp_phase == 1:

      # we aim for a certain angle
      print(f"target angle is {-target_angle_deg * (3.14159 / 180.0)}, actual is {self.mj.get_finger_angle()}")
      if -1 * self.mj.get_finger_angle() < target_angle_deg * (3.14159 / 180.0):
        action = Y_close
      else:
        self.grasp_phase = 2

    # constrict until we feel the squeeze on the object
    if self.grasp_phase == 2:

      # wait for a certain bending
      avg_bend = (bending_readings[0] + bending_readings[1] + bending_readings[2]) / 3.
      print(f"target bending is {target_bend_force_N}, actual is {avg_bend}")
      if avg_bend < target_bend_force_N:
        action = X_close
        # if we have closed as much as we can
        if state_readings[0] < min_x_value:
          print(f"minimum x is {min_x_value}, actual is {state_readings[0]}")
          self.grasp_phase = 3
      else:
        self.grasp_phase = 3

    # advance palm to contact object
    if self.grasp_phase == 3:

      if palm:
        print(f"target palm force is {target_palm_force_N}, actual is {palm_reading}")
        if palm_reading < target_palm_force_N:
          action = Z_plus
        else:
          self.grasp_phase = 4
      else:
        pass

    # lift up object
    if self.grasp_phase == 4:
      action = H_up

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

    # if we have not been given a seed
    if seed is None:
      # if we have previously had a seed, reuse the same one (eg reloading from pickle)
      if self.myseed is not None:
        seed = self.myseed
      else:
        # otherwise, get a random seed from [0, maxint]
        seed = np.random.randint(0, 2_147_483_647)

    # set the python random seed in numpy
    np.random.seed(seed)

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

  def load(self, object_set_name=None, object_set_path=None, index=None, num_segments=None):
    """
    Load and prepare the mujoco environment, uses defaults if arguments are not given.
    This function sets the 'params' for the class as well.
    """

    # if not given an input, use class value
    if num_segments is None: set_N = self.load_num_segments
    else: set_N = num_segments

    # set the finger thickness (changes only applied upon reset(), causes hard_reset() if changed)
    self.mj.set_finger_thickness(self.params.finger_thickness)
    self.params.finger_thickness = self.mj.get_finger_thickness()

    self._load_object_set(name=object_set_name, mjcf_path=object_set_path,
                          num_segments=set_N)
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

    if self.log_level > 1: print("MjEnv step() time was ", t1 - t0)

    return to_return

  def reset(self, hard=None):
    """
    Reset the simulation to the start
    """

    # reset tracking variables
    self.track = MjEnv.Track()

    # there is a small chance we reload a new random task
    if ((np.random.random() < self.params.task_reload_chance or
        self.reload_flag) and not self.test_in_progress):
        self._load_xml()

    # reset the simulation and spawn a new random object
    if hard is True: self.mj.hard_reset()
    else: self.mj.reset()
    self._spawn_object()

    return self._next_observation()

  def render(self):
    """
    Render the simulation to a window
    """

    if not self.disable_rendering:

      self.mj.render()

      if self.log_level > 0:
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

  mj = MjEnv()

  
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
