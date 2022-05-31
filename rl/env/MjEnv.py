#!/usr/bin/env python3

# add the env folder to path
import os
import pickle
import sys

# get the path to this file and insert it to python path (for mjpy.bind)
pathhere = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, pathhere)

# with env in path, we can now import the shared cpp library
from mjpy.bind import MjClass, EventTrack

import time
import gym
import numpy as np
from dataclasses import dataclass

class MjEnv(gym.Env):

  @dataclass
  class Track:
    # tracking variables that are reset after each episode
    current_step: int = 0
    last_action: int = -1
    is_done: bool = False
    last_reward: float = 0
    cumulative_reward: float = 0

  # @dataclass
  class Test:
    # saved after each test trial ends
    obj_idx: int = 0
    obj_trial: int = 0
    reward: float = 0
    steps: int = 0
    object_name: str = ""

    # conditions of the object at episode end
    lifted: bool = False
    stable: bool = False
    oob: bool = False
    target_height: bool = False
    stable_height: bool = False
    palm_force: float = 0
    finger_force: float = 0

    # counts during the episode of events
    cnt_lifted: int = 0
    cnt_object_contact: int = 0
    cnt_palm_force: int = 0
    cnt_exceed_limits: int = 0
    cnt_exceed_axial: int = 0
    cnt_exceed_lateral: int = 0
    cnt_exceed_palm: int = 0

    def add_data(self, cnt): self.cnt = cnt

  def __init__(self):
    """
    A mujoco environment wrapper for OpenAI gym
    """

    super(MjEnv, self).__init__()

    # user defined parameters
    self.max_episode_steps = 100
    self.object_position_noise_mm = 10
    self.disable_rendering = True
    self.task_reload_chance = 1 / 40.
    self.log_level = 0

    # user defined testing parameters
    self.test_in_progress = False
    self.test_completed = False
    self.test_trials_per_obj = 3
    self.test_obj_limit = 1000  # limit number of objects in test, 1000~=no limit
    
    # define file structure
    self.task_xml_folder = "task"
    self.task_xml_template = "gripper_task_{}.xml"
    self.testing_xmls = 1                 # how many files reserved for testing
    
    # create mujoco instance
    self.mj = MjClass()
    self._load_object_set()
    self._load_xml()  

    # auto generated parameters
    self._update_n_actions_obs()

    # reset any lingering goal defaults
    self.mj.reset_goal()

    # initialise tracking variables
    self.track = MjEnv.Track()
    self.test = MjEnv.Test()
    self.prev_test = MjEnv.Test()

    # # limits on gauge readings
    # max_gauge = 50
    # high = np.array(self.state_max 
    #   + [max_gauge for i in range(3 * self.num_gauge_readings)], dtype=np.float32)
    # low = np.array(self.state_min
    #   + [-max_gauge for i in range(3 * self.num_gauge_readings)], dtype=np.float32)

    # # define key gym variables
    # self.action_space = gym.spaces.Discrete(self.num_actions)
    # self.observation_space = gym.spaces.Box(low=low, high=high)

    return

  def _load_xml(self, test=None, index=None):
    """
    Load the mujoco instance with the given mjcf xml file name
    """
    if index:
      filename = self.task_xml_template.format(index)
    elif test:
      # get the random test xml file
      r = np.random.randint(self.training_xmls, self.training_xmls + self.testing_xmls)
      filename = self.task_xml_template.format(r)
    else:
      # get a random task xml file
      r = np.random.randint(0, self.training_xmls)
      filename = self.task_xml_template.format(r)

    if self.log_level > 1: print("loading xml: ", filename)

    # load the new task xml (old model/data are deleted)
    self.mj.load_relative(self.task_xml_folder + '/' + filename)
    self.num_objects = self.mj.get_number_of_objects()

    self.reload_flag = False

  def _load_object_set(self, name=None, mjcf_path=None):
    """
    Determine how many model xml files are in the object set
    """

    # if a mjcf_path is given, override, otherwise we use default
    if mjcf_path != None: self.mj.model_folder_path = mjcf_path

    # if a object set name is given, override, otherwise we use default
    if name != None: self.mj.object_set_name = name

    # check the mjcf_path is correctly formatted
    if self.mj.model_folder_path[-1] != '/':
      self.mj.model_folder_path += '/'

    # now determine the model xml path
    self.xml_path = (self.mj.model_folder_path + self.mj.object_set_name 
                      + '/' + self.task_xml_folder + '/')

    # find out how many xmls are available for training/testing
    xml_files = [x for x in os.listdir(self.xml_path)]
    self.training_xmls = len(xml_files) - self.testing_xmls

    if self.training_xmls < 1:
      raise RuntimeError(f"training xmls failed to be found in MjEnv at: {self.xml_path}")

  def _update_n_actions_obs(self):
    # get an updated number of actions and observations

    self.n_actions = self.mj.get_n_actions()
    self.n_obs = self.mj.get_n_obs()

    # for testing
    print("n_actions is", self.n_actions)
    print("n_obs is", self.n_obs)

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
    if self.track.current_step >= self.max_episode_steps:
      if self.log_level > 1: print("max number of episode steps exceeded, done = true")
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
    obj_idx = np.random.randint(0, self.num_objects)
    x_pos_mm = np.random.randint(-self.object_position_noise_mm, self.object_position_noise_mm + 1)
    y_pos_mm = np.random.randint(-self.object_position_noise_mm, self.object_position_noise_mm + 1)

    # if we are doing a test, chose a specific object
    if self.test_in_progress:
      obj_idx = self.current_test_trial.obj_idx

    # # for testing - fix these random quantities
    # obj_idx = 0
    # x_pos_mm = 0
    # y_pos_mm = 0

    # spawn in the object
    self.mj.spawn_object(obj_idx, x_pos_mm * 1e-3, y_pos_mm * 1e-3)

    return

  def _seed(seed):
    """
    Set the seed for the environment
    """

    np.random.seed(seed)

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
    trial_data.add_data(test_report.cnt)

    # insert information into stored data list
    self.test_trials.append(trial_data)

    # increment object trial
    self.current_test_trial.obj_trial += 1

    # if trials done, move to the next object, reset trial counter
    if self.current_test_trial.obj_trial >= self.test_trials_per_obj:
      self.current_test_trial.obj_idx += 1
      self.current_test_trial.obj_trial = 0

      # if objects finished/exceeded, test is over
      if (self.current_test_trial.obj_idx >= self.num_objects or
          self.current_test_trial.obj_idx >= self.test_obj_limit):
        self._end_test()

  def start_test(self):
    """
    Begin test mode, should be called by class user
    """

    self.current_test_trial = MjEnv.Test()
    self.test_trials = []
    self.test_in_progress = True
    self.test_completed = False
    self._load_xml(test=True)

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
        reward = self._goal_reward(state, goal)
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

  def reset(self):
    """
    Reset the simulation to the start
    """

    # reset tracking variables
    self.track = MjEnv.Track()

    # there is a small chance we reload a new random task
    if ((np.random.random() < self.task_reload_chance or
        self.reload_flag) and not self.test_in_progress):
        self._load_xml()

    # reset the simulation and spawn a new random object
    self.mj.reset()
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

  mj.mj.set.wipe_rewards()
  mj.mj.set.lifted.set(100, 10, 1)
  print(mj._get_cpp_settings())

  with open("test_file.pickle", 'wb') as f:
    pickle.dump(mj, f)
    print("Pickle saved")

  with open("test_file.pickle", 'rb') as f:
    mj = pickle.load(f)
    print("Pickle loaded")

  mj._load_xml(index=0)
  mj._load_xml(index=1)
  mj._load_xml(index=2)

  mj.step(0)

  print(mj._get_cpp_settings())

  # obs = mj.mj.get_observation()
  # print(obs)
