#!/usr/bin/env python3

# add the env folder to path
import os
import sys
sys.path.insert(0, os.path.expanduser('~') + '/mymujoco/rl/env')

# with env in path, we can now import the shared cpp library
from mjpy.bind import MjClass

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

  @dataclass
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
    self.default_path = "/home/luke/gripper_repo_ws/src/gripper_v2/gripper_description/urdf/mujoco/"
    self.task_xml_template = "/task/gripper_task_{}.xml"
    self.default_xml = "gripper_task.xml"
    self.training_xmls = 37
    self.testing_xmls = 1                 # test xml is always the last one
    
    # create mujoco instance
    self.mj = MjClass()
    self._load_xml(random=True)

    # # do we override the c++ default simulator settings - comment out if not
    # self._override_settings()    

    # auto generated parameters
    self._update_n_actions_obs()

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

  def _load_xml(self, path=None, task_id=None, random=None, test=None):
    """
    Load the mujoco instance with the given mjcf xml file name
    """

    # set to default task filename
    filename = self.default_xml
    filepath = self.default_path

    # check user inputs
    if path:
      filepath = path
    if task_id != None:
      filename = self.task_xml_template.format(task_id)
    if random:
      r = np.random.randint(0, self.training_xmls)
      filename = self.task_xml_template.format(r)
    if test:
      # return
      filename = self.task_xml_template.format(self.training_xmls 
                                                + self.testing_xmls - 1)

    if self.log_level > 1: print("loading xml: ", filename)

    # self.mj = MjClass(self.mj.set) # old: create a brand new mjclass instance

    # load the new task xml (old model/data are deleted)
    self.mj.load_relative(filename)
    self.num_objects = self.mj.get_number_of_objects()

    self.reload_flag = False

  def _update_n_actions_obs(self):
    # get an updated number of actions and observations

    self.n_actions = self.mj.get_n_actions()
    self.n_obs = self.mj.get_n_obs()

  def _override_binary(self, field, reward, done, trigger):
    """
    Override a binary reward, field should be self.mj.set.<reward_name>
    """
    field.reward = reward
    field.done = done
    field.trigger = int(trigger)

  def _override_linear(self, field, reward, done, trigger, min, max, overshoot):
    """
    Override a linear reward, field should be self.mj.set.<reward_name>
    """
    field.reward = reward
    field.done = done
    field.trigger = int(trigger)
    field.min = min 
    field.max = max
    field.overshoot = overshoot

  def _override_settings(self):
    """
    Key simulation settings are present in the mjclass.h file. Check there to
    see the defaults - this function may NOT be up to date. This function is
    provided for reference, DO NOT run this function. Instead copy out lines
    """

    never = 1e4

    # reward settings
    self._override_binary(self.mj.set.step_num,         -0.01,  False,  1)
    self._override_binary(self.mj.set.lifted,           0.005,  False,  1)
    self._override_binary(self.mj.set.oob,              0,      1,      1)
    self._override_binary(self.mj.set.dropped,          0,      False,  never)
    self._override_binary(self.mj.set.target_height,    1.0,    1,      never)
    self._override_binary(self.mj.set.exceed_limits,    -0.1,   False,  1)
    self._override_binary(self.mj.set.object_contact,   0.005,  False,  1)
    self._override_binary(self.mj.set.object_stable,    1.0,    1,      3)
    self._override_linear(self.mj.set.exceed_axial,     -0.05,  False,  1,  2.0, 6.0, -1)
    self._override_linear(self.mj.set.exceed_lateral,   -0.05,  False,  1,  4.0, 6.0, -1)
    self._override_linear(self.mj.set.palm_force,       0.05,   False,  1,  1.0, 3.0, 6.0)
    self._override_linear(self.mj.set.exceed_palm,      -0.05,  False,  1,  6.0, 10.0, -1)

    # step() settings
    self.mj.set.gauge_read_rate_hz = 10

    # update_env() settings
    self.mj.set.lift_distance = 1e-3
    self.mj.set.height_target = 50e-3
    self.mj.set.oob_distance = 50e-3
    self.mj.set.stable_finger_force = 1.00
    self.mj.set.stable_palm_force = 2.00

    # is_done() settings
    self.mj.set.max_timeouts = 10

    # set_action() settings
    self.mj.set.action_motor_steps = 100             
    self.mj.set.action_base_translation = 3e-3 
    self.mj.set.max_action_steps = 1000
    self.mj.set.paired_motor_X_step = True

    # action_step() settings
    self.mj.set.render_on_step = False       
    self.mj.set.use_settling = False     

    # render() settings
    self.mj.set.use_render_delay = False       
    self.mj.set.render_delay = 0.5

    raise RuntimeError("_override_settings is provided for reference, it should not be run")

    return

  def _get_cpp_settings(self):
    """
    Return a string of all the cpp simulation settings
    """

    return self.mj.set.get_settings()

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

  def _reward(self):
    """
    Calculate the reward on this step
    """

    return self.mj.reward()

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

    # trial results
    trial_data.object_name = test_report.object_name
    trial_data.reward = self.track.cumulative_reward
    trial_data.steps = self.track.current_step

    # conditions of object at episode end
    trial_data.lifted = bool(test_report.final_cnt.lifted)
    trial_data.stable = bool(test_report.final_cnt.object_stable)
    trial_data.oob = bool(test_report.final_cnt.oob)
    trial_data.target_height = bool(test_report.final_cnt.target_height)
    trial_data.stable_height = bool(test_report.final_cnt.stable_height)
    trial_data.palm_force = test_report.final_palm_force
    trial_data.finger_force = test_report.final_finger_force

    # counts during the episode of events
    trial_data.cnt_lifted = test_report.abs_cnt.lifted
    trial_data.cnt_object_contact = test_report.abs_cnt.object_contact
    trial_data.cnt_palm_force = test_report.abs_cnt.palm_force
    trial_data.cnt_exceed_limits = test_report.abs_cnt.exceed_limits
    trial_data.cnt_exceed_axial = test_report.abs_cnt.exceed_axial
    trial_data.cnt_exceed_lateral = test_report.abs_cnt.exceed_lateral
    trial_data.cnt_exceed_palm = test_report.abs_cnt.exceed_palm

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
    reward = self._reward()
    done = self._is_done()

    self.track.last_action = action
    self.track.last_reward = reward
    self.track.is_done = done
    self.track.cumulative_reward += reward

    # track testing if this result has finished
    if done and self.test_in_progress:
      self._monitor_test()

    t1 = time.time()

    if self.log_level > 1: print("MjEnv step() time was ", t1 - t0)

    return obs, reward, done, {}

  def reset(self):
    """
    Reset the simulation to the start
    """

    # reset tracking variables
    self.track = MjEnv.Track()

    # there is a small chance we reload a new random task
    if ((np.random.random() < self.task_reload_chance or
        self.reload_flag) and not self.test_in_progress):
        self._load_xml(random=True)

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

  # mj._override_binary(mj.mj.set.step_num, 1.0, 3, 10)
  # mj.mj.set.lifted.set(-0.1, 42, 100)
  # mj.mj.set.gauge_read_rate_hz = 100

  # with open("test_file.pickle", 'wb') as f:
  #   pickle.dump(mj, f)

  # with open("test_file.pickle", 'rb') as f:
  #   mj = pickle.load(f)

  mj._load_xml(task_id=0)

  mj.step(0)

  # print(mj._get_cpp_settings())

  # mj.mj.set.wipe_rewards()

  # print(mj._get_cpp_settings())

  obs = mj.mj.get_observation()

  print(obs)
