#!/usr/bin/env python3

# test out the mujoco bindings

from rl.env.mjpy.bind import MjClass
import time

path = "/home/luke/gripper_repo_ws/src/gripper_v2/gripper_description/urdf/mujoco/"
gripper_file = "gripper_mujoco.xml"
panda_file = "panda_mujoco.xml"
both_file = "panda_and_gripper_mujoco.xml"

# create mujoco instance
mj = MjClass(path + gripper_file)

i = 0

while True:

  i += 1

  mj.step()
  if not mj.render(): break
  time.sleep(10/1000)