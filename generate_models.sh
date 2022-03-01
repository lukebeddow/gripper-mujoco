#!/bin/bash

DIR=/home/luke/gripper_repo_ws/src/gripper_v2/gripper_description/urdf/mujoco
FILE=generate_xml.sh

# run the file (pass over first argument)
$DIR/$FILE $1

