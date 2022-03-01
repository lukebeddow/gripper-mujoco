#ifndef MYFUNCTIONS_H_
#define MYFUNCTIONS_H_

#include "mjxmacro.h"
// #include "uitools.h"
#include "mujoco.h"
#include "stdio.h"
#include "string.h"

#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <array>
#include <algorithm>
#include <cstdio>

#include "armadillo"

#include "customtypes.h"
#include "objecthandler.h"

namespace luke
{

/* ----- global variables ----- */

extern Target target_;      // accessed by mysimulate.cpp for gripper sliders

/* ----- functions ----- */

// helper functions
void print_vec(std::vector<bool> v, std::string name);
void print_vec(std::vector<int> v, std::string name);
void print_vec(std::vector<mjtNum> v, std::string name);
void print_vec(std::vector<QPos> v, std::string name);
void print_vec(std::vector<std::string> v, std::string name);

// initialising, setup, and utilities
void init(mjModel* model, mjData* data);
void init_J(mjModel* model, mjData* data);
void print_joint_names(mjModel* model);
void get_joint_indexes(mjModel* model);
void get_joint_addresses(mjModel* model);
void keyframe(mjModel* model, mjData* data, std::string keyframe_name);
void keyframe(mjModel* model, mjData* data, int keyframe_index);
void reset(mjModel* model, mjData* data);
void reload(mjModel* model, mjData* data);
void wipe_settled();

// simulation
void before_step(mjModel* model, mjData* data);
void step(mjModel* model, mjData* data);
void after_step(mjModel* model, mjData* data);

// control
void control(const mjModel* model, mjData* data);
void control_panda(const mjModel* model, mjData* data);
void control_gripper(const mjModel* model, mjData* data, Gripper& target);
void control_base(const mjModel* model, mjData* data);
void update_state(const mjModel* model, mjData* data);
void update_stepper(const mjModel* model, mjData* data);
void update_objects(const mjModel* model, mjData* data);
void update_all(const mjModel* model, mjData* data);

// monitor
void check_settling();
bool is_settled();
bool is_target_reached();
bool is_target_step();
bool within_limits();

// gripper target position
bool set_gripper_target_m(double x, double y, double z);
bool set_gripper_target_m_rad(double x, double th, double z);
bool set_gripper_target_step(int x, int y, int z);
bool move_gripper_target_m(double x, double y, double z);
bool move_gripper_target_m_rad(double x, double th, double z);
bool move_gripper_target_step(int x, int y, int z);
bool move_base_target_m(double x, double y, double z);
void print_target();
void update_target();

// sensing
gfloat read_finger_gauge(const mjData* data, int finger);
std::vector<gfloat> get_gauge_data(const mjModel* model, mjData* data);
std::vector<gfloat> get_panda_state(const mjData* data);
std::vector<gfloat> get_gripper_state(const mjData* data);
std::vector<gfloat> get_target_state();

// environment
Gripper get_gripper_target();
std::vector<std::string> get_objects();
void set_object_pose(mjData* data, int idx, QPos pose);
void reset_object(mjModel* model, mjData* data);
void spawn_object(mjModel* model, mjData* data, std::string name, QPos pose);
void spawn_object(mjModel* model, mjData* data, int idx, QPos pose);
QPos get_object_qpos();
Forces get_object_forces(const mjModel* model, mjData* data);

} // namespace luke

#endif // MYFUNCTIONS_H_