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
#include <memory>
#include <random>

#include "armadillo"

#include "customtypes.h"
#include "objecthandler.h"

// if we have access to boost libraries
#ifndef LUKE_PREVENT_BOOST
  #include "boostdep.h"
#endif

namespace luke
{

template <class T>
struct VectorStruct {
  std::vector<T> panda;
  std::vector<T> gripper;
  std::vector<T> finger;
  std::vector<T> base;

  void reset() {
    // wipe and reallocate
    std::vector<T>().swap(panda);
    std::vector<T>().swap(gripper);
    std::vector<T>().swap(finger);
    std::vector<T>().swap(base);
  }
};

/* ----- global variables ----- */

extern Target target_;      // accessed by mysimulate.cpp for gripper sliders

/* ----- functions ----- */

// helper functions
void print_vec(std::vector<bool> v, std::string name);
void print_vec(std::vector<int> v, std::string name);
void print_vec(std::vector<mjtNum> v, std::string name);
void print_vec(std::vector<gfloat> v, std::string name);
void print_vec(std::vector<QPos> v, std::string name);
void print_vec(std::vector<std::string> v, std::string name);
bool strcmp_w_sub(std::string ref_str, std::string sub_str, int num);

// initialising, setup, and utilities
void init(mjModel* model, mjData* data);
void init_J(mjModel* model, mjData* data);
void print_joint_names(mjModel* model);
void get_joint_indexes(mjModel* model);
void get_joint_addresses(mjModel* model);
void get_geom_indexes(mjModel* model);
bool change_finger_thickness(float thickness);
bool change_finger_width(float width);
void set_finger_stiffness(mjModel* model, mjtNum stiffness);
void set_finger_stiffness(mjModel* model, std::vector<luke::gfloat> stiffness);
void configure_qpos(mjModel* model, mjData* data);
void configure_constraints(mjModel* model, mjData* data);
void keyframe(mjModel* model, mjData* data, std::string keyframe_name);
void keyframe(mjModel* model, mjData* data, int keyframe_index);
void reset(mjModel* model, mjData* data);
void calibrate_reset(mjModel* model, mjData* data);
void get_segment_matrices(mjModel* model, mjData* data);
void set_all_constraints(mjModel* model, mjData* data, bool set_to);
void toggle_constraint(mjModel* model, mjData* data, int id);
void set_constraint(mjModel* model, mjData* data, int id, bool set_as);
void target_constraint(mjModel* model, mjData* data, int id, bool set_as, int type);
void apply_tip_force(mjModel* model, mjData* data, double force, bool reset = false);

void apply_segment_force(mjModel* model, mjData* data, int seg_num, double force,
  double moment = 0);
void set_segment_force(int seg_num, bool set_as, double force);
void set_segment_moment(int seg_num, bool set_as, double moment);
void resolve_segment_forces(mjModel* model, mjData* data);
void apply_UDL(double force_per_m);
void wipe_segment_forces();
void apply_tip_force(double force);
void apply_tip_moment(double moment);

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
void print_state(const mjModel* model, mjData* data);
void update_stepper(mjModel* model, mjData* data);
void update_objects(const mjModel* model, mjData* data);
void update_all(mjModel* model, mjData* data);
void update_constraints(mjModel* model, mjData* data);

// gripper target position
bool set_gripper_target_m(double x, double y, double z);
bool set_gripper_target_m_rad(double x, double th, double z);
bool set_gripper_target_step(int x, int y, int z);
bool set_base_target_m(double x, double y, double z);
bool move_gripper_target_m(double x, double y, double z);
bool move_gripper_target_m_rad(double x, double th, double z);
bool move_gripper_target_step(int x, int y, int z);
bool move_base_target_m(double x, double y, double z);
void set_base_to_max_height(mjData* data);
void print_target();
void update_target();

// sensing
gfloat read_armadillo_gauge(const mjData* data, int finger);
std::vector<gfloat> get_gauge_data(const mjModel* model, mjData* data);
gfloat get_palm_force(const mjModel* model, mjData* data);
std::vector<gfloat> get_panda_state(const mjData* data);
std::vector<gfloat> get_gripper_state(const mjData* data);
std::vector<gfloat> get_target_state();
gfloat get_target_finger_angle();

// environment
Gripper get_gripper_target();
std::vector<std::string> get_objects();
void set_object_pose(mjData* data, int idx, QPos pose);
void reset_object(mjModel* model, mjData* data);
void spawn_object(mjModel* model, mjData* data, std::string name, QPos pose);
void spawn_object(mjModel* model, mjData* data, int idx, QPos pose);
QPos get_object_qpos(mjModel* model, mjData* data);
Forces get_object_forces(const mjModel* model, mjData* data);
Forces_faster get_object_forces_faster(const mjModel* model, mjData* data);
void set_object_colour(mjModel* model, std::vector<float> rgba);
void set_ground_colour(mjModel* model, std::vector<float> rgba);
void randomise_all_colours(mjModel* model, std::shared_ptr<std::default_random_engine> generator);
void default_colours(mjModel* model);
void set_finger_colour(mjModel* model, std::vector<float> rgba, int finger_num);

// other
gfloat verify_armadillo_gauge(const mjData* data, int finger,
  std::vector<float>& vec_joint_x, std::vector<float>& vec_joint_y,
  std::vector<float>& vec_coefficients, std::vector<float>& vec_errors);
gfloat verify_small_angle_model(const mjData* data, int finger,
  std::vector<float>& joint_angles, std::vector<float>& joint_pred,
  std::vector<float>& pred_x, std::vector<float>& pred_y, std::vector<float>& theory_y,
  std::vector<float>& theory_x_curve, std::vector<float>& theory_y_curve,
  float force, float finger_stiffness, int force_style = 0);
void fill_theory_curve(std::vector<float>& theory_X, std::vector<float>& theory_Y, 
  float force, int num, int force_style = 0);
std::vector<float> discretise_curve(std::vector<float> X, std::vector<float> truth_X, 
  std::vector<float> truth_Y);
int last_action_robot();
bool is_sim_unstable(mjModel* model, mjData* data);
int get_N();
float get_finger_thickness();
float calc_yield_point_load();
float get_fingertip_z_height();
std::vector<luke::gfloat> get_stiffnesses();
void print_stiffnesses();

} // namespace luke

#endif // MYFUNCTIONS_H_