#ifndef CUSTOMTYPES_H_
#define CUSTOMTYPES_H_

#include "mujoco.h"

#include <vector>
#include <iostream>
#include <array>

// include large custom types (all in namespace luke already)
#include "slidingwindow.h"
#include "gripper.h"
#include "mynum.h"

// next we will define some more smaller types in the namespace
namespace luke
{

/* ----- type definitions ----- */

// define the precision of the gauges
typedef float gfloat;

/* ----- custom types ----- */

struct Forces {

  /* Here we will save forces extracted from the simulation.

  For the local finger frame: 
    x = axial, y = tangential, z = lateral, all +ve with object contact

  For the local palm frame:
    x = axial, -ve with object contact
  */

  bool empty = false;

  // forces applied on the object
  struct Obj {

    // global frame
    myNum net;           // net force on object, almost always [0,0,weight,0,0,0]
    myNum sum;           // sum of all contact forces, does not always = net or 0
    myNum finger1;
    myNum finger2;
    myNum finger3;
    myNum palm;
    myNum ground;

    // forces felt from the object in their local frame
    myNum finger1_local;  
    myNum finger2_local;
    myNum finger3_local;
    myNum palm_local;

    // need to initialise to zero to enable += running totals
    Obj() : net(6,1), sum(6,1), finger1(6,1), finger2(6,1), finger3(6,1),
            palm(6,1), ground(6,1), finger1_local(3,1), finger2_local(3,1),
            finger3_local(3,1), palm_local(3,1) {}

  } obj;

  // all forces involved (excluding unnamed geoms!)
  struct All {

    // global frame
    myNum finger1;
    myNum finger2;
    myNum finger3;
    myNum palm;

    // local frame
    myNum finger1_local;
    myNum finger2_local;
    myNum finger3_local;
    myNum palm_local;

    // need to initialise to zero to enable += running totals
    All() : finger1(6,1), finger2(6,1), finger3(6,1), palm(6,1), finger1_local(3,1),
            finger2_local(3,1), finger3_local(3,1), palm_local(6,1) {}

  } all;

  // ground forces on gripper fingers
  struct Gnd {

    // global frame
    myNum finger1;
    myNum finger2;
    myNum finger3;

    // local frame
    myNum finger1_local;
    myNum finger2_local;
    myNum finger3_local;

    // need to initialise to zero to enable += running totals
    Gnd() : finger1(6,1), finger2(6,1), finger3(6,1), finger1_local(3,1),
            finger2_local(3,1), finger3_local(3,1) {}

  } gnd;

  void print() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    print_gnd_global();
    print_gnd_local();    
    print_obj_global();
    print_obj_local();
    print_all_global();
    print_all_local();
  }

  void print_obj_global() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    std::cout << "Printing forces on object in global frame:\n";
    std::cout << "net force (mag = " << obj.net.magnitude3() << "):\n"; obj.net.print();
    std::cout << "sum force (mag = " << obj.sum.magnitude3() << "):\n"; obj.sum.print();
    std::cout << "ground force (mag = " << obj.ground.magnitude3() << "):\n"; obj.ground.print();
    std::cout << "finger1 force (mag = " << obj.finger1.magnitude3() << "):\n"; obj.finger1.print();
    std::cout << "finger2 force (mag = " << obj.finger2.magnitude3() << "):\n"; obj.finger2.print();
    std::cout << "finger3 force (mag = " << obj.finger3.magnitude3() << "):\n"; obj.finger3.print();
    std::cout << "palm force (mag = " << obj.palm.magnitude3() << "):\n"; obj.palm.print();
  }

  void print_obj_local() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    std::cout << "Printing forces from object in local frames:\n";
    std::cout << "finger1 local force (mag = " << obj.finger1_local.magnitude3() << "):\n"; obj.finger1_local.print();
    std::cout << "finger2 local force (mag = " << obj.finger2_local.magnitude3() << "):\n"; obj.finger2_local.print();
    std::cout << "finger3 local force (mag = " << obj.finger3_local.magnitude3() << "):\n"; obj.finger3_local.print();
    std::cout << "palm local force (mag = " << obj.palm.magnitude3() << "):\n"; obj.palm_local.print();
  }

  void print_all_global() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    std::cout << "Printing forces from all named geoms in global frame:\n";
    std::cout << "finger1 force (mag = " << all.finger1.magnitude3() << "):\n"; all.finger1.print();
    std::cout << "finger2 force (mag = " << all.finger2.magnitude3() << "):\n"; all.finger2.print();
    std::cout << "finger3 force (mag = " << all.finger3.magnitude3() << "):\n"; all.finger3.print();
    std::cout << "palm force (mag = " << all.palm.magnitude3() << "):\n"; all.palm.print();
  }

  void print_all_local() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    std::cout << "Printing forces from all named geoms on the gripper fingers:\n";
    std::cout << "finger1 local force (mag = " << all.finger1_local.magnitude3() << "):\n"; all.finger1_local.print();
    std::cout << "finger2 local force (mag = " << all.finger2_local.magnitude3() << "):\n"; all.finger2_local.print();
    std::cout << "finger3 local force (mag = " << all.finger3_local.magnitude3() << "):\n"; all.finger3_local.print();
    std::cout << "palm local force (mag = " << all.palm.magnitude3() << "):\n"; all.palm_local.print();
  }

  void print_gnd_global() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    std::cout << "Printing forces from gnd named geoms in global frame:\n";
    std::cout << "finger1 force (mag = " << gnd.finger1.magnitude3() << "):\n"; gnd.finger1.print();
    std::cout << "finger2 force (mag = " << gnd.finger2.magnitude3() << "):\n"; gnd.finger2.print();
    std::cout << "finger3 force (mag = " << gnd.finger3.magnitude3() << "):\n"; gnd.finger3.print();
  }

  void print_gnd_local() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    std::cout << "Printing forces from gnd named geoms in local frames:\n";
    std::cout << "finger1 local force (mag = " << gnd.finger1_local.magnitude3() << "):\n"; gnd.finger1_local.print();
    std::cout << "finger2 local force (mag = " << gnd.finger2_local.magnitude3() << "):\n"; gnd.finger2_local.print();
    std::cout << "finger3 local force (mag = " << gnd.finger3_local.magnitude3() << "):\n"; gnd.finger3_local.print();
 }

};

struct Forces_faster {

  /* Here we will save forces extracted from the simulation.

  For the local finger frame: 
    x = axial, y = tangential, z = lateral, all +ve with object contact

  For the local palm frame:
    x = axial, -ve with object contact
  */

  bool empty = false;

  // forces applied on the object
  struct Obj {

    // global frame
    rawNum net;           // net force on object, almost always [0,0,weight,0,0,0]
    rawNum sum;           // sum of all contact forces, does not always = net or 0
    rawNum finger1;
    rawNum finger2;
    rawNum finger3;
    rawNum palm;
    rawNum ground;

    // forces felt from the object in their local frame
    rawNum finger1_local;  
    rawNum finger2_local;
    rawNum finger3_local;
    rawNum palm_local;

    // need to initialise to zero to enable += running totals
    Obj() : net(6,1), sum(6,1), finger1(6,1), finger2(6,1), finger3(6,1),
            palm(6,1), ground(6,1), finger1_local(3,1), finger2_local(3,1),
            finger3_local(3,1), palm_local(3,1) {}

  } obj;

  // all forces involved (excluding unnamed geoms!)
  struct All {

    // global frame
    rawNum finger1;
    rawNum finger2;
    rawNum finger3;
    rawNum palm;

    // local frame
    rawNum finger1_local;
    rawNum finger2_local;
    rawNum finger3_local;
    rawNum palm_local;

    // need to initialise to zero to enable += running totals
    All() : finger1(6,1), finger2(6,1), finger3(6,1), palm(6,1), finger1_local(3,1),
            finger2_local(3,1), finger3_local(3,1), palm_local(6,1) {}

  } all;

  // ground forces on gripper fingers
  struct Gnd {

    // global frame
    rawNum finger1;
    rawNum finger2;
    rawNum finger3;

    // local frame
    rawNum finger1_local;
    rawNum finger2_local;
    rawNum finger3_local;

    // need to initialise to zero to enable += running totals
    Gnd() : finger1(6,1), finger2(6,1), finger3(6,1), finger1_local(3,1),
            finger2_local(3,1), finger3_local(3,1) {}

  } gnd;

  void print() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    print_gnd_global();
    print_gnd_local();    
    print_obj_global();
    print_obj_local();
    print_all_global();
    print_all_local();
  }

  void print_obj_global() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    std::cout << "Printing forces on object in global frame:\n";
    std::cout << "net force (mag = " << obj.net.magnitude3() << "):\n"; obj.net.print();
    std::cout << "sum force (mag = " << obj.sum.magnitude3() << "):\n"; obj.sum.print();
    std::cout << "ground force (mag = " << obj.ground.magnitude3() << "):\n"; obj.ground.print();
    std::cout << "finger1 force (mag = " << obj.finger1.magnitude3() << "):\n"; obj.finger1.print();
    std::cout << "finger2 force (mag = " << obj.finger2.magnitude3() << "):\n"; obj.finger2.print();
    std::cout << "finger3 force (mag = " << obj.finger3.magnitude3() << "):\n"; obj.finger3.print();
    std::cout << "palm force (mag = " << obj.palm.magnitude3() << "):\n"; obj.palm.print();
  }

  void print_obj_local() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    std::cout << "Printing forces from object in local frames:\n";
    std::cout << "finger1 local force (mag = " << obj.finger1_local.magnitude3() << "):\n"; obj.finger1_local.print();
    std::cout << "finger2 local force (mag = " << obj.finger2_local.magnitude3() << "):\n"; obj.finger2_local.print();
    std::cout << "finger3 local force (mag = " << obj.finger3_local.magnitude3() << "):\n"; obj.finger3_local.print();
    std::cout << "palm local force (mag = " << obj.palm.magnitude3() << "):\n"; obj.palm_local.print();
  }

  void print_all_global() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    std::cout << "Printing forces from all named geoms in global frame:\n";
    std::cout << "finger1 force (mag = " << all.finger1.magnitude3() << "):\n"; all.finger1.print();
    std::cout << "finger2 force (mag = " << all.finger2.magnitude3() << "):\n"; all.finger2.print();
    std::cout << "finger3 force (mag = " << all.finger3.magnitude3() << "):\n"; all.finger3.print();
    std::cout << "palm force (mag = " << all.palm.magnitude3() << "):\n"; all.palm.print();
  }

  void print_all_local() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    std::cout << "Printing forces from all named geoms on the gripper fingers:\n";
    std::cout << "finger1 local force (mag = " << all.finger1_local.magnitude3() << "):\n"; all.finger1_local.print();
    std::cout << "finger2 local force (mag = " << all.finger2_local.magnitude3() << "):\n"; all.finger2_local.print();
    std::cout << "finger3 local force (mag = " << all.finger3_local.magnitude3() << "):\n"; all.finger3_local.print();
    std::cout << "palm local force (mag = " << all.palm.magnitude3() << "):\n"; all.palm_local.print();
  }

  void print_gnd_global() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    std::cout << "Printing forces from gnd named geoms in global frame:\n";
    std::cout << "finger1 force (mag = " << gnd.finger1.magnitude3() << "):\n"; gnd.finger1.print();
    std::cout << "finger2 force (mag = " << gnd.finger2.magnitude3() << "):\n"; gnd.finger2.print();
    std::cout << "finger3 force (mag = " << gnd.finger3.magnitude3() << "):\n"; gnd.finger3.print();
  }

  void print_gnd_local() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    std::cout << "Printing forces from gnd named geoms in local frames:\n";
    std::cout << "finger1 local force (mag = " << gnd.finger1_local.magnitude3() << "):\n"; gnd.finger1_local.print();
    std::cout << "finger2 local force (mag = " << gnd.finger2_local.magnitude3() << "):\n"; gnd.finger2_local.print();
    std::cout << "finger3 local force (mag = " << gnd.finger3_local.magnitude3() << "):\n"; gnd.finger3_local.print();
 }

};

struct Gain {

  // gains for x,y,z motors
  double x = 0;
  double y = 0;
  double z = 0;

  // constructors
  Gain() {};
  Gain(double all) : x(all), y(all), z(all) {};
  Gain(double x, double y, double z) : x(x), y(y), z(z) {};

  // functions
  void set(double all) {
    x = all; y = all; z = all;
  }
  void set(double x_, double y_, double z_) {
    x = x_; y = y_; z = z_;
  }
};

struct QPos {

  // data
  double x, y, z, qx, qy, qz, qw;

  // constructor
  QPos() : x(0), y(0), z(0), qx(0), qy(0), qz(0), qw(1) {}
  QPos(double x, double y, double z, double qx, double qy, double qz, double qw)
    : x(x), y(y), z(z), qx(qx), qy(qy), qz(qz), qw(qw) {}
  void reset() { x = y = z = qx = qy = qz = 0; qw = 1; }

  // update using mjData and qpos address (idx)
  void update(const mjModel* m, const mjData* d, int idx) {
    if (idx == -1) {
      x = y = z = qx = qy = qz = 0; qw = 1;
      return;
    }
    if (idx + 6 > m->nq) {
      throw std::runtime_error("qpos.update() is out of range");
    }
    x = d->qpos[idx];
    y = d->qpos[idx + 1];
    z = d->qpos[idx + 2];
    qx = d->qpos[idx + 3];
    qy = d->qpos[idx + 4];
    qz = d->qpos[idx + 5];
    qw = d->qpos[idx + 6];
  }

  void print() {
    std::printf("QPos: xyz = (%.3f, %.3f, %.3f) quat = (%.3f, %.3f, %.3f, %.3f)\n",
      x, y, z, qx, qy, qz, qw);
  }
};

struct JointStates {

  // list of all possible joints and their state
  double gripper_x { 0.0 };
  double gripper_y { 0.0 };
  double gripper_z { 0.0 };
  double base_x { 0.0 };
  double base_y { 0.0 };
  double base_z { 0.0 };
  double base_roll { 0.0 };
  double base_pitch { 0.0 };
  double base_yaw { 0.0 };
};

struct Base {

  /* describing the gripper base motions */
  
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;

  Base() : x(0), y(0), z(0), roll(0), pitch(0), yaw(0) {}

  Base(double x, double y, double z, double roll, double pitch, double yaw)
    : x(x), y(y), z(z), roll(roll), pitch(pitch), yaw(yaw)
    {}

  void reset() {
    x = y = z = roll = pitch = yaw = 0.0;
  }

  std::vector<double> to_vec() {
    std::vector<double> out { x, y, z, roll, pitch, yaw };
    return out;
  }

};

struct Target {

  Gripper end;                        // final target destination
  Gripper next;                       // target at the next step

  Base base;
  Base base_min;
  Base base_max;

  // std::array<double, 6> base {};      // target of base joints (only [0] used for z)
  std::array<double, 7> panda {};     // target for panda joints (never used)

  // for real life tracking what has just moved
  struct Robot {
    enum {
      none = 0,
      gripper,
      panda
    };
  };

  // what was the last robot motion achieved with
  int last_robot = Robot::none;

  /* ground is at -10mm, max finger tilt from start before tips touch lifts them another
  10mm, then we add 10mm of padding to get +-30mm */
  // static constexpr double base_z_min = -30e-3;
  // static constexpr double base_z_max = 30e-3;

  // calibration value since z height will not be perfect from controller stiffness
  double z_offset = 0;

  // static constexpr std::array<double, 6> base_lims_max {0.1, 1, 1, 1, 1, 1};
  // static constexpr std::array<double, 6> base_lims_min {-0.1, -1, -1, -1, -1, -1};

  // for testing, only used if flag 'log_test_data' is true in update_stepper(...) in myfunctions.cpp
  int datanum = 400;
  SlidingWindow<float> timedata {datanum};
  SlidingWindow<float> target_stepperx {datanum};
  SlidingWindow<float> target_steppery {datanum};
  SlidingWindow<float> target_stepperz {datanum};
  SlidingWindow<float> target_basex {datanum};
  SlidingWindow<float> target_basey {datanum};
  SlidingWindow<float> target_basez {datanum};
  SlidingWindow<float> target_baseroll {datanum};
  SlidingWindow<float> target_basepitch {datanum};
  SlidingWindow<float> target_baseyaw {datanum};
  SlidingWindow<float> actual_stepperx {datanum};
  SlidingWindow<float> actual_steppery {datanum};
  SlidingWindow<float> actual_stepperz {datanum};
  SlidingWindow<float> actual_basex {datanum};
  SlidingWindow<float> actual_basey {datanum};
  SlidingWindow<float> actual_basez {datanum};
  SlidingWindow<float> actual_baseroll {datanum};
  SlidingWindow<float> actual_basepitch {datanum};
  SlidingWindow<float> actual_baseyaw {datanum};

  void reset() {
    end.reset();
    next.reset();
    base.reset();
    panda.fill(0);
  }

  JointStates get_target_m()
  {
    /* returns in metres (or radians for rpy) the joint states */

    JointStates state;

    state.gripper_x = end.get_x_m();
    state.gripper_y = end.get_y_m();
    state.gripper_z = end.get_z_m();
    state.base_x = base.x;
    state.base_y = base.y;
    state.base_z = base.z;
    state.base_roll = base.roll;
    state.base_pitch = base.pitch;
    state.base_yaw = base.yaw;

    return state;
  }

  bool x_moving() {
    if (end.get_x_step() == next.get_x_step()) {
      return false;
    }
    return true;
  }

  bool y_moving() {
    if (end.get_y_step() == next.get_y_step()) {
      return false;
    }
    return true;
  }

  bool z_moving() {
    if (end.get_z_step() == next.get_z_step()) {
      return false;
    }
    return true;
  }

  bool prismatic_moving() {
    return x_moving();
  }

  bool revolute_moving() {

    // if angle changes by more than half a degree
    double tol = 5e-1;

    if (abs(end.get_th_deg() - next.get_th_deg()) < tol) {
      return false;
    }

    return true;
  }

};

} // namespace Luke

#endif