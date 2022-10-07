#include "myfunctions.h"

namespace luke
{

/* ----- helper functions ----- */

void print_vec(std::vector<bool> v, std::string name) {
  std::cout << name << ": { ";
  if (v.size() == 0) {
    std::cout << "}\n";
    return;
  }
  for (int i = 0; i < v.size() - 1; i++) {
    std::cout << v[i] << " ";
  }
  std::cout << v[v.size() - 1] << " }\n";
}
void print_vec(std::vector<int> v, std::string name) {
  std::cout << name << ": { ";
  if (v.size() == 0) {
    std::cout << "}\n";
    return;
  }
  for (int i = 0; i < v.size() - 1; i++) {
    std::cout << v[i] << ", ";
  }
  std::cout << v[v.size() - 1] << " }\n";
}
void print_vec(std::vector<mjtNum> v, std::string name) {
  std::cout << name << ": { ";
  if (v.size() == 0) {
    std::cout << "}\n";
    return;
  }
  // cast mjtNum to float32 and show to 3dp
  for (int i = 0; i < v.size() - 1; i++) {
    printf("%.3f, ", float(v[i]));
  }
  printf("%.3f }\n", float(v[v.size() - 1]));
}
void print_vec(std::vector<gfloat> v, std::string name) {
  std::cout << name << ": { ";
  if (v.size() == 0) {
    std::cout << "}\n";
    return;
  }
  // cast mjtNum to float32 and show to 3dp
  for (int i = 0; i < v.size() - 1; i++) {
    printf("%.3f, ", float(v[i]));
  }
  printf("%.3f }\n", float(v[v.size() - 1]));
}
void print_vec(std::vector<QPos> v, std::string name) {
  std::cout << name << ": { ";
  if (v.size() == 0) {
    std::cout << "}\n";
    return;
  }
  // cast mjtNum to float32 and show to 3dp
  for (int i = 0; i < v.size(); i++) {
    printf("\n\txyz = (%.3f, %.3f, %.3f) quat = (%.3f, %.3f, %.3f, %.3f)",
      v[i].x, v[i].y, v[i].z, v[i].qx, v[i].qy, v[i].qz, v[i].qw);
  }
  std::cout << "\n}\n";
}
void print_vec(std::vector<std::string> v, std::string name) {
  std::cout << name << ": { ";
  if (v.size() == 0) {
    std::cout << "}\n";
    return;
  }
  for (int i = 0; i < v.size() - 1; i++) {
    std::cout << v[i] << ", ";
  }
  std::cout << v[v.size() - 1] << " }\n";
}
template <typename T> int sign(T val) {
    return (T(0) < val) - (val < T(0));
}
bool strcmp_w_sub(std::string ref_str, std::string sub_str, int num) {
  /* check if two strings are equal, but one string having a substitued value.
  This value should be indicated by {X}, and will be swapped with integers
  from 1 ... num. */

  if (sub_str.size() < 3) {
    throw std::runtime_error("string compare with substitution failed as input"
      " string has size less than 3, and the substitution value is '{X}'");
  }

  int sub_idx;
  std::string before_sub_str;
  std::string after_sub_str;
  bool found_sub = false;

  // first find the substitution point
  char c1;
  char c2 = sub_str[0];
  char c3 = sub_str[1];

  for (int i = 2; i < sub_str.size(); i++) {

    c1 = c2;
    c2 = c3; 
    c3 = sub_str[i];

    if (c1 == '{' and c2 == 'X' and c3 == '}') {
      before_sub_str = sub_str.substr(0, i - 2);
      if (sub_str.size() == i + 1) {
        after_sub_str = "";
      }
      else {
        after_sub_str = sub_str.substr(i + 1, sub_str.size() - (i + 1));
      }
      found_sub = true;
      break;
    }
  }

  // if the sub string doesn't contain a substitution marker, do normal strcmp
  if (not found_sub) {
    return (ref_str == sub_str);
  }

  // otherwise perform the comparison with substitution
  for (int i = 1; i < num + 1; i++) {

    std::string to_comp = before_sub_str + std::to_string(i) + after_sub_str;

    if (to_comp == ref_str) {
      return true;
    }
  }

  return false;
}

/* ----- Global variables and settings ----- */

// global settings for joints in the model
struct JointSettings {

  /* ----- user input settings ----- */

  // keyframes (poses) defined in the xml files
  std::string initial_keyframe = "initial pose";
  std::string reset_keyframe = "initial pose";

  // joint names, need to be hardcoded in here for gripper and panda
  struct {
    std::vector<std::string> panda = {
      "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
      "panda_joint5", "panda_joint6", "panda_joint7"
    };
    std::vector<std::string> gripper = {
      "finger_1_prismatic_joint", "finger_1_revolute_joint",
      "finger_2_prismatic_joint", "finger_2_revolute_joint",
      "finger_3_prismatic_joint", "finger_3_revolute_joint",
      "palm_prismatic_joint"
    };
    std::vector<std::string> base = {
      "world_to_base"
    };
    std::vector<std::string> finger;                  // runtime depends
  } names;

  // gripper specific info for joints (determined by the name order above)
  struct {
    std::vector<int> prismatic { 0, 2, 4 };
    std::vector<int> revolute { 1, 3, 5 };
    std::vector<int> palm { 6 };
  } gripper;

  // key dimensions and details
  struct {
    double finger_length = 235e-3;
    double finger_thickness = 0.9e-3;
    double finger_width = 28e-3;
    double E = 200e9;
    double I = (finger_width * std::pow(finger_thickness, 3)) / 12.0;
    double EI = E * I;
    bool fixed_first_segment;                         // runtime depends
    double stiffness_c = 0;                           // runtime depends
    double segment_length = 0;                        // runtime depends
    std::vector<luke::gfloat> joint_stiffness;        // runtime depends
  } dim;

  // strain gauge parameters
  struct {
    bool use_armadillo_gauges = true;          // whether to use curve fitting for finger bending
    int order = 3;
    double xpos = 50e-3;
    double cbrt_xpos = std::cbrt(xpos);         // runtime depends
    double xpos_cubed = xpos * xpos * xpos;     // runtime depends
  } gauge;

  // control parameters
  struct {
    bool stepper = true;
    int num_steps = 10;                         // number of stepper motors steps in one chunk
    double pulses_per_s = 2000;                 // stepper motor pulses per second, this sets speed (2000pps = 300rpm)
    Gain kp {100, 40, 1000};                   // proportional gains for gripper xyz motors {x, y, z}
    Gain kd {1, 1, 1};                          // derivative gains for gripper xyz motors {x, y, z}
    double base_kp = 2000;                      // proportional gain for gripper base motions
    double base_kd = 100;                       // derivative gain for gripper base motions

    double time_per_step = 0.0;                 // runtime depends
  } ctrl;

  // simulation details
  static constexpr struct Sim {
    static constexpr int step_tolerance = 3;    // for detecting equilibrium
    static constexpr int target_tolerance = 10; // for target_reached
    static constexpr int n_arr = 3;             // no. of steps saved in arrays
    static constexpr int n_settle = 10;         // no. of steps to be settled
  } sim{};

  // hardcoded stiffnesses based on numerical solving
  struct {

    // VALID FOR: real data 1, 235x28x0.9mm fingers, EI=0.34
    struct {

      std::vector<float> N5 { 12.351, 4.728, 5.877, 5.030, 1.927 };
      std::vector<float> N6 { 15.230, 5.923, 6.318, 7.202, 6.383, 2.287 };
      std::vector<float> N7 { 18.861, 7.121, 7.336, 9.602, 8.184, 6.218, 2.000 };
      std::vector<float> N8 { 20.645, 8.041, 8.389, 9.570, 10.123, 8.606, 6.252, 2.079 };
      std::vector<float> N9 { 24.818, 9.799, 9.907, 9.872, 12.627, 11.329, 9.560, 7.562, 2.863 };
      std::vector<float> N10 { 27.302, 11.378, 9.575, 10.845, 14.214, 15.567, 10.791, 9.340, 5.922, 1.816 };
      std::vector<float> N15 { 37.103, 22.431, 15.466, 12.927, 23.115, 15.260, 22.107, 20.695, 21.631, 14.546, 16.221, 12.876, 10.205, 6.744, 2.480 };
      std::vector<float> N20 { 75.206, 23.422, 27.987, 21.172, 16.599, 40.647, 28.657, 19.737, 27.249, 28.612, 34.024, 36.741, 25.092, 21.021, 21.680, 18.867, 14.520, 10.797, 6.767, 1.830 };
      std::vector<float> N25 { 503.022, 16.672, 124.905, 27.550, 24.489, 17.904, 63.461, 34.390, 17.560, 35.540, 69.003, 91.560, 70.231, 63.729, 62.175, 22.117, 6.076, 44.070, 40.950, 65.208, 51.254, 44.330, 25.115, 14.184, 0.500 };
      
      // N30 did not converge to a low error value
      std::vector<float> N30 { 685.194, 18.996, 206.246, 38.443, 27.892, 34.549, 17.070, 57.971, 96.573, 58.083, 11.051, 35.880, 81.274, 140.579, 146.179, 184.276, 219.552, 260.973, 277.488, 303.149, 328.203, 356.418, 351.385, 345.677, 327.709, 298.091, 256.786, 205.697, 144.913, 74.690 };

      float t5 = 3.105e-3;
      float t6 = 2.430e-3;
      float t7 = 1.935e-3;
      float t8 = 1.710e-3;
      float t9 = 1.395e-3;
      float t10 = 1.215e-3;
      float t15 = 0.720e-3;
      float t20 = 0.405e-3;

      // not finalised with below 1% error
      float t25 = 0.180e-3;
      float t30 = 0.045e-3;

    } finger_235x28x0p9;

    // VALID FOR: theory data, 235x28x0.9mm fingers, EI=0.34
    struct {

      // done
      std::vector<float> N5 { 13.361, 6.045, 5.883, 5.307, 4.121 }; // 236 loops
      std::vector<float> N6 { 15.925, 7.357, 7.464, 7.060, 6.446, 5.249 }; // 194 loops
      std::vector<float> N7 { 18.977, 8.878, 9.134, 8.811, 8.355, 7.699, 6.356 }; // 189 loops
      std::vector<float> N8 { 22.664, 10.604, 10.951, 10.705, 10.327, 9.852, 9.117, 7.529 }; // 178 loops
      std::vector<float> N9 { 24.711, 11.677, 12.228, 11.941, 11.600, 11.261, 10.834, 10.111, 8.435 }; // 184 loops
      std::vector<float> N10 { 26.829, 12.632, 13.368, 13.206, 12.817, 12.476, 12.184, 11.800, 11.090, 9.325 }; // 196 loops
      std::vector<float> N15 { 46.437, 20.986, 21.369, 21.601, 21.722, 21.728, 21.675, 21.568, 21.396, 21.139, 20.769, 20.229, 19.402, 17.978, 14.822 }; // 43 loops
      std::vector<float> N20 { 60.306, 27.145, 27.533, 27.736, 28.059, 28.104, 28.229, 28.323, 28.333, 28.352, 28.306, 28.211, 28.075, 27.852, 27.540, 27.100, 26.451, 25.441, 23.657, 19.592 }; // 53 loops
      std::vector<float> N25 { 75.270, 33.220, 33.528, 33.823, 34.098, 34.352, 34.582, 34.786, 34.962, 35.108, 35.222, 35.266, 35.285, 35.278, 35.237, 35.155, 35.022, 34.826, 34.549, 34.163, 33.620, 32.833, 31.604, 29.420, 24.405 }; // 62 loops
      std::vector<float> N30 { 93.569, 43.683, 44.020, 44.257, 44.539, 44.649, 44.800, 44.892, 44.953, 45.007, 45.001, 44.995, 44.946, 44.879, 44.786, 44.661, 44.517, 44.332, 44.122, 43.870, 43.578, 43.234, 42.824, 42.331, 41.718, 40.929, 39.853, 38.252, 35.511, 29.385 }; // 10 loops

    } theory_235x28x0p9;

  } hardcoded_c;

  /* ----- automatically generated settings ----- */

  // is this part of the model in use
  struct {
    bool panda = false;
    bool gripper = false;
    bool finger = false;
    bool base = false;
  } in_use;

  // how many joints for each part
  struct {
    int panda = 0;
    int gripper = 0;
    int finger = 0;
    int per_finger = 0;
    int base = 0;
  } num;

  // joint body indexes
  struct {
    std::vector<int> panda;
    std::vector<int> gripper;
    std::vector<int> finger;
    std::vector<int> base;
  } idx;

  // qpos joint indexes
  struct {
    std::vector<int> panda;
    std::vector<int> gripper;
    std::vector<int> finger;
    std::vector<int> base;
  } qposadr;
  
  // qvel joint indexes
  struct {
    std::vector<int> panda;
    std::vector<int> gripper;
    std::vector<int> finger;
    std::vector<int> base;
  } qveladr;

  // qpos for each joint
  struct {
    std::vector<mjtNum> panda;
    std::vector<mjtNum> gripper;
    std::vector<mjtNum> finger;
    std::vector<mjtNum> base;
  } qpos;

  // qvel for each joint
  struct {
    std::vector<mjtNum> panda;
    std::vector<mjtNum> gripper;
    std::vector<mjtNum> finger;
    std::vector<mjtNum> base;
  } qvel;

  // qpos pointer for each joint
  struct {
    std::vector<mjtNum*> panda;
    std::vector<mjtNum*> gripper;
    std::vector<mjtNum*> finger;
    std::vector<mjtNum*> base;
  } to_qpos;

  // qvel pointer for each joint
  struct {
    std::vector<mjtNum*> panda;
    std::vector<mjtNum*> gripper;
    std::vector<mjtNum*> finger;
    std::vector<mjtNum*> base;
  } to_qvel;

  // joint weld constraint indexes (for freezing/fixing joints)
  struct {
    std::vector<int> prismatic;
    std::vector<int> revolute;
    std::vector<int> palm;
  } con_idx;

  // segmented finger geom ids for colour changing fingers
  struct {
    std::vector<int> finger1;
    std::vector<int> finger2;
    std::vector<int> finger3;
    std::vector<int> palm;
  } geom_idx;

  // have the joints settled into equilibrium
  struct {
    std::array<std::array<int, 2>, sim.n_arr> finger1_arr {};
    std::array<std::array<int, 2>, sim.n_arr> finger2_arr {};
    std::array<std::array<int, 2>, sim.n_arr> finger3_arr {};
    std::array<int, sim.n_arr> palm_arr {};
    int counter {};
    bool finger1 {};
    bool finger2 {};
    bool finger3 {};
    bool palm {};
    bool all {};
    bool settled {};
    bool target_reached {};
    bool target_step {};
  } settle;

  /* ----- Member functions ----- */

  // printing functions
  void print_idx() {
    print_vec(idx.panda, "panda joint idx");
    print_vec(idx.gripper, "gripper joint idx");
    print_vec(idx.finger, "finger joint idx");
    print_vec(idx.base, "base joint idx");
  }
  void print_in_use() {
    std::cout << "Using: " 
      << "panda = " << (in_use.panda ? "true" : "false")
      << ", gripper = " << (in_use.gripper ? "true" : "false")
      << ", segmented fingers = " << (in_use.finger ? "true" : "false")
      << ", base = " << (in_use.gripper ? "true" : "false")
      << '\n';
  }
  void print_num() {
    std::cout << "Number of joints for: "
      << "panda = " << num.panda
      << ", gripper = " << num.gripper 
      << ", finger = " << num.finger
      << ", per finger = " << num.per_finger 
      << ", base = " << num.base
      << '\n';
  }
  void print_qposadr() {
    print_vec(qposadr.panda, "panda joint qpos addresses");
    print_vec(qposadr.gripper, "gripper joint qpos addresses");
    print_vec(qposadr.finger, "finger joint qpos addresses");
    print_vec(qposadr.base, "base joint qpos addresses");
  }
  void print_qveladr() {
    print_vec(qveladr.panda, "panda joint qvel addresses");
    print_vec(qveladr.gripper, "gripper joint qvel addresses");
    print_vec(qveladr.finger, "finger joint qvel addresses");
    print_vec(qveladr.base, "base joint qvel addresses");
  }
  void print_qpos() {
    std::cout << "Please note, qpos and qvel are no longer used. To see them for "
      "debugging, please run the function update_state() before printing\n";
    print_vec(qpos.panda, "panda joint qpos");
    print_vec(qpos.gripper, "gripper joint qpos");
    print_vec(qpos.finger, "finger joint qpos");
    print_vec(qpos.base, "base joint qpos");
  }
  void print_qvel() {
    std::cout << "Please note, qpos and qvel are no longer used. To see them for "
      "debugging, please run the function update_state() before printing\n";
    print_vec(qvel.panda, "panda joint qvel");
    print_vec(qvel.gripper, "gripper joint qvel");
    print_vec(qvel.finger, "finger joint qvel");
    print_vec(qvel.base, "base joint qvel");
  }
  void print_geom_idx() {
    print_vec(geom_idx.finger1, "finger1 geom idx");
    print_vec(geom_idx.finger2, "finger2 geom idx");
    print_vec(geom_idx.finger3, "finger3 geom idx");
    print_vec(geom_idx.palm, "palm geom idx");
  }
  void print_settled() {
    std::cout << "Settled: " << settle.finger1 << " " << settle.finger2
      << " " << settle.finger3 << " " << settle.palm << " " 
      << settle.all << " " << settle.settled 
      << ", reached: " << settle.target_reached
      << ", step: " << settle.target_step
      << "\n";
  }
  
};

// global joint settings structure
JointSettings j_;

// create object handler to control graspable objects in simulation
ObjectHandler oh_;

Target target_;     // global state target

// gripper finger state 
// these are not currently used at all!
Gripper finger1_;
Gripper finger2_;
Gripper finger3_;

// time of last stepper step
static double last_step_time_ = 0.0;

// make vectors of pointers for properties we will want to loop over
// these are not currently used at all!
std::vector<Gripper*> fingers_ {&finger1_, &finger2_, &finger3_};
std::vector<std::array<std::array<int, 2>, j_.sim.n_arr>*> finger_arrays_ {
  &j_.settle.finger1_arr, &j_.settle.finger2_arr, &j_.settle.finger3_arr
};
std::vector<bool*> finger_settled_ {
  &j_.settle.finger1, &j_.settle.finger2, &j_.settle.finger3
};

constexpr static bool debug = false; // turn on/off debug mode for this file only

/* ----- initialising, setup, and utilities ----- */

void init(mjModel* model, mjData* data)
{
  /* runs once when model is created */

  // extract model information and store it in our global variable j_
  init_J(model, data);

  // set the model to the inital keyframe
  keyframe(model, data, j_.initial_keyframe);

  // initialise the object handler
  oh_.init(model, data);

  // // assign my control function to the mujoco control fcn pointer
  // mjcb_control = control;

}

void init_J(mjModel* model, mjData* data)
{
  /* initialise our global data structure with joint and model information */

  // wipe the global settings structure
  JointSettings empty;
  j_ = empty;

  // use joint names to get body indexes and qpos/qvel addresses
  get_joint_indexes(model);
  get_joint_addresses(model);
  get_geom_indexes(model);

  if (debug) {
    print_joint_names(model);
  }

  // resize state vectors and find qpos/qvel pointers
  configure_qpos(model, data);

  // calculate constants
  j_.ctrl.time_per_step = j_.ctrl.num_steps / j_.ctrl.pulses_per_s;
  int N = j_.num.per_finger;
  int Ntotal = j_.num.per_finger + j_.dim.fixed_first_segment;
  j_.dim.segment_length = j_.dim.finger_length / float(Ntotal);

  if (j_.dim.fixed_first_segment) {
    j_.dim.stiffness_c = ( j_.dim.EI / (2 * j_.dim.finger_length) ) 
      * ( (float)(N * (N*N + 6*N + 11)) / (float)((N + 1) * (N + 1)) );
  }
  else {
    j_.dim.stiffness_c = ( j_.dim.EI / (2 * j_.dim.finger_length) ) 
       * ( (float)((N + 1)*(N + 2)) / N);
  }

  if (debug) {
    std::cout << "Number of finger joints N is " << j_.num.per_finger << '\n';
    std::cout << "Joint stiffness c is " << j_.dim.stiffness_c << '\n';
  }

  configure_constraints(model, data);
}

void reset(mjModel* model, mjData* data)
{
  /* reset the simulation */

  // reset the targets and disable any constraints
  target_.reset();
  set_all_constraints(model, data, false);

  // wipe object positions and reset
  mj_resetData(model, data);
  keyframe(model, data, j_.reset_keyframe);
  reset_object(model, data);

  // recalculate all object positions/forces
  mj_forward(model, data);
  update_all(model, data);

  // briefly override
  j_.settle.settled = true;
  j_.settle.target_reached = true;

  // set the joints to the equilibrium position
  calibrate_reset(model, data);

  // now enable all constraints at equilibrium position
  set_all_constraints(model, data, true);
}

void print_joint_names(mjModel* model)
{
  /* print joint names to the terminal */

  for (int i = 0; i < model->njnt; i++) {
    auto x = mj_id2name(model, mjOBJ_JOINT, i);
    std::cout << "i = " << i << " gives jnt name = " << x << '\n';
  }
}

void get_joint_indexes(mjModel* model)
{
  /* Get the indexes of the different joint groups */

  for (std::string name : j_.names.panda) {
    int idx = mj_name2id(model, mjOBJ_JOINT, name.c_str());
    j_.idx.panda.push_back(idx);
  }
  for (std::string name : j_.names.gripper) {
    int idx = mj_name2id(model, mjOBJ_JOINT, name.c_str());
    j_.idx.gripper.push_back(idx);
  }
  for (std::string name : j_.names.base) {
    int idx = mj_name2id(model, mjOBJ_JOINT, name.c_str());
    j_.idx.base.push_back(idx);
  }

  if (j_.idx.panda[0] != -1) j_.in_use.panda = true;
  else j_.in_use.panda = false;

  if (j_.idx.gripper[0] != -1) j_.in_use.gripper = true;
  else j_.in_use.gripper = false;

  if (j_.idx.base[0] != -1) j_.in_use.base = true;
  else j_.in_use.base = false;

  // determine how many joints are being used for each part
  j_.num.panda = j_.names.panda.size() * j_.in_use.panda;
  j_.num.gripper = j_.names.gripper.size() * j_.in_use.gripper;
  j_.num.base = j_.names.base.size() * j_.in_use.base;

  // count how many segment joints we have
  j_.num.finger = 0;
  j_.dim.fixed_first_segment = true;
  for (int i = 0; i < model->njnt; i++) {
    std::string x = mj_id2name(model, mjOBJ_JOINT, i);
    if (x.substr(0,6) == "finger" and x.substr(9, 13) == "segment_joint") {
      j_.num.finger += 1;
      // if we have a segment_joint_0 then there is not a fixed first joint
      if (x.substr(9, 15) == "segment_joint_0") {
        j_.dim.fixed_first_segment = false;
      }
    }
  }

  // // for testing
  // if (j_.num.finger != 27) throw std::runtime_error("j_.num.finger != 27");

  // hence per finger is this divided by 3
  j_.num.per_finger = j_.num.finger / 3;

  if (j_.num.finger > 0) {

    j_.in_use.finger = true;
    int ffs = (int) j_.dim.fixed_first_segment;

    // add the names of every finger joint to the global vector
    for (int i = 1; i <= 3; i++) {
      for (int k = ffs; k < j_.num.per_finger + ffs; k++) {
        // create the joint name string and add it to the vector
        std::string next = "finger_" + std::to_string(i)
          + "_segment_joint_" + std::to_string(k);
        j_.names.finger.push_back(next);
      }
    }

    // now, we want the indexes corresponding to each joint
    for (std::string name : j_.names.finger) {
      int idx = mj_name2id(model, mjOBJ_JOINT, name.c_str());
      j_.idx.finger.push_back(idx);
    }

    // extra safety check
    if (j_.names.finger.size() == 0 or j_.idx.finger[0] == -1) {
      printf("Error: Finger joints not found\n");
      j_.in_use.finger = false;
      j_.num.finger = 0;
    }
  }

  if (debug) {
    j_.print_in_use();
    j_.print_num();
    j_.print_idx();
    std::cout << "Fixed first segment is: " << j_.dim.fixed_first_segment << '\n';
  }
}

void get_geom_indexes(mjModel* model)
{
  /* get the indexes of the geoms for the fingers */

  // each geom has both a 'collision' and 'visual' version, so we collect both
  std::vector<std::string> geom_suffixes { "collision", "visual" };

  int ffs = j_.dim.fixed_first_segment;

  for (std::string geom_tag : geom_suffixes) {

    for (int i = 0; i < j_.num.finger; i++) {

      std::string geom_name = "finger_" + std::to_string(i / j_.num.per_finger + 1)  // finger_X, X=1,2,3
        + "_segment_link_" + std::to_string(i % j_.num.per_finger + 1 + ffs)         // links go 2-10 for 10 segments
        + "_geom_" + geom_tag;

      int x = mj_name2id(model, mjOBJ_GEOM, geom_name.c_str());

      if (i < j_.num.per_finger) {
        j_.geom_idx.finger1.push_back(x);
      }
      else if (i < 2 * j_.num.per_finger) {
        j_.geom_idx.finger2.push_back(x);
      }
      else if (i < 3 * j_.num.per_finger) {
        j_.geom_idx.finger3.push_back(x);
      }
      else {
        throw std::runtime_error("get_geom_indexes() found inconsistent finger segment numbers");
      }
    }

    // now add the hook links
    std::string f1_hook = "finger_1_segment_link_" + std::to_string(j_.num.per_finger + ffs)
      + "_geom_hook_" + geom_tag;
    std::string f2_hook = "finger_2_segment_link_" + std::to_string(j_.num.per_finger + ffs)
      + "_geom_hook_" + geom_tag;
    std::string f3_hook = "finger_3_segment_link_" + std::to_string(j_.num.per_finger + ffs)
      + "_geom_hook_" + geom_tag;

    j_.geom_idx.finger1.push_back(mj_name2id(model, mjOBJ_GEOM, f1_hook.c_str()));
    j_.geom_idx.finger2.push_back(mj_name2id(model, mjOBJ_GEOM, f2_hook.c_str()));
    j_.geom_idx.finger3.push_back(mj_name2id(model, mjOBJ_GEOM, f3_hook.c_str()));

    // now add the palm link
    std::string palm_geom_name = "palm_geom_" + geom_tag;
    j_.geom_idx.palm.push_back(mj_name2id(model, mjOBJ_GEOM, palm_geom_name.c_str()));
  }

  if (debug) {
    j_.print_geom_idx();
  }
}

void get_joint_addresses(mjModel* model)
{
  /* Get the qpos and qvel addresses for each active joint */

  // old (incorrect) code used: model->jnt_qposadr[model->body_jntadr[idx]]

  if (j_.in_use.panda) {
    for (int idx : j_.idx.panda) {
      j_.qposadr.panda.push_back(model->jnt_qposadr[idx]);
      j_.qveladr.panda.push_back(model->jnt_dofadr[idx]);
    }
  }

  if (j_.in_use.gripper) {
    for (int idx : j_.idx.gripper) {
      j_.qposadr.gripper.push_back(model->jnt_qposadr[idx]);
      j_.qveladr.gripper.push_back(model->jnt_dofadr[idx]);
    }
  }

  if (j_.in_use.finger) {
    for (int idx : j_.idx.finger) {
      j_.qposadr.finger.push_back(model->jnt_qposadr[idx]);
      j_.qveladr.finger.push_back(model->jnt_dofadr[idx]);
    }
  }

  if (j_.in_use.base) {
    for (int idx : j_.idx.base) {
      j_.qposadr.base.push_back(model->jnt_qposadr[idx]);
      j_.qveladr.base.push_back(model->jnt_dofadr[idx]);
    }
  }

  if (debug) {
    j_.print_qposadr();
    j_.print_qveladr();
  }
}

void set_finger_stiffness(mjModel* model, std::vector<luke::gfloat> stiffness)
{
  /* set the finger stiffness to a vector sequence of values */

  constexpr bool local_debug = false; // debug
  
  int N = j_.num.per_finger;

  if (stiffness.size() != N) {
    throw std::runtime_error("wrong number of joint stiffnesses passed to set_finger_stiffness(...)");
  }

  // loop over all three fingers
  for (int i = 0; i < 3; i++) {

    // loop from n=1 to N
    for (int n = 1; n < N + 1; n++) {

      int idx = j_.idx.finger[i * N + (n - 1)];

      if (local_debug and i == 0) 
        std::cout << "Finger joint stiffness joint " << i << " is " << stiffness[n - 1] << '\n';

      if (i == 0) {
        j_.dim.joint_stiffness[n - 1] = stiffness[n - 1];
      }

      model->jnt_stiffness[idx] = stiffness[n - 1];

    }
  }
}

void set_finger_stiffness(mjModel* model, mjtNum stiffness)
{
  /* set the stiffness of the flexible finger joints. The input value for stiffness
  determines the behaviour of this function.
  
  stiffness > 0       -> all joints are set equally to this stiffness value
  stiffness -1 to 0   -> stiffness is calculated using the model and EI
  stiffness -2 to -1  -> stiffness is calculated using model and EI but adjusted with tuned params
  stiffness -3 to -2  -> stiffness is calculated based on FEA curve fit derivation
  stiffness -4 to -3  -> stiffness is calculated from Bisshopp equation for end tip angle, attempt 1
  stiffness -5 to -4  -> stiffness is calculated from Bisshopp equation for end tip angle, attempt 2
  stiffness -6 to -5  -> stiffness is calculated from new attempt at theory, equating angles
  stiffness -7 to -6  -> stiffness is calculated from new attempt at theory, equating angles PLUS calculation of a

  stiffness is -100   -> use hardcoded stiffness values, convergence on real 0.9mm bending @ 300g

  */

  constexpr bool local_debug = false;

  if (stiffness > 0) {
    if (local_debug) std::cout << "Finger joint stiffness ALL set to " << stiffness << '\n';
    for (int i : j_.idx.finger) {
      model->jnt_stiffness[i] = stiffness;
    }
  }
  else {

    int N = j_.num.per_finger;

    j_.dim.joint_stiffness.clear();
    j_.dim.joint_stiffness.resize(N);

    // use the derived model from EI and euler bending
    if (stiffness > -1 and stiffness < 0) {

      if (local_debug) std::cout << "Finger joint stiffness set using EI derivation\n";

      // loop over all three fingers
      for (int i = 0; i < 3; i++) {

        // loop from n=1 to N
        for (int n = 1; n < N + 1; n++) {

          int idx = j_.idx.finger[i * N + (n - 1)];

          // determine the joint stiffness using the model
          float c = (j_.dim.stiffness_c * (N - n + 1)) / (float) n;

          if (i == 0) {
            j_.dim.joint_stiffness[n - 1] = c;
          }

          if (local_debug and i == 0) {
            std::cout << "finger joint " << n << " has c_n = " << c << '\n';
          }

          model->jnt_stiffness[idx] = c;
        }
      }
    }

    // use the euler model but adjust values
    else if (stiffness > -2 and stiffness < -1) {

      if (local_debug) std::cout << "Finger joint stiffness set using EI derivation and THEN adjusted\n";

      // loop over all three fingers
      for (int i = 0; i < 3; i++) {

        // loop from n=1 to N
        for (int n = 1; n < N + 1; n++) {

          int idx = j_.idx.finger[i * N + (n - 1)];

          float c = (j_.dim.stiffness_c * (N - n + 1)) / (float) n;

          // // TESTING: smooth out stiffness - this results in good for 5, okay 10, not that good for 30
          // float max_reduction = 0.55 - (0.015*N);
          // float max_increase = 1.0 + (0.03*N);
          // float this_frac = (float) (n - 1) / (float) (N - 1);
          // float this_change = this_frac * (max_increase - max_reduction) + max_reduction;

          // TESTING: smooth out stiffess symetrrically, this is not working
          int half_way = N / 2.0;
          float this_change;
          float this_frac;
          float max_reduction = 0.40 - (0.015*N); // only non-zero N < 20
          float max_increase = 0.80 + (0.2*N);
          float middle_point = 1.0; //(max_increase + max_reduction) / 2.0;
          if (n <= half_way) {
            this_frac = (float) (n - 1) / (half_way - 1);
            this_change = this_frac * (middle_point - max_reduction) + max_reduction;
          }
          else {
            this_frac = (float) (n - half_way) / (float) (N - half_way);
            this_change = this_frac * (max_increase - middle_point) + middle_point;
          }

          // make adjustment to stiffness
          c *= this_change;

          if (i == 0) {
            j_.dim.joint_stiffness[n - 1] = c;
          }

          if (local_debug and i == 0) {

            // FOR TESTING: echo stiffnesses
            // std::cout << "idx " << idx << " has c_n = " << c << ", -> ";
            // std::cout << "factor 1 " << factor1 << ", factor 2 " << factor2 << ", factor 3 " << factor3 << '\n';
            std::cout << "finger joint " << n << " has c_n = " << c << " which is now " << c * this_change << " (this change is " << this_change << ")" << '\n';
          }

          model->jnt_stiffness[idx] = c;
        }
      }
    }
  
    // use new model based on FEA curve fit
    else if (stiffness > -3 and stiffness < -2) {

      if (local_debug) std::cout << "Finger joint stiffness set using FEA curve fit constants\n";

      // loop over all three fingers
      for (int i = 0; i < 3; i++) {

        // loop from n=1 to N
        for (int n = 1; n < N + 1; n++) {

          int idx = j_.idx.finger[i * N + (n - 1)];

          // TESTING FEA CURVE FIT based on cn equation
          double factor1 = ((N + 1) * (N + 2)) / (float) (6 * N);
          double A = -5.042e-7 * 1e6;
          double B = 3.531e-4 * 1e3;
          double factor2 = ((n * n) / (float) (N * N)) * (((n * j_.dim.finger_length * A) / (float) N) + B);
          double factor3 = (float) (N - n + 1) / (float) n;
          double c = (factor1 / factor2) * factor3;

          // // TESTING FEA CURVE FIT based on c not cn equation
          // double factor1 = (1/12.0) * N * (N + 1) * (N + 2) * (N + 3);
          // double A = -5.042e-7 * 1e6;
          // double B = 3.531e-4 * 1e3;
          // double factor2 = ((n * n) / (float) (N * N)) * (((n * j_.dim.finger_length * A) / (float) N) + B);
          // double factor3 = (N - n + 1) / (float) (n * n + n);
          // double c = (factor1 / factor2) * factor3; 

          // // TESTING FEA CURVE FIT cubic fit constants in angle term
          // double A = -5.042e-7 * 1e6;
          // double B = 3.531e-4 * 1e3;
          // double factor1 = (1/6.0) * N * (N + 1) * (A * (N + 2) + 3 * B);
          // // double factor2 = ((N * N) / (float) (N * N)) * (((N * j_.dim.finger_length * A) / (float) N) + B);
          // double factor2 = ((j_.dim.finger_length * j_.dim.EI) / 3.0);
          // double factor3 = (float) (N - n + 1) / (float) (A * n + B);
          // double c = (factor1 * factor2) * factor3;

          // // test ...
          // c /= (float) (N / 1.667) * n;

          if (i == 0) {
            j_.dim.joint_stiffness[n - 1] = c;
          }

          if (local_debug and i == 0) {

            // FOR TESTING: echo stiffnesses
            std::cout << "finger joint " << n << " has c_n = " << c << ", -> ";
            std::cout << "factor 1 " << factor1 << ", factor 2 " << factor2 << ", factor 3 " << factor3 << '\n';
          }

          model->jnt_stiffness[idx] = c;
        }
      }
    }

    // use the Bisshopp equation, attempt one
    else if (stiffness > -4 and stiffness < -3) {

      if (local_debug) std::cout << "Finger joint stiffness set using Bisshopp equation - attempt 1 FAILED\n";

      // loop over all three fingers
      for (int i = 0; i < 3; i++) {

        float sum_cinv = 0;

        // loop from n=1 to N
        for (int n = 1; n < N + 1; n++) {

          int idx = j_.idx.finger[i * N + (n - 1)];

          float ratio = (float) n / (float) N;
          float fsq = 10;
          float term1 = (std::pow(ratio, 2)) / (2 * j_.dim.EI);
          float term2 = (std::pow(ratio, 6) * fsq * std::pow(j_.dim.finger_length, 4)) / (24 * std::pow(j_.dim.EI, 3));
 
          float cinv = (term1 - term2);
          float cdiff = cinv;// - sum_cinv;
          float c = 1.0 / cdiff;
          // float c = 1.0 / (term1 - term2);

          // // invert, minus, and then invert again
          // float invert_cn = 1.0 / cn; //std::cout << "invert cn is " << invert_cn << '\n';
          // float invert_c = invert_cn - sum_invert_ci;
          // float c = 1.0 / invert_c;; //(1.0 / invert_c); //1.0 / cn;

          if (i == 0) {
            j_.dim.joint_stiffness[n - 1] = c;
          }

          if (local_debug and i == 0) {
            // FOR TESTING: echo stiffnesses
            // std::cout << "idx " << idx << " has c_n = " << c << ", -> ";
            // std::cout << "factor 1 " << factor1 << ", factor 2 " << factor2 << ", factor 3 " << factor3 << '\n';
            std::cout << "finger joint " << n << " has c_n = " << c << ", term1 is " << term1 << " and term2 is " 
              << term2 << ", sum_cinv is " << sum_cinv << '\n';
          }

          // increment the sum
          sum_cinv += cinv;

          model->jnt_stiffness[idx] = c;
        }
      }
    }

    // use the Bisshopp equation, attempt two
    else if (stiffness > -5 and stiffness < -4) {

      if (local_debug) std::cout << "Finger joint stiffness set using Bisshopp equation - attempt 2\n";

      // loop over all three fingers
      for (int i = 0; i < 3; i++) {

        // loop from n=1 to N
        for (int n = 1; n < N + 1; n++) {

          int idx = j_.idx.finger[i * N + (n - 1)];

          float m = n;
          float m1 = n - 1;

          if (m1 < 0) m1 = 0;

          float fsq = 25;

          float diff2 = std::pow(m, 2) - std::pow(m1, 2);
          float diff6 = std::pow(m, 6) - std::pow(m1, 6);

          float frac1 = 1.0 / (2 * N * N * j_.dim.EI);
          float frac2 = (fsq * std::pow(j_.dim.finger_length, 4)) / (24 * std::pow(N, 6) * std::pow(j_.dim.EI, 3));

          float term1 = frac1 * diff2;
          float term2 = frac2 * diff6;

          float c = (1.0) / (term1 - term2);

          // // do we convert the values to the virtual work expression?
          // float convert = (float)(N - n + 1) / (N * j_.dim.finger_length);
          // c *= convert;

          // float n5 = std::pow(n, 5);
          // float n4 = std::pow(n, 4);
          // float n3 = std::pow(n, 3);
          // float n2 = std::pow(n, 2);
          // float poly = 6*n5 - 15*n4 + 20*n3 - 15*n2 + 6*m - 1;
          // float denom1 = N * N * j_.dim.EI;
          // float denom2 = 12 * std::pow(N, 6) * std::pow(j_.dim.EI, 3);

          // float term1 = (2*m - 1) / denom1;
          // float term2 = (poly * fsq * std::pow(j_.dim.finger_length, 4)) / denom2;

          // float c_inv = term1 - term2;
          // float c = 2.0 / c_inv;

          if (i == 0) {
            j_.dim.joint_stiffness[n - 1] = c;
          }

          if (local_debug and i == 0) {
            // FOR TESTING: echo stiffnesses
            // std::cout << "idx " << idx << " has c_n = " << c << ", -> ";
            // std::cout << "factor 1 " << factor1 << ", factor 2 " << factor2 << ", factor 3 " << factor3 << '\n';
            std::cout << "finger joint " << n << " has c_n = " << c << ", term1 is " << term1 << " and term2 is " 
              << term2 << "(" << (term2 / term1) * 100 << " %)" << '\n';

            // std::cout << "convert is " << convert << '\n';
          }

          model->jnt_stiffness[idx] = c;
        }
      }
    }

    else if (stiffness > -6 and stiffness < -5) {

      if (local_debug) std::cout << "Finger joint stiffness set using new basic theory attempt equating angles\n";

      // loop over all three fingers
      for (int i = 0; i < 3; i++) {

        // loop from n=1 to N
        for (int n = 1; n < N + 1; n++) {

          int idx = j_.idx.finger[i * N + (n - 1)];

          // determine the joint stiffness using the model
          float m = n - 0.5;
          float mm1 = m - 1;
          if (mm1 < 0) mm1 = 0;
          float sq_diff = (m * m) - (mm1 * mm1);
          float diff = m - mm1;

          float factor1 = (2 * j_.dim.EI) / j_.dim.finger_length;
          float factor2 = (N - n + 1) / (float) N;

          float factor3 = ((sq_diff) / (float) (N * N)) - ((2 * diff) / (float) N);
          

          float c = (factor1 * factor2) / -factor3;

          if (i == 0) {
            j_.dim.joint_stiffness[n - 1] = c;
          }

          if (local_debug and i == 0) {
            std::cout << "finger joint " << n << " has c_n = " << c 
              << ",f1 = " << factor1 << ", f2 = " << factor2 << ", f3 = " << factor3 << '\n';
          }

          model->jnt_stiffness[idx] = c;
        }
      }
    }

    else if (stiffness > -7 and stiffness < -6) {

      if (local_debug) std::cout << "Finger joint stiffness set using new basic theory attempt equating angles PLUS calculation of a\n";

      // loop over all three fingers
      for (int i = 0; i < 3; i++) {

        // loop from n=1 to N
        for (int n = 1; n < N + 1; n++) {

          int idx = j_.idx.finger[i * N + (n - 1)];

          // determine the correction factor a
          // float L = j_.dim.finger_length;
          // float h = (L / (float) N);
          // float x = h * (n - 1);
          // float A = 3;
          // float B = 6 * (x - L);
          // float C = 6*x*x - 8*L*x + h*x - 3*L*h*h + h*h*h;

          // determine the correction factor b (where b=1-a, so a = 1-b
          float L = j_.dim.finger_length;
          float h = (L / (float) N);
          float x = h * (n - 1);
          float A = 3;
          float B = 6 * (x - L);
          // float C = 6*x*x - 12*L*x + 3*h*x - 3*L*h + h*h;
          float C = 3*h*(L - x) - h*h;

          float Bsq = B * B;
          float fAC = 4 * A * C;

          if (fAC > Bsq) throw std::runtime_error("set_finger_stiffness() has math error on quadratic formula");

          float b = (1 / (2*A)) * (-B + sqrt(Bsq - fAC));

          // invert
          float a = 1 - b;

          // a = 0.5;

          // determine the joint stiffness using the model
          float m = n - a;
          float mm1 = m - 1;
          if (mm1 < 0) mm1 = 0;
          float sq_diff = (m * m) - (mm1 * mm1);
          float diff = m - mm1;

          float factor1 = (2 * j_.dim.EI) / j_.dim.finger_length;
          float factor2 = (N - n + 1) / (float) N;

          float factor3 = ((sq_diff) / (float) (N * N)) - ((2 * diff) / (float) N);
          
          float c = (factor1 * factor2) / -factor3;

          if (i == 0) {
            j_.dim.joint_stiffness[n - 1] = c;
          }

          if (local_debug and i == 0) {
            std::cout << "finger joint " << n << " has c_n = " << c 
              << ",f1 = " << factor1 << ", f2 = " << factor2 << ", f3 = " << factor3 
              << '\n';
          }

          model->jnt_stiffness[idx] = c;
        }
      }
    }

    else if (stiffness > -8 and stiffness < -7) {

      if (local_debug) std::cout << "Finger joint stiffness set using new basic theory attempt equating angles PLUS calculation of a, 2nd try\n";

      // loop over all three fingers
      for (int i = 0; i < 3; i++) {

        // loop from n=1 to N
        for (int n = 1; n < N + 1; n++) {

          int idx = j_.idx.finger[i * N + (n - 1)];

          // // determine the correction factor b (where b=1-a, so a = 1-b
          // float L = j_.dim.finger_length;
          // float X = std::pow(N, 3) * ( ((6*n - 3)/(float)(N*N)) - ((3*n*n - 3*n + 1)/(float)(N*N*N)) );
          // float A = 1;
          // float B = 3 * (2*N - n);
          // float C = n*n - 6*N*n + X;
          // float Bsq = B * B;
          // float fAC = 4 * A * C;
          // if (fAC > Bsq) throw std::runtime_error("set_finger_stiffness() has math error on quadratic formula");
          // float a = (1 / (2*A)) * (-B + sqrt(Bsq - fAC));
          // // determine the joint stiffness using the model
          // float m = n - a;
          // float mm1 = n - 1;
          // if (mm1 < 0) mm1 = 0;
          // float sq_diff = (m * m) - (mm1 * mm1);
          // float diff = m - mm1;

          // float yn_over_c = ((3*n*n)/(float)(N*N)) - ((n*n*n)/(float)(N*N*N));
          // float ynm1_over_c = ((3*(n-1)*(n-1))/(float)(N*N)) - (((n-1)*(n-1)*(n-1))/(float)(N*N*N));
          // float phi_over_c = (3.0 / j_.dim.finger_length) * ( ((2*(n-1))/(float)N) - (((n-1)*(n-1))/(float)(N*N)) );
          // float theta_m_over_c = (N / j_.dim.finger_length) * (yn_over_c - ynm1_over_c) - phi_over_c;

          float L = j_.dim.finger_length;
          float x1 = ((n - 1) / (float) N) * L;
          float x2 = ((n) / (float) N) * L;
          float yn_over_c = (3*L*std::pow(x2, 2) - std::pow(x2, 3));
          float ynm1_over_c = (3*L*std::pow(x1, 2) - std::pow(x1, 3));
          float phi_over_c = (3.0) * (2*L*x1 - std::pow(x1, 2));
          float theta_m_over_c = (N / L) * (yn_over_c - ynm1_over_c) - phi_over_c;

          // float theta_m = 

          // float X1 = (N / 3.0) * ( ((6*n - 3)/(float)(N*N)) - ((3*n*n - 3*n + 1)/(float)(N*N*N)) );
          // float X2 = ((2*(n-1))/(float)N) - (((n-1)*(n-1))/(float)(N*N));

          float factor1 = (6 * j_.dim.EI) / std::pow(j_.dim.finger_length, 2);
          float factor2 = (N - n + 1) / (float) N;
          // float factor3 = ((sq_diff) / (float) (N * N)) - ((2 * diff) / (float) N);
          float factor3 = theta_m_over_c;
          float c = (factor1 * factor2) / factor3;

          if (i == 0) {
            j_.dim.joint_stiffness[n - 1] = c;
          }

          if (local_debug and i == 0) {
            std::cout << "finger joint " << n << " has c_n = " << c 
              << ",f1 = " << factor1 << ", f2 = " << factor2 << ", f3 = " << factor3 
              << '\n';
          }

          model->jnt_stiffness[idx] = c;
        }
      }
    }

    else if (stiffness > -100.5 and stiffness < -99.5) {

      if (local_debug) std::cout << "Finger joint stiffness set using hardcoding for 235x28x0.9mm fingers from real data\n";

      switch (N) {

        case 5:  set_finger_stiffness(model, j_.hardcoded_c.finger_235x28x0p9.N5); break;
        case 6:  set_finger_stiffness(model, j_.hardcoded_c.finger_235x28x0p9.N6); break;
        case 7:  set_finger_stiffness(model, j_.hardcoded_c.finger_235x28x0p9.N7); break;
        case 8:  set_finger_stiffness(model, j_.hardcoded_c.finger_235x28x0p9.N8); break;
        case 9:  set_finger_stiffness(model, j_.hardcoded_c.finger_235x28x0p9.N9); break;
        case 10: set_finger_stiffness(model, j_.hardcoded_c.finger_235x28x0p9.N10); break;
        case 15: set_finger_stiffness(model, j_.hardcoded_c.finger_235x28x0p9.N15); break;
        case 20: set_finger_stiffness(model, j_.hardcoded_c.finger_235x28x0p9.N20); break;
        case 25: set_finger_stiffness(model, j_.hardcoded_c.finger_235x28x0p9.N25); break;
        case 30: set_finger_stiffness(model, j_.hardcoded_c.finger_235x28x0p9.N30); break;

        default:
          std::cout << "N is " << N << '\n';
          throw std::runtime_error("no hardcoded stiffness values for this N");
      }

    }

    else if (stiffness > -101.5 and stiffness < -100.5) {

      if (local_debug) std::cout << "Finger joint stiffness set using hardcoding for 235x28x0.9mm fingers from theory predictions\n";

      switch (N) {

        case 5:  set_finger_stiffness(model, j_.hardcoded_c.theory_235x28x0p9.N5); break;
        case 6:  set_finger_stiffness(model, j_.hardcoded_c.theory_235x28x0p9.N6); break;
        case 7:  set_finger_stiffness(model, j_.hardcoded_c.theory_235x28x0p9.N7); break;
        case 8:  set_finger_stiffness(model, j_.hardcoded_c.theory_235x28x0p9.N8); break;
        case 9:  set_finger_stiffness(model, j_.hardcoded_c.theory_235x28x0p9.N9); break;
        case 10: set_finger_stiffness(model, j_.hardcoded_c.theory_235x28x0p9.N10); break;
        case 15: set_finger_stiffness(model, j_.hardcoded_c.theory_235x28x0p9.N15); break;
        case 20: set_finger_stiffness(model, j_.hardcoded_c.theory_235x28x0p9.N20); break;
        case 25: set_finger_stiffness(model, j_.hardcoded_c.theory_235x28x0p9.N25); break;
        case 30: set_finger_stiffness(model, j_.hardcoded_c.theory_235x28x0p9.N30); break;

        default:
          std::cout << "N is " << N << '\n';
          throw std::runtime_error("no hardcoded stiffness values for this N");
      }

    }
  }

  if (local_debug)
    print_vec(j_.dim.joint_stiffness, "joint stiffness vector");
}

void configure_qpos(mjModel* model, mjData* data)
{
  /* sort out qpos and qvel information */

  // resize state vectors
  j_.qpos.panda.resize(j_.num.panda);
  j_.qpos.gripper.resize(j_.num.gripper);
  j_.qpos.finger.resize(j_.num.finger);
  j_.qpos.base.resize(j_.num.base);

  j_.qvel.panda.resize(j_.num.panda);
  j_.qvel.gripper.resize(j_.num.gripper);
  j_.qvel.finger.resize(j_.num.finger);
  j_.qvel.base.resize(j_.num.base);

  // resize pointer vectors
  j_.to_qpos.panda.resize(j_.num.panda);
  j_.to_qpos.gripper.resize(j_.num.gripper);
  j_.to_qpos.finger.resize(j_.num.finger);
  j_.to_qpos.base.resize(j_.num.base);

  j_.to_qvel.panda.resize(j_.num.panda);
  j_.to_qvel.gripper.resize(j_.num.gripper);
  j_.to_qvel.finger.resize(j_.num.finger);
  j_.to_qvel.base.resize(j_.num.base);

  // insert the pointers
  if (j_.in_use.panda) {
    for (int i = 0; i < j_.num.panda; i++) {
      j_.to_qpos.panda[i] = &data->qpos[j_.qposadr.panda[i]];
      j_.to_qvel.panda[i] = &data->qvel[j_.qveladr.panda[i]];
    }
  }

  if (j_.in_use.gripper) {
    for (int i = 0; i < j_.num.gripper; i++) {
      j_.to_qpos.gripper[i] = &data->qpos[j_.qposadr.gripper[i]];
      j_.to_qvel.gripper[i] = &data->qvel[j_.qveladr.gripper[i]];
    }
  }
  
  if (j_.in_use.finger) {
    for (int i = 0; i < j_.num.finger; i++) {
      j_.to_qpos.finger[i] = &data->qpos[j_.qposadr.finger[i]];
      j_.to_qvel.finger[i] = &data->qvel[j_.qveladr.finger[i]];
    }
  }

  if (j_.in_use.base) {
    for (int i = 0; i < j_.num.base; i++) {
      j_.to_qpos.base[i] = &data->qpos[j_.qposadr.base[i]];
      j_.to_qvel.base[i] = &data->qvel[j_.qveladr.base[i]];
    }
  }
}

void configure_constraints(mjModel* model, mjData* data)
{
  /* configure equality constraints for gripper motors */

  constexpr char pris_b1[] = "gripper_base_link";
  constexpr char pris_b2[] = "finger_{X}_intermediate";
  constexpr char rev_b1[] = "finger_{X}_intermediate";
  constexpr char rev_b2[] = "finger_{X}";
  constexpr char palm_b1[] = "gripper_base_link";
  constexpr char palm_b2[] = "palm";

  for (int i = 0; i < model->neq; i++) {

    std::string name1 = mj_id2name(model, mjOBJ_BODY, model->eq_obj1id[i]);
    std::string name2 = mj_id2name(model, mjOBJ_BODY, model->eq_obj2id[i]);


    if (debug) {
      std::printf("Constraint %d has ids %d and %d, which are bodies %s and %s\n", 
        i, model->eq_obj1id[i], model->eq_obj2id[i], name1.c_str(), name2.c_str());
    }

    // detect if it is a prismatic joint constraint
    if ((strcmp_w_sub(name1, pris_b1, 3) or strcmp_w_sub(name1, pris_b2, 3)) and
        (strcmp_w_sub(name2, pris_b1, 3) or strcmp_w_sub(name2, pris_b2, 3))) {
      j_.con_idx.prismatic.push_back(i);
    }

    // detect if it is a revolute joint constraint
    if ((strcmp_w_sub(name1, rev_b1, 3) or strcmp_w_sub(name1, rev_b2, 3)) and
        (strcmp_w_sub(name2, rev_b1, 3) or strcmp_w_sub(name2, rev_b2, 3))) {
      j_.con_idx.revolute.push_back(i);
    }

    // detect if it is a palm joint constraint
    if ((strcmp_w_sub(name1, palm_b1, 3) or strcmp_w_sub(name1, palm_b2, 3)) and
        (strcmp_w_sub(name2, palm_b1, 3) or strcmp_w_sub(name2, palm_b2, 3))) {
      j_.con_idx.palm.push_back(i);
    }

    // set constraint to default value of false
    model->eq_active[i] = false;
  }

  if (debug) {
    print_vec(j_.con_idx.prismatic, "prismatic joint constraints");
    print_vec(j_.con_idx.revolute, "revolute joint constraints");
    print_vec(j_.con_idx.palm, "palm joint constraints");
  }
}

void set_all_constraints(mjModel* model, mjData* data, bool set_to)
{
  /* reset all constraints to false */

  for (int i : j_.con_idx.prismatic) {
    set_constraint(model, data, i, set_to);
  }
  for (int i : j_.con_idx.revolute) {
    set_constraint(model, data, i, set_to);
  }
  for (int i : j_.con_idx.palm) {
    set_constraint(model, data, i, set_to);
  }
}

void toggle_constraint(mjModel* model, mjData* data, int id)
{
  set_constraint(model, data, id, not model->eq_active[id]);
}

void set_constraint(mjModel* model, mjData* data, int id, bool set_as)
{
  /* toggle a constraint, if active lock the body in place relative to another */

  if (set_as) {

    // prepare and get indexes of position/rotation data
    mjtNum body1_pos[3];
    mjtNum body2_pos[3];
    mjtNum body1_rot[9];
    mjtNum body2_rot[9];
    int con_id = id * mjNEQDATA; // index where we insert constraint data
    int b1_pos_id = model->eq_obj1id[id] * 3;
    int b2_pos_id = model->eq_obj2id[id] * 3;
    int b1_rot_id = model->eq_obj1id[id] * 9;
    int b2_rot_id = model->eq_obj2id[id] * 9;

    // get the global rotation of the two bodies
    for (int i = 0; i < 9; i++) {
      body1_rot[i] = data->xmat[b1_rot_id + i];
      body2_rot[i] = data->xmat[b2_rot_id + i];
    }

    // get the global position of the two bodies
    for (int i = 0; i < 3; i++) {
      body1_pos[i] = data->xpos[b1_pos_id + i];
      body2_pos[i] = data->xpos[b2_pos_id + i];
    }

    // now find the local rotation, R12 = (R01)^T * R02
    mjtNum R12[9];
    mju_mulMatTMat(R12, body1_rot, body2_rot, 3, 3, 3);

    // subract the vectors from each other, then rotate into frame 1 (from 0)
    mjtNum vdiff[3];
    mjtNum vec12[3];
    vdiff[0] = body2_pos[0] - body1_pos[0];
    vdiff[1] = body2_pos[1] - body1_pos[1];
    vdiff[2] = body2_pos[2] - body1_pos[2];
    mju_mulMatVec(vec12, body1_rot, vdiff, 3, 3);

    // convert local rotation into a quaternion
    mjtNum quat12[4];
    mju_mat2Quat(quat12, R12);

    // insert this info into the constraint
    model->eq_data[con_id + 0] = vec12[0];
    model->eq_data[con_id + 1] = vec12[1];
    model->eq_data[con_id + 2] = vec12[2];
    model->eq_data[con_id + 3] = quat12[0];
    model->eq_data[con_id + 4] = quat12[1];
    model->eq_data[con_id + 5] = quat12[2];
    model->eq_data[con_id + 6] = quat12[3];

    // activate the constraint
    set_as = true;

    /* for testing
    std::cout << "Body 1 position is: ";
    for (int i = 0; i < 3; i++) {
      std::cout << body1_pos[i] << ", ";
    }
    std::cout << '\n';
    std::cout << "Body 2 position is: ";
    for (int i = 0; i < 3; i++) {
      std::cout << body2_pos[i] << ", ";
    }
    std::cout << '\n';
    std::cout << "Body 1 rotation is: ";
    mjtNum b1Quat[4];
    mju_mat2Quat(b1Quat, body1_rot);
    for (int i = 0; i < 4; i++) {
      std::cout << b1Quat[i] << ", ";
    }
    std::cout << '\n';
    std::cout << "Body 2 rotation is: ";
    mjtNum b2Quat[4];
    mju_mat2Quat(b2Quat, body2_rot);
    for (int i = 0; i < 4; i++) {
      std::cout << b2Quat[i] << ", ";
    }
    std::cout << '\n';
    std::cout << "Constraint data is: ";
    for (int i = 0; i < 7; i++) {
      std::cout << model->eq_data[con_id + i] << ", ";
    }
    std::cout << "\n\n";
    */
    
  }

  // set the constraint either active or inactive
  model->eq_active[id] = set_as;
}

void target_constraint(mjModel* model, mjData* data, int id, bool set_as, int type)
{
  /* set a constraint to send a motor to a target position */

  /* THIS FUNCTION IS UNFINISHED AND NOT CURRENTLY IN USE */

  if (set_as) {

    // int con_id = id * mjNEQDATA; // index where we insert constraint data

    // prepare and get indexes of position/rotation data
    mjtNum body1_pos[3];
    mjtNum body2_pos[3];
    mjtNum body1_rot[9];
    mjtNum body2_rot[9];
    int con_id = id * mjNEQDATA; // index where we insert constraint data
    int b1_pos_id = model->eq_obj1id[id] * 3;
    int b2_pos_id = model->eq_obj2id[id] * 3;
    int b1_rot_id = model->eq_obj1id[id] * 9;
    int b2_rot_id = model->eq_obj2id[id] * 9;

    // get the global rotation of the two bodies
    for (int i = 0; i < 9; i++) {
      body1_rot[i] = data->xmat[b1_rot_id + i];
      body2_rot[i] = data->xmat[b2_rot_id + i];
    }

    // get the global position of the two bodies
    for (int i = 0; i < 3; i++) {
      body1_pos[i] = data->xpos[b1_pos_id + i];
      body2_pos[i] = data->xpos[b2_pos_id + i];
    }

    // now find the local rotation, R12 = (R01)^T * R02
    mjtNum R12[9];
    mju_mulMatTMat(R12, body1_rot, body2_rot, 3, 3, 3);

    // subract the vectors from each other, then rotate into frame 1 (from 0)
    mjtNum vdiff[3];
    mjtNum vec12[3];
    vdiff[0] = body2_pos[0] - body1_pos[0];
    vdiff[1] = body2_pos[1] - body1_pos[1];
    vdiff[2] = body2_pos[2] - body1_pos[2];
    mju_mulMatVec(vec12, body1_rot, vdiff, 3, 3);

    // convert local rotation into a quaternion
    mjtNum quat12[4];
    mju_mat2Quat(quat12, R12);

    // insert this info into the constraint
    model->eq_data[con_id + 0] = vec12[0];
    model->eq_data[con_id + 1] = vec12[1];
    model->eq_data[con_id + 2] = vec12[2];
    model->eq_data[con_id + 3] = quat12[0];
    model->eq_data[con_id + 4] = quat12[1];
    model->eq_data[con_id + 5] = quat12[2];
    model->eq_data[con_id + 6] = quat12[3];

    // gripper prismatic
    if (type == 0) {
      model->eq_data[con_id + 0] = 0;
      model->eq_data[con_id + 1] = target_.end.x;
      model->eq_data[con_id + 2] = 0;
      model->eq_data[con_id + 3] = 0;
      model->eq_data[con_id + 4] = 0;
      model->eq_data[con_id + 5] = 0;
      model->eq_data[con_id + 6] = 1;
    }

    // // gripper revolute
    // else if (type == 1) {
    //   float axis[3] = { 0, 0, -1 };
    //   float half_angle = 0.5 * target_.end.get_revolute_joint();
    //   model->eq_data[con_id + 0] = 0;
    //   model->eq_data[con_id + 1] = 0;
    //   model->eq_data[con_id + 2] = 0;
    //   model->eq_data[con_id + 3] = axis[0] * std::sin(half_angle);
    //   model->eq_data[con_id + 4] = axis[1] * std::sin(half_angle);
    //   model->eq_data[con_id + 5] = axis[2] * std::sin(half_angle);
    //   model->eq_data[con_id + 6] = std::cos(half_angle);
    // }

    // // gripper palm
    // else if (type == 2) {
    //   model->eq_data[con_id + 0] = target_.end.z;
    //   model->eq_data[con_id + 1] = 0;
    //   model->eq_data[con_id + 2] = 0;
    //   model->eq_data[con_id + 3] = 0;
    //   model->eq_data[con_id + 4] = 0;
    //   model->eq_data[con_id + 5] = 0;
    //   model->eq_data[con_id + 6] = 1;
    // }

    /* for testing */
    std::cout << "Body 1 position is: ";
    for (int i = 0; i < 3; i++) {
      std::cout << body1_pos[i] << ", ";
    }
    std::cout << '\n';
    std::cout << "Body 2 position is: ";
    for (int i = 0; i < 3; i++) {
      std::cout << body2_pos[i] << ", ";
    }
    std::cout << '\n';
    std::cout << "Body 1 rotation is: ";
    mjtNum b1Quat[4];
    mju_mat2Quat(b1Quat, body1_rot);
    for (int i = 0; i < 4; i++) {
      std::cout << b1Quat[i] << ", ";
    }
    std::cout << '\n';
    std::cout << "Body 2 rotation is: ";
    mjtNum b2Quat[4];
    mju_mat2Quat(b2Quat, body2_rot);
    for (int i = 0; i < 4; i++) {
      std::cout << b2Quat[i] << ", ";
    }
    std::cout << '\n';
    std::cout << "Constraint data is: ";
    for (int i = 0; i < 7; i++) {
      std::cout << model->eq_data[con_id + i] << ", ";
    }
    std::cout << "\n\n";
    
  }

  // set the constraint either active or inactive
  model->eq_active[id] = set_as;
}

void keyframe(mjModel* model, mjData* data, std::string keyframe_name)
{
  /* overload with keyframe name */

  // set model to keyframe with the given name
  int key = mj_name2id(model, mjOBJ_KEY, keyframe_name.c_str());

  keyframe(model, data, key);
}

void keyframe(mjModel* model, mjData* data, int key)
{
  /* to run once to initialise the model to a desired state.
  Code is adapted from testspeed.cc line 117 */

  if (key >= 0) {
    data->time = model->key_time[key];
    mju_copy(data->qpos, model->key_qpos + key * model->nq, model->nq);
    mju_copy(data->qvel, model->key_qvel + key * model->nv, model->nv);
    mju_copy(data->act, model->key_act + key * model->na, model->na);
    // mju_copy(d->mocap_pos, m->key_mpos+i*3*m->nmocap, 3*m->nmocap);
		// mju_copy(d->mocap_quat, m->key_mquat+i*4*m->nmocap, 4*m->nmocap);
  }
  else {
    throw std::runtime_error("keyframe does not exist");
  }

  last_step_time_ = model->key_time[key];
}

void calibrate_reset(mjModel* model, mjData* data)
{
  /* find the equilibrium start position and set the simulation to that */

  static bool first_call = true;
  static std::vector<mjtNum> control_signals;
  static std::vector<mjtNum> qpos_positions;

  if (first_call) {

    constexpr int settle_number = 400; // found using mysimulate and visual inspection

    // loop to settle the simulation ~86ms
    for (int i = 0; i < settle_number; i++) {
      before_step(model, data);
      step(model, data);
      after_step(model, data);
    }

    // see where the joints have settled to equilibrium
    for (int i = 0; i < j_.num.panda; i++) {
      qpos_positions.push_back(*j_.to_qpos.panda[i]);
    }

    for (int i = 0; i < j_.num.base; i++) {
      qpos_positions.push_back(*j_.to_qpos.base[i]);
    }

    for (int i = 0; i < j_.num.gripper; i++) {
      qpos_positions.push_back(*j_.to_qpos.gripper[i]);
    }

    first_call = false;
  }

  int k = 0;

  // apply the equilibrium positions to the joints
  for (int i = 0; i < j_.num.panda; i++) { 
    (*j_.to_qpos.panda[i]) = qpos_positions[k]; 
    k += 1;
  }

  for (int i = 0; i < j_.num.base; i++) {
    (*j_.to_qpos.base[i]) = qpos_positions[k]; 
    k += 1;
  }

  for (int i = 0; i < j_.num.gripper; i++) {
    (*j_.to_qpos.gripper[i]) = qpos_positions[k]; 
    k += 1;
  }

}

void apply_tip_force(mjModel* model, mjData* data, double force, bool reset)
{
  /* apply a horizontal force to the tip of the finger */
  
  static bool first_call = true;
  static std::vector<int> tip_idx;
  static mjtNum tip_mat1[9];
  static mjtNum tip_mat2[9];
  static mjtNum tip_mat3[9];
  static std::vector<mjtNum*> tip_mat{ tip_mat1, tip_mat2, tip_mat3 };

  // the first time this function is called, find the tip body indexes
  if (first_call or reset) {

    if (reset) tip_idx.clear();

    // get the name of the last finger link (hook link is removed by mujoco as fixed joint)
    int tip_num = j_.num.per_finger + j_.dim.fixed_first_segment;
    std::string tip_name = "finger_{X}_segment_link_" + std::to_string(tip_num);

    for (int i = 0; i < model->nbody; i++) {
      std::string name = mj_id2name(model, mjOBJ_BODY, i);
      if (strcmp_w_sub(name, tip_name, 3)) {
        tip_idx.push_back(i);
      }
    }

    if (debug) {
      print_vec(tip_idx, "finger body tip_idx");
    }
    
    if (tip_idx.size() != 3) {
      throw std::runtime_error("tip_idx vector in apply_tip_force(...) has size != 3");
    }

    // find the starting body orientation
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 9; j++) {
        tip_mat[i][j] = data->xmat[tip_idx[i] * 9 + j];
      }
    }

    first_call = false;
  }

  // for lock the fingers in place
  for (int i : j_.con_idx.prismatic) {
    set_constraint(model, data, i, true);
  }
  for (int i : j_.con_idx.revolute) {
    set_constraint(model, data, i, true);
  }

  // loop through and apply force to fingertips
  for (int i = 0; i < 3; i++) {

    // prepare to apply force outwards on fingertips
    mjtNum fvec[3] = { 0, 0, -force };
    mjtNum rotfvec[3];

    // rotate into the tip frame to pull directly horizontal
    mju_mulMatVec(rotfvec, tip_mat[i], fvec, 3, 3);

    // apply force in cartesian space (joint space is qfrc_applied)
    data->xfrc_applied[tip_idx[i] * 6 + 0] = rotfvec[0];
    data->xfrc_applied[tip_idx[i] * 6 + 1] = rotfvec[1];
    data->xfrc_applied[tip_idx[i] * 6 + 2] = rotfvec[2];
  }
}

void wipe_settled()
{
  /* wipes the settled and target reached states, to give an action time to
  affect the simulation state */

  j_.settle.settled = false;
  j_.settle.target_reached = false;
  j_.settle.target_step = false;
  j_.settle.counter = 0;
}

/* ----- simulation ----- */

void before_step(mjModel* model, mjData* data)
{
  /* before making a simulation step */

  mju_zero(data->ctrl, model->nu);
  mju_zero(data->qfrc_applied, model->nv);
  mju_zero(data->xfrc_applied, 6 * model->nbody);
}

void step(mjModel* model, mjData* data)
{
  /* make a simulation step */

  // mj_step(model, data);

  mj_step1(model, data);

  /* 
  To make the 'leadscrews' non-backdriveable, we want no forces to be
  transferred from the finger to the finger platform/joints. So we will
  try wiping any forces, and trust that the momentum forces are sufficiently
  small.

  The joints to wipe are either:
    finger_1_revolute_joint (1 + panda)
    finger_2_revolute_joint (12 + panda)
    finger_3_revolute_joint (23 + panda)
  or:
    finger_1_segment_joint_1 (2 + panda)
    finger_2_segment_joint_2 (13 + panda)
    finger_3_segment_joint_3 (24 + panda)

  */   

  static std::vector<int> to_wipe {
    j_.idx.gripper[j_.gripper.prismatic[0]],  // 0
    j_.idx.gripper[j_.gripper.prismatic[1]],  // 11
    j_.idx.gripper[j_.gripper.prismatic[2]],  // 22
    j_.idx.gripper[j_.gripper.revolute[0]],   // 1
    j_.idx.gripper[j_.gripper.revolute[1]],   // 12
    j_.idx.gripper[j_.gripper.revolute[2]],   // 23
  };

  control(model, data);   // since ctrl pntr not assigned

  for (int i = 0; i < to_wipe.size(); i++) {
    // all are (nv * 1), and nv = 34 for gripper, which = njnts
    // data->qfrc_passive[to_wipe[i]] = 0;  // passive force
    // data->efc_vel[to_wipe[i]] = 0;       // velocity in constraint space: J*qvel
    // data->efc_aref[to_wipe[i]] = 0;      // reference pseudo-acceleartion
    // data->qfrc_bias[to_wipe[i]] = 0;     // C(qpos, qvel)
    // data->cvel[to_wipe[i]] = 0;          // com-based velcotiy [3D rot; 3D tran]

    // data->qfrc_unc[to_wipe[i]] = 0;
    // data->qacc_unc[to_wipe[i]] = 0;
  }

  // // for testing, applly known force to the end of the finger
  // data->xfrc_applied[11 * 6 + 1] = 10;

  mj_step2(model, data);
  return;

  mj_fwdActuation(model, data);
  mj_fwdAcceleration(model, data);
  mj_fwdConstraint(model, data);


  std::vector<mjtNum> qfrc;
  // std::cout << "qfrc constraint is ";
  for (int i = 0; i < to_wipe.size(); i++) {

    // // wipe forces arising from constraints (contacts)
    // qfrc.push_back(data->qfrc_constraint[to_wipe[i]]);
    // data->qfrc_constraint[to_wipe[i]] = 0;
    // // std::cout << data->qfrc_constraint[to_wipe[i]] << " ";
  } 
  // std::cout << "\n";

  // int j = 0;
  // for (int i : j_.gripper.prismatic) {
  //   data->ctrl[j_.num.panda + i] += -qfrc[j];
  //   j += 1;
  // }
  // for (int i : j_.gripper.revolute) {
  //   data->ctrl[j_.num.panda + i] += -qfrc[j];
  //   j += 1;
  // }
  // mj_fwdActuation(model, data);
  // mj_fwdAcceleration(model, data);
  // mj_fwdConstraint(model, data);

  mj_sensorAcc(model, data);
  mj_checkAcc(model, data);

  // compare forward and inverse solutions if enabled
  // if( mjENABLED(mjENBL_FWDINV) )
  if (model->opt.enableflags and mjENBL_FWDINV)
      mj_compareFwdInv(model, data);

  mj_Euler(model, data);


}

void after_step(mjModel* model, mjData* data)
{
  /* after making a simulation step */

  // compute the contact forces on all bodies
  mj_rnePostConstraint(model, data);

  update_all(model, data);
}

/* ----- control ----- */

void control(const mjModel* model, mjData* data)
{
  /* Control function for mujoco */

  // disable this warning as added objects contribute nv but not nu
  if (false and model->nu != model->nv) {
    printf("Warning from Luke's control function: "
      "model nu (num ctrl inputs) does not equal nv (num DoF). "
      "nu = %i, nv = %i\n", model->nu, model->nv);
    throw std::runtime_error("nu != nv for your model");
  }

  if (j_.in_use.panda) {
    control_panda(model, data);
  }

  if (j_.in_use.gripper) {
    control_gripper(model, data, target_.next);
  }

  if (j_.in_use.base) {
    control_base(model, data);
  }
}

void control_panda(const mjModel* model, mjData* data)
{
  /* control the panda joints */

  // mju_scl(data->ctrl, data->qvel, -0.1 * 100, model->nv);

}

void control_gripper(const mjModel* model, mjData* data, Gripper& target)
{
  /* control the gripper joints */

  double u = 0;

  double force_lim = 100;

  int n = j_.num.panda + j_.num.base;

  // input the control signals
  for (int i : j_.gripper.prismatic) {
    u = ((*j_.to_qpos.gripper[i]) - target.x) * j_.ctrl.kp.x 
      + (*j_.to_qvel.gripper[i]) * j_.ctrl.kd.x;
    // if (abs(u) > force_lim) {
    //   std::cout << "x frc limited from " << u << " to ";
    //   u = force_lim * sign(u);
    //   std::cout << u << '\n';
    // }
    data->ctrl[n + i] = -u;
  }

  for (int i : j_.gripper.revolute) {
    u = ((*j_.to_qpos.gripper[i]) - target.th) * j_.ctrl.kp.y 
      + (*j_.to_qvel.gripper[i]) * j_.ctrl.kd.y;
    // if (abs(u) > force_lim) {
    //   std::cout << "y frc limited from " << u << " to ";
    //   u = force_lim * sign(u);
    //   std::cout << u << '\n';
    // }
    data->ctrl[n + i] = -u;
  }
  
  for (int i : j_.gripper.palm) {
    u = ((*j_.to_qpos.gripper[i]) - target.z) * j_.ctrl.kp.z 
      + (*j_.to_qvel.gripper[i]) * j_.ctrl.kd.z;
    // if (abs(u) > force_lim) {
    //   std::cout << "z frc limited from " << u << " to ";
    //   u = force_lim * sign(u);
    //   std::cout << u << '\n';
    // }
    data->ctrl[n + i] = -u;
  }
}

void control_base(const mjModel* model, mjData* data)
{
  /* control the base joint */

  double u = 0;
  int n = j_.num.panda;

  if (j_.num.base != 1) {
    throw std::runtime_error("base dof does not equal 1");
  }

  for (int i = 0; i < j_.num.base; i++) {
    u = ((*j_.to_qpos.base[i]) - target_.base[i]) * j_.ctrl.base_kp
      + (*j_.to_qvel.base[i]) * j_.ctrl.base_kd;
    data->ctrl[n + i] = -u;
  }
}

void update_state(const mjModel* model, mjData* data)
{
  /* update our record of the model state */

  /* this function has been replaced by accessing qpos and qvel directly via
  their pointers. It can still be used for helpful printing and debugging
  but should never be called in the main loop */

  if (j_.in_use.panda) {
    for (int i = 0; i < j_.num.panda; i++) {
      j_.qpos.panda[i] = data->qpos[j_.qposadr.panda[i]];
      j_.qvel.panda[i] = data->qvel[j_.qveladr.panda[i]];
    }
  }

  if (j_.in_use.gripper) {
    for (int i = 0; i < j_.num.gripper; i++) {
      j_.qpos.gripper[i] = data->qpos[j_.qposadr.gripper[i]];
      j_.qvel.gripper[i] = data->qvel[j_.qveladr.gripper[i]];
    }
  }

  if (j_.in_use.finger) {
    for (int i = 0; i < j_.num.finger; i++) {
      j_.qpos.finger[i] = data->qpos[j_.qposadr.finger[i]];
      j_.qvel.finger[i] = data->qvel[j_.qveladr.finger[i]];
    }
  }

  if (j_.in_use.base) {
    for (int i = 0; i < j_.num.base; i++) {
      j_.qpos.base[i] = data->qpos[j_.qposadr.base[i]];
      j_.qvel.base[i] = data->qvel[j_.qveladr.base[i]];
    }
  }

  // // report state for testing
  // j_.print_qpos();
  // j_.print_qvel();

}

void print_state(const mjModel* model, mjData* data)
{
  /* print the qpos of all the joints */

  update_state(model, data);
  j_.print_qpos();
  j_.print_qvel();
}

void update_all(mjModel* model, mjData* data)
{
  /* update the state of everything in the simulation */

  // update_state(model, data); // NO LONGER NEEDED
  // update_objects(model, data); // NO LONGER NEEDED

  if (j_.ctrl.stepper) {
    update_stepper(model, data);
  }
  else {
    throw std::runtime_error("non-stepper not implemented");
  }

  // // for testing
  // update_state(model, data);
  // j_.print_qpos();
  // target_.end.print();
}

void update_stepper(mjModel* model, mjData* data)
{
  /* update the gripper joint positions and determine equilibirum/target_reached 
  assuming a stepper motor style */

  constexpr static bool log_test_data = true;

  bool stepped = false;

  if (data->time > last_step_time_ + j_.ctrl.time_per_step) {

    // we can optionally log position data to see motor response to steps
    if (log_test_data) {
      Gripper temp;
      temp.set_xyz_m_rad(*j_.to_qpos.gripper[0], *j_.to_qpos.gripper[1], *j_.to_qpos.gripper[6]);
      target_.timedata.add(data->time);
      target_.target_stepperx.add(target_.next.x * 1e6);
      target_.target_steppery.add(target_.next.y * 1e6);
      target_.target_stepperz.add(target_.next.z * 1e6);
      target_.target_basez.add(target_.base[0] * 1e6);
      target_.actual_stepperx.add(temp.x * 1e6);
      target_.actual_steppery.add(temp.y * 1e6);
      target_.actual_stepperz.add(temp.z * 1e6);
      target_.actual_basez.add(*j_.to_qpos.base[0] * 1e6);
    }

    // check if motors are moving, if not, lock them
    update_constraints(model, data);

    // apply a step to any motors still not at the target
    last_step_time_ = data->time;
    target_.next.step_to(target_.end, j_.ctrl.num_steps);
    stepped = true;

    // uncomment these to see ratio of steps to waits
    // std::cout << "step!\n";
  }
  else {
    // std::cout << "wait-";
  }
}

void update_objects(const mjModel* model, mjData* data)
{
  /* update the position of the objects in the simulation */

  for (int i = 0; i < oh_.names.size(); i++) {
    if (oh_.in_use[i]) {
      oh_.qpos[i].update(model, data, oh_.qposadr[i]);
    }
  }

  // // for testing
  // QPos test = get_object_qpos();
  // printf("qpos is xyz (%.3f, %.3f, %.3f)\n", test.x, test.y, test.z);

  // get_object_contact_forces(model, data);
}

void update_constraints(mjModel* model, mjData* data)
{
  /* control toggling of constraints which log motor positions once they finish
  moving */

  static bool old_x = true;
  static bool old_y = true;
  static bool old_z = true;

  bool new_x = target_.x_moving();
  bool new_y = target_.y_moving();
  bool new_z = target_.z_moving();

  // // FOR TESTING - this work was not finished
  // for (int i : j_.con_idx.prismatic) {
  //   target_constraint(model, data, i, not new_x, 0);
  // }
  // for (int i : j_.con_idx.revolute) {
  //   target_constraint(model, data, i, not new_y, 1);
  // }
  // for (int i : j_.con_idx.palm) {
  //   target_constraint(model, data, i, not new_z, 2);
  // }

  // return;

  if (new_x != old_x) {
    if (new_x) {
      // constraint enable is true
      for (int i : j_.con_idx.prismatic) {
        set_constraint(model, data, i, false);
      }
    }
    else {
      // constraint enable is false
      for (int i : j_.con_idx.prismatic) {
        set_constraint(model, data, i, true);
      }
    }
    old_x = new_x;
  }
  
  if (new_y != old_y) {
    if (new_y) {
      // constraint enable is true
      for (int i : j_.con_idx.revolute) {
        set_constraint(model, data, i, false);
      }
    }
    else {
      // constraint enable is false
      for (int i : j_.con_idx.revolute) {
        set_constraint(model, data, i, true);
      }
    }
    old_y = new_y;
  }

  if (new_z != old_z) {
    if (new_z) {
      // constraint enable is true
      for (int i : j_.con_idx.palm) {
        set_constraint(model, data, i, false);
      }
    }
    else {
      // constraint enable is false
      for (int i : j_.con_idx.palm) {
        set_constraint(model, data, i, true);
      }
    }
    old_z = new_z;
  }

  // // for testing
  // std::cout << "Prismatic constraints: ";
  // for (int i : j_.con_idx.prismatic) {
  //   std::cout << (int) model->eq_active[i] << ", ";
  // }
  // std::cout << "\n";
}

/* ----- monitor simulation ----- */

void check_settling()
{
  /* check if the simulation has settled to steady state */

  /* THIS FUNCTION IS CURRENTLY NOT CALLED AND WILL NOT WORK IF IT IS
  AS finger1_, finger2_, and finger3_ ARE NOT UPDATED EVER */
  throw std::runtime_error("check_settling() function is not expecting to be called");

  static constexpr int n = j_.sim.n_arr;

  // loop through each finger
  for (int i = 0; i < fingers_.size(); i++) {

    (*finger_settled_[i]) = true;

    // first shift the window
    for (int j = 0; j < n - 1; j++) {
      for (int k = 0; k < 2; k++) {
        (*finger_arrays_[i])[j][k] = (*finger_arrays_[i])[j + 1][k];
      }
    }

    // now update with the most recent results
    (*finger_arrays_[i])[n - 1][0] = fingers_[i]->step.x;
    (*finger_arrays_[i])[n - 1][1] = fingers_[i]->step.y;

    // now check if all the values are settled
    for (int j = 0; j < n - 1; j++) {
      for (int k = 0; k < 2; k++) {
        if (abs((*finger_arrays_[i])[j][k] - (*finger_arrays_[i])[n - 1][k])
              > j_.sim.step_tolerance) {
          (*finger_settled_[i]) = false;
        }
      }
    }
  }
  
  // determine if the palm has settled
  j_.settle.palm = true;
  for (int j = 0; j < n - 1; j++) {
    j_.settle.palm_arr[j] = j_.settle.palm_arr[j + 1];
  }
  j_.settle.palm_arr[n - 1] = finger1_.step.z;
  for (int j = 0; j < n - 1; j++) {
    if (abs(j_.settle.palm_arr[j] - j_.settle.palm_arr[n - 1])
        > j_.sim.step_tolerance) {
      j_.settle.palm = false;
    }
  }

  // determine if everything is settled for n_arr steps
  j_.settle.all = j_.settle.finger1 * j_.settle.finger2 * j_.settle.finger3
    * j_.settle.palm;

  // determine if it has been settled for n_settle steps
  j_.settle.counter = (j_.settle.counter + 1) * j_.settle.all;
  if (j_.settle.counter >= j_.sim.n_settle) {
    j_.settle.settled = true;
    j_.settle.counter = j_.sim.n_settle;
  }
  else {
    j_.settle.settled = false;
  }

  // determine if we have reached our target
  if (j_.settle.settled) {
    if (finger1_.is_at(target_.end, j_.sim.target_tolerance) and
        finger2_.is_at(target_.end, j_.sim.target_tolerance) and 
        finger3_.is_at(target_.end, j_.sim.target_tolerance)) {
      j_.settle.target_reached = true;
    }
  }
  else j_.settle.target_reached = false;

  // determine if we have reached the correct step target
  if (target_.next.is_at(target_.end)) {
    j_.settle.target_step = true;
  }
  else j_.settle.target_step = false;

  /* FOR TESTING
  // print the arrays
  for (int i = 0; i < 3; i++) {
    std::cout << "finger " << i << " array: {";
    for (int j = 0; j < n; j++) {
      std::cout << "{ ";
      for (int k = 0; k < 2; k++) {
        std::cout << (*finger_arrays_[i])[j][k] << " ";
      }
      std::cout << "}";
    }
    std::cout << "}\n";
  }
  std::cout << "palm array: { ";
  for (int j = 0; j < n; j++) {
    std::cout << j_.settle.palm_arr[j] << " ";
  }
  std::cout << "}\n";
  */

  // j_.print_settled();
}

bool is_settled()
{
  /* has the gripper fingers and palm stopped moving */

  return j_.settle.settled;
}

bool is_target_reached()
{
  /* have we reached our target - note this currently has steady state error
  issues */

  return j_.settle.target_reached;
}

bool is_target_step()
{
  /* has the gripper target become the actual step target */

  return j_.settle.target_step;
}

/* ----- set gripper target ----- */

bool set_gripper_target_step(int x, int y, int z)
{
  /* set a step target for the gripper motors */

  target_.last_robot = Target::Robot::gripper;

  // return false if target is outside motor limits
  return target_.end.set_xyz_step(x, y, z);
}

bool set_gripper_target_m(double x, double y, double z)
{
  /* set a motor state target for the gripper, returns true when reached */

  target_.last_robot = Target::Robot::gripper;

  // return false if the target is outside motor limits
  return target_.end.set_xyz_m(x, y, z);
}

bool set_gripper_target_m_rad(double x, double th, double z)
{
  /* sets a joint state target for the gripper, returns true when reached */

  target_.last_robot = Target::Robot::gripper;

  // return false if the target is outside motor limits
  return target_.end.set_xyz_m(x, th, z);
}

bool move_gripper_target_step(int x, int y, int z)
{
  /* adjust the gripper target by the indicated number of steps */

  target_.last_robot = Target::Robot::gripper;

  // return false if the target is outside motor limits
  return target_.end.set_xyz_step(target_.end.step.x + x, target_.end.step.y + y, 
    target_.end.step.z + z);
}

bool move_gripper_target_m(double x, double y, double z)
{
  /* adjust gripper target by the indicated distances in metres */

  target_.last_robot = Target::Robot::gripper;

  // return false if the target is outside motor limits
  return target_.end.set_xyz_m(target_.end.x + x, target_.end.y + y,
    target_.end.z + z);
}

bool move_gripper_target_m_rad(double x, double th, double z)
{
  /* adjust the gripper joint values */

  target_.last_robot = Target::Robot::gripper;

  // return false if the target is outside motor limits
  return target_.end.set_xyz_m_rad(target_.end.x + x, target_.end.th + th, 
    target_.end.z + z);
}

bool move_base_target_m(double x, double y, double z)
{
  /* move the base target in x, y, z */

  target_.last_robot = Target::Robot::panda;

  /* only z motion currently implemented */
  target_.base[0] += z;

  // check limits, currently only z movements supported
  double z_min = luke::Target::base_z_min;
  double z_max = luke::Target::base_z_max;

  // check if we have gone outside the limits
  if (target_.base[0] > z_max) {
    target_.base[0] = z_max;
    return false;
  }
  if (target_.base[0] < z_min) {
    target_.base[0] = z_min;
    return false;
  }

  return true;
}

void print_target()
{
  std::cout << "The target gripper state is:";
  target_.end.print();
}

void update_target()
{
  target_.end.update();
}

/* ----- sensing ------ */

gfloat read_armadillo_gauge(const mjData* data, int finger)
{
  /* read the virtual strain gauge for one finger */

  arma::vec joint_values(j_.num.per_finger, arma::fill::zeros);

  // get the joint values for this finger
  for (int i = 0; i < j_.num.per_finger; i++) {
    joint_values(i) = 
      data->qpos[j_.idx.finger[i + finger * j_.num.per_finger]];
  }

  // next convert this into X and Y coordinates
  arma::vec cumulative(j_.num.per_finger, arma::fill::zeros);
  arma::mat finger_xy(j_.num.per_finger + 1, 2, arma::fill::zeros);

  // if first segment is locked
  if (j_.dim.fixed_first_segment)
    finger_xy(0, 0) = j_.dim.segment_length;
  else
    finger_xy(0, 0) = 0;

  for (int i = 0; i < j_.num.per_finger; i++) {

    // keep cumulative total of angular sum
    if (i == 0) {
      cumulative(i) = joint_values(i);
    }
    else {
      cumulative(i) = cumulative(i - 1) + joint_values(i);
    }

    // calculate cartesian coordinates of each joint
    finger_xy(i + 1, 0) = finger_xy(i, 0)
      + j_.dim.segment_length * std::cos(cumulative(i));
    finger_xy(i + 1, 1) = finger_xy(i, 1) 
      + j_.dim.segment_length * std::sin(cumulative(i));
  }

  // polyfit a cubic curve to these joint positions
  arma::vec coeff = arma::polyfit(finger_xy.col(0), finger_xy.col(1), j_.gauge.order);

  // evaluate y at the gauge x position
  gfloat y = 0.0;
  for (int i = 0; i <= j_.gauge.order; i++) {
    y += coeff(i) * std::pow(j_.gauge.xpos, j_.gauge.order - i);
  }

  /* The equation relating the force P to deflection delta is:
        delta = (P * l^3) / (3 * E * I)
     We can approximate the P / 3EI as proportional to our strain, k:
        delta = k * l^3
            k = delta / l^3
     Hence we our approximated strain, k, as our gauge reading
  */

  // calculate the approximated gauge reading
  gfloat k = y / j_.gauge.xpos_cubed;

  // transfer to SI units for force (optional)
  gfloat P = k * (3 * j_.dim.EI);

  /* the SI result is not accurate because the finger stiffness is not
  accurate (here we do not have the right E). However, tuning the stiffness
  to be perfect is not helpful as the simulation can become unstable and the
  interaction with the simulated 'motors' is already not realistic */

  /* Finally, we want to scale this data. To get an idea of the size of k, lets
     take some default values:
      xpos = 50mm
      y_max == xpos -> this is from the finger bending to 45deg
     Hence, an absolute maximum value would be:
      k = xpos / cbrt(xpos)
      k = 0.136
     Lets scale this to the range -100, +100

     Lets scale the data to the range -1, +1. First, what is the maximum
     expected force?

     k = P / 3EI
     E = 200e9 Pa
     I = 1/12 * hb^3 where h = 28mm and b = 0.9mm
     I = 1.71e-12 m^4

     hence, with maximum expected force of 20N, we get:

     k = 20 / (3 * 200 * 1.71) * (10e-12 * 10e-9)
     k = 19.6 m^-2
  */

  k *= (100.0 / 0.136);

  return P;
}

gfloat verify_armadillo_gauge(const mjData* data, int finger,
  std::vector<float>& vec_joint_x, std::vector<float>& vec_joint_y,
  std::vector<float>& vec_coefficients, std::vector<float>& vec_errors)
{
  /* read the virtual strain gauge for one finger */

  arma::vec joint_values(j_.num.per_finger, arma::fill::zeros);

  // get the joint values for this finger
  for (int i = 0; i < j_.num.per_finger; i++) {
    // joint_values(i) = 
    //   data->qpos[j_.idx.finger[i + finger * j_.num.per_finger]];

    joint_values(i) = *j_.to_qpos.finger[i + finger * j_.num.per_finger];
  }

  // next convert this into X and Y coordinates
  arma::vec cumulative(j_.num.per_finger, arma::fill::zeros);
  arma::mat finger_xy(j_.num.per_finger + 1, 2, arma::fill::zeros);

  // if first segment is locked
  if (j_.dim.fixed_first_segment)
    finger_xy(0, 0) = j_.dim.segment_length;
  else
    finger_xy(0, 0) = 0;

  for (int i = 0; i < j_.num.per_finger; i++) {

    // keep cumulative total of angular sum
    if (i == 0) {
      cumulative(i) = joint_values(i);
    }
    else {
      cumulative(i) = cumulative(i - 1) + joint_values(i);
    }

    // calculate cartesian coordinates of each joint
    finger_xy(i + 1, 0) = finger_xy(i, 0)
      + j_.dim.segment_length * std::cos(cumulative(i));
    finger_xy(i + 1, 1) = finger_xy(i, 1) 
      + j_.dim.segment_length * std::sin(cumulative(i));
  }

  // polyfit a cubic curve to these joint positions
  arma::vec coeff = arma::polyfit(finger_xy.col(0), finger_xy.col(1), j_.gauge.order);

  // evaluate y at the gauge x position
  gfloat y = 0.0;
  for (int i = 0; i <= j_.gauge.order; i++) {
    y += coeff(i) * std::pow(j_.gauge.xpos, j_.gauge.order - i);
  }

  /* The equation relating the force P to deflection delta is:
        delta = (P * l^3) / (3 * E * I)
     We can approximate the P / 3EI as proportional to our strain, k:
        delta = k * l^3
            k = delta / l^3
     Hence we our approximated strain, k, as our gauge reading
  */

  // calculate the approximated gauge reading
  gfloat k = y / j_.gauge.xpos_cubed;

  // transfer to SI units for force (optional)
  gfloat P = k * (3 * j_.dim.EI);

  /* the SI result is not accurate because the finger stiffness is not
  accurate (here we do not have the right E). However, tuning the stiffness
  to be perfect is not helpful as the simulation can become unstable and the
  interaction with the simulated 'motors' is already not realistic */

  /* Finally, we want to scale this data. To get an idea of the size of k, lets
     take some default values:
      xpos = 50mm
      y_max == xpos -> this is from the finger bending to 45deg
     Hence, an absolute maximum value would be:
      k = xpos / cbrt(xpos)
      k = 0.136
     Lets scale this to the range -100, +100

     Lets scale the data to the range -1, +1. First, what is the maximum
     expected force?

     k = P / 3EI
     E = 200e9 Pa
     I = 1/12 * hb^3 where h = 28mm and b = 0.9mm
     I = 1.71e-12 m^4

     hence, with maximum expected force of 20N, we get:

     k = 20 / (3 * 200 * 1.71) * (10e-12 * 10e-9)
     k = 19.6 m^-2
  */

  k *= (100.0 / 0.136);

  /* ----- only difference between read/verfiy is as follows ----- */

  int num_points = j_.num.per_finger + 1;
  int num_coeff = j_.gauge.order + 1;

  vec_joint_x.resize(num_points);
  vec_joint_y.resize(num_points);
  vec_errors.resize(num_points);
  vec_coefficients.resize(num_coeff);

  // save the joint angles
  for (int i = 0; i < num_points; i++) {
    vec_joint_x[i] = finger_xy(i, 0);
    vec_joint_y[i] = finger_xy(i, 1);
  }

  // save the cubic coefficients
  for (int i = 0; i < num_coeff; i++) {
    vec_coefficients[i] = coeff(i);
  }

  // evaluate the predicted y position and resulting error
  float cum_error = 0;
  for (int i = 0; i < num_points; i++) {
    float y = 0.0;
    for (int j = 0; j < num_coeff; j++) {
      y += vec_coefficients[j] * std::pow(vec_joint_x[i], j_.gauge.order - j);
    }
    float error = vec_joint_y[i] - y;
    vec_errors[i] = error;
    cum_error += abs(error);
  }

  return cum_error / num_points;

  /* ----- end read/verify differences ----- */
}

gfloat verify_small_angle_model(const mjData* data, int finger,
  std::vector<float>& joint_angles, std::vector<float>& joint_pred,
  std::vector<float>& pred_x, std::vector<float>& pred_y, std::vector<float>& theory_y,
  std::vector<float>& theory_x_curve, std::vector<float>& theory_y_curve,
  float force, float finger_stiffness)
{
  /* evaluate the difference in joint angle between the actual and model
  predicted values */

  // CONVERT FORCE TO GRAM FORCE
  force *= 0.981;

  int ffs =  j_.dim.fixed_first_segment;

  int N = j_.num.per_finger;
  joint_angles.resize(N);
  joint_pred.resize(N);
  pred_x.resize(N + 1);
  pred_y.resize(N + 1);
  theory_y.resize(N + 1 + ffs);
  std::vector<float> theory_x(N + 1 + ffs);

  int theory_N = 50;
  float theory_step = j_.dim.finger_length / (float) theory_N;
  theory_x_curve.resize(theory_N);
  theory_y_curve.resize(theory_N);

  std::vector<float> joint_errors(N);

  float cum_error = 0;
  float cum_pred_angle = 0;

  if (j_.dim.fixed_first_segment) {
    pred_x[0] = j_.dim.segment_length;
    theory_x[1] = j_.dim.segment_length;
    theory_y[1] = (force * std::pow(theory_x[1], 3)) / (3 * j_.dim.EI); 
  }
  else {
    pred_x[0] = 0; 
  }

  pred_y[0] = 0;
  theory_x[0] = 0;
  theory_y[0] = 0;
  theory_x_curve[0] = 0;
  theory_y_curve[0] = 0;

  // get the joint values for this finger
  for (int i = 0; i < N; i++) {

    // determine the joint stiffness of this joint
    int n = i + 1;
    // float c = (j_.dim.stiffness_c * (N - n + 1)) / (float)(n + 1);
    // float c = (j_.dim.stiffness_c * (N - n + 1)) / (float) n;
    float c = j_.dim.joint_stiffness[i];

    // actual joint values
    joint_angles[i] = *j_.to_qpos.finger[i + finger * N];

    // predicted joint values
    // joint_pred[i] = ((force * std::pow(j_.dim.finger_length, 2)) / (c));
    // joint_pred[i] = ((float)(N - n + 1) / (float)(N + 1)) 
    //                     * ((force * j_.dim.finger_length) / (c));

    joint_pred[i] = ((N - n + 1) * force * j_.dim.finger_length) / (N * c);
    // joint_pred[i] = ((N - n + 1) * force * j_.dim.finger_length) / ((N + 1) * c);

    // joint angle error
    joint_errors[i] = joint_angles[i] - joint_pred[i];
    cum_error += abs(joint_errors[i]);

    // predicted xy positions
    cum_pred_angle += joint_pred[i];
    pred_x[i + 1] = pred_x[i] + j_.dim.segment_length * std::cos(cum_pred_angle);
    pred_y[i + 1] = pred_y[i] + j_.dim.segment_length * std::sin(cum_pred_angle);

    // // theory y position
    // theory_x[i + 1 + ffs] = theory_x[i + ffs] + j_.dim.segment_length;
    // theory_y[i + 1 + ffs] = (force * std::pow(theory_x[i + 1 + ffs], 3)) / (3 * j_.dim.EI); 

    // basic theory attempt 2
    double theory_factor = (force * 1) / (6.0 * j_.dim.EI);
    double x = (i + 1) * j_.dim.segment_length;
    theory_x[i + 1 + ffs] = x;
    theory_y[i + 1 + ffs] = theory_factor * (-std::pow(x, 3) + 3 * j_.dim.finger_length * std::pow(x, 2));  
    
  }

  fill_theory_curve(theory_x_curve, theory_y_curve, force, theory_N);

  // // approximate free end tangent angle
  // double B = (force * std::pow(j_.dim.finger_length, 2)) / (j_.dim.EI);
  // double phi_0 = 0.5 * B * (1.0 - (1.0/12.0) * std::pow(B, 2));
  // double gamma = M_PI_2;

  // // factors for basic theory
  // double f1 = -force / 6.0;
  // double f2 = (force * std::pow(j_.dim.finger_length, 2)) / 2.0;
  // double f3 = -(force * std::pow(j_.dim.finger_length, 3)) / 3.0;

  // // create theory curve
  // for (int i = 0; i < theory_N - 1; i++) {

  //   // // proportional to L cubed, basic
  //   // theory_x_curve[i + 1] = theory_x_curve[i] + theory_step;
  //   // theory_y_curve[i + 1] = (force * std::pow(theory_x_curve[i + 1], 3)) / (3 * j_.dim.EI); 

  //   // basic theory attempt 2
  //   // double x = j_.dim.finger_length * (1.0 - (i / (float) (theory_N - 2)));
  //   // theory_x_curve[i + 1] = theory_x_curve[i] + theory_step;
  //   // theory_y_curve[i + 1] = (-1.0 / j_.dim.EI) * ((f1 * std::pow(x, 3)) + (f2 * x) + f3); 

  //   double theory_factor = (force / (6.0 * j_.dim.EI));
  //   double x = j_.dim.finger_length * ((i / (float) (theory_N - 2)));
  //   theory_x_curve[i + 1] = x;
  //   theory_y_curve[i + 1] = theory_factor * (-std::pow(x, 3) + 3 * j_.dim.finger_length * std::pow(x, 2));  

  //   // following Bisshopp end angle approximation ... how to get cartesian?

  //   // // // Batista paper, analytical solution
  //   // #ifndef LUKE_PREVENT_BOOST
  //   //   double s = 1.0 - (i / (float) (theory_N - 2));
  //   //   double M0 = 0.0;
  //   //   double alpha = 1.8785;
  //   //   luke_boost::ArcPoint p = luke_boost::get_point(s, force, M0, j_.dim.finger_length,
  //   //     j_.dim.EI, alpha);
  //   //   theory_x_curve[i + 1] = p.x;
  //   //   theory_y_curve[i + 1] = p.y;
  //   // #endif
  // }

  // return average error
  return (gfloat) cum_error / N;
}

void fill_theory_curve(std::vector<float>& theory_X, std::vector<float>& theory_Y, 
  float force, int num)
{
  /* take two vectors (which are wiped) and fill them with the theory curve, this
  is basic bending theory for Euler-Bernoulli beam. Force should be given in NEWTONS */

  theory_X.clear();
  theory_Y.clear();

  theory_X.resize(num);
  theory_Y.resize(num);

  // factors for basic theory
  double f1 = -force / 6.0;
  double f2 = (force * std::pow(j_.dim.finger_length, 2)) / 2.0;
  double f3 = -(force * std::pow(j_.dim.finger_length, 3)) / 3.0;

  // create theory curve
  for (int i = 0; i < num; i++) {

    double theory_factor = (force / (6.0 * j_.dim.EI));
    double x = j_.dim.finger_length * ((i / (float) (num - 2)));
    theory_X[i] = x;
    theory_Y[i] = theory_factor * (-std::pow(x, 3) + 3 * j_.dim.finger_length * std::pow(x, 2)); 
    // theory_Y[i] = theory_factor * (-std::pow(x, 3) + 3 * std::pow(j_.dim.finger_length, 2) * std::pow(x, 1));  
  }
}

std::vector<gfloat> get_gauge_data(const mjModel* model, mjData* data)
{
  /* Get the position of the finger joints */

  if (not j_.in_use.finger) {
    printf("Error: gauge data has been request without using segments\n");
    return std::vector<gfloat>{0, 0, 0};
  }

  std::vector<gfloat> readings(3);

  // use armadillo to detect finger bending
  if (j_.gauge.use_armadillo_gauges) {
    for (int i = 0; i < 3; i++) {
      readings[i] = read_armadillo_gauge(data, i);
    }
  }
  // use fingertip forces
  else {
    Forces forces = get_object_forces(model, data);
    readings[0] = (gfloat) forces.all.finger1_local[1];
    readings[1] = (gfloat) forces.all.finger2_local[1];
    readings[2] = (gfloat) forces.all.finger3_local[1];
  }

  return readings;  
}

gfloat get_palm_force(const mjModel* model, mjData* data)
{
  /* get the axial force on the palm */

  return (gfloat) oh_.get_palm_force(model, data);
}

std::vector<gfloat> get_panda_state(const mjData* data)
{
  /* Get the state of the panda joints */

  if (not j_.in_use.panda) return std::vector<gfloat>{ 0 };

  std::vector<gfloat> joint_values(j_.num.panda);

  for (int i = 0; i < j_.num.panda; i++) {
    joint_values[i] = data->qpos[j_.idx.panda[i]];
  }

  return joint_values;
}

std::vector<gfloat> get_gripper_state(const mjData* data)
{
  /* Get the state of the gripper joints */

  if (not j_.in_use.gripper) return std::vector<gfloat>{ 0 };

  std::vector<gfloat> joint_values(j_.num.gripper);

  for (int i = 0; i < j_.num.gripper; i++) {
    joint_values[i] = data->qpos[j_.idx.gripper[i]];
  }

  return joint_values;
}

std::vector<gfloat> get_target_state()
{
  /* Get the state of the gripper target */

  return target_.get_target_m();

  // // old code
  // // target_.end.update();
  // gfloat x = target_.end.x;
  // gfloat y = target_.end.y; // or theta?
  // gfloat z = target_.end.z;

  // std::vector<gfloat> target_joint_values = { x, y, z };

  // return target_joint_values;
}

int get_N() 
{
  return j_.num.per_finger + j_.dim.fixed_first_segment;
}

std::vector<luke::gfloat> get_stiffnesses()
{
  return j_.dim.joint_stiffness;
}

/* ----- environment ----- */

Gripper get_gripper_target()
{
  /* get the target state of the gripper */

  return target_.end;
}

std::vector<std::string> get_objects()
{
  /* get the names of objects in the simulation scene */

  std::vector<std::string> objects;

  for (int i = 0; i < oh_.names.size(); i++) {
    if (oh_.in_use[i]) {
      objects.push_back(oh_.names[i]);
    }
  }
  
  return objects;
}

void reset_object(mjModel* model, mjData* data)
{
  /* reset the live object to its starting position outside the task area */

  oh_.reset_live(model, data);
}

void spawn_object(mjModel* model, mjData* data, std::string name, QPos pose)
{
  /* overload to pass object name not index */

  for (int i = 0; i < oh_.names.size(); i++) {
    if (oh_.names[i] == name and oh_.in_use[i]) {
      spawn_object(model, data, i, pose);
      return;
    }
  }

  throw std::runtime_error("name not found");
}

void spawn_object(mjModel* model, mjData* data, int idx, QPos pose)
{
  /* spawn an object in the simulation with the given pose, and always wipes qvel */

  oh_.spawn_object(model, data, idx, pose);
}

QPos get_object_qpos(mjModel* model, mjData* data)
{
  /* returns the position of the live object in the simulation */

  if (oh_.live_object == -1) {
    QPos empty;
    return empty;
    throw std::runtime_error("no live object");
  }

  if (oh_.live_object >= oh_.names.size())
    throw std::runtime_error("live object exceeds number of named objects");

  // // for testing
  // QPos test = get_object_qpos();
  // printf("qpos is xyz (%.3f, %.3f, %.3f)\n", test.x, test.y, test.z);

  // old, when qpos was updated
  // return oh_.qpos[oh_.live_object];

  return oh_.get_live_qpos(model, data);
}

Forces get_object_forces(const mjModel* model, mjData* data)
{
  /* get the contact forces on the live object */

  // use the faster version of the extract_forces() function
  return oh_.extract_forces(model, data);
}

Forces_faster get_object_forces_faster(const mjModel* model, mjData* data)
{
  /* get the contact forces on the live object */

  // use the faster version of the extract_forces() function
  return oh_.extract_forces_faster(model, data);
}

void set_object_colour(mjModel* model, std::vector<float> rgba)
{
  /* set the colour of the main object */

  oh_.set_colour(model, rgba);
}

void set_ground_colour(mjModel* model, std::vector<float> rgba)
{
  /* randomise the colour of the ground */

  oh_.set_ground_colour(model, rgba);
}

void randomise_all_colours(mjModel* model, std::shared_ptr<std::default_random_engine> generator)
{
  /* randomise the colour of every object but not the ground*/

  oh_.randomise_all_colours(model, generator);
}

void default_colours(mjModel* model)
{
  /* restore colours to default values */

  oh_.default_colours(model);

  std::vector<float> rgba_default { 0.5, 0.5, 0.5, 1.0 };
  set_finger_colour(model, rgba_default, 1);
  set_finger_colour(model, rgba_default, 2);
  set_finger_colour(model, rgba_default, 3);
  set_finger_colour(model, rgba_default, 4); // 4 means palm
}

void set_finger_colour(mjModel* model, std::vector<float> rgba, int finger_num)
{
  /* set the segmented finger all to one colour, finger_num = 1,2,3, or 4 (4 means palm) */

  if (rgba.size() != 3 and rgba.size() != 4) {
    throw std::runtime_error("set_finger_colour() not given a rgba vector of size 3 or 4");
  }

  // make a pointer to a vector so we can flexibly swap between different fingers
  std::vector<int>* fptr;

  // assign this pointer to one of the following options
  if (finger_num == 1) fptr = &j_.geom_idx.finger1;
  else if (finger_num == 2) fptr = &j_.geom_idx.finger2;
  else if (finger_num == 3) fptr = &j_.geom_idx.finger3;
  else if (finger_num == 4) fptr = &j_.geom_idx.palm;
  else {
    throw std::runtime_error("set_finger_colour() expects finger_num equal to either 1,2,3 or 4 (4 is palm)");
  }

  // loop through the vector we assigned and update the colour
  for (int i : *fptr) {
    model->geom_rgba[i * 4 + 0] = rgba[0];
    model->geom_rgba[i * 4 + 1] = rgba[1];
    model->geom_rgba[i * 4 + 2] = rgba[2];

    // if an a value is given, set this too
    if (rgba.size() == 4)
      model->geom_rgba[i * 4 + 3] = rgba[3];
  }
}

/* ----- misc ----- */

int last_action_robot()
{
  /* which robot was the last robot used for a change in target.
        0 = none,
        1 = gripper,
        2 = panda
  */

  return target_.last_robot;
}

bool is_sim_unstable(mjModel* model, mjData* data)
{
  /* detect if the simulation has become unstable */

  if (data->warning[mjWARN_BADQACC].number > 0) {
    return true;
  }

  return false;
}

void print_stiffnesses()
{
  print_vec(get_stiffnesses(), "Joint stiffnesses");
}

} // namespace luke