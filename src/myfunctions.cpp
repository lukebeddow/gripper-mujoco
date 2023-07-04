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

/* ----- global variables and settings ----- */

// global settings for joints in the model
struct JointSettings {

  /* ----- user input settings ----- */

  // keyframes (poses) defined in the xml files
  std::string initial_keyframe = "initial pose";
  std::string reset_keyframe = "initial pose";

  // joint names, need to be hardcoded in here for gripper and panda
  struct Names {
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
    std::vector<std::string> base_z = {
      "world_to_base"
    };
    std::vector<std::string> base_xyz = {
      "base_X_joint", "base_Y_joint", "base_Z_joint"
    };
    std::vector<std::string> finger;                  // runtime depends
    std::vector<std::string> hook;                    // runtime depends
  } names;

  // gripper specific info for joints (determined by the name order above)
  struct {
    std::vector<int> prismatic { 0, 2, 4 };
    std::vector<int> revolute { 1, 3, 5 };
    std::vector<int> palm { 6 };
  } gripper;

  // key dimensions and details
  struct Dim {
    
    double finger_length = 235e-3;
    double finger_thickness = 0.9e-3;
    double finger_width = 28e-3;
    double E = 200e9;
    double I = (finger_width * std::pow(finger_thickness, 3)) / 12.0;
    double EI = E * I;
    double yield_stress = 215e6;
    double gripper_distance_above_ground = 10e-3;
    bool fixed_first_segment;                         // runtime depends
    bool fixed_hook_segment;                          // runtime depends
    double stiffness_c = 0;                           // runtime depends
    double segment_length = 0;                        // runtime depends
    std::vector<luke::gfloat> joint_stiffness;        // runtime depends

    void update_EI() {
      I = (finger_width * std::pow(finger_thickness, 3)) / 12.0;
      EI = E * I;
    }

    void reset() {
      joint_stiffness.clear();
      update_EI();
    }

  } dim;

  // base operating limits
  struct BaseLims {

    double x_min { -50e-3 };
    double x_max { 50e-3 };
    double y_min { -50e-3 };
    double y_max { 50e-3 };
    double z_min { -30e-3 };
    double z_max { 30e-3 };
    
    double roll_min { -0.5 };
    double roll_max { 0.5 };
    double pitch_min { -0.5 };
    double pitch_max { 0.5 };
    double yaw_min { -M_PI / 2 };
    double yaw_max { M_PI / 2 };

  } baseLims;

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
    Gain gripper_kp {100, 40, 1000};            // proportional gains for gripper xyz motors {x, y, z}
    Gain gripper_kd {1, 1, 1};                  // derivative gains for gripper xyz motors {x, y, z}
    Gain base_xyz_kp {500, 500, 2000};          // proportional gains for base xyz motions {x, y, z}
    Gain base_xyz_kd {80, 80, 100};             // derivative gains for base xyz motions {x, y, z}

    double time_per_step = 0.0;                 // runtime depends
  } ctrl;

  /* ----- automatically generated settings ----- */

  // is this part of the model in use
  struct InUse {
    bool panda = false;
    bool gripper = false;
    bool finger = false;
    bool base_z = false;
    bool base_xyz = false;
  } in_use;

  // how many joints for each part
  struct JointNum {
    int panda = 0;
    int gripper = 0;
    int finger = 0;
    int per_finger = 0;
    int base = 0;
  } num;

  VectorStruct<int> idx;
  VectorStruct<int> qposadr;
  VectorStruct<int> qveladr;
  VectorStruct<mjtNum> qpos;
  VectorStruct<mjtNum> qvel;
  VectorStruct<mjtNum*> to_qpos;
  VectorStruct<mjtNum*> to_qvel;

  // joint weld constraint indexes (for freezing/fixing joints)
  struct ConIdx {
    std::vector<int> prismatic;
    std::vector<int> revolute;
    std::vector<int> palm;
  } con_idx;

  // segmented finger geom ids for colour changing fingers
  struct GeomIdx {
    std::vector<int> finger1;
    std::vector<int> finger2;
    std::vector<int> finger3;
    std::vector<int> palm;
    std::vector<int> main_body;
  } geom_idx;

  // segement matrix (3x3) orientations
  struct SegmentMatrices {
    mjtNum finger1[9];
    mjtNum finger2[9];
    mjtNum finger3[9];
    // std::vector<mjtNum*> fingers { finger1, finger2, finger3 };
    int idx_size;
    std::vector<int> f1_idx;
    std::vector<int> f2_idx;
    std::vector<int> f3_idx;
    std::vector<int> apply_flags;
    std::vector<float> force;
    std::vector<float> moment;
  } segmentMatrices;

  /* ----- Member functions ----- */

  // only resets the automatically generated settings
  void reset() {

    // special case, reset joint stiffness vector
    dim.reset();

    // reset the VectorStructs
    idx.reset();
    qposadr.reset();
    qveladr.reset();
    qpos.reset();
    qvel.reset();
    to_qpos.reset();
    to_qvel.reset();

    // reset other custom structs
    Names names_reset;
    names = names_reset;

    ConIdx con_idx_reset;
    con_idx = con_idx_reset;

    GeomIdx geom_idx_reset;
    geom_idx = geom_idx_reset;

    InUse in_use_reset;
    in_use = in_use_reset;

    JointNum joint_num_reset;
    num = joint_num_reset;

    SegmentMatrices seg_mat_reset;
    segmentMatrices = seg_mat_reset;
  }

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
      << ", base z = " << (in_use.base_z ? "true" : "false")
      << ", base xyz = " << (in_use.base_xyz ? "true" : "false")
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
    print_vec(geom_idx.main_body, "main body geom idx");
  }
  
};

// global joint settings structure
JointSettings j_;

// create object handler to control graspable objects in simulation
ObjectHandler oh_;

// global state target for gripper joints
Target target_;

// time of last stepper step
static double last_step_time_ = 0.0;

// turn on/off debug mode for this file only
constexpr static bool debug_ = true; 

/* ----- initialising, setup, and utilities ----- */

void init(mjModel* model, mjData* data)
{
  /* runs once when model is created */

  last_step_time_ = 0.0;

  // set the model to the inital keyframe
  keyframe(model, data, j_.initial_keyframe);

  // calculate all object positions/forces
  mj_forward(model, data);

  // extract model information and store it in our global variable j_
  init_J(model, data);

  // initialise the object handler
  oh_.init(model, data);

  // // assign my control function to the mujoco control fcn pointer
  // mjcb_control = control;

  // update the base joint limits
  update_base_limits();
}

void init_J(mjModel* model, mjData* data)
{
  /* initialise our global data structure with joint and model information */

  // wipe the global settings structure
  j_.reset();

  // use joint names to get body indexes and qpos/qvel addresses
  get_joint_indexes(model);
  get_joint_addresses(model);
  get_geom_indexes(model);
  get_segment_matrices(model, data);

  if (debug_) {
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

  if (debug_) {
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
  wipe_segment_forces();
  set_all_constraints(model, data, false);

  // wipe object positions and reset
  mj_resetData(model, data);
  keyframe(model, data, j_.reset_keyframe);
  reset_object(model, data);

  // recalculate all object positions/forces
  mj_forward(model, data);
  update_all(model, data);

  // update the base joint limits
  update_base_limits();

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
  for (std::string name : j_.names.base_z) {
    int idx = mj_name2id(model, mjOBJ_JOINT, name.c_str());
    if (idx == -1) continue; // skip in case this joint does not exist
    j_.idx.base.push_back(idx);
  }
  for (std::string name : j_.names.base_xyz) {
    int idx = mj_name2id(model, mjOBJ_JOINT, name.c_str());
    if (idx == -1) continue; // skip in case this joint does not exist
    j_.idx.base.push_back(idx);
  }

  if (j_.idx.panda[0] != -1) j_.in_use.panda = true;
  else j_.in_use.panda = false;

  if (j_.idx.gripper[0] != -1) j_.in_use.gripper = true;
  else j_.in_use.gripper = false;

  // determine what number of joints are used in the base
  if (j_.idx.base[0] != -1) {
    j_.num.base = j_.idx.base.size();
    if (j_.num.base == 1) {
      j_.in_use.base_z = true;
    }
    else if (j_.num.base == 3) {
      j_.in_use.base_xyz = true;
    }
    else {
      throw std::runtime_error("base joints number does not equal 1 or 3");
    }
  }
  else {
    j_.in_use.base_z = false;
    j_.in_use.base_xyz = false;
  }

  // determine how many joints are being used for the other parts
  j_.num.panda = j_.names.panda.size() * j_.in_use.panda;
  j_.num.gripper = j_.names.gripper.size() * j_.in_use.gripper;

  // count how many segment joints we have
  j_.num.finger = 0;
  j_.dim.fixed_first_segment = true;
  j_.dim.fixed_hook_segment = true;
  for (int i = 0; i < model->njnt; i++) {
    std::string x = mj_id2name(model, mjOBJ_JOINT, i);
    if (x.substr(0,6) == "finger" and x.substr(9, 13) == "segment_joint") {
      j_.num.finger += 1;
      // if we have a segment_joint_0 then there is not a fixed first joint
      if (x.substr(9, 15) == "segment_joint_0") {
        j_.dim.fixed_first_segment = false;
      }
    }
    // if we have a hook_joint then there is no fixed hook joint
    if (x.substr(9, 17) == "finger_hook_joint") {
      j_.dim.fixed_hook_segment = false;
    }
  }

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

  if (debug_) {
    j_.print_in_use();
    j_.print_num();
    j_.print_idx();
    std::cout << "Fixed first segment is: " << j_.dim.fixed_first_segment << '\n';
    std::cout << "Fixed hook segment is: " << j_.dim.fixed_hook_segment << '\n';
  }
}

void get_geom_indexes(mjModel* model)
{
  /* get the indexes of the geoms for the fingers */

  bool debug_geoms = false;

  // do we have an object set where all the main body geoms are named
  bool main_body_geoms_are_named = false;
  auto x = mj_id2name(model, mjOBJ_GEOM, 0); // see if geom 0 is named
  if (x != NULL) main_body_geoms_are_named = true;

  std::vector<std::string> main_body_names {
    "gripper_base_link_geom",
    "finger_1_intermediate_geom",
    "finger_1_geom",
    "finger_2_intermediate_geom",
    "finger_2_geom",
    "finger_3_intermediate_geom",
    "finger_3_geom",
  };

  // each geom has both a 'collision' and 'visual' version, so we collect both
  std::vector<std::string> geom_suffixes { "collision", "visual" };

  int ffs = j_.dim.fixed_first_segment;

  for (std::string geom_tag : geom_suffixes) {

    // loop over every single geom looking for the main body
    if (main_body_geoms_are_named) {
      for (int i = 0; i < model->ngeom; i++) {
        std::string x = mj_id2name(model, mjOBJ_GEOM, i);
        if (debug_geoms) std::cout << "Geom " << i << " has name " << x << '\n';
        for (int j = 0; j < main_body_names.size(); j++) {
          if (main_body_names[j] + "_" + geom_tag == x) {
            j_.geom_idx.main_body.push_back(i);
          }
        }
      }
    }

    // now search specifically for finger segment geoms
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

  if (debug_) {
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

  if (j_.in_use.base_z or j_.in_use.base_xyz) {
    for (int idx : j_.idx.base) {
      j_.qposadr.base.push_back(model->jnt_qposadr[idx]);
      j_.qveladr.base.push_back(model->jnt_dofadr[idx]);
    }
  }

  if (debug_) {
    j_.print_qposadr();
    j_.print_qveladr();
  }
}

bool change_finger_thickness(float thickness)
{
  /* set a new finger thickness, and correspondingly change EI, requires reset after 
  to set new finger stiffnesses */

  constexpr bool local_debug = debug_;

  if (local_debug) {
    std::cout << "About to change finger thickness from " << j_.dim.finger_thickness
      << " to " << thickness << ", EI is " << j_.dim.EI << '\n';
  }

  // check if thickness is greater than 5mm
  if (thickness > 2e-3) {
    std::cout << "thickness given = " << thickness << '\n';
    throw std::runtime_error("change_finger_thickness() got value above 2mm - make sure you are using SI units!");
  }

  constexpr float tol = 1e-5;

  if (abs(thickness - j_.dim.finger_thickness) < tol) {
    if (local_debug) std::cout << "Finger thickness is the same as current, not changing\n";
    return false;
  }

  j_.dim.finger_thickness = thickness;
  j_.dim.I = (j_.dim.finger_width * std::pow(j_.dim.finger_thickness, 3)) / 12.0;
  j_.dim.EI = j_.dim.E * j_.dim.I;

  if (local_debug) {
    std::cout << "Finger thickness changed, now is " << j_.dim.finger_thickness
      << ", EI is " << j_.dim.EI << '\n';
  }

  return true;
}

bool change_finger_width(float width)
{
  /* set a new finger wdith, and correspondingly change EI, no reset required, but
  the loaded urdf should be changed if the width changes which would require a hard_reset */

  constexpr bool local_debug = debug_;

  if (local_debug) {
    std::cout << "About to change finger width from " << j_.dim.finger_width
      << " to " << width << ", EI is " << j_.dim.EI << '\n';
  }

  // check if thickness is greater than 1 metre
  if (width > 1) {
    std::cout << "width given = " << width << '\n';
    throw std::runtime_error("change_finger_width() got value above 1 metre - make sure you are using SI units!");
  }

  constexpr float tol = 1e-5;

  if (abs(width - j_.dim.finger_width) < tol) {
    if (local_debug) std::cout << "Finger width is the same as current, not changing\n";
    return false;
  }

  j_.dim.finger_width = width;
  j_.dim.I = (j_.dim.finger_width * std::pow(j_.dim.finger_thickness, 3)) / 12.0;
  j_.dim.EI = j_.dim.E * j_.dim.I;

  if (local_debug) {
    std::cout << "Finger width changed, now is " << j_.dim.finger_width
      << ", EI is " << j_.dim.EI << '\n';
  }

  return true;
}

bool change_youngs_modulus(float E)
{
  /* set a new E value */

  constexpr bool local_debug = true; // debug_;

  if (local_debug) {
    std::cout << "About to change Youngs modulus from " << j_.dim.E
      << " to " << E << '\n';
  }

  // check we have the right order of magnitude
  if (E < 10e9 or E > 1000e9) {
    std::cout << "Youngs modulus given, E = " << E << '\n';
    throw std::runtime_error("change_youngs_modulus() got an E value not in the right magnitude range, should be E = 200e9");
  }

  constexpr float tol = 1e6;

  if (abs(E - j_.dim.E) < tol) {
    if (local_debug) std::cout << "Youngs modulus, E, is the same as current, not changing\n";
    return false;
  }

  j_.dim.E = E;
  j_.dim.EI = j_.dim.E * j_.dim.I;

  if (local_debug) {
    std::cout << "Youngs modulus changed, now is " << j_.dim.E
      << ", EI is " << j_.dim.EI << '\n';
  }

  return true;
}

void set_finger_stiffness(mjModel* model, std::vector<luke::gfloat> stiffness)
{
  /* set the finger stiffness to a vector sequence of values */

  constexpr bool local_debug = false;
  
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

  INPUT OPTIONS
  stiffness > 0       -> all joints are set to this stiffness value
  stiffness is -7.5   -> stiffness is calculated with finalised theory derviation
  stiffness is -100   -> use hardcoded stiffness values, convergence on real data (only 0.9mm bending @ 300g)
  stiffness is -101   -> use hardcoded stiffness values, convergence on theory (0.8/0.9/1.0mm @ 300g)
  stiffness is -102   -> use hardcoded stiffness values, testing area

  */

  // start of function proper
  constexpr bool local_debug = false;

  int N = j_.num.per_finger;

  // prepare a convenience vector to record stiffness values for each finger
  j_.dim.joint_stiffness.clear();
  j_.dim.joint_stiffness.resize(N);

  if (stiffness > 0) {
    if (local_debug) std::cout << "Finger joint stiffness ALL set to " << stiffness << '\n';
    for (int i : j_.idx.finger) {
      model->jnt_stiffness[i] = stiffness;
    }
    // save stiffness values for one finger, even though they are all the same
    for (int i = 0; i < N; i++) j_.dim.joint_stiffness[i] = stiffness;
  }

  else if (stiffness > -8 and stiffness < -7) {

    if (local_debug) std::cout << "Finger joint stiffness set using finalised theory method (EI*N)/L\n";

    // loop over all three fingers
    for (int i = 0; i < 3; i++) {

      float angle_sum = 0;

      // loop from n=1 to N
      for (int n = 1; n < N + 1; n++) {

        int idx = j_.idx.finger[i * N + (n - 1)];

        // calculate the stiffness for each joint
        float c;
        if (n == 1) {
          c = ((2 * j_.dim.EI) / j_.dim.finger_length) * ((N*N) / (double)(N - (1.0/3.0)));
        }
        else {
          c = (N * j_.dim.EI) / j_.dim.finger_length;
        }

        // save stiffness values for 1st finger
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

  else {
    std::cout << "set_finger_stiffness(...) input stiffness was " << stiffness << '\n';
    throw  std::runtime_error("set_finger_stiffness(...) input stiffness not valid");
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

  if (j_.in_use.base_z or j_.in_use.base_xyz) {
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


    if (debug_) {
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

  if (debug_) {
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

void get_segment_matrices(mjModel* model, mjData* data)
{
  /* find the matrix orientation for each of the segments of the fingers */

  j_.segmentMatrices.f1_idx.clear();
  j_.segmentMatrices.f2_idx.clear();
  j_.segmentMatrices.f3_idx.clear();

  // get the name of the last finger link (hook link is removed by mujoco as fixed joint)
  int tip_num = j_.num.per_finger + j_.dim.fixed_first_segment + (not j_.dim.fixed_hook_segment);
  j_.segmentMatrices.idx_size = tip_num;

  // reset vectors back to empty
  j_.segmentMatrices.apply_flags.clear(); // wipe all flags
  j_.segmentMatrices.apply_flags.resize(tip_num, 0); // set all to 0 (false)
  j_.segmentMatrices.force.clear();
  j_.segmentMatrices.force.resize(tip_num, 0.0);
  j_.segmentMatrices.moment.clear();
  j_.segmentMatrices.moment.resize(tip_num, 0.0);

  std::string f1_names = "finger_1_segment_link_{X}";
  std::string f2_names = "finger_2_segment_link_{X}";
  std::string f3_names = "finger_3_segment_link_{X}";
  std::string f1_hook = "finger_1_finger_hook_link";
  std::string f2_hook = "finger_2_finger_hook_link";
  std::string f3_hook = "finger_3_finger_hook_link";

  // get the segment joints
  for (int i = 0; i < model->nbody; i++) {

    std::string name = mj_id2name(model, mjOBJ_BODY, i);

    // we assume the order must be correct without checking
    if (strcmp_w_sub(name, f1_names, tip_num)) {
      j_.segmentMatrices.f1_idx.push_back(i);
    }
    if (strcmp_w_sub(name, f2_names, tip_num)) {
      j_.segmentMatrices.f2_idx.push_back(i);
    }
    if (strcmp_w_sub(name, f3_names, tip_num)) {
      j_.segmentMatrices.f3_idx.push_back(i);
    }

    // if the finger hook is active (these will be added last)
    if (f1_hook == name) j_.segmentMatrices.f1_idx.push_back(i);
    if (f2_hook == name) j_.segmentMatrices.f2_idx.push_back(i);
    if (f3_hook == name) j_.segmentMatrices.f3_idx.push_back(i);
  }

  if (debug_) {
    print_vec(j_.segmentMatrices.f1_idx, "finger 1 segment idx");
    print_vec(j_.segmentMatrices.f2_idx, "finger 2 segment idx");
    print_vec(j_.segmentMatrices.f3_idx, "finger 3 segment idx");
  }
  
  if (j_.segmentMatrices.f1_idx.size() != tip_num) {
    throw std::runtime_error("f1_idx vec in get_segment_matrices() has size != num segments");
  }
  if (j_.segmentMatrices.f2_idx.size() != tip_num) {
    throw std::runtime_error("f2_idx vec in get_segment_matrices() has size != num segments");
  }
  if (j_.segmentMatrices.f3_idx.size() != tip_num) {
    throw std::runtime_error("f3_idx vec in get_segment_matrices() has size != num segments");
  }

  // std::cout << "get_segment_matrices() tip idx are " << j_.segmentMatrices.f1_idx[tip_num - 1]
  //   << ", " << j_.segmentMatrices.f2_idx[tip_num - 1]
  //   << ", " << j_.segmentMatrices.f3_idx[tip_num - 1] << '\n';

  // find the starting body orientation
  for (int j = 0; j < 9; j++) {

    // std::cout << "matrix values finger 2: " << data->xmat[j_.segmentMatrices.f2_idx[tip_num - 1] * 9 + j] << '\n';

    // use frame orientation of first joint for orientation of entire finger
    j_.segmentMatrices.finger1[j] = data->xmat[j_.segmentMatrices.f1_idx[0] * 9 + j];
    j_.segmentMatrices.finger2[j] = data->xmat[j_.segmentMatrices.f2_idx[0] * 9 + j];
    j_.segmentMatrices.finger3[j] = data->xmat[j_.segmentMatrices.f3_idx[0] * 9 + j];
  }
}

void set_segment_force(int seg_num, bool set_as, double force)
{
  /* toggle whether to apply force and/or moment to a given segment */

  if (seg_num < 0 or seg_num >= j_.segmentMatrices.idx_size) {
    std::cout << "ERROR: seg_num is " << seg_num << '\n';
    throw std::runtime_error("apply_segment_force() recieved seg_num out of bounds");
  }

  // std::cout << "set segement force, seg_num = " << seg_num << ", force = " << force << "\n";

  j_.segmentMatrices.apply_flags[seg_num] = set_as;
  j_.segmentMatrices.force[seg_num] = force;
}

void set_segment_moment(int seg_num, bool set_as, double moment)
{
  /* toggle whether to apply force and/or moment to a given segment */

  if (seg_num < 0 or seg_num >= j_.segmentMatrices.idx_size) {
    std::cout << "ERROR: seg_num is " << seg_num << '\n';
    throw std::runtime_error("apply_segment_moment() recieved seg_num out of bounds");
  }

  // std::cout << "set segement moment, seg_num = " << seg_num << ", moment = " << moment << "\n";

  j_.segmentMatrices.apply_flags[seg_num] = set_as;
  j_.segmentMatrices.moment[seg_num] = moment;
}

void resolve_segment_forces(mjModel* model, mjData* data)
{
  /* apply forces to segments specified, should be called every step */

  // std::cout << "resolve_segment_forces():\n";
  // print_vec(j_.segmentMatrices.apply_flags, "apply flags");
  // print_vec(j_.segmentMatrices.force, "force");
  // std::cout << "\n\n";

  for (int i = 0; i < j_.segmentMatrices.apply_flags.size(); i++) {
    if (j_.segmentMatrices.apply_flags[i]) {
      apply_segment_force(model, data, i, j_.segmentMatrices.force[i], j_.segmentMatrices.moment[i]);
      // std::cout << "applying force " << j_.segmentMatrices.force[i] << " on segment " << i << '\n';
      // std::cout << "applying moment " << j_.segmentMatrices.moment[i] << " on segment " << i << '\n';
    }
  }
}

void apply_segment_force(mjModel* model, mjData* data, int seg_num, double force, double moment)
{
  /* apply a horizontal force to a given segment from 1..N. Can also apply a 
  moment around the joint axis, this = 0 by default*/

  if (seg_num < 0 or seg_num >= j_.segmentMatrices.idx_size) {
    std::cout << "ERROR: seg_num is " << seg_num << '\n';
    throw std::runtime_error("apply_segment_force() recieved seg_num out of bounds");
  }

  // lock the fingers in place
  for (int i : j_.con_idx.prismatic) {
    set_constraint(model, data, i, true);
  }
  for (int i : j_.con_idx.revolute) {
    set_constraint(model, data, i, true);
  }

  std::vector<std::vector<int>*> idx_vecs {
    &j_.segmentMatrices.f1_idx,
    &j_.segmentMatrices.f2_idx,
    &j_.segmentMatrices.f3_idx
  };

  std::vector<mjtNum*> mats {
    j_.segmentMatrices.finger1,
    j_.segmentMatrices.finger2,
    j_.segmentMatrices.finger3
  };

  // loop through and apply force to the given segment
  for (int i = 0; i < 3; i++) {

    // prepare to apply force outwards on fingertips
    mjtNum fvec[3] = { 0, 0, -force };
    mjtNum mvec[3] = { 0, moment, 0 };
    mjtNum rotfvec[3];
    mjtNum rotmvec[3];

    // std::cout << "finger " << i + 1 << " matrix values in apply: ";
    // for (int j = 0; j < 9; j++) std::cout << mats[i][j] << ", ";
    // std::cout << "\n";

    // rotate into the tip frame to pull directly horizontal
    mju_mulMatVec(rotfvec, mats[i], fvec, 3, 3);
    mju_mulMatVec(rotmvec, mats[i], mvec, 3, 3);

    // apply force in cartesian space (joint space is qfrc_applied)
    data->xfrc_applied[(*idx_vecs[i])[seg_num] * 6 + 0] = rotfvec[0];
    data->xfrc_applied[(*idx_vecs[i])[seg_num] * 6 + 1] = rotfvec[1];
    data->xfrc_applied[(*idx_vecs[i])[seg_num] * 6 + 2] = rotfvec[2];
    data->xfrc_applied[(*idx_vecs[i])[seg_num] * 6 + 3] = rotmvec[0];
    data->xfrc_applied[(*idx_vecs[i])[seg_num] * 6 + 4] = rotmvec[1];
    data->xfrc_applied[(*idx_vecs[i])[seg_num] * 6 + 5] = rotmvec[2];

    // std::cout << "apply_segment_force() finger " << i + 1 << " rotfvec is " << rotfvec[0] << ", " << rotfvec[1]
    //   << ", " << rotfvec[2] << '\n';
  }
}

void apply_UDL(double force_per_m)
{
  /* apply a uniformally distributed load with a force per metre W */

  // find total force applied over the finger length
  double total_force = force_per_m * j_.dim.finger_length;

  // std::cout << "total force applied in UDL is " << total_force << '\n';

  // do we apply force to joint 0 (which will have no effect)
  bool ignore_first_seg = false; // yes we do, otherwise the UDL is not uniform
  float force_per = total_force / (float) (j_.segmentMatrices.idx_size - ignore_first_seg);

  // add force also to first segment for visual consistency in mujoco, it has no effect
  for (int i = 0; i < j_.segmentMatrices.idx_size; i++) {
    set_segment_force(i, true, force_per);
  }
}

void wipe_segment_forces()
{
  /* remove all segment fores */

  for (int i = 0; i < j_.segmentMatrices.idx_size; i++) {
    set_segment_force(i, false, 0.0);
  }
}

void apply_tip_force(double force)
{
  /* apply a force at the tip of the finger */

  set_segment_force(j_.segmentMatrices.idx_size - 1, true, force);
}

void apply_tip_moment(double moment)
{
  /* apply a moment at the tip of the finger */

  set_segment_moment(j_.segmentMatrices.idx_size - 1, true, moment);
}

void apply_tip_force(mjModel* model, mjData* data, double force, bool reset)
{
  /* apply a horizontal force to the tip of the finger */

  /* OLD CODE: THIS FUNCTION SHOULD NO LONGER BE CALLED, it has been replaced
  by the apply_tip_force(double_force) version */

  std::runtime_error("apply_tip_force(model, data, force, reset) is depreciated and should not be called");
  
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

    /* this method does not work if the hook link is in use, as it is the final joint not
    one of the segment joints. Hence this function is depreciated and should not be called */

    int tip_num = j_.num.per_finger + j_.dim.fixed_first_segment;
    std::string tip_name = "finger_{X}_segment_link_" + std::to_string(tip_num);

    for (int i = 0; i < model->nbody; i++) {
      std::string name = mj_id2name(model, mjOBJ_BODY, i);
      if (strcmp_w_sub(name, tip_name, 3)) {
        tip_idx.push_back(i);
      }
    }

    if (debug_) {
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

  // mj_step(model, data); // needs ctrl ptr assigned

  mj_step1(model, data);
  control(model, data); // since ctrl ptr not assigned
  mj_step2(model, data);

  return;

  // mj_step1(model, data);

  // /* 
  // To make the 'leadscrews' non-backdriveable, we want no forces to be
  // transferred from the finger to the finger platform/joints. So we will
  // try wiping any forces, and trust that the momentum forces are sufficiently
  // small.

  // The joints to wipe are either:
  //   finger_1_revolute_joint (1 + panda)
  //   finger_2_revolute_joint (12 + panda)
  //   finger_3_revolute_joint (23 + panda)
  // or:
  //   finger_1_segment_joint_1 (2 + panda)
  //   finger_2_segment_joint_2 (13 + panda)
  //   finger_3_segment_joint_3 (24 + panda)

  // */   

  // static std::vector<int> to_wipe {
  //   j_.idx.gripper[j_.gripper.prismatic[0]],  // 0
  //   j_.idx.gripper[j_.gripper.prismatic[1]],  // 11
  //   j_.idx.gripper[j_.gripper.prismatic[2]],  // 22
  //   j_.idx.gripper[j_.gripper.revolute[0]],   // 1
  //   j_.idx.gripper[j_.gripper.revolute[1]],   // 12
  //   j_.idx.gripper[j_.gripper.revolute[2]],   // 23
  // };

  // control(model, data);   // since ctrl pntr not assigned

  // for (int i = 0; i < to_wipe.size(); i++) {
  //   // all are (nv * 1), and nv = 34 for gripper, which = njnts
  //   // data->qfrc_passive[to_wipe[i]] = 0;  // passive force
  //   // data->efc_vel[to_wipe[i]] = 0;       // velocity in constraint space: J*qvel
  //   // data->efc_aref[to_wipe[i]] = 0;      // reference pseudo-acceleartion
  //   // data->qfrc_bias[to_wipe[i]] = 0;     // C(qpos, qvel)
  //   // data->cvel[to_wipe[i]] = 0;          // com-based velcotiy [3D rot; 3D tran]

  //   // data->qfrc_unc[to_wipe[i]] = 0;
  //   // data->qacc_unc[to_wipe[i]] = 0;
  // }

  // // // for testing, applly known force to the end of the finger
  // // data->xfrc_applied[11 * 6 + 1] = 10;

  // mj_step2(model, data);
  // return;

  // mj_fwdActuation(model, data);
  // mj_fwdAcceleration(model, data);
  // mj_fwdConstraint(model, data);


  // std::vector<mjtNum> qfrc;
  // // std::cout << "qfrc constraint is ";
  // for (int i = 0; i < to_wipe.size(); i++) {

  //   // // wipe forces arising from constraints (contacts)
  //   // qfrc.push_back(data->qfrc_constraint[to_wipe[i]]);
  //   // data->qfrc_constraint[to_wipe[i]] = 0;
  //   // // std::cout << data->qfrc_constraint[to_wipe[i]] << " ";
  // } 
  // // std::cout << "\n";

  // // int j = 0;
  // // for (int i : j_.gripper.prismatic) {
  // //   data->ctrl[j_.num.panda + i] += -qfrc[j];
  // //   j += 1;
  // // }
  // // for (int i : j_.gripper.revolute) {
  // //   data->ctrl[j_.num.panda + i] += -qfrc[j];
  // //   j += 1;
  // // }
  // // mj_fwdActuation(model, data);
  // // mj_fwdAcceleration(model, data);
  // // mj_fwdConstraint(model, data);

  // mj_sensorAcc(model, data);
  // mj_checkAcc(model, data);

  // // compare forward and inverse solutions if enabled
  // // if( mjENABLED(mjENBL_FWDINV) )
  // if (model->opt.enableflags and mjENBL_FWDINV)
  //     mj_compareFwdInv(model, data);

  // mj_Euler(model, data);


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

  if (j_.in_use.base_z or j_.in_use.base_xyz) {
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
    u = ((*j_.to_qpos.gripper[i]) - target.x) * j_.ctrl.gripper_kp.x 
      + (*j_.to_qvel.gripper[i]) * j_.ctrl.gripper_kd.x;
    // if (abs(u) > force_lim) {
    //   std::cout << "x frc limited from " << u << " to ";
    //   u = force_lim * sign(u);
    //   std::cout << u << '\n';
    // }
    data->ctrl[n + i] = -u;
  }

  for (int i : j_.gripper.revolute) {
    u = ((*j_.to_qpos.gripper[i]) - target.th) * j_.ctrl.gripper_kp.y 
      + (*j_.to_qvel.gripper[i]) * j_.ctrl.gripper_kd.y;
    // if (abs(u) > force_lim) {
    //   std::cout << "y frc limited from " << u << " to ";
    //   u = force_lim * sign(u);
    //   std::cout << u << '\n';
    // }
    data->ctrl[n + i] = -u;
  }
  
  for (int i : j_.gripper.palm) {
    u = ((*j_.to_qpos.gripper[i]) - target.z) * j_.ctrl.gripper_kp.z 
      + (*j_.to_qvel.gripper[i]) * j_.ctrl.gripper_kd.z;
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

  if (j_.in_use.base_z and j_.num.base != 1) {
    throw std::runtime_error("base dof does not equal 1 but in_use.base_z is true");
  }
  if (j_.in_use.base_xyz and j_.num.base != 3) {
    throw std::runtime_error("base dof does not equal 3 but in_use.base_xyz is true");
  }

  // for (int i = 0; i < j_.num.base; i++) {
  //   u = ((*j_.to_qpos.base[i]) - target_.base[i]) * j_.ctrl.base_kp
  //     + (*j_.to_qvel.base[i]) * j_.ctrl.base_kd;
  //   data->ctrl[n + i] = -u;
  // }

  if (j_.in_use.base_z) {

    // z movement only
    u = ((*j_.to_qpos.base[0]) - target_.base.z) * j_.ctrl.base_xyz_kp.z
      + (*j_.to_qvel.base[0]) * j_.ctrl.base_xyz_kd.z;
    data->ctrl[n + 0] = -u;

  }
  else if (j_.in_use.base_xyz) {

    // x movement
    u = ((*j_.to_qpos.base[0]) - target_.base.x) * j_.ctrl.base_xyz_kp.x
      + (*j_.to_qvel.base[0]) * j_.ctrl.base_xyz_kd.x;
    data->ctrl[n + 0] = -u;

    // y movement
    u = ((*j_.to_qpos.base[1]) - target_.base.y) * j_.ctrl.base_xyz_kp.y
      + (*j_.to_qvel.base[1]) * j_.ctrl.base_xyz_kd.y;
    data->ctrl[n + 1] = -u;

    // z movement
    u = ((*j_.to_qpos.base[2]) - target_.base.z) * j_.ctrl.base_xyz_kp.z
      + (*j_.to_qvel.base[2]) * j_.ctrl.base_xyz_kd.z;
    data->ctrl[n + 2] = -u;
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

  if (j_.in_use.base_z or j_.in_use.base_xyz) {
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

  // disable motor position data logging if hook isn't fixed (ie deflection test set)
  static bool log_test_data = true * j_.dim.fixed_hook_segment;

  bool stepped = false;

  if (data->time > last_step_time_ + j_.ctrl.time_per_step) {

    // we can optionally log position data to see motor response to steps
    if (log_test_data) {

      target_.timedata.add(data->time);

      // save current target information
      target_.target_stepperx.add(target_.next.x * 1e6);
      target_.target_steppery.add(target_.next.y * 1e6);
      target_.target_stepperz.add(target_.next.z * 1e6);
      target_.target_basez.add(target_.base.z * 1e6);
      if (j_.in_use.base_xyz) {
        target_.target_basex.add(target_.base.x * 1e6);
        target_.target_basey.add(target_.base.y * 1e6);
      }

      // now save actual position data direct from mujoco
      Gripper temp;
      temp.set_xyz_m_rad(*j_.to_qpos.gripper[0], *j_.to_qpos.gripper[1], *j_.to_qpos.gripper[6]);
      target_.actual_stepperx.add(temp.x * 1e6);
      target_.actual_steppery.add(temp.y * 1e6);
      target_.actual_stepperz.add(temp.z * 1e6);
      if (j_.in_use.base_z) {
        target_.actual_basez.add(*j_.to_qpos.base[0] * 1e6);
      }
      else if (j_.in_use.base_xyz) {
        target_.actual_basex.add(*j_.to_qpos.base[0] * 1e6);
        target_.actual_basey.add(*j_.to_qpos.base[1] * 1e6);
        target_.actual_basez.add(*j_.to_qpos.base[2] * 1e6);
      }
      
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

void update_base_limits()
{
  /* update the operating limits of the gripper base, only done on reset */

  target_.base_min.x = j_.baseLims.x_min;
  target_.base_max.x = j_.baseLims.x_max;

  target_.base_min.y = j_.baseLims.y_min;
  target_.base_max.y = j_.baseLims.y_max;

  target_.base_min.z = j_.baseLims.z_min;
  target_.base_max.z = j_.baseLims.z_max;

  target_.base_min.roll = j_.baseLims.roll_min;
  target_.base_max.roll = j_.baseLims.roll_max;

  target_.base_min.pitch = j_.baseLims.pitch_min;
  target_.base_max.pitch = j_.baseLims.pitch_max;

  target_.base_min.yaw = j_.baseLims.yaw_min;
  target_.base_max.yaw = j_.baseLims.yaw_max;
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

bool set_base_target_m(double x, double y, double z)
{
  /* specify an x,y,z base target */

  target_.last_robot = Target::Robot::panda;

  // apply the new positions to the targets
  target_.base.x = x;
  target_.base.y = y;
  target_.base.z = z;

  bool within_limits = true;

  // x base limits
  if (target_.base.x > target_.base_max.x) {
    target_.base.x = target_.base_max.x;
    within_limits = false;
  }
  if (target_.base.x < target_.base_min.x) {
    target_.base.x = target_.base_min.x;
    within_limits = false;
  }
  // y base limits
  if (target_.base.y > target_.base_max.y) {
    target_.base.y = target_.base_max.y;
    within_limits = false;
  }
  if (target_.base.y < target_.base_min.y) {
    target_.base.y = target_.base_min.y;
    within_limits = false;
  }
  // z base limits
  if (target_.base.z > target_.base_max.z) {
    target_.base.z = target_.base_max.z;
    within_limits = false;
  }
  if (target_.base.z < target_.base_min.z) {
    target_.base.z = target_.base_min.z;
    within_limits = false;
  }

  return within_limits;
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

  // add the incoming position changes to our target
  target_.base.x += x;
  target_.base.y += y;
  target_.base.z += z;

  bool within_limits = true;

  // x base limits
  if (target_.base.x > target_.base_max.x) {
    target_.base.x = target_.base_max.x;
    within_limits = false;
  }
  if (target_.base.x < target_.base_min.x) {
    target_.base.x = target_.base_min.x;
    within_limits = false;
  }
  // y base limits
  if (target_.base.y > target_.base_max.y) {
    target_.base.y = target_.base_max.y;
    within_limits = false;
  }
  if (target_.base.y < target_.base_min.y) {
    target_.base.y = target_.base_min.y;
    within_limits = false;
  }
  // z base limits
  if (target_.base.z > target_.base_max.z) {
    target_.base.z = target_.base_max.z;
    within_limits = false;
  }
  if (target_.base.z < target_.base_min.z) {
    target_.base.z = target_.base_min.z;
    within_limits = false;
  }

  return within_limits;
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

void set_base_to_max_height(mjData* data)
{
  /* moves the base position to maximum height, should only ber used for specific
  tests and not during any grasping */

  // confusingly, for the base down is +ve and up is -ve
  float max_height = target_.base_min.z;

  // set the base target to maximum
  set_base_target_m(0, 0, max_height);

  // override qpos for the base to snap model to maximum
  if (j_.in_use.base_xyz) {
    (*j_.to_qpos.base[2]) = max_height; 
  }
  else if (j_.in_use.base_z) {
    (*j_.to_qpos.base[0]) = max_height; 
  } 
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

  // // calculate the approximated gauge reading
  // gfloat k = y / j_.gauge.xpos_cubed;

  // // transfer to SI units for force (optional)
  // // THIS IS NOT ACCURATE as L^3 only applies at tip of beam
  // // BETTER TO NOT PROCESS TO SI as it removes this functions dependence on j_.dim.EI
  // gfloat P = k * (3 * j_.dim.EI);

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

  // k *= (100.0 / 0.136);

  // return y value in millimeters, unprocessed (OLD: return P;)
  return y * 1000;
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

  // // calculate the approximated gauge reading
  // gfloat k = y / j_.gauge.xpos_cubed;

  // // transfer to SI units for force (optional)
  // gfloat P = k * (3 * j_.dim.EI);

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

  // k *= (100.0 / 0.136);

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
  float force, float finger_stiffness, int force_style)
{
  /* evaluate the difference in joint angle between the actual and model
  predicted values
  
  force style: 0 = point end load
               1 = UDL
               2 = pure end moment
  */

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
    theory_y[1] = (force * std::pow(theory_x[1], 3)) / (3 * j_.dim.EI); // WRONG EQUATION!
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
    float c = j_.dim.joint_stiffness[i];

    // actual joint values
    joint_angles[i] = *j_.to_qpos.finger[i + finger * N];

    // predicted joint values
    joint_pred[i] = ((N - n + 1) * force * j_.dim.finger_length) / (N * c);

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
    if (force_style == 0) {
      double theory_factor = (force) / (6.0 * j_.dim.EI);
      double x = (i + 1) * j_.dim.segment_length;
      theory_x[i + 1 + ffs] = x;
      theory_y[i + 1 + ffs] = theory_factor * (-std::pow(x, 3) + 3 * j_.dim.finger_length * std::pow(x, 2)); 
    }
    else if (force_style == 1) {
      double theory_factor = (force / (24.0 * j_.dim.EI));
      double x = (i + 1) * j_.dim.segment_length;
      theory_x[i + 1 + ffs] = x;
      theory_y[i + 1 + ffs] = theory_factor * 
        (std::pow(x, 4) - 4 * j_.dim.finger_length * std::pow(x, 3) 
          + 6 * j_.dim.finger_length * j_.dim.finger_length * std::pow(x, 2)); 
    }
    else if (force_style == 2) {
      double theory_factor = ((force) / (2.0 * j_.dim.EI));
      double x = (i + 1) * j_.dim.segment_length;
      theory_x[i + 1 + ffs] = x;
      theory_y[i + 1 + ffs] = theory_factor * std::pow(x, 2);
    }
    else {
      std::cout << "force_style = " << force_style << '\n';
      throw std::runtime_error("force style was not valid in verify_small_angle_model(...)");
    }
    
  }

  fill_theory_curve(theory_x_curve, theory_y_curve, force, theory_N, force_style);
  
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
  float force, int num, int force_style)
{
  /* take two vectors (which are wiped) and fill them with the theory curve, this
  is basic bending theory for Euler-Bernoulli beam. Force should be given in NEWTONS.
  This does a point load on a cantilever 
  
  force style: 0 = point end load
               1 = UDL
               2 = pure end moment
  */

  theory_X.clear();
  theory_Y.clear();

  theory_X.resize(num);
  theory_Y.resize(num);

  float L = j_.dim.finger_length;

  if (force_style == 0) {

    // create point end load theory curve
    for (int i = 0; i < num; i++) {

      double theory_factor = (force / (6.0 * j_.dim.EI));
      double x = L * ((i / (float) (num - 1)));
      theory_X[i] = x;
      theory_Y[i] = theory_factor * (-std::pow(x, 3) + 3 * L * std::pow(x, 2)); 
    }
  }
  else if (force_style == 1) {

    // float W = force / L; // if we are doing UDL

    // create UDL theory curve
    for (int i = 0; i < num; i++) {

      double theory_factor = (force / (24.0 * j_.dim.EI));
      double x = L * ((i / (float) (num - 1)));
      theory_X[i] = x;
      theory_Y[i] = theory_factor * (std::pow(x, 4) - 4 * L * std::pow(x, 3) + 6 * L * L * std::pow(x, 2)); 
    }
  }
  else if (force_style == 2) {

    // create pure end moment theory curve
    for (int i = 0; i < num; i++) {

      double theory_factor = (force / (2.0 * j_.dim.EI));
      double x = L * ((i / (float) (num - 1)));
      theory_X[i] = x;
      theory_Y[i] = theory_factor * (std::pow(x, 2)); 
    }
  }
  else {
    std::cout << "force_style = " << force_style << '\n';
      throw std::runtime_error("force style was not valid in fill_theory_curve(...)");
  }
}

std::vector<float> discretise_curve(std::vector<float> X, std::vector<float> truth_X, 
  std::vector<float> truth_Y)
{
  /* takes a detailed curve (truth_X, truth_Y) and finds the more coarse points
  (X, Y), returning the vector Y when given X */

  int n_profile = X.size();
  int n_truth = truth_X.size();

  std::vector<float> Y(n_profile);

  int last = 0;
  bool found = false;

  for (int i = 0; i < n_profile; i++) {

    // find the closest X point in the 'truth'
    for (int j = last; j < n_truth; j++) {

      if (truth_X[j] > X[i]) {
        last = j - 1;
        found = true;
        break;
      }
    }

    float truth_X_val;
    float truth_Y_val;

    if (not found) {
      truth_X_val = truth_X[n_truth - 1];
      truth_Y_val = truth_Y[n_truth - 1];
    }
    else if (last < 0) {
      truth_X_val = truth_X[0];
      truth_Y_val = truth_Y[0];
    }
    else {
      // interpolate
      float interval = truth_X[last + 1] - truth_X[last];
      float a = (truth_X[last + 1] - X[i]) / interval;
      float b = 1 - a;
      truth_X_val = (a * truth_X[last] + b * truth_X[last + 1]);
      truth_Y_val = (a * truth_Y[last] + b * truth_Y[last + 1]);
    }

    // save this Y value
    Y[i] = truth_Y_val;

    // prepare to loop
    found = false;
  }

  return Y;
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

JointStates get_target_state()
{
  /* Get the state of the gripper target in as a named structure */

  return target_.get_target_m();
}

std::vector<gfloat> get_target_state_vector()
{
  /* Get a vector of the gripper target positions, only including active joints */

  JointStates state = target_.get_target_m();

  std::vector<gfloat> state_vec;

  state_vec.push_back(state.gripper_x);
  state_vec.push_back(state.gripper_x);
  state_vec.push_back(state.gripper_x);
  if (j_.in_use.base_xyz) {
    state_vec.push_back(state.base_x);
    state_vec.push_back(state.base_y);
  }
  state_vec.push_back(state.base_z);

  // state_vec.push_back(state.base_roll);
  // state_vec.push_back(state.base_pitch);
  // state_vec.push_back(state.base_yaw);

  return state_vec;
}

std::vector<double> get_base_min()
{
  /* return the base joint minimums */

  return target_.base_min.to_vec();
}

std::vector<double> get_base_max()
{
  /* return the base joint minimums */

  return target_.base_max.to_vec();
}

bool use_base_xyz()
{
  /* are we using full base xyz movement */

  return j_.in_use.base_xyz;
}

int get_N() 
{
  return j_.num.per_finger + j_.dim.fixed_first_segment;
}

float get_finger_thickness()
{
  return j_.dim.finger_thickness;
}

float get_finger_width()
{
  return j_.dim.finger_width;
}

float get_finger_length()
{
  /* return the current finger length */

  return j_.dim.finger_length;
}

std::vector<luke::gfloat> get_stiffnesses()
{
  return j_.dim.joint_stiffness;
}

gfloat get_target_finger_angle()
{
  /* return the target finger angle in radians */

  return target_.end.get_th_rad();
}

float calc_yield_point_load()
{
  /* return the vertical point load to yield the cantilever */

  float M_max = (j_.dim.yield_stress * j_.dim.I) / (0.5 * j_.dim.finger_thickness);
  float F_max = M_max / j_.dim.finger_length;

  return F_max;
}

float calc_yield_point_load(float thickness, float width)
{
  /* calculate the yield load given a particular thickness and width */

  float I = (width * std::pow(thickness, 3)) / 12.0;
  float M_max = (j_.dim.yield_stress * I) / (0.5 * thickness);
  float F_max = M_max / j_.dim.finger_length;

  return F_max;
}

float get_fingertip_z_height()
{
  /* returns the current fingertip height with 0 being the starting value before
  any actions, and negative values meaning the fingertips are going down. Since
  the gripper starts usually at 10mm height, a value of -10e-3 indicates the tips
  have hit the ground */

  float straight_finger_distance = -target_.base_min.z - target_.base.z;
  float tip_lift = j_.dim.finger_length * (1 - std::cos(target_.end.get_th_rad()));
  float height_above_min = straight_finger_distance + tip_lift;

  return height_above_min + target_.base_min.z;
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

void set_all_objects_colour(mjModel* model, std::vector<float> rgba)
{
  /* set the colour of all the objects */

  oh_.set_all_colours(model, rgba);
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

void set_main_body_colour(mjModel* model, std::vector<float> rgba)
{
  /* set the colour of the gripper main body. This only works if the main body
  geoms are named (newer object sets), otherwise it has no effect */

  // loop through the vector we assigned and update the colour
  for (int i : j_.geom_idx.main_body) {
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