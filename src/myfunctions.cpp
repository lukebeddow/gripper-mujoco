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
    double segment_length = 0;                        // runtime depends
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
    int num_steps = 10;
    double pulses_per_s = 2000;
    double stepper_kp = 10.0;
    double stepper_kd = 0.5;
    double servo_kp = 4.0;
    double servo_kd = 0.4;
    // Gain kp {10, 10, 100};
    // Gain kd {0.5, 0.5, 0.5};
    Gain kp {100, 100, 1000};
    Gain kd {1, 1, 1};
    double base_kp = 2000;
    double base_kd = 100;
  } ctrl;

  // simulation details
  static constexpr struct Sim {
    static constexpr int step_tolerance = 3;    // for detecting equilibrium
    static constexpr int target_tolerance = 10; // for target_reached
    static constexpr int n_arr = 3;             // no. of steps saved in arrays
    static constexpr int n_settle = 10;         // no. of steps to be settled
  } sim{};

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

constexpr static bool debug = false;

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

  if (debug) print_joint_names(model);

  // resize state vectors and find qpos/qvel pointers
  configure_qpos(model, data);

  // calculate constants
  j_.dim.segment_length = j_.dim.finger_length / float(j_.num.per_finger);

  // // initialise the gains depending on our choice of control
  // if (j_.ctrl.stepper) {
  //   j_.ctrl.kp.set(j_.ctrl.stepper_kp);
  //   j_.ctrl.kd.set(j_.ctrl.stepper_kd);
  // }
  // else {
  //   j_.ctrl.kp.set(j_.ctrl.servo_kp);
  //   j_.ctrl.kd.set(j_.ctrl.servo_kd);
  // }

  // initialise the settling arrays to our default finger values
  for (int i = 0; i < fingers_.size(); i++) {
    for (int j = 0; j < j_.sim.n_arr; j++) {
      (*finger_arrays_[i])[j][0] = fingers_[i]->step.x;
      (*finger_arrays_[i])[j][1] = fingers_[i]->step.y;
    }
  }
}

void reset(mjModel* model, mjData* data)
{
  /* reset the simulation */

  // reset the targets
  target_.reset();

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

  // set the joints to the eqilibrium position
  calibrate_reset(model, data);
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
  for (int i = 0; i < model->njnt; i++) {
    std::string x = mj_id2name(model, mjOBJ_JOINT, i);
    if (x.substr(0,6) == "finger" and x.substr(9, 13) == "segment_joint") {
      j_.num.finger += 1;
    }
  }

  // for testing
  if (j_.num.finger != 27) throw std::runtime_error("j_.num.finger != 27");

  // hence per finger is this divided by 3
  j_.num.per_finger = j_.num.finger / 3;

  if (j_.num.finger > 0) {

    j_.in_use.finger = true;

    // add the names of every finger joint to the global vector
    for (int i = 1; i <= 3; i++) {
      for (int k = 1; k <= j_.num.per_finger; k++) {
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

void add_base_joint_noise(std::vector<luke::gfloat> noise)
{
  /* add noise in metres to resultant joint position */

  if (noise.size() == 1) {
    target_.base_noise[0] = noise[0];
  }

  else {
    throw std::runtime_error("base noise must be specified as a 1 element vector"
     " in add_gripper_joint_noise()");
  }
}

void add_gripper_joint_noise(std::vector<luke::gfloat> noise)
{
  /* add noise in metres to resultant joint position */

  if (noise.size() == 3) {
    target_.gripper_noise[j_.gripper.prismatic[0]] = noise[0];
    target_.gripper_noise[j_.gripper.prismatic[1]] = noise[0];
    target_.gripper_noise[j_.gripper.prismatic[2]] = noise[0];
    target_.gripper_noise[j_.gripper.revolute[0]] = noise[1];
    target_.gripper_noise[j_.gripper.revolute[1]] = noise[1];
    target_.gripper_noise[j_.gripper.revolute[2]] = noise[1];
    target_.gripper_noise[j_.gripper.palm[0]] = noise[2];
  }

  else if (noise.size() == 7) {
    target_.gripper_noise[j_.gripper.prismatic[0]] = noise[0];
    target_.gripper_noise[j_.gripper.prismatic[1]] = noise[2];
    target_.gripper_noise[j_.gripper.prismatic[2]] = noise[4];
    target_.gripper_noise[j_.gripper.revolute[0]] = target_.end.calc_th(0, noise[1]);
    target_.gripper_noise[j_.gripper.revolute[1]] = target_.end.calc_th(0, noise[3]);
    target_.gripper_noise[j_.gripper.revolute[2]] = target_.end.calc_th(0, noise[5]);
    target_.gripper_noise[j_.gripper.palm[0]] = noise[6];
  }

  else {
    throw std::runtime_error("gripper noise must either be specified as a 3 element"
     " or 7 element vector in add_gripper_joint_noise()");
  }
}

void snap_to_target()
{
  /* force all of the joints to their target position (including noise) */

  // snap the base, only z supported, note we have an offset
  (*j_.to_qpos.base[0]) = target_.base[0] - target_.base_noise[0] - target_.z_offset; 

  for (int i = 0; i < j_.num.gripper; i++) {
    (*j_.to_qpos.gripper[i]) = target_.end.x + target_.gripper_noise[0];
  }

  // snap gripper joints
  for (int i : j_.gripper.prismatic) {
    (*j_.to_qpos.gripper[i]) = target_.end.x + target_.gripper_noise[i];
  }

  for (int i : j_.gripper.revolute) {
    (*j_.to_qpos.gripper[i]) = target_.end.th + target_.gripper_noise[i];
  }

  for (int i : j_.gripper.palm) {
    (*j_.to_qpos.gripper[i]) = target_.end.z + target_.gripper_noise[i];
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

  double force_lim = 10000;

  int n = j_.num.panda + j_.num.base;

  // input the control signals
  for (int i : j_.gripper.prismatic) {
    u = ((*j_.to_qpos.gripper[i]) + target_.gripper_noise[i] - target.x) * j_.ctrl.kp.x 
      + (*j_.to_qvel.gripper[i]) * j_.ctrl.kd.x;
    if (abs(u) > force_lim) {
      std::cout << "x frc limited from " << u << " to ";
      u = force_lim * sign(u);
      std::cout << u << '\n';
    }
    data->ctrl[n + i] = -u;
  }

  for (int i : j_.gripper.revolute) {
    u = ((*j_.to_qpos.gripper[i]) + target_.gripper_noise[i] - target.th) * j_.ctrl.kp.y 
      + (*j_.to_qvel.gripper[i]) * j_.ctrl.kd.y;
    if (abs(u) > force_lim) {
      std::cout << "y frc limited from " << u << " to ";
      u = force_lim * sign(u);
      std::cout << u << '\n';
    }
    data->ctrl[n + i] = -u;
  }
  
  for (int i : j_.gripper.palm) {
    u = ((*j_.to_qpos.gripper[i]) + target_.gripper_noise[i] - target.z) * j_.ctrl.kp.z 
      + (*j_.to_qvel.gripper[i]) * j_.ctrl.kd.z;
    if (abs(u) > force_lim) {
      std::cout << "z frc limited from " << u << " to ";
      u = force_lim * sign(u);
      std::cout << u << '\n';
    }
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
    u = ((*j_.to_qpos.base[i]) - target_.base_noise[i] - target_.base[i]) * j_.ctrl.base_kp
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

void update_all(const mjModel* model, mjData* data)
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

void update_stepper(const mjModel* model, mjData* data)
{
  /* update the gripper joint positions and determine equilibirum/target_reached 
  assuming a stepper motor style */

  static double time_per_step = j_.ctrl.num_steps / j_.ctrl.pulses_per_s;

  bool stepped = false;

  if (data->time > last_step_time_ + time_per_step) {
    last_step_time_ = data->time;
    target_.next.step_to(target_.end, j_.ctrl.num_steps);
    stepped = true;
    // std::cout << "step!\n";
  }
  else {
    // std::cout << "wait-";
  }

  // // extract the state of each finger - this is only used for check_settling(), NOT NEEDED
  // finger1_.set_xyz_m_rad(*j_.to_qpos.gripper[0], *j_.to_qpos.gripper[1], *j_.to_qpos.gripper[6]);
  // finger2_.set_xy_m_rad(*j_.to_qpos.gripper[2], *j_.to_qpos.gripper[3]);
  // finger3_.set_xy_m_rad(*j_.to_qpos.gripper[4], *j_.to_qpos.gripper[5]);

  // // next we will check for settling, only worth checking after steps made
  // if (stepped) {
  //   check_settling();
  // }
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

  // return false if target is outside motor limits
  return target_.end.set_xyz_step(x, y, z);
}

bool set_gripper_target_m(double x, double y, double z)
{
  /* set a motor state target for the gripper, returns true when reached */

  // return false if the target is outside motor limits
  return target_.end.set_xyz_m(x, y, z);
}

bool set_gripper_target_m_rad(double x, double th, double z)
{
  /* sets a joint state target for the gripper, returns true when reached */

  // return false if the target is outside motor limits
  return target_.end.set_xyz_m(x, th, z);
}

bool move_gripper_target_step(int x, int y, int z)
{
  /* adjust the gripper target by the indicated number of steps */

  // return false if the target is outside motor limits
  return target_.end.set_xyz_step(target_.end.step.x + x, target_.end.step.y + y, 
    target_.end.step.z + z);
}

bool move_gripper_target_m(double x, double y, double z)
{
  /* adjust gripper target by the indicated distances in metres */

  // return false if the target is outside motor limits
  return target_.end.set_xyz_m(target_.end.x + x, target_.end.y + y,
    target_.end.z + z);
}

bool move_gripper_target_m_rad(double x, double th, double z)
{
  /* adjust the gripper joint values */

  // return false if the target is outside motor limits
  return target_.end.set_xyz_m_rad(target_.end.x + x, target_.end.th + th, 
    target_.end.z + z);
}

bool move_base_target_m(double x, double y, double z)
{
  /* move the base target in x, y, z */

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
  constexpr gfloat E = 200e9;
  constexpr gfloat I = (28e-3 * std::pow(0.9e-3, 3)) / 12.0;
  gfloat P = k * (3 * E * I);

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
    joint_values(i) = 
      data->qpos[j_.idx.finger[i + finger * j_.num.per_finger]];
  }

  // next convert this into X and Y coordinates
  arma::vec cumulative(j_.num.per_finger, arma::fill::zeros);
  arma::mat finger_xy(j_.num.per_finger + 1, 2, arma::fill::zeros);

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
  constexpr gfloat E = 200e9;
  constexpr gfloat I = (28e-3 * std::pow(0.9e-3, 3)) / 12.0;
  gfloat P = k * (3 * E * I);

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
  for (int i = 0; i < num_points; i++) {
    float y = 0.0;
    for (int j = 0; j < num_coeff; j++) {
      y += vec_coefficients[j] * std::pow(vec_joint_x[i], j_.gauge.order - j);
    }
    float error = vec_joint_y[i] - y;
    vec_errors[i] = error;
  }

  /* ----- end read/verify differences ----- */

  return P;
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

} // namespace luke