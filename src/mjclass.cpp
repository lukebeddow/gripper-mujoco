#include "mjclass.h"

/* ----- constructors, destructor, and initialisers ----- */

MjClass::MjClass()
{
  /* constructor */

  model = NULL;
  data = NULL;
}

MjClass::MjClass(std::string file_path)
{
  /* constructor */

  model = NULL;
  data = NULL;

  load(file_path);
}

MjClass::MjClass(mjModel* m, mjData* d)
{
  /* constructor */

  model = m;
  data = d;

  init();
}

MjClass::MjClass(MjType::Settings settings_to_use)
{
  /* initialise with predefined settings */

  model = NULL;
  data = NULL;

  s_ = settings_to_use;
}

MjClass::~MjClass()
{
  /* destructor */

  // free any existing data
  if (data) mj_deleteData(data);
  if (model) mj_deleteModel(model);
}

void MjClass::init()
{
  /* initialise everything after load() has been called */

  if (not model or not data) {
    throw std::runtime_error("init() has been called before loading a model");
  }

  // initialise keyframe positions, joint, and object settings
  luke::init(model, data);

  // reset to double check everything is ready
  reset();
  
  // get the objects available for grasping
  env_.object_names = luke::get_objects();
}

void MjClass::init(mjModel* m, mjData* d)
{
  // free any existing data
  if (model) mj_deleteModel(model);
  if (data) mj_deleteData(data);

  // assign new pointers
  model = m;
  data = d;

  // note that no loadpath was used
  current_load_path = "init(m, d) used - no load path";
  
  init();
}

void MjClass::configure_settings()
{
  /* apply simulation settings */

  /* check what actions are set */
  action_options.clear();
  action_options.resize(MjType::Action::count, -1);

  // what actions are valid - MUST be same order as action enums
  int i = 0;
  if (not s_.paired_motor_X_step) {
    action_options[i] = MjType::Action::x_motor_positive;
    action_options[i + 1] = MjType::Action::x_motor_negative;
    i += 2;
  }
  else {
    action_options[i] = MjType::Action::prismatic_positive;
    action_options[i + 1] = MjType::Action::prismatic_negative;
    i += 2;
  }
  if (true) {
    action_options[i] = MjType::Action::y_motor_positive;
    action_options[i + 1] = MjType::Action::y_motor_negative;
    i += 2;
  }
  if (s_.use_palm_action) {
    action_options[i] = MjType::Action::z_motor_positive;
    action_options[i + 1] = MjType::Action::z_motor_negative;
    i += 2;
  }
  if (s_.use_height_action) {
    action_options[i] = MjType::Action::height_positive;
    action_options[i + 1] = MjType::Action::height_negative;
    i += 2;
  }

  n_actions = i;

}

/* ----- core functionality ----- */

void MjClass::load(std::string model_path)
{
  /* load a model into the simulation from an xml path */

  // free any existing data
  if (data) mj_deleteData(data);
  if (model) mj_deleteModel(model);
  data = NULL;
  model = NULL;

  // load the model from an XML file
  char error[500] = "";
  model = mj_loadXML(model_path.c_str(), 0, error, 500);

  if (not model) {
    mju_error_s("Load model error: %s", error);
  }

  data = mj_makeData(model);

  // save the loadpath used
  current_load_path = model_path;

  init();
}

void MjClass::load_relative(std::string relative_path)
{
  /* load a model with a relative path, using compiled defaults */

  if (model_folder_path != "" and object_set_name != "") {

    if (model_folder_path.back() != '/') {
      model_folder_path += "/";
    }

    if (relative_path[0] != '/') {
      relative_path = "/" + relative_path;
    }

    load(model_folder_path + object_set_name + relative_path);
  }
  else {
    throw std::runtime_error(
      "Cannot use MjClass::load_relative() as LUKE_MJCF_PATH or LUKE_DEFAULTOBJECTS not set"
    );
  }
}

void MjClass::reset()
{
  /* reset the simulation back to the start */

  // reset the simulation
  luke::reset(model, data);

  // reset this class to defaults
  timeout_count = 0;
  last_read_time = 0.0;
  finger1_gauge.reset();
  finger2_gauge.reset();
  finger3_gauge.reset();
  gauge_timestamps.reset();
  palm_sensor.reset();
  env_.reset();

  // reset the test report
  MjType::TestReport blank_report;
  testReport_ = blank_report;

  // // empty any curve validation data
  // if (s_.curve_validation) {
  //   curve_validation_data_.entries.clear();
  // }

  // ensure the simulation settings are all ready to go
  configure_settings();
}

void MjClass::step()
{
  /* step the simulation forwards once */

  // tick();

  luke::before_step(model, data);
  luke::step(model, data);
  luke::after_step(model, data);

  // check for new gauge data
  monitor_gauges();

  if (s_.render_on_step) {
    render();
  }

  // std::cout << "Time for MjClass::step " << tock() << '\n';

}

#if defined(LUKE_CLUSTER)

bool MjClass::render() 
{
  std::cout << "Rendering disabled on cluster\n";
  return false;
}

#else

bool MjClass::render()
{
  /* Render a frame of the simulation to the screen */

  // safety catch, we are unable to close the window properly
  static bool window_closed = false;
  if (window_closed) {
    return false;
  }

  // if the render window has not yet been initialised
  if (not render_init) {
    render::init(model, data);
    render_init = true;
  }

  bool window_open = true;

  // if we are rendering for a set period of time
  if (s_.use_render_delay) {

    auto start_time = time_::now();

    // render repeatedly
    while (window_open and std::chrono::duration_cast<std::chrono::milliseconds>
      (time_::now() - start_time).count() < s_.render_delay * 1000) { 
      
      window_open = render::render(model, data);
    }
  }
  else {
    // just render once
    window_open = render::render(model, data);
  }

  // if the window has been closed
  if (not window_open) {
    render::finish();
    render_init = false;
    window_closed = true;
  }
  
  return window_open;
}

#endif

/* ----- sensing ----- */

bool MjClass::monitor_gauges()
{
  /* check with set rate whether gauges have new data */

  // read and save the gauge data
  double time_between_reads = 1.0 / s_.gauge_read_rate_hz;

  if (data->time > last_read_time + time_between_reads) {

    std::vector<luke::gfloat> gauges = read_gauges();

    finger1_gauge.add(gauges[0]);
    finger2_gauge.add(gauges[1]);
    finger3_gauge.add(gauges[2]);

    if (s_.use_palm_sensor) 
      palm_sensor.add(read_palm());

    gauge_timestamps.add(data->time);
    last_read_time = data->time;

    // // for testing the curve validation
    // if (s_.curve_validation) {
    //   // extract the finger data
    //   MjType::CurveFitData::PoseData pose;
    //   luke::verify_armadillo_gauge(data, 0,
    //     pose.f1.x, pose.f1.y, pose.f1.coeff, pose.f1.errors);
    //   luke::verify_armadillo_gauge(data, 1,
    //     pose.f2.x, pose.f2.y, pose.f2.coeff, pose.f2.errors);
    //   luke::verify_armadillo_gauge(data, 2,
    //     pose.f3.x, pose.f3.y, pose.f3.coeff, pose.f3.errors);
    //   // save
    //   curve_validation_data_.entries.push_back(pose);
    // }

    return true;
  }

  return false;
}

std::vector<luke::gfloat> MjClass::read_gauges()
{
  /* read the strain gauges of each finger, readings only arrive with set Hz */

  std::vector<luke::gfloat> gauges = luke::get_gauge_data(model, data);

  // scale and clip the data to fall between -1 and 1
  for (int i = 0; i < 3; i++) {

    gauges[i] = gauges[i] / s_.normalising_force;

    if (gauges[i] > 1) gauges[i] = 1;
    else if (gauges[i] < -1) gauges[i] = -1;

  }

  return gauges;
}

luke::gfloat MjClass::read_palm()
{
  /* read from the palm sensor */

  if (s_.use_palm_sensor) {

    luke::gfloat palm_reading = luke::get_palm_force(model, data);

    // bumper sensor
    if (s_.palm_force_normalise < 0) {
      if (palm_reading > ftol) {
        palm_reading = 1;
      }
      else {
        palm_reading = -1;
      }
    }

    // force sensor
    else {
      palm_reading = palm_reading / s_.palm_force_normalise;
      if (palm_reading > 1) palm_reading = 1;
      else if (palm_reading < -1) palm_reading = -1;
    }

    return palm_reading;
  }
  else {
    return 0.0;
  }
}

std::vector<luke::gfloat> MjClass::get_gripper_state()
{
  /* get the state of the joints */

  // old: get the true joint positions
  // return luke::get_gripper_state(data);

  // new: get the target joint positions (ie what we hope they are)
  return luke::get_target_state();
}

bool MjClass::is_target_reached()
{
  /* have we reached the target state - note this suffers from steady state error */

  return luke::is_target_reached();
}

bool MjClass::is_settled()
{
  /* is the gripper in equilibrium */

  return luke::is_settled();
}

void MjClass::update_env()
{
  /* track the position and state of the object */

  /* ----- get information from simulation ----- */

  // get information about the object from the simluation
  env_.obj.qpos = luke::get_object_qpos();
  env_.grp.target = luke::get_gripper_target();
  luke::Forces forces = luke::get_object_forces(model, data);
  
  // // for testing
  // forces.print();

  // save the forces on the object in local frames (from gripper perspective)
  env_.obj.finger1_force = forces.obj.finger1_local;
  env_.obj.finger2_force = forces.obj.finger2_local;
  env_.obj.finger3_force = forces.obj.finger3_local;
  env_.obj.palm_force = forces.obj.palm_local;
  env_.obj.ground_force = forces.obj.ground;

  // save forces on the gripper fingers (include all contacts, not just object)
  env_.grp.finger1_force = forces.all.finger1_local;
  env_.grp.finger2_force = forces.all.finger2_local;
  env_.grp.finger3_force = forces.all.finger3_local;

  // calculate finger and palm force magnitudes
  float finger1_force_mag = env_.obj.finger1_force.magnitude3();
  float finger2_force_mag = env_.obj.finger2_force.magnitude3();
  float finger3_force_mag = env_.obj.finger3_force.magnitude3();
  float palm_force_mag = env_.obj.palm_force.magnitude3();

  // get palm force on object (x = axial in local frame, +ve for compression)
  env_.obj.palm_axial_force = +1 * env_.obj.palm_force[0];

  // get average finger force on object
  env_.obj.avg_finger_force = 0.333 * (finger1_force_mag + finger2_force_mag
    + finger3_force_mag);

  // get the highest axial finger force (x = axial in local frame, -ve for comp.)
  env_.grp.peak_finger_axial_force = -1 * std::min({ 
    forces.gnd.finger1_local[0], forces.gnd.finger2_local[0], forces.gnd.finger3_local[0]
  });

  // get highest outwards lateral force on finger
  env_.grp.peak_finger_lateral_force = -1 * std::min({
    forces.obj.finger1_local[1], forces.obj.finger2_local[1], forces.obj.finger3_local[1]
  });

  /* ----- detect state of key events ----- */

  // lifted is true if ground force is 0 and lift distance is exceeded
  bool lifted = false;
  if (env_.obj.ground_force.magnitude3() < ftol)
    lifted = true;

  // check if the object has gone out of bounds
  bool out_of_bounds = false;
  if (env_.obj.qpos.x > s_.oob_distance or env_.obj.qpos.x < -s_.oob_distance or
      env_.obj.qpos.y > s_.oob_distance or env_.obj.qpos.y < -s_.oob_distance)
    out_of_bounds = true;

  // has the object been lifted above the target height and not oob
  bool target_height = false;
  if (env_.obj.qpos.z > env_.start_qpos.z + s_.height_target
      and not out_of_bounds)
    target_height = true;

  // check if the finger limit axial force is exceeded
  bool exceed_axial = false;
  if (env_.grp.peak_finger_axial_force > s_.exceed_axial.min)
    exceed_axial = true;

  // check if the finger lateral force limit is exceeded
  bool exceed_lateral = false;
  if (env_.grp.peak_finger_lateral_force > s_.exceed_lateral.min)
    exceed_lateral = true;

  // detect if we are in a good palm force range (must be lifted)
  bool palm_force = false;
  if (env_.obj.palm_axial_force > s_.palm_force.min and
      env_.obj.palm_axial_force < s_.palm_force.overshoot and lifted)
    palm_force = true;

  // detect if we exceed safe limits for palm force
  bool exceed_palm = false;
  if (env_.obj.palm_axial_force > s_.exceed_palm.min)
    exceed_palm = true;

  // detect any contact with the object
  bool object_contact = false;
  if (finger1_force_mag > ftol or
      finger2_force_mag > ftol or
      finger3_force_mag > ftol or
      palm_force_mag > ftol)
    object_contact = true;

  // detect finger force on object
  bool finger_force = false;
  if (env_.obj.avg_finger_force > s_.finger_force.min)
    finger_force = true;

  // check if object is stable (must also be lifted)
  bool object_stable = false;
  if (finger1_force_mag > s_.stable_finger_force and
      finger2_force_mag > s_.stable_finger_force and
      finger3_force_mag > s_.stable_finger_force and
      palm_force_mag > s_.stable_palm_force and lifted)
    object_stable = true;

  bool stable_height = false;
  if (object_stable and target_height)
    stable_height = true;

  /* ----- update count of events in a row ----- */

  env_.cnt.step_num += 1;

  // update dropped - this is a complex boolean expression
  // if dropped==false, lifted==false, and cnt.lifted==true, set dropped=1
  env_.cnt.dropped = ((not env_.cnt.dropped * not lifted * env_.cnt.lifted) ? 1
    // otherwise, if lifted==true, set dropped=0, else true+=1 and false=0
    : (lifted ? 0 : (env_.cnt.dropped ? env_.cnt.dropped + 1 : 0)));

  // update the rest, if=0 do nothing, if!=0, increment by 1
  env_.cnt.lifted = env_.cnt.lifted * lifted + lifted;
  env_.cnt.oob = env_.cnt.oob * out_of_bounds + out_of_bounds;
  env_.cnt.target_height = env_.cnt.target_height * target_height + target_height;
  env_.cnt.object_stable = env_.cnt.object_stable * object_stable + object_stable;
  env_.cnt.exceed_axial = env_.cnt.exceed_axial * exceed_axial + exceed_axial;
  env_.cnt.exceed_lateral = env_.cnt.exceed_lateral * exceed_lateral + exceed_lateral;
  env_.cnt.palm_force = env_.cnt.palm_force * palm_force + palm_force;
  env_.cnt.object_contact = env_.cnt.object_contact * object_contact + object_contact;
  env_.cnt.exceed_palm = env_.cnt.exceed_palm * exceed_palm + exceed_palm;
  env_.cnt.finger_force = env_.cnt.finger_force * finger_force + finger_force;
  env_.cnt.stable_height = env_.cnt.stable_height * stable_height + stable_height;
  // env_.cnt.exceed_limits is set in set_action()

  if (s_.debug) { std::cout << "cnt: "; env_.cnt.print(); }

  /* ----- update absolute count of events ----- */

  env_.abs_cnt.step_num = env_.cnt.step_num;
  if (env_.cnt.lifted) env_.abs_cnt.lifted += 1;
  if (env_.cnt.oob) env_.abs_cnt.oob += 1;
  if (env_.cnt.dropped) env_.abs_cnt.dropped += 1;
  if (env_.cnt.target_height) env_.abs_cnt.target_height += 1;
  if (env_.cnt.exceed_limits) env_.abs_cnt.exceed_limits += 1;
  if (env_.cnt.exceed_axial) env_.abs_cnt.exceed_axial += 1;
  if (env_.cnt.exceed_lateral) env_.abs_cnt.exceed_lateral += 1;
  if (env_.cnt.object_stable) env_.abs_cnt.object_stable += 1;
  if (env_.cnt.palm_force) env_.abs_cnt.palm_force += 1;
  if (env_.cnt.object_contact) env_.abs_cnt.object_contact += 1;
  if (env_.cnt.exceed_palm) env_.abs_cnt.exceed_palm += 1;
  if (env_.cnt.finger_force) env_.abs_cnt.finger_force += 1;
  if (env_.cnt.stable_height) env_.abs_cnt.stable_height += 1;

  if (s_.debug) { std::cout << "abs_cnt: "; env_.abs_cnt.print(); }
  
  return;
}

/* ----- control ----- */

// these all return false if the target is outside the motor limits

bool MjClass::set_joint_target(double x, double th, double z)
{
  /* set a joint value target for the gripper */

  return luke::set_gripper_target_m_rad(x, th, z);
}

bool MjClass::set_motor_target(double x, double y, double z)
{
  /* set a motor position target for the gripper */

  return luke::set_gripper_target_m(x, y, z);
}

bool MjClass::set_step_target(int x, int y, int z)
{
  /* set a motor step target for the gripper */

  return luke::set_gripper_target_step(x, y, z);
}

bool MjClass::move_motor_target(double x, double y, double z)
{
  /* move the gripper motor position by xyz metres */

  return luke::move_gripper_target_m(x, y, z);
}

bool MjClass::move_joint_target(double x, double th, double z)
{
  /* move the gripper joint position by xz metres and th radians */

  return luke::move_gripper_target_m_rad(x, th, z);
}

bool MjClass::move_step_target(int x, int y, int z)
{
  /* move the gripper step position by xyz steps */

  return luke::move_gripper_target_step(x, y, z);
}

/* ----- learning functions ----- */

bool MjClass::action_step()
{
  /* step until the simulation settles */

  if (s_.debug) tick();

  // ensure the simulation is not settled to start with
  bool timeout = s_.use_settling;
  luke::wipe_settled();

  bool settled = false;
  bool target_reached = false;
  bool target_step = false;

  for (int i = 0; i < s_.max_action_steps; i++) {
    
    step();

    settled = luke::is_settled();
    target_step = luke::is_target_step();
    target_reached = luke::is_target_reached();

    // if the simulation is in steady state and target commands aren't changing
    if (s_.use_settling and settled and target_step) {
      timeout = false;
      if (s_.debug) std::cout << "action_step() settled after " << i << " steps\n";
      break;
    }
  }

  if (timeout and s_.debug and s_.use_settling) 
    std::cout << "action_step() timeout after limit of " 
      << s_.max_action_steps << " steps\n";
  else if (s_.debug and not s_.use_settling)
    std::cout << "action_step() complete after " << s_.max_action_steps
      << " steps\n";

  // track whether action step is settling, or timing out
  timeout_count = (timeout_count + 1) * timeout;

  // track the object and environment
  update_env();
  
  env_.num_action_steps += 1;

  if (s_.debug) std::cout << "time for action_step() was " << tock() << " seconds\n";

  return timeout;
}

void MjClass::set_action(int action)
{
  /* sets an action in the simulation, but does not step at all */

  bool wl = true; // within limits

  int action_code = action_options[action];

  // // testing::print
  // luke::print_vec(action_options, "action options");
  // std::cout << "The action code is " << action_code << '\n';

  switch (action_code) {

    case MjType::Action::x_motor_positive:
      wl = luke::move_gripper_target_step(s_.action_motor_steps, 0, 0);
      break;
    case MjType::Action::x_motor_negative:
      wl = luke::move_gripper_target_step(-s_.action_motor_steps, 0, 0);
      break;

    case MjType::Action::prismatic_positive:
      wl = luke::move_gripper_target_step(s_.action_motor_steps, s_.action_motor_steps, 0);
      break;
    case MjType::Action::prismatic_negative:
      wl = luke::move_gripper_target_step(-s_.action_motor_steps, -s_.action_motor_steps, 0);
      break;

    case MjType::Action::y_motor_positive:
      wl = luke::move_gripper_target_step(0, s_.action_motor_steps, 0);
      break;
    case MjType::Action::y_motor_negative:
      wl = luke::move_gripper_target_step(0, -s_.action_motor_steps, 0);
      break;

    case MjType::Action::z_motor_positive:
      wl = luke::move_gripper_target_step(0, 0, s_.action_motor_steps);
      break;
    case MjType::Action::z_motor_negative:
      wl = luke::move_gripper_target_step(0, 0, -s_.action_motor_steps);
      break;

    case MjType::Action::height_positive:
      wl = luke::move_base_target_m(0, 0, s_.action_base_translation);
      break;
    case MjType::Action::height_negative:
      wl = luke::move_base_target_m(0, 0, -s_.action_base_translation);
      break;

    default:
      std::cout << "Action value received is " << action_code << '\n';
      std::cout << "Number of actions is " << n_actions << '\n';
      throw std::runtime_error("MjClass::set_action() received out of bounds int");
   
  }

  // int num_steps = s_.action_motor_steps;
  // switch (action) {
  //   case 0:
  //     if (s_.paired_motor_X_step)
  //       wl = luke::move_gripper_target_step(num_steps, num_steps, 0);
  //     else
  //       wl = luke::move_gripper_target_step(num_steps, 0, 0);
  //     break;
  //   case 1:
  //     wl = luke::move_gripper_target_step(0, num_steps, 0);
  //     break;
  //   case 2:
  //     wl = luke::move_gripper_target_step(0, 0, num_steps);
  //     break;
  //   case 3:
  //     if (s_.paired_motor_X_step)
  //       wl = luke::move_gripper_target_step(-num_steps, -num_steps, 0);
  //     else
  //       wl = luke::move_gripper_target_step(-num_steps, 0, 0);
  //     break;
  //   case 4:
  //     wl = luke::move_gripper_target_step(0, -num_steps, 0);
  //     break;
  //   case 5:
  //     wl = luke::move_gripper_target_step(0, 0, -num_steps);
  //     break;
  //   case 6:
  //     wl = luke::move_base_target_m(0, 0, s_.action_base_translation);
  //     break;
  //   case 7:
  //     wl = luke::move_base_target_m(0, 0, -s_.action_base_translation);
  //     break;
  //   default:
  //     throw std::runtime_error("MjClass::set_action() received out of bounds int");
  // }

  // save if we exceeded limits
  bool exceed_limits = not wl;
  env_.cnt.exceed_limits *= exceed_limits;  // wipe to zero if false
  env_.cnt.exceed_limits += exceed_limits;  // increment by one if true
}

bool MjClass::is_done()
{
  /* determine if an episode should end */

  // if the object has been lifted for long enough
  if (s_.lifted.done and env_.cnt.lifted >= s_.lifted.done) {
    if (s_.debug) std::cout << "The has been lifted long enough, is_done() = true"
      << " (lifted limit of " << s_.lifted.done << " exceeded)\n";
    return true;
  }
  // if the object has been dropped
  if (s_.dropped.done and env_.cnt.dropped >= s_.dropped.done) {
    if (s_.debug) std::cout << "The object has been dropped, is_done() = true"
      << " (dropped limit of " << s_.dropped.done << " exceeded)\n";
    return true;
  }
  // if the object is out of bounds
  if (s_.oob.done and env_.cnt.oob >= s_.oob.done) {
    if (s_.debug) std::cout << "The object is out of bounds, is_done() = true"
      << " (oob limit of " << s_.oob.done << " exceeded)\n";
    return true;
  }
  // if the object has been lifted to the target height
  if (s_.target_height.done and env_.cnt.target_height >= s_.target_height.done) {
    if (s_.debug) std::cout << "Object has reached target height, is_done() = true"
      << " (target_height limit of " << s_.target_height.done << " exceeded)\n";
    return true;
  }
  // if the limits are exceeded
  if (s_.exceed_limits.done and env_.cnt.exceed_limits >= s_.exceed_limits.done) {
    if (s_.debug) std::cout << "Gripper limits exceeded, is_done() = true"
      << " (exceed_limits limit of " << s_.exceed_limits.done << " exceeded)\n";
    return true;
  }
  // if the finger axial force is too high
  if (s_.exceed_axial.done and env_.cnt.exceed_axial >= s_.exceed_axial.done) {
    if (s_.debug) std::cout << "Finger axial force too high, is_done() = true"
      << " (exceed_axial limit of " << s_.exceed_axial.done << " exceeded)\n";
    return true;
  }
  // if the finger lateral force is too high
  if (s_.exceed_lateral.done and env_.cnt.exceed_lateral >= s_.exceed_lateral.done) {
    if (s_.debug) std::cout << "Finger lateral force too high, is_done() = true"
      << " (exceed_lateral limit of " << s_.exceed_lateral.done << " exceeded)\n";
    return true;
  }
  // if the palm force is too high
  if (s_.exceed_palm.done and env_.cnt.exceed_palm >= s_.exceed_palm.done) {
    if (s_.debug) std::cout << "Palm force too high, is_done() = true"
      << " (exceed_palm limit of " << s_.exceed_palm.done << " exceeded)\n";
    return true;
  }
  // if object is stable
  if (s_.object_stable.done and env_.cnt.object_stable >= s_.object_stable.done) {
    if (s_.debug) std::cout << "Object stable long enough, is_done() = true"
      << " (object_stable limit of " << s_.object_stable.done << " exceeded)\n";
    return true;
  }
  // if object is stable and at target height
  if (s_.stable_height.done and env_.cnt.stable_height >= s_.stable_height.done) {
    if (s_.debug) std::cout << "Object stable and at target height, is_done() = true"
      << " (stable_height limit of " << s_.stable_height.done << " exceeded)\n";
    return true;
  }

  // step_num: not implemented, done should never be true
  // object_contact: not implemented, done should never be true
  // finger_force: not implemented, done should never be true
  // palm_force: not implemented, done should never be true

  // if the cumulative reward drops below a given threshold
  if (env_.cumulative_reward < s_.quit_on_reward_below) {
    if (s_.debug) std::printf("Reward dropped below limit of %.3f, is_done() = true\n",
      s_.quit_on_reward_below);
    return true;
  }

  // if the simulation is unstable (action_step() not settling with use_setting=true)
  if (timeout_count > s_.max_timeouts) {
    if (s_.debug) std::cout << "The max unsettled steps is exceeded, is_done() = true"
      << " (max_timeouts of " << s_.max_timeouts << " exceeded)\n";
    return true;
  }

  return false;
}

std::vector<luke::gfloat> MjClass::get_observation()
{
  /* overload */

  if (s_.obs_raw_data) {
    throw std::runtime_error("get_observation() needs the number of samples n for raw data");
  }

  return get_observation(0);
}

std::vector<luke::gfloat> MjClass::get_observation(int n)
{
  /* get an observation with n samples from the gauges */

  std::vector<luke::gfloat> sensor_output;
  std::vector<luke::gfloat> state_output;
  std::vector<luke::gfloat> observation;

  if (s_.obs_raw_data) {

    // get the raw observations from the gauges and the gripper state
    std::vector<luke::gfloat> f1 = finger1_gauge.read(n);
    std::vector<luke::gfloat> f2 = finger2_gauge.read(n);
    std::vector<luke::gfloat> f3 = finger3_gauge.read(n);

    sensor_output.reserve(3 * n);
    sensor_output.insert(sensor_output.end(), f1.begin(), f1.end());
    sensor_output.insert(sensor_output.end(), f2.begin(), f2.end());
    sensor_output.insert(sensor_output.end(), f3.begin(), f3.end());

    // get the gripper state
    state_output = get_gripper_state();

  }
  else {

    /* the default mujoco timestep is 0.002 seconds */
    constexpr double mujoco_timestep = 0.002;
    double time_per_step = mujoco_timestep * s_.max_action_steps;
    double readings_since_step = time_per_step * s_.gauge_read_rate_hz;
    // round to int (if between round up) to include the last reading
    int n_readings = std::ceil(readings_since_step);

    // create vector of pointers to iterate over
    std::vector<luke::SlidingWindow<luke::gfloat>*> data_ptrs;
    if (s_.use_palm_sensor) {
      data_ptrs = { &finger1_gauge, &finger2_gauge, &finger3_gauge, &palm_sensor };
    }
    else {
      data_ptrs = { &finger1_gauge, &finger2_gauge, &finger3_gauge };
    }

    // create the output vector, 3 elements for each sensor
    int data_per_sensor = 3;
    sensor_output.resize(data_per_sensor * data_ptrs.size());

    for (int i = 0; i < data_ptrs.size(); i++) {
      
      luke::gfloat old_reading = data_ptrs[i]->read_element(n_readings);
      luke::gfloat new_reading = data_ptrs[i]->read_element();
      luke::gfloat change = new_reading - old_reading;

      if (change > 1) change = 1;
      else if (change < -1) change = -1;

      // save the data for this sensor
      sensor_output[i * data_per_sensor + 0] = old_reading;
      sensor_output[i * data_per_sensor + 1] = change;
      sensor_output[i * data_per_sensor + 2] = new_reading;
    }

    // get the gripper state
    state_output = get_gripper_state();

    // normalise to the range -1, +1 for each state output
    for (int i = 0; i < 6; i++) {
      state_output[i] = normalise_between(
        state_output[i], luke::Gripper::xy_min, luke::Gripper::xy_max
      );
    }
    state_output[6] = normalise_between(state_output[6], luke::Gripper::z_min, luke::Gripper::z_max);
  }

  // finally, build the observation as state + sensor data  
  observation.reserve(sensor_output.size() + state_output.size());
  observation.insert(observation.end(), state_output.begin(), state_output.end());
  observation.insert(observation.end(), sensor_output.begin(), sensor_output.end());
  
  return observation;
}

void MjClass::reset_object()
{
  /* remove any object from the scene */

  luke::reset_object(model, data);
}

void MjClass::spawn_object(int index)
{
  /* overload, default x and y positions */

  constexpr double default_x = 0.0;
  constexpr double default_y = 0.0;

  spawn_object(index, default_x, default_y);
}

void MjClass::spawn_object(int index, double xpos, double ypos)
{
  /* spawn an object beneath the gripper */

  if (index < 0 or index >= env_.object_names.size()) {
    throw std::runtime_error("bad index to spawn_object()");
  }

  // save info on object to be spawned
  env_.obj.name = env_.object_names[index];

  // set the position to be spawned
  luke::QPos spawn_pos;
  spawn_pos.x = xpos;
  spawn_pos.y = ypos;
  spawn_pos.z = -1;     // will automatically be set to keyframe value

  // spawn the object and save its start position
  luke::spawn_object(model, data, index, spawn_pos);
  env_.start_qpos = luke::get_object_qpos();

  // update everything for rendering
  forward();
}

float MjClass::reward()
{
  /* calculate the reward available at the current simulation state */

  float reward = 0;

  /* ----- binary rewards ----- */

  // reward per step
  if (env_.cnt.step_num >= s_.step_num.trigger) {
    if (s_.debug) std::printf("Step made, reward += %.4f\n", s_.step_num.reward);
    reward += s_.step_num.reward;
  }

  // reward for object contact
  if (env_.cnt.object_contact >= s_.object_contact.trigger) {
    if (s_.debug) std::printf("Contact made, reward += %.4f\n", s_.object_contact.reward);
    reward += s_.object_contact.reward;
  }

  // is the object currently lifted
  if (env_.cnt.lifted >= s_.lifted.trigger) {
    if (s_.debug) std::printf("Object lifted, reward += %.4f\n", s_.lifted.reward);
    reward += s_.lifted.reward;
  }

  // is the object out of bounds
  if (env_.cnt.oob >= s_.oob.trigger) {
    if (s_.debug) std::printf("Object oob, reward += %.4f\n", s_.oob.reward);
    reward += s_.oob.reward;
  }
  
  // has the object reached the target height for the first time
  if (env_.cnt.target_height >= s_.target_height.trigger) {
    if (s_.debug) std::printf("Object reached target height, reward += %.4f\n", s_.target_height.reward);
    reward += s_.target_height.reward;
  }

  // is the object grasped stably (do we make sure this can only be applied once?)
  if (env_.cnt.object_stable >= s_.object_stable.trigger) {
    if (s_.debug) std::printf("Object grasped stably, reward += %.4f\n", s_.object_stable.reward);
    reward += s_.object_stable.reward;
  }

  // is the object grasped stably and at the target height
  if (env_.cnt.stable_height >= s_.stable_height.trigger) {
    if (s_.debug) std::printf("Object stable and at target height, reward += %.4f\n", s_.stable_height.reward);
    reward += s_.stable_height.reward;
  }

  // has the object been dropped
  if (env_.cnt.dropped >= s_.dropped.trigger) {
    if (s_.debug) std::printf("Object dropped, reward += %.4f\n", s_.dropped.reward);
    reward += s_.dropped.reward;
  }

  // has the gripper or base exceeded its limits
  if (env_.cnt.exceed_limits >= s_.exceed_limits.trigger) {
    if (s_.debug) std::printf("Limits exceeded, reward += %.4f\n", s_.exceed_limits.reward);
    reward += s_.exceed_limits.reward;
  }

  /* ----- linear rewards ----- */

  // reward based on achieving a target palm force (must be lifted)
  if (env_.cnt.palm_force >= s_.palm_force.trigger) {
    // linearly scale reward
    float fraction = linear_reward(env_.obj.palm_axial_force, s_.palm_force.min,
      s_.palm_force.max, s_.palm_force.overshoot);
    float r = s_.palm_force.reward * fraction;
    if (s_.debug) std::printf("Palm force of %.1f gets reward += %.4f\n",
      env_.obj.palm_axial_force, r);
    reward += r;
  }

  // reward for all fingers exerting force on the object
  if (env_.cnt.finger_force >= s_.finger_force.trigger) {
    // linearly scale reward
    float fraction = linear_reward(env_.obj.avg_finger_force, s_.finger_force.min,
      s_.finger_force.max, s_.finger_force.overshoot);
    float r = s_.finger_force.reward * fraction;
    if (s_.debug) std::printf("Finger force avg of %.1f gets reward += %.4f\n",
      env_.obj.avg_finger_force, r);
    reward += r;
  }

  // penalty for exceeding safe amount of palm force
  if (env_.cnt.exceed_palm >= s_.exceed_palm.trigger) {
    // linearly scale reward
    float fraction = linear_reward(env_.obj.palm_axial_force, s_.exceed_palm.min,
      s_.exceed_palm.max, s_.exceed_palm.overshoot);
    float r = s_.exceed_palm.reward * fraction;
    if (s_.debug) std::printf("Palm force of %.1f gets reward += %.4f\n",
      env_.obj.palm_axial_force, r);
    reward += r;
  }

  // penalty based on high axial force on the fingers
  if (env_.cnt.exceed_axial >= s_.exceed_axial.trigger) {
    // linearly scale reward
    float fraction = linear_reward(env_.grp.peak_finger_axial_force,
      s_.exceed_axial.min, s_.exceed_axial.max, s_.exceed_axial.overshoot);
    float r = s_.exceed_axial.reward * fraction;
    if (s_.debug) std::printf("Max finger axial force of %.1f gets reward += %.4f\n",
      env_.grp.peak_finger_axial_force, r);
    reward += r;
  }

  // penalty based on high lateral force on the fingers
  if (env_.cnt.exceed_lateral >= s_.exceed_lateral.trigger) {
    float fraction = linear_reward(env_.grp.peak_finger_lateral_force,
      s_.exceed_lateral.min, s_.exceed_lateral.max, s_.exceed_lateral.overshoot);
    float r = s_.exceed_lateral.reward * fraction;
    if (s_.debug) std::printf("Max finger lateral force of %.1f gets reward += %.4f\n",
      env_.grp.peak_finger_lateral_force, r);
    reward += r;
  }

  // useful for testing, this value is not used in python
  env_.cumulative_reward += reward;

  // if we are capping the maximum cumulative negative reward
  if (env_.cumulative_reward < s_.quit_on_reward_below) {
    if (s_.quit_reward_capped) {
      // reduce the reward to not put us below the cap
      reward += s_.quit_on_reward_below - env_.cumulative_reward - ftol;
      env_.cumulative_reward = s_.quit_on_reward_below - ftol;
    }
  }

  return reward;
}

int MjClass::get_n_actions()
{
  /* get the number of possible actions */
  
  // recheck the settings to ensure we get it right
  configure_settings();

  return n_actions;
}

int MjClass::get_n_obs()
{
  /* get the number of observations */

  return get_observation().size();
}

/* ----- misc ----- */

void MjClass::tick()
{
  /* start the clock */

  start_time_ = time_::now();
}

float MjClass::tock()
{
  /* returns seconds elapsed since last tick() call */

  float elapsed = (std::chrono::duration_cast<std::chrono::microseconds>
      (time_::now() - start_time_).count()) / 1e6;

  return elapsed;
}

MjType::TestReport MjClass::get_test_report()
{
  /* fills out and returns the test report */

  testReport_.object_name = env_.obj.name;
  testReport_.num_steps = env_.num_action_steps;
  testReport_.cumulative_reward = env_.cumulative_reward;

  testReport_.final_palm_force = env_.obj.palm_force.magnitude3();
  testReport_.final_finger_force = (env_.obj.finger1_force.magnitude3() +
    env_.obj.finger2_force.magnitude3() + env_.obj.finger3_force.magnitude3()) / 3;

  testReport_.final_cnt = env_.cnt;
  testReport_.abs_cnt = env_.abs_cnt;

  return testReport_;
}

/* ------ utility functions ----- */

float linear_reward(float val, float min, float max, float overshoot)
{
  /* returns a value from 0-1 depending on the linear interpolation of val
  between min and max. Overshoot specifies what should be done if val > max.
  overshoot = -1 means return 1.0, overshoot > max mean linearly interpolate
  downwards towards the overshoot value */

  if (val < min) return 0.0;
  if (val > max) {
    if (overshoot < max) {
      // saturate high
      return 1.0;
    }
    else {
      if (val > overshoot) {
        // saturate low
        return 0.0;
      }
      else {
        // reverse and interpolate to max
        min = 0;
        max = overshoot - max;
        val = overshoot - val;
      }
    }
  }

  return (val - min) / (max - min);
}

float normalise_between(float val, float min, float max)
{
  /* project val to the interval -1 +1 from the interval min max */

  if (val > max) return 1.0;
  else if (val < min) return -1.0;
  
  return 2 * (val - min) / (max - min) - 1;
}

std::string MjType::Settings::get_settings()
{
  /* this function returns a string detailing all of the settings */

  // strings we will use within our macros to hold data
  std::string output_str = "c++ simulation settings:\n\n";
  std::string str = "";
  std::string type_str = "";
  std::string name_str = "";
  std::string val_str = "";
  std::string pad = "";

  // define spacings so that values all line up
  constexpr int name_chars = 25;
  constexpr int val_chars = 5;
  constexpr int type_chars = 14;
  constexpr int float_bonus = 6;

  // create the column headers, type first
  type_str = "Type";
  pad.resize(type_chars - type_str.size(), ' ');
  str += type_str + pad;
  // next name
  name_str = "Name";
  pad.clear(); pad.resize(name_chars - name_str.size(), ' ');
  str += name_str + pad;
  // now values
  val_str = "Value/s";
  str += val_str + "\n";
  // now add headers to output
  output_str += str;

  // we will use our macro to build one large string
  #define XX(NAME, TYPE, VALUE) \
            str.clear();\
            /* type first */\
            type_str.clear(); type_str += #TYPE;\
            pad.clear(); pad.resize(type_chars - type_str.size(), ' ');\
            str += type_str + pad;\
            /* name next */\
            name_str.clear(); name_str += #NAME;\
            pad.clear(); pad.resize(name_chars - name_str.size(), ' ');\
            str += name_str + pad;\
            /* value last */\
            val_str.clear(); val_str += std::to_string((TYPE)NAME);\
            str += "{ " + val_str + " }\n";\
            /* add to output */\
            output_str += str;

  #define SS(NAME, IN_USE, NORM, READRATE) \
            str.clear();\
            /* type first */\
            type_str.clear(); type_str += "Sensor";\
            pad.clear(); pad.resize(type_chars - type_str.size(), ' ');\
            str += type_str + pad;\
            /* name next */\
            name_str.clear(); name_str += #NAME;\
            pad.clear(); pad.resize(name_chars - name_str.size(), ' ');\
            str += name_str + pad;\
            /* now values */\
            val_str.clear(); val_str += std::to_string((bool)NAME.in_use);\
            pad.clear(); pad.resize(val_chars - val_str.size(), ' ');\
            str += "{" + pad + val_str + ", ";\
            val_str.clear(); val_str += std::to_string((float)NAME.normalise);\
            pad.clear(); pad.resize(val_chars + float_bonus - val_str.size(), ' ');\
            str += pad + val_str + ", ";\
            val_str.clear(); val_str += std::to_string((float)NAME.read_rate);\
            pad.clear(); pad.resize(val_chars + float_bonus - val_str.size(), ' ');\
            str += pad + val_str + " }\n";\
            /* add to output */\
            output_str += str;
            
  #define BR(NAME, REWARD, DONE, TRIGGER) \
            str.clear();\
            /* type first */\
            type_str.clear(); type_str += "BinaryReward";\
            pad.clear(); pad.resize(type_chars - type_str.size(), ' ');\
            str += type_str + pad;\
            /* name next */\
            name_str.clear(); name_str += #NAME;\
            pad.clear(); pad.resize(name_chars - name_str.size(), ' ');\
            str += name_str + pad;\
            /* now values */\
            val_str.clear(); val_str += std::to_string((float)NAME.reward);\
            pad.clear(); pad.resize(val_chars + float_bonus - val_str.size(), ' ');\
            str += "{" + pad + val_str + ", ";\
            val_str.clear(); val_str += std::to_string((int)NAME.done);\
            pad.clear(); pad.resize(val_chars - val_str.size(), ' ');\
            str += pad + val_str + ", ";\
            val_str.clear(); val_str += std::to_string((int)NAME.trigger);\
            pad.clear(); pad.resize(val_chars - val_str.size(), ' ');\
            str += pad + val_str + " }\n";\
            /* add to output */\
            output_str += str;

  #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT) \
            str.clear();\
            /* type first */\
            type_str.clear(); type_str += "LinearReward";\
            pad.clear(); pad.resize(type_chars - type_str.size(), ' ');\
            str += type_str + pad;\
            /* name next */\
            name_str.clear(); name_str += #NAME;\
            pad.clear(); pad.resize(name_chars - name_str.size(), ' ');\
            str += name_str + pad;\
            /* now values */\
            val_str.clear(); val_str += std::to_string((float)NAME.reward);\
            pad.clear(); pad.resize(val_chars + float_bonus - val_str.size(), ' ');\
            str += "{" + pad + val_str + ", ";\
            val_str.clear(); val_str += std::to_string((int)NAME.done);\
            pad.clear(); pad.resize(val_chars - val_str.size(), ' ');\
            str += pad + val_str + ", ";\
            val_str.clear(); val_str += std::to_string((int)NAME.trigger);\
            pad.clear(); pad.resize(val_chars - val_str.size(), ' ');\
            str += pad + val_str + ", ";\
            val_str.clear(); val_str += std::to_string((float)NAME.min);\
            pad.clear(); pad.resize(val_chars + float_bonus - val_str.size(), ' ');\
            str += pad + val_str + ", ";\
            val_str.clear(); val_str += std::to_string((float)NAME.max);\
            pad.clear(); pad.resize(val_chars + float_bonus - val_str.size(), ' ');\
            str += pad + val_str + ", ";\
            val_str.clear(); val_str += std::to_string((float)NAME.overshoot);\
            pad.clear(); pad.resize(val_chars + float_bonus - val_str.size(), ' ');\
            str += pad + val_str + " }\n";\
            /* now add to output */\
            output_str += str;

    // now run the macros
    LUKE_MJSETTINGS
  #undef XX
  #undef SS
  #undef BR
  #undef LR

  return output_str;
}

void MjType::Settings::wipe_rewards()
{
  /* wipe all the reward settings to defaults which never trigger */

  int never = 10000;

  #define XX(NAME, TYPE, VALUE)
  #define SS(NAME, IN_USE, NORM, READRATE)
  #define BR(NAME, REWARD, DONE, TRIGGER) \
            NAME.reward = 0.0;\
            NAME.done = false;\
            NAME.trigger = never;
  #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT) \
            NAME.reward = 0.0;\
            NAME.done = false;\
            NAME.trigger = never;\
            NAME.min = 0.0;\
            NAME.max = 0.0;\
            NAME.overshoot = -1;
  
    // run the macro and wipe the rewards
    LUKE_MJSETTINGS
  
  #undef XX
  #undef SS
  #undef BR
  #undef LR
}

void MjType::EventTrack::print()
{
  
  #define XX(name, type, value)
  #define SS(name, in_use, normalise, readrate)
  #define BR(name, reward, done, trigger) << #name << " = " << name << "; "
  #define LR(name, reward, done, trigger, min, max, overshoot) \
            << #name << " = " << name << "; "
    
    // run the macro and print all the event track fields
    std::cout LUKE_MJSETTINGS << "\n";

  #undef XX
  #undef SS
  #undef BR
  #undef LR
}

// end