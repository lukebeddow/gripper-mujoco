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

  // set the simulation timestep
  model->opt.timestep = s_.mujoco_timestep;

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

  // what sample function will we use for sampling regular sensor data
  switch (s_.sensor_sample_mode) {
    case MjType::Sample::raw: {
      sampleFcnPtr = &MjType::Sensor::raw_sample;
      break;
    }
    case MjType::Sample::change: {
      sampleFcnPtr = &MjType::Sensor::change_sample;
      break;
    }
    case MjType::Sample::average: {
      sampleFcnPtr = &MjType::Sensor::average_sample;
      break;
    }
    default: {
      throw std::runtime_error("s_.sensor_sample_mode not set to legal value");
    }
  }

  // what sample function will we use for sampling state data
  switch (s_.state_sample_mode) {
    case MjType::Sample::raw: {
      stateFcnPtr = &MjType::Sensor::raw_sample;
      break;
    }
    case MjType::Sample::change: {
      stateFcnPtr = &MjType::Sensor::change_sample;
      break;
    }
    case MjType::Sample::average: {
      stateFcnPtr = &MjType::Sensor::average_sample;
      break;
    }
    default: {
      throw std::runtime_error("s_.state_sample_mode not set to legal value");
    }
  }

  // // update the goal given the settings
  // goal_.goal_reward = s_.goal_reward;
  // goal_.binary_goal = s_.binary_goal_vector;

  // enforce HER goals to trigger at 1 always
  default_goal_event_triggering();

  // safety check
  if (s_.motor_state_sensor.read_rate >= 0)
    throw std::runtime_error("motor_state_sensor read_rate must be a negative number");
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

  if (s_.debug)
    std::cout << "Loading xml at path: " << model_path << '\n';

  // load the model from an XML file
  char error[500] = "";
  model = mj_loadXML(model_path.c_str(), 0, error, 500);

  if (not model) {
    std::cout << "MjClass load model error when trying to load file: "
      << model_path << '\n';
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

  // reset sensor saved data
  finger1_gauge.reset();
  finger2_gauge.reset();
  finger3_gauge.reset();
  palm_sensor.reset();
  finger1_axial_gauge.reset();
  finger2_axial_gauge.reset();
  finger3_axial_gauge.reset();
  wrist_X_sensor.reset();
  wrist_Y_sensor.reset();
  wrist_Z_sensor.reset();
  x_motor_position.reset();
  y_motor_position.reset();
  z_motor_position.reset();

  // reset timestamps for sensor readings
  gauge_timestamps.reset();
  axial_timestamps.reset();
  palm_timestamps.reset();
  wristXY_timestamps.reset();
  wristZ_timestamps.reset();

  // reset sensor last read times
  s_.bending_gauge.reset();
  s_.axial_gauge.reset();
  s_.palm_sensor.reset();
  s_.wrist_sensor_XY.reset();
  s_.wrist_sensor_Z.reset();
  s_.motor_state_sensor.reset();

  // reset data structures
  env_.reset();
  MjType::TestReport blank_report;
  testReport_ = blank_report;

  // empty any curve validation data
  if (s_.curve_validation) {
    curve_validation_data_.entries.clear();
  }

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

  // check for new sensor data
  monitor_sensors();

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

void MjClass::monitor_sensors()
{
  /* check all the sensors and take readings if possible */

  bool retrieved_forces = false;
  luke::Forces forces;

  // check the bending strain gauges
  if (s_.bending_gauge.ready_to_read(data->time)) {

    // read
    std::vector<luke::gfloat> gauges = luke::get_gauge_data(model, data);

    // normalise
    gauges[0] = s_.bending_gauge.apply_normalisation(gauges[0]);
    gauges[1] = s_.bending_gauge.apply_normalisation(gauges[1]);
    gauges[2] = s_.bending_gauge.apply_normalisation(gauges[2]);

    // save
    finger1_gauge.add(gauges[0]);
    finger2_gauge.add(gauges[1]);
    finger3_gauge.add(gauges[2]);

    // record time
    gauge_timestamps.add(data->time);

    // whilst testing: are we validating finger curvature?
    if (s_.curve_validation) validate_curve();
  }

  // check the axial strain gauges
  if (s_.axial_gauge.ready_to_read(data->time)) {

    if (not retrieved_forces) {
      forces = luke::get_object_forces(model, data);
      retrieved_forces = true;
    }

    // read
    std::vector<luke::gfloat> axial_gauges {
      (luke::gfloat)forces.all.finger1_local[0],
      (luke::gfloat)forces.all.finger2_local[0],
      (luke::gfloat)forces.all.finger3_local[0]
    };

    // normalise
    axial_gauges[0] = s_.axial_gauge.apply_normalisation(axial_gauges[0]);
    axial_gauges[1] = s_.axial_gauge.apply_normalisation(axial_gauges[1]);
    axial_gauges[2] = s_.axial_gauge.apply_normalisation(axial_gauges[2]);

    // save
    finger1_axial_gauge.add(axial_gauges[0]);
    finger2_axial_gauge.add(axial_gauges[1]);
    finger3_axial_gauge.add(axial_gauges[2]);

    // record time
    axial_timestamps.add(data->time);
  }

  // check the palm sensor
  if (s_.palm_sensor.ready_to_read(data->time)) {

    if (not retrieved_forces) {
      forces = luke::get_object_forces(model, data);
      retrieved_forces = true;
    }

    // read
    luke::gfloat palm_reading = forces.all.palm_local[0];

    // normalise
    palm_reading = s_.axial_gauge.apply_normalisation(palm_reading);

    // save
    palm_sensor.add(palm_reading);

    // record time
    palm_timestamps.add(data->time);
  }

  // check the wrist sensor XY force
  if (s_.wrist_sensor_XY.ready_to_read(data->time)) {

    // read
    luke::gfloat x = data->userdata[0];
    luke::gfloat y = data->userdata[1];

    // normalise
    x = s_.wrist_sensor_XY.apply_normalisation(x);
    y = s_.wrist_sensor_XY.apply_normalisation(y);

    // save
    wrist_X_sensor.add(x);
    wrist_Y_sensor.add(y);
    
    // record time
    wristXY_timestamps.add(data->time);
    
  }

  // check the wrist sensor Z force
  if (s_.wrist_sensor_Z.ready_to_read(data->time)) {

    // read
    luke::gfloat z = data->userdata[2];

    // normalise
    z = s_.wrist_sensor_Z.apply_normalisation(z);

    // save
    wrist_Z_sensor.add(z);

    // record time
    wristZ_timestamps.add(data->time);
  }
}

void MjClass::sense_gripper_state()
{
  /* save the gripper xyz motor state position */

  // get position we think each motor should be (NOT luke::get_gripper_state(data)!)
  std::vector<luke::gfloat> xyz_pos = luke::get_target_state();

  // normalise { x, y, z } joint values
  xyz_pos[0] = normalise_between(
    xyz_pos[0], luke::Gripper::xy_min, luke::Gripper::xy_max);
  xyz_pos[1] = normalise_between(
    xyz_pos[1], luke::Gripper::xy_min, luke::Gripper::xy_max);
  xyz_pos[2] = normalise_between(
    xyz_pos[2], luke::Gripper::z_min, luke::Gripper::z_max);

  // save reading
  x_motor_position.add(xyz_pos[0]);
  y_motor_position.add(xyz_pos[1]);
  z_motor_position.add(xyz_pos[2]);
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

  // calculate finger and palm force magnitudes on the object
  float finger1_force_mag = env_.obj.finger1_force.magnitude3();
  float finger2_force_mag = env_.obj.finger2_force.magnitude3();
  float finger3_force_mag = env_.obj.finger3_force.magnitude3();
  float palm_force_mag = env_.obj.palm_force.magnitude3();
  float ground_force_mag = env_.obj.ground_force.magnitude3();

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

  /* ----- detect state of binary events ----- */

  // another step has been made
  env_.cnt.step_num.value = true;

  // lifted is true if ground force is 0 and lift distance is exceeded
  if (ground_force_mag < ftol)
    env_.cnt.lifted.value = true;

  // check if the object has gone out of bounds
  if (env_.obj.qpos.x > s_.oob_distance or env_.obj.qpos.x < -s_.oob_distance or
      env_.obj.qpos.y > s_.oob_distance or env_.obj.qpos.y < -s_.oob_distance)
    env_.cnt.oob.value = true;

  // check if the object has been dropped (env_.cnt.lifted must already be set)
  env_.cnt.dropped.value = 
    // if lastdropped==false, newlifted==false, and lastlifted==true, set dropped=1
    ((not env_.cnt.dropped.row * not env_.cnt.lifted.value * env_.cnt.lifted.row) ? 1
    // else if lifted==true -> set dropped=0
    : (env_.cnt.lifted.value ? 0 
    // else if lastdropped==true +=1 to it, otherwise -> set dropped=0
    : (env_.cnt.dropped.row ? env_.cnt.dropped.row + 1 : 0)));

  // lifted above the target height and not oob (env_.cnt.oob must be set)
  if (env_.obj.qpos.z > env_.start_qpos.z + s_.done_height
      and not env_.cnt.oob.value)
    env_.cnt.target_height.value = true;

  // detect any gripper contact with the object
  if (finger1_force_mag > ftol or
      finger2_force_mag > ftol or
      finger3_force_mag > ftol or
      palm_force_mag > ftol)
    env_.cnt.object_contact.value = true;

  // check if object is stable (must also be lifted and env_.cnt.lifted set)
  if (finger1_force_mag > s_.stable_finger_force and
      finger2_force_mag > s_.stable_finger_force and
      finger3_force_mag > s_.stable_finger_force and
      palm_force_mag > s_.stable_palm_force and env_.cnt.lifted.value)
    env_.cnt.object_stable.value = true;

  // if stable and lifted to target (need env_.cnt.object_stable and target_height set)
  if (env_.cnt.object_stable.value and env_.cnt.target_height.value)
    env_.cnt.stable_height.value = true;

  /* ----- detect state of linear events (also save reward scaled value) ----- */

  env_.cnt.exceed_axial.value = env_.grp.peak_finger_axial_force;
  env_.cnt.exceed_lateral.value = env_.grp.peak_finger_lateral_force;
  env_.cnt.palm_force.value = env_.obj.palm_axial_force * env_.cnt.lifted.value; // must be lifted
  env_.cnt.exceed_palm.value = env_.obj.palm_axial_force;
  env_.cnt.finger_force.value = env_.obj.avg_finger_force;
  
  // testing for linear goals
  env_.cnt.finger1_force.value = finger1_force_mag;
  env_.cnt.finger2_force.value = finger2_force_mag;
  env_.cnt.finger3_force.value = finger3_force_mag;
  env_.cnt.ground_force.value = ground_force_mag;

  // update the counts of these events
  update_events(env_.cnt, s_);

  if (s_.debug) env_.cnt.print();

  // // for testing
  // std::cout << "Testing EventTrack\n";
  // luke::print_vec(env_.cnt.vectorise(), "vectorise");
  // MjType::EventTrack blank;
  // blank.unvectorise(env_.cnt.vectorise());
  // std::cout << "now unvectorise\n";
  // blank.print();
  // std::cout << "Reward from unvectorised\n";
  // float r = reward(blank);
  // std::cout << "The reward is " << r << '\n';
  
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

void MjClass::action_step()
{
  /* step until the simulation settles */

  if (s_.debug) tick();

  bool target_reached = false;
  bool target_step = false;

  for (int i = 0; i < s_.sim_steps_per_action; i++) {
    
    step();

    target_step = luke::is_target_step();
    target_reached = luke::is_target_reached();
  }

  if (s_.debug)
    std::cout << "action_step() complete after " << s_.sim_steps_per_action
      << " steps\n";

  // save the new gripper position
  sense_gripper_state();

  // track the object and environment
  update_env();
  
  env_.num_action_steps += 1;

  if (s_.debug) std::cout << "time for action_step() was " << tock() << " seconds\n";

  return;
}

void MjClass::set_action(int action)
{
  /* sets an action in the simulation, but does not step at all */

  bool wl = true; // within limits

  int action_code = action_options[action];

  // // for testing
  // luke::print_vec(action_options, "action options");
  // std::cout << "The action code is " << action_code << '\n';

  if (s_.debug) std::cout << "Action number " << action << " received, name = ";

  switch (action_code) {

    case MjType::Action::x_motor_positive:
      if (s_.debug) std::cout << "x_motor_positive";
      wl = luke::move_gripper_target_step(s_.action_motor_steps, 0, 0);
      break;
    case MjType::Action::x_motor_negative:
      if (s_.debug) std::cout << "x_motor_negative";
      wl = luke::move_gripper_target_step(-s_.action_motor_steps, 0, 0);
      break;

    case MjType::Action::prismatic_positive:
      if (s_.debug) std::cout << "prismatic_positive";
      wl = luke::move_gripper_target_step(s_.action_motor_steps, s_.action_motor_steps, 0);
      break;
    case MjType::Action::prismatic_negative:
      if (s_.debug) std::cout << "prismatic_negative";
      wl = luke::move_gripper_target_step(-s_.action_motor_steps, -s_.action_motor_steps, 0);
      break;

    case MjType::Action::y_motor_positive:
      if (s_.debug) std::cout << "y_motor_positive";
      wl = luke::move_gripper_target_step(0, s_.action_motor_steps, 0);
      break;
    case MjType::Action::y_motor_negative:
      if (s_.debug) std::cout << "y_motor_negative";
      wl = luke::move_gripper_target_step(0, -s_.action_motor_steps, 0);
      break;

    case MjType::Action::z_motor_positive:
      if (s_.debug) std::cout << "z_motor_positive";
      wl = luke::move_gripper_target_step(0, 0, s_.action_motor_steps);
      break;
    case MjType::Action::z_motor_negative:
      if (s_.debug) std::cout << "z_motor_negative";
      wl = luke::move_gripper_target_step(0, 0, -s_.action_motor_steps);
      break;

    case MjType::Action::height_positive:
      if (s_.debug) std::cout << "height_positive";
      wl = luke::move_base_target_m(0, 0, s_.action_base_translation);
      break;
    case MjType::Action::height_negative:
      if (s_.debug) std::cout << "height_negative";
      wl = luke::move_base_target_m(0, 0, -s_.action_base_translation);
      break;

    default:
      std::cout << "Action value received is " << action_code << '\n';
      std::cout << "Number of actions is " << n_actions << '\n';
      throw std::runtime_error("MjClass::set_action() received out of bounds int");
   
  }

  // // save if we exceeded limits
  // bool exceed_limits = not wl;
  // env_.cnt.exceed_limits *= exceed_limits;  // wipe to zero if false
  // env_.cnt.exceed_limits += exceed_limits;  // increment by one if true

  if (s_.debug)
    std::cout << ", within_limits = " << wl << '\n';

  env_.cnt.exceed_limits.value = not wl;
}

bool MjClass::is_done()
{
  /* determine if an episode should end */

  // general and sensor settings not used
  #define XX(NAME, TYPE, VALUE)
  #define SS(NAME, USE, NORM, READRATE)

  /* do NOT use other fields than name as it will pull values from simsettings.h not s_,
     eg instead of using TRIGGER we need to use s_.NAME.trigger */
  #define BR(NAME, DONTUSE1, DONTUSE2, DONTUSE3)                                \
            if (s_.NAME.done and env_.cnt.NAME.row >= s_.NAME.done) {           \
              if (s_.debug) std::cout << "is_done() = true, "                   \
                << #NAME << " limit of " << s_.NAME.done << " exceeded\n";      \
              return true;                                                      \
            }                                                                   
   
  #define LR(NAME, DONTUSE1, DONTUSE2, DONTUSE3, DONTUSE4, DONTUSE5, DONTUSE6)  \
            if (s_.NAME.done and env_.cnt.NAME.row >= s_.NAME.done) {           \
              if (s_.debug) std::cout << "is_done() = true, "                   \
                << #NAME << " limit of " << s_.NAME.done << " exceeded\n";      \
              return true;                                                      \
            }                                                                   
            
    // run the macro to create the code
    LUKE_MJSETTINGS

  #undef XX
  #undef SS
  #undef BR
  #undef LR

  // the above macro produces code snippets equivalent to these below examples

  // if (s_.lifted.done and env_.cnt.lifted.row >= s_.lifted.done) {
  //   if (s_.debug) std::cout << "is_done() = true, "         
  //     << "lifted" << " limit of " << s_.lifted.done << " exceeded\n";  
  //   return true;
  // }

  // if (s_.oob.done and env_.cnt.oob.row >= s_.oob.done) {
  //   if (s_.debug) std::cout << "is_done() = true, "
  //     << "oob" << " limit of " << s_.oob.done << " exceeded)\n";
  //   return true;
  // }

  // if the cumulative reward drops below a given threshold
  if (env_.cumulative_reward < s_.quit_on_reward_below) {
    if (s_.debug) std::printf("Reward dropped below limit of %.3f, is_done() = true\n",
      s_.quit_on_reward_below);
    return true;
  }

  return false;
}

std::vector<luke::gfloat> MjClass::get_observation()
{
  /* get an observation with n samples from the gauges */

  std::vector<luke::gfloat> observation;

  // how much time has elapsed since the last state
  double time_per_step = model->opt.timestep * s_.sim_steps_per_action;

  // get bending strain gauge sensor output
  if (s_.bending_gauge.in_use) {

    // sample data
    std::vector<luke::gfloat> f1 = 
      (s_.bending_gauge.*sampleFcnPtr)(finger1_gauge, time_per_step);
    std::vector<luke::gfloat> f2 = 
      (s_.bending_gauge.*sampleFcnPtr)(finger2_gauge, time_per_step);
    std::vector<luke::gfloat> f3 = 
      (s_.bending_gauge.*sampleFcnPtr)(finger3_gauge, time_per_step);

    // insert data into observation output
    observation.insert(observation.end(), f1.begin(), f1.end());
    observation.insert(observation.end(), f2.begin(), f2.end());
    observation.insert(observation.end(), f3.begin(), f3.end());
  }

  // get axial strain gauge sensor output
  if (s_.axial_gauge.in_use) {

    // sample data
    std::vector<luke::gfloat> a1 = 
      (s_.axial_gauge.*sampleFcnPtr)(finger1_axial_gauge, time_per_step);
    std::vector<luke::gfloat> a2 = 
      (s_.axial_gauge.*sampleFcnPtr)(finger2_axial_gauge, time_per_step);
    std::vector<luke::gfloat> a3 = 
      (s_.axial_gauge.*sampleFcnPtr)(finger3_axial_gauge, time_per_step);

    // insert data into observation output
    observation.insert(observation.end(), a1.begin(), a1.end());
    observation.insert(observation.end(), a2.begin(), a2.end());
    observation.insert(observation.end(), a3.begin(), a3.end());
  }

  // get palm sensor output
  if (s_.palm_sensor.in_use) {

    // sample data
    std::vector<luke::gfloat> p1 = 
      (s_.palm_sensor.*sampleFcnPtr)(palm_sensor, time_per_step);

    // insert data into observation output
    observation.insert(observation.end(), p1.begin(), p1.end());
  }

  // get wrist sensor XY output
  if (s_.wrist_sensor_XY.in_use) {

    // sample data
    std::vector<luke::gfloat> wX =
      (s_.wrist_sensor_XY.*sampleFcnPtr)(wrist_X_sensor, time_per_step);
    std::vector<luke::gfloat> wY =
      (s_.wrist_sensor_XY.*sampleFcnPtr)(wrist_Y_sensor, time_per_step);

    // insert data into observation output
    observation.insert(observation.end(), wX.begin(), wX.end());
    observation.insert(observation.end(), wY.begin(), wY.end());
  }

  // get wrist sensor Z output
  if (s_.wrist_sensor_Z.in_use) {
    
    // sample data
    std::vector<luke::gfloat> wZ =
      (s_.wrist_sensor_XY.*sampleFcnPtr)(wrist_Z_sensor, time_per_step);

    // insert data into observation output
    observation.insert(observation.end(), wZ.begin(), wZ.end());
  }

  // get motor state output
  if (s_.motor_state_sensor.in_use) {

    // sample data
    std::vector<luke::gfloat> s1 = 
      (s_.motor_state_sensor.*stateFcnPtr)(x_motor_position, time_per_step);
    std::vector<luke::gfloat> s2 = 
      (s_.motor_state_sensor.*stateFcnPtr)(y_motor_position, time_per_step);
    std::vector<luke::gfloat> s3 = 
      (s_.motor_state_sensor.*stateFcnPtr)(z_motor_position, time_per_step);

    // insert data into observation output
    observation.insert(observation.end(), s1.begin(), s1.end());
    observation.insert(observation.end(), s2.begin(), s2.end());
    observation.insert(observation.end(), s3.begin(), s3.end());
  }
  
  return observation;
}

std::vector<float> MjClass::get_event_state()
{
  /* get the full state of the simulation */

  return env_.cnt.vectorise();
}

std::vector<float> MjClass::get_goal()
{
  /* get a vector form of the current goal*/

  std::vector<float> goal_vec = goal_.vectorise();

  if (goal_vec.size() == 0) {
    throw std::runtime_error("MjClass::goal is empty, it can be set with"
      " MjClass.goal.<eventname>.involved = True\n");
  }

  return goal_vec;
}

std::vector<float> MjClass::assess_goal()
{
  /* assess how the current events fit with the desired goal */

  std::vector<float> event_vec = env_.cnt.vectorise();

  return assess_goal(event_vec);
}

std::vector<float> MjClass::assess_goal(std::vector<float> event_vec)
{
  /* assess which goals are accomplished given an event vector */

  MjType::Goal new_goal = score_goal(goal_, event_vec, s_);

  return new_goal.vectorise();
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

  float transition_reward = 0.0;

  // how are we calculating the reward
  if (s_.use_HER) {
    // transition_reward = goal_rewards(env_.cnt, s_, goal_);
    throw std::runtime_error("reward() was called with s_.use_HER = true, "
      "this is currently disabled, please called reward(goal, state) instead\n");
  }
  else {
    transition_reward = calc_rewards(env_.cnt, s_);
  }
   
  // useful for testing, this value is not used in python
  env_.cumulative_reward += transition_reward;

  // if we are capping the maximum cumulative negative reward
  if (env_.cumulative_reward < s_.quit_on_reward_below) {
    if (s_.quit_reward_capped) {
      // reduce the reward to not put us below the cap
      transition_reward += s_.quit_on_reward_below - env_.cumulative_reward - ftol;
      env_.cumulative_reward = s_.quit_on_reward_below - ftol;
    }
  }

  if (s_.debug)
    std::cout << "Cumulative cpp reward is " << env_.cumulative_reward << '\n';

  return transition_reward;
}

float MjClass::reward(std::vector<float> goal_vec, std::vector<float> event_vec)
{
  /* calculate the reward using vector representations */

  MjType::EventTrack event;
  event.unvectorise(event_vec);

  MjType::Goal goal(goal_);
  goal.unvectorise(goal_vec);

  float transition_reward = goal_rewards(event, s_, goal);

  // useful for testing, this value is not used in python
  env_.cumulative_reward += transition_reward;

  if (s_.debug)
    std::cout << "Cumulative cpp reward is " << env_.cumulative_reward << '\n';

  return transition_reward;
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

  int obs_size = get_observation().size();

  if (s_.use_HER) {
    int goal_size = get_goal().size();
    obs_size += goal_size;
  }

  return obs_size;
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
  testReport_.cumulative_reward = env_.cumulative_reward;
  testReport_.cnt = env_.cnt;

  return testReport_;
}

void MjClass::validate_curve()
{
  /* for testing the curvature of the fingers */

  // extract the finger data
  MjType::CurveFitData::PoseData pose;
  luke::verify_armadillo_gauge(data, 0,
    pose.f1.x, pose.f1.y, pose.f1.coeff, pose.f1.errors);
  luke::verify_armadillo_gauge(data, 1,
    pose.f2.x, pose.f2.y, pose.f2.coeff, pose.f2.errors);
  luke::verify_armadillo_gauge(data, 2,
    pose.f3.x, pose.f3.y, pose.f3.coeff, pose.f3.errors);
  // save
  curve_validation_data_.entries.push_back(pose);
}

MjType::EventTrack MjClass::add_events(MjType::EventTrack& e1, MjType::EventTrack& e2)
{
  /* add the absolute count and last value of two events, all else is ignored */

  MjType::EventTrack out;

  #define XX(NAME, TYPE, VALUE)
  #define SS(NAME, USED, NORMALISE, READ_RATE)

  #define BR(NAME, REWARD, DONE, TRIGGER)                                      \
            out.NAME.abs = e1.NAME.abs + e2.NAME.abs;                          \
            out.NAME.last_value = e1.NAME.last_value + e2.NAME.last_value;

  #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT)                 \
            out.NAME.abs = e1.NAME.abs + e2.NAME.abs;                          \
            out.NAME.last_value = e1.NAME.last_value + e2.NAME.last_value;  

    // run the macro to create the code
    LUKE_MJSETTINGS
    
  #undef XX
  #undef SS 
  #undef BR
  #undef LR

  return out;
}

void MjClass::reset_goal()
{
  /* wipe the desired goal completely */

  goal_.reset(true);
}

void MjClass::default_goal_event_triggering()
{
  /* set all goal event triggers to default */

  // this value should not be changed
  constexpr int default_trigger = 1;

  #define XX(NAME, TYPE, VALUE)
  #define SS(NAME, USED, NORMALISE, READ_RATE)

  #define BR(NAME, REWARD, DONE, TRIGGER)                                      \
            if (goal_.NAME.involved) { s_.NAME.trigger = default_trigger; }    

  #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT)                 \
            if (goal_.NAME.involved) { s_.NAME.trigger = default_trigger; }    

    // run the macro to create the code
    LUKE_MJSETTINGS
    
  #undef XX
  #undef SS 
  #undef BR
  #undef LR
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

  /* be aware when using macro fields other than name as it will pull values 
     from simsettings.h not s_, instead of using TRIGGER we need to use 
     s_.NAME.trigger */

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
            
  #define BR(NAME, DONTUSE1, DONTUSE2, DONTUSE3) \
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

  #define LR(NAME, DONTUSE1, DONTUSE2, DONTUSE3, DONTUSE4, DONTUSE5, DONTUSE6) \
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

  /* do NOT use other fields than name as it will pull values from simsettings.h not s_,
     eg instead of using TRIGGER we need to use s_.NAME.trigger */
  #define XX(NAME, TYPE, VALUE)
  #define SS(NAME, IN_USE, NORM, READRATE)
  #define BR(NAME, DONTUSE1, DONTUSE2, DONTUSE3) \
            NAME.reward = 0.0;\
            NAME.done = false;\
            NAME.trigger = never;
  #define LR(NAME, DONTUSE1, DONTUSE2, DONTUSE3, DONTUSE4, DONTUSE5, DONTUSE6) \
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

void MjType::Settings::scale_rewards(float scale)
{
  /* scale all of the rewards by a given value */

  /* do NOT use other fields than name as it will pull values from simsettings.h not s_,
     eg instead of using TRIGGER we need to use s_.NAME.trigger */
  #define XX(NAME, TYPE, VALUE)
  #define SS(NAME, IN_USE, NORM, READRATE)
  #define BR(NAME, DONTUSE1, DONTUSE2, DONTUSE3) NAME.reward *= scale;
  #define LR(NAME, DONTUSE1, DONTUSE2, DONTUSE3, DONTUSE4, DONTUSE5, DONTUSE6) \
            NAME.reward *= scale;
  
    // run the macro and scale the rewards
    LUKE_MJSETTINGS
  
  #undef XX
  #undef SS
  #undef BR
  #undef LR
}

void MjType::EventTrack::print()
{
  /* print out the event track information */

  calculate_percentage();

  std::cout << "EventTrack = row (abs); "

  #define XX(NAME, TYPE, VALUE)
  #define SS(NAME, USED, NORMALISE, READ_RATE)

  #define BR(NAME, REWARD, DONE, TRIGGER)                               \
            << #NAME << " = " << NAME.row << " (" << NAME.abs << "); "
  #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT)          \
            << #NAME << " = " << NAME.row << " (" << NAME.abs << ", " << NAME.last_value << "); "

    // run the macro to create the code
    LUKE_MJSETTINGS << "\n";

  #undef XX
  #undef SS
  #undef BR
  #undef LR

}

void update_events(MjType::EventTrack& events, MjType::Settings& settings)
{
  /* update the count of each event and reset recent event information */

  bool active = false; // is a linear reward active

  #define XX(NAME, TYPE, VALUE)
  #define SS(NAME, USED, NORMALISE, READ_RATE)
  #define BR(NAME, REWARD, DONE, TRIGGER)                                    \
            events.NAME.row = events.NAME.row *                              \
                                  events.NAME.value + events.NAME.value;     \
            events.NAME.abs += events.NAME.value;                            \
            events.NAME.last_value = events.NAME.value;                      \
            events.NAME.value = false; // reset

  #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT)               \
            active = false;                                                  \
            if (events.NAME.value > settings.NAME.min and                    \
                (events.NAME.value < settings.NAME.overshoot or              \
                 settings.NAME.overshoot < 0))                               \
              { active = true; }                                             \
            events.NAME.row = events.NAME.row * active + active;             \
            events.NAME.abs += active;                                       \
            events.NAME.last_value = events.NAME.value;                      \
            events.NAME.value = 0.0; // reset

    // run the macro to create the code
    LUKE_MJSETTINGS

  #undef XX
  #undef SS
  #undef BR
  #undef LR
}

float calc_rewards(MjType::EventTrack& events, MjType::Settings& settings)
{
  /* calculate the reward of one transition based on the simulation events */

  float reward = 0;

  // general and sensor settings not used
  #define XX(NAME, TYPE, VALUE)
  #define SS(NAME, USE, NORM, READRATE)

  /* do NOT use other fields than name as it will pull values from simsettings.h,
     eg instead of using TRIGGER we need to use settings.NAME.trigger */
  #define BR(NAME, DONTUSE1, DONTUSE2, DONTUSE3)                                \
            if (events.NAME.row >= settings.NAME.trigger) {                     \
              if (settings.debug)                                               \
                std::printf("%s triggered, reward += %.4f\n",                   \
                  #NAME, settings.NAME.reward);                                 \
              reward += settings.NAME.reward;                                   \
            }
        
  #define LR(NAME, DONTUSE1, DONTUSE2, DONTUSE3, DONTUSE4, DONTUSE5, DONTUSE6)  \
            if (events.NAME.row >= settings.NAME.trigger) {                     \
              float fraction = linear_reward(events.NAME.last_value,            \
                settings.NAME.min, settings.NAME.max, settings.NAME.overshoot); \
              float scaled_reward = settings.NAME.reward * fraction;            \
              if (settings.debug)                                               \
                std::printf("%s triggered by value %.1f, reward += %.4f\n",     \
                  #NAME, events.NAME.last_value, settings.NAME.reward);         \
              reward += scaled_reward;                                          \
            }
            
    // run the macro to create the code
    LUKE_MJSETTINGS

  #undef XX
  #undef SS
  #undef BR
  #undef LR

  /* example of binary reward snippet from above macro */
  // if (events.row.step_num >= settings.step_num.trigger) {
  //   if (settings.debug)                                                       
  //     std::printf("%s triggered, reward += %.4f\n", "step_num", settings.step_num.reward);
  //   reward += settings.step_num.reward;
  // }

  /* example of linear reward snippet from above macro */
  // if (events.row.palm_force >= settings.palm_force.trigger) {
  //   float fraction = linear_reward(settings.palm_force.value, settings.palm_force.min,
  //     settings.palm_force.max, settings.palm_force.overshoot);
  //   float scaled_reward = settings.palm_force.reward * fraction;
  //   if (settings.debug)
  //     std::printf("%s triggered by value %.1f, reward += %.4f\n",
  //       "palm_force", settings.palm_force.value, settings.palm_force.reward); 
  //   reward += scaled_reward;
  // }

  if (settings.debug) {
    std::cout << "Transition reward is: " << reward << '\n';
  }

  return reward;
}

float goal_rewards(MjType::EventTrack& events, MjType::Settings& settings,
  MjType::Goal goal)
{
  /* calculate rewards based on goals in HER */

  float reward = 0;
  float goal_reward = settings.goal_reward;

  // is the reward evenly split between multiple goals
  if (settings.divide_goal_reward) {
    goal_reward /= goal.vectorise().size();
  }

  if (settings.debug) {
    std::cout << "Reward given per goal is: " << goal_reward << '\n';
    std::cout << "Goal performance is: "; goal.print_verbose();
  }

  // general and sensor settings not used
  #define XX(NAME, TYPE, VALUE)
  #define SS(NAME, USE, NORM, READRATE)

  /* do NOT use other fields than name as it will pull values from simsettings.h,
     eg instead of using TRIGGER we need to use settings.NAME.trigger */
  #define BR(NAME, DONTUSE1, DONTUSE2, DONTUSE3)                                \
            if (events.NAME.row >= settings.NAME.trigger                        \
                and goal.NAME.involved) {                                       \
              if (settings.debug)                                               \
                std::printf("%s triggered, reward += %.4f\n",                   \
                  "goal: " #NAME, goal_reward);                                 \
              reward += goal_reward;                                            \
            }                                                               
        
  #define LR(NAME, DONTUSE1, DONTUSE2, DONTUSE3, DONTUSE4, DONTUSE5, DONTUSE6)  \
            if (events.NAME.row >= settings.NAME.trigger                        \
                and goal.NAME.involved) {                                       \
              if (settings.binary_goal_vector) {                                \
                if (settings.debug)                                             \
                  std::printf("%s triggered, reward += %.4f\n",                 \
                    "goal: " #NAME, goal_reward);                               \
                reward += goal_reward;                                          \
              }                                                                 \
              else {                                                            \
                float fraction = linear_reward(events.NAME.last_value,          \
                    settings.NAME.min, settings.NAME.max,                       \
                    settings.NAME.overshoot);                                   \
                float reward_to_give = fraction * goal_reward;                  \
                if (settings.debug)                                             \
                  std::printf("%s triggered with value %.1f, reward += %.4f\n", \
                    "goal: " #NAME, events.NAME.last_value, reward_to_give);    \
                reward += reward_to_give;                                       \
              }                                                                 \
            }         
            
    // run the macro to create the code
    LUKE_MJSETTINGS

  #undef XX
  #undef SS
  #undef BR
  #undef LR

  if (settings.debug) {
    std::cout << "Goal transition reward is: " << reward << '\n';
  }

  return reward;
}

std::vector<float> MjType::EventTrack::vectorise()
{
  /* turn the state into a vector */

  std::vector<float> out;

  #define XX(NAME, TYPE, VALUE)
  #define SS(NAME, USED, NORMALISE, READ_RATE)

  #define BR(NAME, REWARD, DONE, TRIGGER)                                      \
            out.push_back(NAME.row);

  #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT)                 \
            out.push_back(NAME.row);                                           \
            out.push_back(NAME.last_value);  

    // run the macro to create the code
    LUKE_MJSETTINGS
    
  #undef XX
  #undef SS 
  #undef BR
  #undef LR

  return out;
}

void MjType::EventTrack::unvectorise(std::vector<float> in)
{
  /* fill in event track details from an input vector. Check to make sure this
  matches with vectorise() */

  // reset ourselves first, to prevent mixed data
  reset();

  if (in.size() != vectorise().size()) {
    throw std::runtime_error("EventTrack::unvectorise() input not correct length"
      ", check to make sure the right vector has been passed");
  }

  int i = 0;

  #define XX(NAME, TYPE, VALUE)
  #define SS(NAME, USED, NORMALISE, READ_RATE)

  #define BR(NAME, REWARD, DONE, TRIGGER)                                      \
            NAME.row = in[i] + 0.5; /* casts float -> int */                   \
            i++;

  #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT)                 \
            NAME.row = in[i] + 0.5; /* casts float -> int */                   \
            i++;                                                               \
            NAME.last_value = in[i];                                           \
            i++;                                                               

    // run the macro to create the code
    LUKE_MJSETTINGS
    
  #undef XX
  #undef SS 
  #undef BR
  #undef LR

}

std::vector<float> MjType::Goal::vectorise() const
{
  /* return a vector of the goal state, which must map from [-1,+1]*/

  std::vector<float> out;

  #define XX(NAME, TYPE, VALUE)
  #define SS(NAME, USED, NORMALISE, READ_RATE)

  #define BR(NAME, REWARD, DONE, TRIGGER)                                      \
            if (NAME.involved) {                                               \
              if (NAME.state > 1) out.push_back(1);                            \
              else if (NAME.state < -1) out.push_back(-1);                     \
              else out.push_back(NAME.state);                                  \
            }

  #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT)                 \
            if (NAME.involved) {                                               \
              if (NAME.state > 1) out.push_back(1);                            \
              else if (NAME.state < -1) out.push_back(-1);                     \
              else out.push_back(NAME.state);                                  \
            }

    // run the macro to create the code
    LUKE_MJSETTINGS
    
  #undef XX
  #undef SS 
  #undef BR
  #undef LR

  return out;
}

void MjType::Goal::unvectorise(std::vector<float> vec)
{
  /* fill in the goal object with the vector values */

  if (vec.size() != vectorise().size()) {
    throw std::runtime_error("Goal::unvectorise() input not correct length"
      ", check to make sure the right vector has been passed");
  }

  int i = 0;

  #define XX(NAME, TYPE, VALUE)
  #define SS(NAME, USED, NORMALISE, READ_RATE)

  #define BR(NAME, REWARD, DONE, TRIGGER)                                      \
            if (NAME.involved) {                                               \
              NAME.state = vec[i];                                             \
              i++;                                                             \
            }

  #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT)                 \
            if (NAME.involved) {                                               \
              NAME.state = vec[i];                                             \
              i++;                                                             \
            }

    // run the macro to create the code
    LUKE_MJSETTINGS
    
  #undef XX
  #undef SS 
  #undef BR
  #undef LR
}

void MjType::Goal::print()
{
  /* print the goal state vector */

  luke::print_vec(vectorise(), "Goal vector");
}

void MjType::Goal::print_verbose()
{
  /* print a goal including the name of each field */

  std::vector<float> values = vectorise();
  std::vector<std::string> names = goal_names();

  int num = values.size() - 1;

  std::cout << "Goal = value; ";

  for (int i = 0; i < num; i++) {
    std::cout << names[i] << " = " << values[i] << "; ";
  }

  if (num > 0) {
    std::cout << names[num] << " = " << values[num] << "\n";
  }
  else {
    std::cout << "no active goals\n";
  }
}

MjType::Goal score_goal(MjType::Goal const goal, std::vector<float> event_vec, 
  MjType::Settings settings)
{
  /* change the goal to fit with the observed events */

  MjType::EventTrack event;
  event.unvectorise(event_vec);

  return score_goal(goal, event, settings);
}

MjType::Goal score_goal(MjType::Goal const goal, MjType::EventTrack event, 
  MjType::Settings settings)
{
  /* change the goal to fit with the observed events */

  MjType::Goal new_goal;

  #define XX(NAME, TYPE, VALUE)
  #define SS(NAME, USED, NORMALISE, READ_RATE)

  #define BR(NAME, REWARD, DONE, TRIGGER)                                      \
            if (goal.NAME.involved) {                                          \
              new_goal.NAME.involved = true;                                   \
              if (event.NAME.row >= settings.NAME.trigger) {                   \
                new_goal.NAME.state = 1.0;                                     \
              }                                                                \
              else {                                                           \
                new_goal.NAME.state = -1.0;                                    \
              }                                                                \
            }                                                                  \
            else {                                                             \
              new_goal.NAME.involved = false;                                  \
              new_goal.NAME.state = -1.0;                                      \
            }

  #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT)                 \
            if (goal.NAME.involved) {                                          \
              new_goal.NAME.involved = true;                                   \
              if (event.NAME.row >= settings.NAME.trigger) {                   \
                if (settings.binary_goal_vector) {                             \
                  new_goal.NAME.state = 1.0;                                   \
                }                                                              \
                else {                                                         \
                  float fraction = linear_reward(event.NAME.last_value,        \
                    settings.NAME.min, settings.NAME.max,                      \
                    settings.NAME.overshoot);                                  \
                  /* map [0,1] to [-1,1] */                                    \
                  new_goal.NAME.state = (2 * fraction - 1);                    \
                }                                                              \
              }                                                                \
              else {                                                           \
                new_goal.NAME.state = -1.0;                                    \
              }                                                                \
            }                                                                  \
            else {                                                             \
              new_goal.NAME.involved = false;                                  \
              new_goal.NAME.state = -1.0;                                      \
            }

    // run the macro to create the code
    LUKE_MJSETTINGS

  #undef XX
  #undef SS
  #undef BR
  #undef LR

  return new_goal;
}

std::vector<std::string> MjType::Goal::goal_names()
{
  /* get the names of the currently active goals */

  std::vector<std::string> goal_names;

  #define XX(NAME, TYPE, VALUE)
  #define SS(NAME, USED, NORMALISE, READ_RATE)

  #define BR(NAME, REWARD, DONE, TRIGGER)                                      \
            if (NAME.involved) { goal_names.push_back(#NAME); }

  #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT)                 \
            if (NAME.involved) { goal_names.push_back(#NAME); }

    // run the macro to create the code
    LUKE_MJSETTINGS

  #undef XX
  #undef SS
  #undef BR
  #undef LR

  return goal_names;
}

std::string MjType::Goal::get_goal_info()
{
  /* get information about which goals are active */

  std::string goal_info = "HER goal uses the following events { ";

  int num = vectorise().size();
  int i = 0;

  #define XX(NAME, TYPE, VALUE)
  #define SS(NAME, USED, NORMALISE, READ_RATE)

  #define BR(NAME, REWARD, DONE, TRIGGER)                                      \
            if (NAME.involved) {                                               \
              goal_info += #NAME;                                              \
              i++;                                                             \
              if (i < num) { goal_info += ", "; }                              \
            }

  #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT)                 \
            if (NAME.involved) {                                               \
              goal_info += #NAME;                                              \
              i++;                                                             \
              if (i < num) { goal_info += ", "; }                              \
            }

    // run the macro to create the code
    LUKE_MJSETTINGS

  #undef XX
  #undef SS
  #undef BR
  #undef LR

  goal_info += " }\n";

  return goal_info;
}

// end