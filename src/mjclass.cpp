#include "mjclass.h"

// declare the random number generator
std::shared_ptr<std::default_random_engine> MjType::generator;

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

  // if we are randomising the colour of simulated objects
  if (s_.randomise_colours) {
    luke::randomise_all_colours(model, MjType::generator);
    randomise_ground_colour();
  }
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

  // set gauge calibrations
  calibrate_.offset.g1 = 0.70e6;
  calibrate_.offset.g2 = -0.60e6;
  calibrate_.offset.g3 = -0.56e6;
  calibrate_.scale.g1 = calibrate_.scale.g2 = calibrate_.scale.g3 = 1.258e-6;
  calibrate_.norm.g1 = calibrate_.norm.g2 = calibrate_.norm.g3 = 2;

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
    case MjType::Sample::median: {
      sampleFcnPtr = &MjType::Sensor::median_sample;
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
    case MjType::Sample::median: {
      stateFcnPtr = &MjType::Sensor::median_sample;
      break;
    }
    default: {
      throw std::runtime_error("s_.state_sample_mode not set to legal value");
    }
  }

  // enforce HER goals to trigger at 1 always
  default_goal_event_triggering();

  // update the sensor number of readings based on time per step
  double time_per_step = model->opt.timestep * s_.sim_steps_per_action;
  s_.update_sensor_settings(time_per_step);

  // update the finger spring stiffness
  luke::set_finger_stiffness(model, s_.finger_stiffness);

  // if we have a new random seed, create a new random generator
  static uint old_random_seed = s_.random_seed + 1;
  if (old_random_seed != s_.random_seed) {
    MjType::generator.reset(new std::default_random_engine(s_.random_seed));
    old_random_seed = s_.random_seed;
  }

  /* start of automatic settings changes */

  bool echo_auto_changes = true; // s_.debug;

  // if we have not initialised the automatic flags, do it now
  if (not resetFlags.flags_init) {
    resetFlags.auto_calibrate = s_.auto_calibrate_gauges;
    resetFlags.auto_simsteps = s_.auto_sim_steps;
    resetFlags.auto_timestep = s_.auto_set_timestep;
    resetFlags.flags_init = true;
    // resetFlags.some_set = resetFlags.all_done();
  }

  // find timestep automatically, this change must be done before calibrate_gauges()
  if (resetFlags.auto_timestep) {
    resetFlags.auto_timestep = false;                     // disable auto timestep immediately due to recursion
    resetFlags.auto_calibrate = false;                    // disable calibration before timestep found
    resetFlags.auto_simsteps = false;                     // disable simsteps before timestep found
    s_.mujoco_timestep = find_highest_stable_timestep();  // find the timestep, calls configure_settings() recursively
    resetFlags.auto_calibrate = s_.auto_calibrate_gauges; // re-enable calibration after timestep found
    resetFlags.auto_simsteps = s_.auto_sim_steps;         // re-enable simsteps after timestep found
    if (echo_auto_changes) std::cout << "MjClass auto-setting: Mujoco timestep set to: " << s_.mujoco_timestep << '\n';
  }

  // set the simulation timestep in mujoco
  model->opt.timestep = s_.mujoco_timestep;

  // calibrate the gauges, requires timestep to be stable
  if (resetFlags.auto_calibrate) {
    resetFlags.auto_calibrate = false;
    calibrate_gauges();
    if (echo_auto_changes) std::cout << "MjClass auto-setting: Bending gauge normalisation set to: " << s_.bending_gauge.normalise << '\n';

    // one additional reset() as fingers will be wobbling from the calibration
    reset();
  }

  // find the sim settings per action automatically, requires timestep to be finalised
  if (resetFlags.auto_simsteps) {
    resetFlags.auto_simsteps = false;
    s_.sim_steps_per_action = std::ceil(s_.time_for_action / s_.mujoco_timestep);
    if (echo_auto_changes) std::cout << "MjClass auto-setting: Sim steps per action set to: " << s_.sim_steps_per_action << '\n';
  }
  
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

  // we have loaded a new file
  render_reload = true;

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
  z_base_position.reset();

  // reset timestamps for sensor readings
  step_timestamps.reset();
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
  s_.base_state_sensor.reset();

  // reset data structures
  env_.reset();
  MjType::TestReport blank_report;
  testReport_ = blank_report;

  // reset variables for use with real gripper
  samples_since_last_obs = 0;

  // empty any curve validation data
  curve_validation_data_.reset();

  // ensure the simulation settings are all ready to go
  configure_settings();
}

void MjClass::hard_reset()
{
  /* a complete reset, use this when you want to load a new model file
  which has a different number of gripper joints eg a different number
  of segments */

  // reinitialise the joint settings structure
  luke::init_J(model, data);

  // reset the tip force
  luke::apply_tip_force(model, data, 0, true);

  // we want to reset the auto setting flags to original values
  resetFlags.flags_init = false;

  // regular reset code
  reset();
}

void MjClass::step()
{
  /* step the simulation forwards once */

  // tick();

  luke::before_step(model, data);

  // if doing curve validation, can apply a set tip force
  if (s_.curve_validation < 0) {
    luke::apply_tip_force(model, data, -1 * s_.curve_validation);
  }

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
  else if (render_reload) {
    render::reload_for_rendering(model, data);
  }

  // init and reload perform the same job, so we no longer need to reload
  render_reload = false;

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
  luke::Forces_faster forces;

  // check the bending strain gauges
  if (s_.bending_gauge.ready_to_read(data->time)) {

    // read
    std::vector<luke::gfloat> gauges = luke::get_gauge_data(model, data);

    // normalise
    gauges[0] = s_.bending_gauge.apply_normalisation(gauges[0]);
    gauges[1] = s_.bending_gauge.apply_normalisation(gauges[1]);
    gauges[2] = s_.bending_gauge.apply_normalisation(gauges[2]);

    // apply noise (can be gaussian based on sensor settings, if std_dev > 0)
    gauges[0] = s_.bending_gauge.apply_noise(gauges[0], uniform_dist);
    gauges[1] = s_.bending_gauge.apply_noise(gauges[1], uniform_dist);
    gauges[2] = s_.bending_gauge.apply_noise(gauges[2], uniform_dist);

    // save
    finger1_gauge.add(gauges[0]);
    finger2_gauge.add(gauges[1]);
    finger3_gauge.add(gauges[2]);

    // record time
    gauge_timestamps.add(data->time);
  }

  // check the axial strain gauges
  if (s_.axial_gauge.ready_to_read(data->time)) {

    if (not retrieved_forces) {
      forces = luke::get_object_forces_faster(model, data);
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

    // apply noise (can be gaussian based on sensor settings, if std_dev > 0)
    axial_gauges[0] = s_.axial_gauge.apply_noise(axial_gauges[0], uniform_dist);
    axial_gauges[1] = s_.axial_gauge.apply_noise(axial_gauges[1], uniform_dist);
    axial_gauges[2] = s_.axial_gauge.apply_noise(axial_gauges[2], uniform_dist);

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
      forces = luke::get_object_forces_faster(model, data);
      retrieved_forces = true;
    }

    // read
    luke::gfloat palm_reading = forces.all.palm_local[0];

    // normalise
    palm_reading = s_.palm_sensor.apply_normalisation(palm_reading);

    // apply noise (can be gaussian based on sensor settings, if std_dev > 0)
    palm_reading = s_.palm_sensor.apply_noise(palm_reading, uniform_dist);

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

    // apply noise (can be gaussian based on sensor settings, if std_dev > 0)
    x = s_.wrist_sensor_XY.apply_noise(x, uniform_dist);
    y = s_.wrist_sensor_XY.apply_noise(y, uniform_dist);

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

    // apply noise (can be gaussian based on sensor settings, if std_dev > 0)
    z = s_.wrist_sensor_Z.apply_noise(z, uniform_dist);

    // save
    wrist_Z_sensor.add(z);

    // record time
    wristZ_timestamps.add(data->time);
  }
}

void MjClass::sense_gripper_state()
{
  /* save the end target state position of the gripper and base */

  // get position we think each motor should be (NOT luke::get_gripper_state(data)!)
  std::vector<luke::gfloat> state_vec = luke::get_target_state();

  // normalise { x, y, z } joint values
  state_vec[0] = normalise_between(
    state_vec[0], luke::Gripper::xy_min, luke::Gripper::xy_max);
  state_vec[1] = normalise_between(
    state_vec[1], luke::Gripper::xy_min, luke::Gripper::xy_max);
  state_vec[2] = normalise_between(
    state_vec[2], luke::Gripper::z_min, luke::Gripper::z_max);
  state_vec[3] = normalise_between(
    state_vec[3], luke::Target::base_z_min, luke::Target::base_z_max);

  // apply noise (can be gaussian based on sensor settings, if std_dev > 0)
  state_vec[0] = s_.motor_state_sensor.apply_noise(state_vec[0], uniform_dist);
  state_vec[1] = s_.motor_state_sensor.apply_noise(state_vec[1], uniform_dist);
  state_vec[2] = s_.motor_state_sensor.apply_noise(state_vec[2], uniform_dist);
  state_vec[3] = s_.base_state_sensor.apply_noise(state_vec[3], uniform_dist);

  // save reading
  x_motor_position.add(state_vec[0]);
  y_motor_position.add(state_vec[1]);
  z_motor_position.add(state_vec[2]);
  z_base_position.add(state_vec[3]);

  // save the time the reading was made
  step_timestamps.add(data->time);
}

void MjClass::update_env()
{
  /* Update tracking of the simulation environement to determine whether events
  have occured. These events are how reward() and is_done() determine progress.
  
  This function is split into four parts:

    1. extract information from the simulation, forces, positions, etc into the env_ variable
    2. determine if binary events have triggered eg is the object lifted up
    3. input values for linear events eg what is the palm force
    4. resolve all events and counts, done automatically

  To add a new event, edit part 1. to calculate the values needed to judge your
  event. Then add logic to get the value of your event in part 2. if binary or
  part 3. if linear. Done! Ensure your event is also added in simsettings.h
  The rest is done automatically by macros.

  */

  /* ----- get information from simulation (EDIT here to get extra information) ----- */

  // get information about the object from the simluation
  env_.obj.qpos = luke::get_object_qpos(model, data);
  env_.grp.target = luke::get_gripper_target();
  luke::Forces_faster forces = luke::get_object_forces_faster(model, data);
  
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

  /* ----- detect state of binary events (EDIT here to add a binary event) ----- */

  // another step has been made
  env_.cnt.step_num.value = true;

  // lifted is true if both the object and gripper have zero axial force from the ground
  if (ground_force_mag < ftol and 
      env_.grp.peak_finger_axial_force < ftol)
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

  /* ----- input state value of linear events (EDIT here to add a linear event) ----- */

  env_.cnt.exceed_axial.value = env_.grp.peak_finger_axial_force;
  env_.cnt.exceed_lateral.value = env_.grp.peak_finger_lateral_force;
  env_.cnt.palm_force.value = env_.obj.palm_axial_force * env_.cnt.lifted.value; // must be lifted
  env_.cnt.exceed_palm.value = env_.obj.palm_axial_force;
  env_.cnt.finger_force.value = env_.obj.avg_finger_force;
  
  // testing: track info for linear goals
  env_.cnt.finger1_force.value = finger1_force_mag;
  env_.cnt.finger2_force.value = finger2_force_mag;
  env_.cnt.finger3_force.value = finger3_force_mag;
  env_.cnt.ground_force.value = ground_force_mag;

  /* ----- resolve linear events and update counts of all events (no editing needed) ----- */

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

std::vector<float> MjClass::set_action(int action)
{
  /* sets an action in the simulation, but does not step at all. Returns the
  new state vector */

  bool wl = true; // within limits

  int action_code = action_options[action];

  // // for testing
  // luke::print_vec(action_options, "action options");
  // std::cout << "The action code is " << action_code << '\n';

  if (s_.debug) std::cout << "Action number " << action << " received, name = ";

  switch (action_code) {

    case MjType::Action::x_motor_positive:
      if (s_.debug) std::cout << "x_motor_positive";
      if (s_.XYZ_action_mm_rad) 
        wl = luke::move_gripper_target_m(-s_.X_action_mm * 1e-3, 0, 0); // -ve since home is 134mm and end is 50mm
      else
        wl = luke::move_gripper_target_step(s_.action_motor_steps, 0, 0);
      break;
    case MjType::Action::x_motor_negative:
      if (s_.debug) std::cout << "x_motor_negative";
      if (s_.XYZ_action_mm_rad) 
        wl = luke::move_gripper_target_m(s_.X_action_mm * 1e-3, 0, 0);
      else
        wl = luke::move_gripper_target_step(-s_.action_motor_steps, 0, 0);
      break;

    case MjType::Action::prismatic_positive:
      if (s_.debug) std::cout << "prismatic_positive";
      if (s_.XYZ_action_mm_rad) 
        wl = luke::move_gripper_target_m_rad(-s_.X_action_mm * 1e-3, 0, 0);
      else
        wl = luke::move_gripper_target_step(s_.action_motor_steps, s_.action_motor_steps, 0);
      break;
    case MjType::Action::prismatic_negative:
      if (s_.debug) std::cout << "prismatic_negative";
      if (s_.XYZ_action_mm_rad) 
        wl = luke::move_gripper_target_m_rad(s_.X_action_mm * 1e-3, 0, 0);
      else
        wl = luke::move_gripper_target_step(-s_.action_motor_steps, -s_.action_motor_steps, 0);
      break;

    case MjType::Action::y_motor_positive:
      if (s_.debug) std::cout << "y_motor_positive";
      if (s_.XYZ_action_mm_rad) 
        wl = luke::move_gripper_target_m_rad(0, -s_.Y_action_rad, 0); // -ve as angle convention is swapped
      else 
        wl = luke::move_gripper_target_step(0, s_.action_motor_steps, 0);
      break;
    case MjType::Action::y_motor_negative:
      if (s_.debug) std::cout << "y_motor_negative";
      if (s_.XYZ_action_mm_rad) 
        wl = luke::move_gripper_target_m_rad(0, s_.Y_action_rad, 0);
      else
        wl = luke::move_gripper_target_step(0, -s_.action_motor_steps, 0);
      break;

    case MjType::Action::z_motor_positive:
      if (s_.debug) std::cout << "z_motor_positive";
      if (s_.XYZ_action_mm_rad) 
        wl = luke::move_gripper_target_m_rad(0, 0, s_.Z_action_mm * 1e-3);
      else
        wl = luke::move_gripper_target_step(0, 0, s_.action_motor_steps);
      break;
    case MjType::Action::z_motor_negative:
      if (s_.debug) std::cout << "z_motor_negative";
      if (s_.XYZ_action_mm_rad) 
        wl = luke::move_gripper_target_m_rad(0, 0, -s_.Z_action_mm * 1e-3);
      else
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

  if (s_.debug)
    std::cout << ", within_limits = " << wl << '\n';

  env_.cnt.exceed_limits.value = not wl;

  // get the target state and return it
  return luke::get_target_state();
}

bool MjClass::is_done()
{
  /* determine if an episode should end */

  // general and sensor settings not used
  #define XX(NAME, TYPE, VALUE)
  #define SS(NAME, IN_USE, NORM, READRATE)

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

  // use for printing detailed observation debug information
  constexpr bool debug_obs = false;

  std::vector<luke::gfloat> observation;

  // // FOR TESTING
  // std::cout << "Last 10 gauge 1 readings: ";
  // finger1_gauge.print(10);

  if (debug_obs) {
    std::cout << "Observation information:\n";
  }

  // get bending strain gauge sensor output
  if (s_.bending_gauge.in_use) {

    // sample data
    std::vector<luke::gfloat> f1 = 
      (s_.bending_gauge.*sampleFcnPtr)(finger1_gauge);
    std::vector<luke::gfloat> f2 = 
      (s_.bending_gauge.*sampleFcnPtr)(finger2_gauge);
    std::vector<luke::gfloat> f3 = 
      (s_.bending_gauge.*sampleFcnPtr)(finger3_gauge);

    // insert data into observation output
    observation.insert(observation.end(), f1.begin(), f1.end());
    observation.insert(observation.end(), f2.begin(), f2.end());
    observation.insert(observation.end(), f3.begin(), f3.end());

    if (debug_obs) {
      luke::print_vec(f1, "Bending gauge 1");
      luke::print_vec(f2, "Bending gauge 2");
      luke::print_vec(f3, "Bending gauge 3");
    }
  }

  // get axial strain gauge sensor output
  if (s_.axial_gauge.in_use) {

    // sample data
    std::vector<luke::gfloat> a1 = 
      (s_.axial_gauge.*sampleFcnPtr)(finger1_axial_gauge);
    std::vector<luke::gfloat> a2 = 
      (s_.axial_gauge.*sampleFcnPtr)(finger2_axial_gauge);
    std::vector<luke::gfloat> a3 = 
      (s_.axial_gauge.*sampleFcnPtr)(finger3_axial_gauge);

    // insert data into observation output
    observation.insert(observation.end(), a1.begin(), a1.end());
    observation.insert(observation.end(), a2.begin(), a2.end());
    observation.insert(observation.end(), a3.begin(), a3.end());

    if (debug_obs) {
      luke::print_vec(a1, "Axial gauge 1");
      luke::print_vec(a2, "Axial gauge 2");
      luke::print_vec(a3, "Axial gauge 3");
    }
  }

  // get palm sensor output
  if (s_.palm_sensor.in_use) {

    // sample data
    std::vector<luke::gfloat> p1 = 
      (s_.palm_sensor.*sampleFcnPtr)(palm_sensor);

    // insert data into observation output
    observation.insert(observation.end(), p1.begin(), p1.end());

    if (debug_obs) {
      luke::print_vec(p1, "Palm gauge");
    }
  }

  // get wrist sensor XY output
  if (s_.wrist_sensor_XY.in_use) {

    // sample data
    std::vector<luke::gfloat> wX =
      (s_.wrist_sensor_XY.*sampleFcnPtr)(wrist_X_sensor);
    std::vector<luke::gfloat> wY =
      (s_.wrist_sensor_XY.*sampleFcnPtr)(wrist_Y_sensor);

    // insert data into observation output
    observation.insert(observation.end(), wX.begin(), wX.end());
    observation.insert(observation.end(), wY.begin(), wY.end());

    if (debug_obs) {
      luke::print_vec(wX, "Wrist X");
      luke::print_vec(wY, "Wrist Y");
    }
  }

  // get wrist sensor Z output
  if (s_.wrist_sensor_Z.in_use) {
    
    // sample data
    std::vector<luke::gfloat> wZ =
      (s_.wrist_sensor_XY.*sampleFcnPtr)(wrist_Z_sensor);

    // insert data into observation output
    observation.insert(observation.end(), wZ.begin(), wZ.end());

    if (debug_obs) {
      luke::print_vec(wZ, "Wrist Z");
    }
  }

  // get motor state output
  if (s_.motor_state_sensor.in_use) {

    // sample data
    std::vector<luke::gfloat> s1 = 
      (s_.motor_state_sensor.*stateFcnPtr)(x_motor_position);
    std::vector<luke::gfloat> s2 = 
      (s_.motor_state_sensor.*stateFcnPtr)(y_motor_position);
    std::vector<luke::gfloat> s3 = 
      (s_.motor_state_sensor.*stateFcnPtr)(z_motor_position);

    // insert data into observation output
    observation.insert(observation.end(), s1.begin(), s1.end());
    observation.insert(observation.end(), s2.begin(), s2.end());
    observation.insert(observation.end(), s3.begin(), s3.end());
    
    if (debug_obs) {
      luke::print_vec(s1, "Motor state X");
      luke::print_vec(s2, "Motor state Y");
      luke::print_vec(s3, "Motor state Z");
    }
  }

  // get base state
  if (s_.base_state_sensor.in_use) {

    // sample data
    std::vector<luke::gfloat> bZ = 
      (s_.base_state_sensor.*stateFcnPtr)(z_base_position);

    // insert data into observation output
    observation.insert(observation.end(), bZ.begin(), bZ.end());

    if (debug_obs) {
      luke::print_vec(bZ, "Base state Z");
    }
  }

  if (debug_obs) {
    std::cout << "End of observation (n_obs = " << observation.size() << ")\n";
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

  constexpr double default_x_pos = 0.0;
  constexpr double default_y_pos = 0.0;
  constexpr double default_z_rot = 0.0;

  spawn_object(index, default_x_pos, default_y_pos, default_z_rot);
}

void MjClass::spawn_object(int index, double xpos, double ypos, double zrot)
{
  /* spawn an object beneath the gripper at (xpos, ypos) with a given rotation
  zrot in radians about the vertical axis */

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

  // set the rotation to be spawned
  double x1 = spawn_pos.qx;
  double y1 = spawn_pos.qy;
  double z1 = spawn_pos.qz;
  double w1 = spawn_pos.qw;
  double x2 = 1 * 1 * sin(zrot / 2) - 0 * 0 * cos(zrot / 2);
  double y2 = 0 * 1 * cos(zrot / 2) - 1 * 0 * sin(zrot / 2);
  double z2 = 1 * 0 * cos(zrot / 2) + 0 * 1 * sin(zrot / 2);
  double w2 = 1 * 1 * cos(zrot / 2) + 0 * 0 * sin(zrot / 2);
  spawn_pos.qw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
  spawn_pos.qx = w1 * x2 + x1 * w2 - y1 * z2 + z1 * y2;
  spawn_pos.qy = w1 * y2 + x1 * z2 + y1 * w2 - z1 * x2;
  spawn_pos.qz = w1 * z2 - x1 * y2 + y1 * x2 + z1 * w2;

  // spawn the object and save its start position
  luke::spawn_object(model, data, index, spawn_pos);
  env_.start_qpos = luke::get_object_qpos(model, data);

  // update everything for rendering
  forward();
}

void MjClass::randomise_object_colour()
{
  /* randomise the colour of the object */

  std::vector<float> rgb(3);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  rgb[0] = distribution(*MjType::generator);
  rgb[1] = distribution(*MjType::generator);
  rgb[2] = distribution(*MjType::generator);

  luke::set_object_colour(model, rgb);
}

void MjClass::randomise_ground_colour()
{
  /* randomise the colour of the ground plane */

  std::vector<float> rgb(3);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  rgb[0] = distribution(*MjType::generator);
  rgb[1] = distribution(*MjType::generator);
  rgb[2] = distribution(*MjType::generator);

  luke::set_ground_colour(model, rgb);
}

void MjClass::randomise_finger_colours()
{
  /* give all the fingers the same random colour */

  std::vector<float> rgb(3);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  rgb[0] = distribution(*MjType::generator);
  rgb[1] = distribution(*MjType::generator);
  rgb[2] = distribution(*MjType::generator);

  // set each finger in turn to be the same colour
  luke::set_finger_colour(model, rgb, 1);
  luke::set_finger_colour(model, rgb, 2);
  luke::set_finger_colour(model, rgb, 3);
  luke::set_finger_colour(model, rgb, 4); // 4 means palm
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
   
  // this value is not used in python
  env_.cumulative_reward += transition_reward;

  // if we are capping the maximum cumulative negative reward
  if (env_.cumulative_reward < s_.quit_on_reward_below) {
    if (s_.quit_reward_capped) {
      // reduce the reward to not put us below the cap
      transition_reward += s_.quit_on_reward_below - env_.cumulative_reward - ftol;
      env_.cumulative_reward = s_.quit_on_reward_below - ftol;
    }
  }

  // if we are capping the maximum cumulative positive reward
  if (env_.cumulative_reward > s_.quit_on_reward_above) {
    if (s_.quit_reward_capped) {
      // reduce the reward to not put us above the cap
      transition_reward += s_.quit_on_reward_above - env_.cumulative_reward - ftol;
      env_.cumulative_reward = s_.quit_on_reward_above - ftol;
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

/* ----- real gripper functions ----- */

std::vector<float> MjClass::get_finger_gauge_data()
{
  /* report the most recent gauge data in a vector { f1, f2, f3 } */

  std::vector<float> out;

  out.push_back(finger1_gauge.read_element());
  out.push_back(finger2_gauge.read_element());
  out.push_back(finger3_gauge.read_element());

  return out;
}

void MjClass::input_real_data(std::vector<float> state_data, 
  std::vector<float> sensor_data, float timestamp)
{
  /* insert real data */

  // flag to ensure we configure before running with real data
  static bool configured = false;
  if (not configured) {
    configure_settings();
    configured = true;
  }

  // count data inputs
  samples_since_last_obs += 1;

  // add state data
  int i = 0;

  if (s_.motor_state_sensor.in_use) {

    // normalise and save state data
    state_data[i] = normalise_between(
      state_data[i], luke::Gripper::xy_min, luke::Gripper::xy_max);
    x_motor_position.add(state_data[i]); ++i;

    state_data[i] = normalise_between(
      state_data[i], luke::Gripper::xy_min, luke::Gripper::xy_max);
    y_motor_position.add(state_data[i]); ++i;

    state_data[i] = normalise_between(
      state_data[i], luke::Gripper::xy_min, luke::Gripper::xy_max);
    z_motor_position.add(state_data[i]); ++i;

  }
  
  if (s_.base_state_sensor.in_use) {

    state_data[i] = normalise_between(
      state_data[i], luke::Target::base_z_min, luke::Target::base_z_max);
    z_base_position.add(state_data[i]); ++i;

  }

  // add sensor data
  int j = 0;

  if (s_.bending_gauge.in_use) {

    // scale, normalise, and save gauge data
    sensor_data[j] = (sensor_data[j] + calibrate_.offset.g1) * calibrate_.scale.g1;
    sensor_data[j] = normalise_between(
      sensor_data[j], -calibrate_.norm.g1, calibrate_.norm.g1);
    finger1_gauge.add(sensor_data[j]); ++j;

    sensor_data[j] = (sensor_data[j] + calibrate_.offset.g2) * calibrate_.scale.g2;
    sensor_data[j] = normalise_between(
      sensor_data[j], -calibrate_.norm.g2, calibrate_.norm.g2);
    finger2_gauge.add(sensor_data[j]); ++j;
  
    sensor_data[j] = (sensor_data[j] + calibrate_.offset.g3) * calibrate_.scale.g3;
    sensor_data[j] = normalise_between(
      sensor_data[j], -calibrate_.norm.g3, calibrate_.norm.g3);
    finger3_gauge.add(sensor_data[j]); ++j;

  }

  if (s_.palm_sensor.in_use) {

    // scale, normalise, and save gauge data
    sensor_data[j] = (sensor_data[j] + calibrate_.offset.palm) * calibrate_.scale.palm;
    sensor_data[j] = normalise_between(
      sensor_data[j], -calibrate_.norm.palm, calibrate_.norm.palm);
    palm_sensor.add(sensor_data[j]); ++j;

  }

  // add timestamp data - not used currently
  gauge_timestamps.add(timestamp);
  palm_timestamps.add(timestamp);

}

std::vector<float> MjClass::get_real_observation()
{
  /* get an observation of real data, samples_since_last_obs should NOT be inclusive, 
  so if before we had [0,1,2] and now we have [0,1,2,3,4,5] then n=3 */

  // manually set reading settings to ensure correctness
  s_.bending_gauge.update_n_readings(samples_since_last_obs, s_.state_n_prev_steps);
  s_.base_state_sensor.update_n_readings(samples_since_last_obs, s_.state_n_prev_steps);
  s_.palm_sensor.update_n_readings(samples_since_last_obs, s_.sensor_n_prev_steps);
  s_.wrist_sensor_Z.update_n_readings(samples_since_last_obs, s_.sensor_n_prev_steps);

  // reset as we are about to return an observation
  samples_since_last_obs = 0;

  return get_observation();
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

bool MjClass::last_action_gripper()
{
  /* was the last action performed on the gripper */

  if (luke::last_action_robot() == luke::Target::Robot::gripper)
    return true;
  else
    return false;
}

bool MjClass::last_action_panda()
{
  /* was the last action performed on the panda */

  if (luke::last_action_robot() == luke::Target::Robot::panda)
    return true;
  else 
    return false;
}

MjType::TestReport MjClass::get_test_report()
{
  /* fills out and returns the test report */

  testReport_.object_name = env_.obj.name;
  testReport_.cumulative_reward = env_.cumulative_reward;
  testReport_.cnt = env_.cnt;

  return testReport_;
}

MjType::CurveFitData::PoseData MjClass::validate_curve()
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

  luke::verify_small_angle_model(data, 0, pose.f1.joints, 
    pose.f1.pred_j, pose.f1.pred_x, pose.f1.pred_y, pose.f1.theory_y,
    pose.f1.theory_x_curve, pose.f1.theory_y_curve,
    -1 * s_.curve_validation, s_.finger_stiffness);
  luke::verify_small_angle_model(data, 1, pose.f2.joints, 
    pose.f2.pred_j, pose.f2.pred_x, pose.f2.pred_y, pose.f2.theory_y,
    pose.f2.theory_x_curve, pose.f2.theory_y_curve,
    -1 * s_.curve_validation, s_.finger_stiffness);
  luke::verify_small_angle_model(data, 2, pose.f3.joints, 
    pose.f3.pred_j, pose.f3.pred_x, pose.f3.pred_y, pose.f3.theory_y,
    pose.f3.theory_x_curve, pose.f3.theory_y_curve,
    -1 * s_.curve_validation, s_.finger_stiffness);

  // calculate errors
  pose.calc_error();

  // save
  curve_validation_data_.entries.push_back(pose);

  return pose;
}

MjType::CurveFitData::PoseData MjClass::validate_curve_under_force(int force)
{
  /* determine the cubic fit error and displacement of the fingers under
  a given force */

  // set the force
  s_.curve_validation = -1 * force;

  // step the simulation to allow the forces to settle
  float time_to_settle = 2;
  int steps_to_make = time_to_settle / s_.mujoco_timestep;
  // std::cout << "Stepping for " << steps_to_make << " steps to allow settling\n";
  for (int i = 0; i < steps_to_make; i++) {
    step();
  }

  // evaluate the finger pose
  MjType::CurveFitData::PoseData pose;
  pose = validate_curve();
  pose.tag_string = "Force is " + std::to_string(force) + " N";
  
  return pose;
}

MjType::CurveFitData MjClass::curve_validation_regime(bool print)
{
  /* peform test battery to validate finger bending, print is false by default */

  MjType::CurveFitData curvedata;

  bool debug_state = s_.debug;

  s_.debug = false;

  int max_force = 5;

  // for testing
  std::vector<float> readings(max_force);
  std::vector<float> norms(max_force);

  for (int i = 1; i < max_force + 1; i++) {
    
    MjType::CurveFitData::PoseData pose;
    pose = validate_curve_under_force(i);
    if (print) pose.print();
    curvedata.entries.push_back(pose);

    // for testing
    readings[i - 1] = luke::read_armadillo_gauge(data, 0);
    norms[i - 1] = finger1_gauge.read_element();
  }

  // for testing
  luke::print_vec(readings, "Gauge readings for 1-5N");
  luke::print_vec(norms, "normalised readings");

  s_.debug = debug_state;

  // overwrite the internal curve validation data
  curve_validation_data_ = curvedata;

  return curvedata;
}

void MjClass::calibrate_gauges()
{
  /* run a calibration scheme to normalise gauge outputs for a set force */

  validate_curve_under_force(s_.bend_gauge_normalise);
  luke::gfloat max_gauge_reading = luke::read_armadillo_gauge(data, 0);
  s_.bending_gauge.normalise = max_gauge_reading;

  // turn off curve validation mode
  s_.curve_validation = 0;
}

MjType::EventTrack MjClass::add_events(MjType::EventTrack& e1, MjType::EventTrack& e2)
{
  /* add the absolute count and last value of two events, all else is ignored */

  MjType::EventTrack out;

  #define XX(NAME, TYPE, VALUE)
  #define SS(NAME, IN_USE, NORM, READRATE)

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
  #define SS(NAME, IN_USE, NORM, READRATE)

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

float MjClass::find_highest_stable_timestep()
{
  /* find the highest timestep where the simulation is stable after 0.5 seconds */

  constexpr bool debug = true;

  float increment = 50e-6;   // 0.05 milliseconds
  float start_value = 3e-3;  // 30 millseconds
  float test_time = 1.0;     // 0.5 seconds

  float next_timestep = start_value;
  bool unstable = false;

  // find stable timestep
  while (next_timestep > increment) {

    int num_steps = (test_time / next_timestep) + 1;
    s_.mujoco_timestep = next_timestep;
    reset();

    for (int i = 0; i < num_steps; i++) {

      step();

      if (luke::is_sim_unstable(model, data)) {
        if (debug) {
          std::printf("Timestep of %.3f milliseconds unstable after %i steps (or %.3f seconds)\n",
            next_timestep * 1000, i, i * next_timestep);
        }
        unstable = true;
        break;
      }
    }

    if (unstable) {
      next_timestep -= increment;
      unstable = false;
      continue;
    }
    else {
      break;
    }
  }

  if (debug) {
    std::printf("Timestep of %.3f milliseconds remained stable for %.2f seconds\n",
            next_timestep * 1000, test_time);
  }

  // for safety, reduce timestep by 10 percent
  float final_timestep = next_timestep * 0.9;

  // round the timestep to a whole number of microseconds
  final_timestep = (float) ((int)(final_timestep * 1e6) * 1e-6);

  if (debug) {
    std::printf("Stable timestep is now set to %.3f milliseconds\n", final_timestep * 1000);
  }

  // set this timestep
  s_.mujoco_timestep = final_timestep;
  reset();

  return final_timestep;
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

void MjType::Settings::disable_sensors()
{
  /* do NOT use other fields than name as it will pull values from simsettings.h not s_,
     eg instead of using TRIGGER we need to use s_.NAME.trigger */

  #define XX(NAME, TYPE, VALUE)
  #define SS(NAME, IN_USE, NORM, READRATE) NAME.in_use = false;
  #define BR(NAME, DONTUSE1, DONTUSE2, DONTUSE3)
  #define LR(NAME, DONTUSE1, DONTUSE2, DONTUSE3, DONTUSE4, DONTUSE5, DONTUSE6)
  
    // run the macro and disable all the sensors
    LUKE_MJSETTINGS
  
  #undef XX
  #undef SS
  #undef BR
  #undef LR
}

void MjType::Settings::set_sensor_prev_steps_to(int prev_steps)
{
  /* do NOT use other fields than name as it will pull values from simsettings.h not s_,
     eg instead of using TRIGGER we need to use s_.NAME.trigger */

  #define XX(NAME, TYPE, VALUE)
  #define SS(NAME, IN_USE, NORM, READRATE) NAME.prev_steps = prev_steps;
  #define BR(NAME, DONTUSE1, DONTUSE2, DONTUSE3)
  #define LR(NAME, DONTUSE1, DONTUSE2, DONTUSE3, DONTUSE4, DONTUSE5, DONTUSE6)
  
    // run the macro and disable all the sensors
    LUKE_MJSETTINGS
  
  #undef XX
  #undef SS
  #undef BR
  #undef LR
}

void MjType::Settings::update_sensor_settings(double time_since_last_sample)
{
  /* updates the number of readings each sensor is taking based on time between
  samples and read rate */

  // turn on normalisation behaviour for all sensors
  set_use_normalisation(true);

  // apply noise settings
  set_use_noise(all_sensors_use_noise);
  apply_noise_params(); // currently prevents customising sensors differently

  // set the number of previous steps to sample back for all sensors
  set_sensor_prev_steps_to(sensor_n_prev_steps);

  // manually override state sensors
  int state_sensor_n_readings_per_step = 1;
  motor_state_sensor.update_n_readings(state_sensor_n_readings_per_step, state_n_prev_steps);
  base_state_sensor.update_n_readings(state_sensor_n_readings_per_step, state_n_prev_steps);

  #define XX(NAME, TYPE, VALUE)
  
  // update n_readings for all except state sensors
  #define SS(NAME, IN_USE, NORM, READRATE)                       \
            if (#NAME != "motor_state_sensor" and                \
                #NAME != "base_state_sensor")                    \
              NAME.update_n_readings(time_since_last_sample);

  #define BR(NAME, DONTUSE1, DONTUSE2, DONTUSE3)
  #define LR(NAME, DONTUSE1, DONTUSE2, DONTUSE3, DONTUSE4, DONTUSE5, DONTUSE6)
  
    // run the macro and update all the sensors
    LUKE_MJSETTINGS
  
  #undef XX
  #undef SS
  #undef BR
  #undef LR
}

void MjType::Settings::set_use_normalisation(bool set_as)
{
  /* do NOT use other fields than name as it will pull values from simsettings.h not s_,
     eg instead of using TRIGGER we need to use s_.NAME.trigger */

  #define XX(NAME, TYPE, VALUE)
  #define SS(NAME, IN_USE, NORM, READRATE) NAME.use_normalisation = set_as;
  #define BR(NAME, DONTUSE1, DONTUSE2, DONTUSE3)
  #define LR(NAME, DONTUSE1, DONTUSE2, DONTUSE3, DONTUSE4, DONTUSE5, DONTUSE6)
  
    // run the macro and disable all the sensors
    LUKE_MJSETTINGS
  
  #undef XX
  #undef SS
  #undef BR
  #undef LR
}

void MjType::Settings::set_use_noise(bool set_as)
{
  /* do NOT use other fields than name as it will pull values from simsettings.h not s_,
     eg instead of using TRIGGER we need to use s_.NAME.trigger */

  #define XX(NAME, TYPE, VALUE)
  #define SS(NAME, IN_USE, NORM, READRATE) NAME.use_noise = set_as;
  #define BR(NAME, DONTUSE1, DONTUSE2, DONTUSE3)
  #define LR(NAME, DONTUSE1, DONTUSE2, DONTUSE3, DONTUSE4, DONTUSE5, DONTUSE6)
  
    // run the macro and disable all the sensors
    LUKE_MJSETTINGS
  
  #undef XX
  #undef SS
  #undef BR
  #undef LR
}

void MjType::Settings::apply_noise_params()
{
  /* do NOT use other fields than name as it will pull values from simsettings.h not s_,
     eg instead of using TRIGGER we need to use s_.NAME.trigger */

  #define XX(NAME, TYPE, VALUE)

  #define SS(NAME, IN_USE, NORM, READRATE)       \
            NAME.noise_mag = sensor_noise_mag;   \
            NAME.noise_std = sensor_noise_std;   \
            NAME.noise_mu = sensor_noise_mu;

  #define BR(NAME, DONTUSE1, DONTUSE2, DONTUSE3)
  #define LR(NAME, DONTUSE1, DONTUSE2, DONTUSE3, DONTUSE4, DONTUSE5, DONTUSE6)
  
    // run the macro and disable all the sensors
    LUKE_MJSETTINGS
  
  #undef XX
  #undef SS
  #undef BR
  #undef LR

  // manually override the state sensors
  motor_state_sensor.noise_mag = state_noise_mag;
  motor_state_sensor.noise_mu = state_noise_mu;
  motor_state_sensor.noise_std = state_noise_std;
  base_state_sensor.noise_mag = state_noise_mag;
  base_state_sensor.noise_mu = state_noise_mu;
  base_state_sensor.noise_std = state_noise_std;
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
  #define SS(NAME, IN_USE, NORM, READRATE)

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
  #define SS(NAME, IN_USE, NORM, READRATE)
  #define BR(NAME, REWARD, DONE, TRIGGER)                                    \
            events.NAME.row = events.NAME.row *                              \
                                  events.NAME.value + events.NAME.value;     \
            events.NAME.abs += events.NAME.value;                            \
            events.NAME.last_value = events.NAME.value;                      \
            events.NAME.value = false; // reset for next step

  #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT)               \
            active = false;                                                  \
            if (events.NAME.value > settings.NAME.min and                    \
                (events.NAME.value < settings.NAME.overshoot or              \
                 settings.NAME.overshoot < 0))                               \
              { active = true; }                                             \
            events.NAME.row = events.NAME.row * active + active;             \
            events.NAME.abs += active;                                       \
            events.NAME.last_value = events.NAME.value;                      \
            events.NAME.value = 0.0; // reset for next step

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
  #define SS(NAME, IN_USE, NORM, READRATE)

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
  #define SS(NAME, IN_USE, NORM, READRATE)

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
  #define SS(NAME, IN_USE, NORM, READRATE)

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
  #define SS(NAME, IN_USE, NORM, READRATE)

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
  #define SS(NAME, IN_USE, NORM, READRATE)

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
  #define SS(NAME, IN_USE, NORM, READRATE)

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
  #define SS(NAME, IN_USE, NORM, READRATE)

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
  #define SS(NAME, IN_USE, NORM, READRATE)

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
  #define SS(NAME, IN_USE, NORM, READRATE)

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