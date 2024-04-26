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
    randomise_every_colour();
  }
  else {
    set_neat_colours();
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

  // if finger stiffness has changed, we should recalibrate timestep/sim steps/gauges etc
  if (resetFlags.finger_EI_changed) {
    resetFlags.finger_EI_changed = false;
    hard_reset(); // this calls reset()->configure_settings()
    return;
  }

  /* check what actions are set */
  action_options.clear();
  action_options.resize(MjType::Action::count, -1);

  // set all the actions discrete or continous
  s_.set_all_action_continous(s_.continous_actions);

  // use macros to determine which actions are in use
  int i = 0;
  #define AA(NAME, USED, VALUE, SIGN)                                    \
    if (s_.NAME.in_use) {                                                           \
      if (s_.NAME.continous) {                                                      \
        action_options[i] = MjType::Action::TOKEN_CONCAT(NAME, CONTINOUS_TOKEN);    \
        i += 1;                                                                     \
      }                                                                             \
      else {                                                                        \
        action_options[i] = MjType::Action::TOKEN_CONCAT(NAME, POSITIVE_TOKEN);     \
        action_options[i + 1] = MjType::Action::TOKEN_CONCAT(NAME, NEGATIVE_TOKEN); \
        i += 2;                                                                     \
      }                                                                             \
    }
    
    // run the macro to create the code
    LUKE_MJSETTINGS_ACTION

  #undef AA

  // are we using a termination signal action
  if (s_.use_termination_action) {
    action_options[i] = MjType::Action::termination_signal;
    i += 1;
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
    case MjType::Sample::sign: {
      sampleFcnPtr = &MjType::Sensor::sign_sample;
      break;
    }
    case MjType::Sample::scaled_change: {
      sampleFcnPtr = &MjType::Sensor::scaled_change_sample;
      break;
    }
    case MjType::Sample::scaled_change_sq: {
      sampleFcnPtr = &MjType::Sensor::scaled_change_sq_sample;
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
    case MjType::Sample::sign: {
      stateFcnPtr = &MjType::Sensor::sign_sample;
      break;
    }
    case MjType::Sample::scaled_change: {
      stateFcnPtr = &MjType::Sensor::scaled_change_sample;
      break;
    }
    case MjType::Sample::scaled_change_sq: {
      sampleFcnPtr = &MjType::Sensor::scaled_change_sq_sample;
      break;
    }
    default: {
      throw std::runtime_error("s_.state_sample_mode not set to legal value");
    }
  }

  // enforce HER goals to trigger at 1 always
  default_goal_event_triggering();

  // update the finger spring stiffness
  luke::set_finger_stiffness_using_model(model);

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
    resetFlags.auto_exceed_lateral_lim = s_.auto_exceed_lateral_lim;
    resetFlags.flags_init = true;
  }

  // it is best also to do timestep first as it spams the terminal
  // find timestep automatically, this change must be done before calibrate_simulated_sensors()
  if (resetFlags.auto_timestep) {
    
    // save initial state of resetFlags variables
    static bool auto_timestep;
    static bool auto_calibrate;
    static bool auto_simsteps;
    auto_calibrate = resetFlags.auto_calibrate;
    auto_simsteps = resetFlags.auto_simsteps;
    
    resetFlags.auto_timestep = false;                     // disable auto timestep immediately due to recursion
    resetFlags.auto_calibrate = false;                    // disable calibration before timestep found
    resetFlags.auto_simsteps = false;                     // disable simsteps before timestep found
    s_.mujoco_timestep = find_highest_stable_timestep();  // find the timestep, calls configure_settings() recursively
    resetFlags.auto_calibrate = auto_calibrate;           // re-enable calibration after timestep found
    resetFlags.auto_simsteps = auto_simsteps;             // re-enable simsteps after timestep found
    if (echo_auto_changes) std::cout << "MjClass auto-setting: Mujoco timestep set to: " << s_.mujoco_timestep << '\n';
  }

  // set the simulation timestep in mujoco
  model->opt.timestep = s_.mujoco_timestep;

  // automatically set the exceed lateral force limits (ie safe bending limits)
  if (resetFlags.auto_exceed_lateral_lim) {
    resetFlags.auto_exceed_lateral_lim = false;
    s_.exceed_lateral.min = s_.exceed_lat_min_factor * yield_load();
    s_.exceed_lateral.max = s_.exceed_lat_max_factor * yield_load();
    if (echo_auto_changes) std::cout << "MjClass auto-setting: yield load is " 
      << yield_load() << ", exceed lateral (min, max) is ("
      << s_.exceed_lateral.min << ", " << s_.exceed_lateral.max << ")\n";
  }

  // calibrate the gauges and wrist Z sensor, requires timestep to be stable
  if (resetFlags.auto_calibrate) {
    resetFlags.auto_calibrate = false;
    // determine the safe finger loads based on the yield of this finger stiffness
    /* mysimluate typically gives saturation at +20% from this value, likely due 
    to different loading (point load in theory but 4/5th length loading in practice) */
    float bend_gauge_normalise = s_.saturation_yield_factor * yield_load();
    calibrate_simulated_sensors(bend_gauge_normalise);
    // save the calibration ratio to estimate SI outputs from the gauges
    sim_gauge_raw_to_N_factor = bend_gauge_normalise / s_.bending_gauge.normalise;
    if (echo_auto_changes) {
      std::cout << "MjClass auto-setting: Bending gauge normalisation set to: " 
        << s_.bending_gauge.normalise << " (NOT SI), based on saturation load of " << bend_gauge_normalise << " newtons\n";
      std::cout << "MjClass auto-setting: Wrist Z sensor offset set to: " 
        << s_.wrist_sensor_Z.raw_value_offset << '\n';
    }

    // one additional reset() as fingers will be wobbling from the calibration
    reset();
  }

  // if the 'time_for_action' setting is changed, recalibrate sim steps per action
  if (s_.auto_sim_steps) {
    static float time_for_action = s_.time_for_action;
    static float tol = 1e-4;
    if (abs(time_for_action - s_.time_for_action) > tol) {
      resetFlags.auto_simsteps = true;
      time_for_action = s_.time_for_action;
    }
  }

  // find the sim settings per action automatically, requires timestep to be finalised
  if (resetFlags.auto_simsteps) {
    resetFlags.auto_simsteps = false;
    s_.sim_steps_per_action = std::ceil(s_.time_for_action / s_.mujoco_timestep);
    if (echo_auto_changes) std::cout << "MjClass auto-setting: Sim steps per action set to: " << s_.sim_steps_per_action << '\n';
  }

  // update the sensor number of readings based on time per step
  double time_per_step = model->opt.timestep * s_.sim_steps_per_action;
  s_.apply_noise_params(uniform_dist); // if sensor mu>0, randomises mu
  s_.update_sensor_settings(time_per_step);
}

std::string MjClass::file_from_from_command_line(int argc, char **argv)
{
  /* load a model based on command line arguments and flags. Valid flags are:
        -g, --gripper [gripper],          eg gripper_N8_28
        -N, --segments [segments],        eg -N 8 = use 8 segments
        -w, --width [width in mm],        eg -w 28 = use 28mm width
        -o, --object-set [object set],    eg -o set6_fullset_800_50i
        -t, --task [task number],         eg -t 2 = use task 2
        -p, --path [path to object set],  eg -p /home/luke/mymujoco/mjcf
  */

  MjClassInputParser input(argc, argv);

  std::string gripper = input.getCmdFromList("-g", "--gripper");
  std::string segments = input.getCmdFromList("-N", "--segments");
  std::string width = input.getCmdFromList("-w", "--width");
  std::string object_set = input.getCmdFromList("-o", "--object-set");
  std::string task = input.getCmdFromList("-t", "--task");
  std::string path = input.getCmdFromList("-p", "--path");
  std::string thickness = input.getCmdFromList("T", "--thickness");

  // debug information
  std::cout << "MjClass::load_from_command_line(...) recieved the following:\n";
  if (not gripper.empty())    std::cout << "    -> gripper     " << gripper << '\n';
  if (not segments.empty())   std::cout << "    -> segments    " << segments << '\n';
  if (not width.empty())      std::cout << "    -> width       " << width << '\n';
  if (not object_set.empty()) std::cout << "    -> object_set  " << object_set << '\n';
  if (not task.empty())       std::cout << "    -> task        " << task << '\n';
  if (not path.empty())       std::cout << "    -> path        " << path << '\n';
  if (not thickness.empty())  std::cout << "    -> thickness   " << thickness << '\n';
  if (not gripper.empty() and (not width.empty() or not segments.empty())) {
    std::cout << "Warning: gripper overrides values of width and segments\n";
  }

  // apply defaults on empty fields (gripper overrides width and segments)
  if (segments.empty()) { segments = "8"; };
  if (width.empty()) { width = "28"; };
  if (gripper.empty()) { gripper = "gripper_N" + segments + "_" + width; };
  if (object_set.empty()) { object_set = "set9_fullset"; };
  if (task.empty()) { task = "0"; };
  if (path.empty()) { path = LUKE_MJCF_PATH; };

  // apply given thickness
  if (not thickness.empty()) {
    double thickness_double = std::stod(thickness);
    set_finger_thickness(thickness_double * 1e-3);
  }

  // use default templates to assemble filepath
  if (path.back() != '/') { path += '/'; };
  if (object_set.back() != '/') { object_set += '/'; };
  if (gripper.back() != '/') { gripper += '/'; };
  task = "gripper_task_" + task + ".xml";
  std::string fullpath = path + object_set + gripper + task;

  std::cout << "Full filepath: " << fullpath << '\n';
  return fullpath;
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
  sim_sensors_.reset();
  sim_sensors_SI_.reset();
  real_sensors_.reset();

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
  s_.base_state_sensor_XY.reset();
  s_.base_state_sensor_Z.reset();
  s_.base_state_sensor_yaw.reset();
  s_.cartesian_contacts_XYZ.reset();

  // refetch the gripper base limits in case they have changed
  base_min_ = luke::get_base_min();
  base_max_ = luke::get_base_max();

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

  // move the base to a random new position
  random_base_Z_movement(s_.base_position_noise);
}

void MjClass::hard_reset()
{
  /* a complete reset, use this when you want to load a new model file
  which has a different number of gripper joints eg a different number
  of segments */

  // reinitialise the joint settings structure
  luke::init_J(model, data);

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

  if (s_.curve_validation) {

    // can apply forces on finger segments, eg tip force
    luke::resolve_segment_forces(model, data);
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

bool MjClass::rendering_enabled()
{
  return false;
}

bool MjClass::render() 
{
  std::cout << "mjclass::render() disabled in compilation settings\n";
  return false;
}

void MjClass::close_render()
{
  std::cout << "mjclass::close_render() disabled in compilation settings\n";
  return;
}

void MjClass::init_rgbd()
{
  std::cout << "mjclass::init_rgbd() disabled in compilation settings\n";
  return false;
}

luke::RGBD MjClass::render_and_get_RGBD_image()
{
  std::cout << "mjclass::render_and_get_RGBD_image() disabled in compilation settings\n";
  throw std::runtime_error("mjclass::render_and_get_RGBD_image() disabled in compilation settings\n")
  return;
}

luke::RGBD MjClass::render_and_get_RGBD_image(int height, int width)
{
  std::cout << "mjclass::render_and_get_RGBD_image() disabled in compilation settings\n";
  throw std::runtime_error("mjclass::render_and_get_RGBD_image() disabled in compilation settings\n")
  return;
}

void MjClass::render_RGBD_image()
{
  std::cout << "mjclass::render_RGBD_image() disabled in compilation settings\n";
  throw std::runtime_error("mjclass::render_RGBD_image() disabled in compilation settings\n")
  return;
}

luke::RGBD MjClass::get_RGBD_image()
{
  std::cout << "mjclass::get_RGBD_image() disabled in compilation settings\n";
  throw std::runtime_error("mjclass::get_RGBD_image() disabled in compilation settings\n")
  return;
}

void MjClass::set_RGBD_image_size()
{
  std::cout << "mjclass::set_RGBD_image_size() disabled in compilation settings\n";
  throw std::runtime_error("mjclass::set_RGBD_image_size() disabled in compilation settings\n")
  return;
}

#else

bool MjClass::rendering_enabled()
{
  return true;
}

bool MjClass::render()
{
  /* Render a frame of the simulation to the screen */

  // safety catch, we are unable to close the window properly
  static bool window_closed = false;
  if (window_closed) {
    return false;
  }

  // if the render window has not yet been initialised
  if (not render_window_init) {
    render::init_window(*this);
    render_window_init = true;
  }
  else if (render_reload) {
    render::reload_for_rendering(*this);
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
      
      // window_open = render::render(model, data);
      window_open = render::render_window();
    }
  }
  else {
    // just render once
    window_open = render::render_window();
  }

  // if the window has been closed
  if (not window_open) {
    render::finish_window();
    render_window_init = false;
    window_closed = true;
  }

  return window_open;
}

void MjClass::close_render()
{
  /* close the rendering window */

  if (render_window_init) render::finish_window();
  render_window_init = false;
}

bool MjClass::init_rgbd()
{
  /* initialise an rgbd camera */

  render::init_camera(*this);
  render_camera_init = true;
  render_reload = false;

  return true;
}

luke::RGBD MjClass::get_RGBD()
{
  /* perform a render and then return an RGBD image */

  render_RGBD();
  return read_existing_RGBD();
}

luke::RGBD MjClass::get_mask()
{
  /* perform a render of a segmentation mask then return an RGBD image */

  render_mask();
  return render::read_mask();
}

void MjClass::render_RGBD()
{
  /* render a single frame of the current state, this can be used to bypass the
  normal render() and keep the window hidden. render() displays the window to
  the screen */

  // if the render window has not yet been initialised
  if (not render_camera_init) {
    init_rgbd();
  }
  else if (render_reload) {
    render::reload_for_rendering(*this);
    render_reload = false;
  }

  render::render_camera();
}

void MjClass::render_mask()
{
  /* render a single frame of the current state, this can be used to bypass the
  normal render() and keep the window hidden. render() displays the window to
  the screen. Renders with a segmentation mask */

  // if the render window has not yet been initialised
  if (not render_camera_init) {
    init_rgbd();
  }
  else if (render_reload) {
    render::reload_for_rendering(*this);
    render_reload = false;
  }

  render::render_camera_with_seg_mask();
}

luke::RGBD MjClass::read_existing_RGBD()
{
  /* return an existing RGBD image from the current state of the simulation,
  without re-rendering */

  return render::read_rgbd();
}

void MjClass::set_RGBD_size(int width, int height)
{
  /* set how many pixels the RGBD image should be */

  render::resize_camera_window(width, height);
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

    // save SI (for gauges our raw data is NOT SI so we best approximate with calibration)
    sim_sensors_SI_.finger1_gauge.add(gauges[0] * sim_gauge_raw_to_N_factor);
    sim_sensors_SI_.finger2_gauge.add(gauges[1] * sim_gauge_raw_to_N_factor);
    sim_sensors_SI_.finger3_gauge.add(gauges[2] * sim_gauge_raw_to_N_factor);

    // normalise
    gauges[0] = s_.bending_gauge.apply_normalisation(gauges[0]);
    gauges[1] = s_.bending_gauge.apply_normalisation(gauges[1]);
    gauges[2] = s_.bending_gauge.apply_normalisation(gauges[2]);

    // apply noise (can be gaussian based on sensor settings, if std_dev > 0)
    gauges[0] = s_.bending_gauge.apply_noise(gauges[0], uniform_dist, 1);
    gauges[1] = s_.bending_gauge.apply_noise(gauges[1], uniform_dist, 2);
    gauges[2] = s_.bending_gauge.apply_noise(gauges[2], uniform_dist, 3);

    // save
    sim_sensors_.finger1_gauge.add(gauges[0]);
    sim_sensors_.finger2_gauge.add(gauges[1]);
    sim_sensors_.finger3_gauge.add(gauges[2]);

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

    // save SI
    sim_sensors_SI_.finger1_axial_gauge.add(axial_gauges[0]);
    sim_sensors_SI_.finger2_axial_gauge.add(axial_gauges[1]);
    sim_sensors_SI_.finger3_axial_gauge.add(axial_gauges[2]);

    // normalise
    axial_gauges[0] = s_.axial_gauge.apply_normalisation(axial_gauges[0]);
    axial_gauges[1] = s_.axial_gauge.apply_normalisation(axial_gauges[1]);
    axial_gauges[2] = s_.axial_gauge.apply_normalisation(axial_gauges[2]);

    // apply noise (can be gaussian based on sensor settings, if std_dev > 0)
    axial_gauges[0] = s_.axial_gauge.apply_noise(axial_gauges[0], uniform_dist, 1);
    axial_gauges[1] = s_.axial_gauge.apply_noise(axial_gauges[1], uniform_dist, 2);
    axial_gauges[2] = s_.axial_gauge.apply_noise(axial_gauges[2], uniform_dist, 3);

    // save
    sim_sensors_.finger1_axial_gauge.add(axial_gauges[0]);
    sim_sensors_.finger2_axial_gauge.add(axial_gauges[1]);
    sim_sensors_.finger3_axial_gauge.add(axial_gauges[2]);

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

    // save SI
    sim_sensors_SI_.palm_sensor.add(palm_reading);

    // normalise
    palm_reading = s_.palm_sensor.apply_normalisation(palm_reading);

    // apply noise (can be gaussian based on sensor settings, if std_dev > 0)
    palm_reading = s_.palm_sensor.apply_noise(palm_reading, uniform_dist);

    // save
    sim_sensors_.palm_sensor.add(palm_reading);

    // record time
    palm_timestamps.add(data->time);
  }

  // // check the wrist sensor XY force
  // if (s_.wrist_sensor_XY.ready_to_read(data->time)) {

  //   // read
  //   luke::gfloat x = data->userdata[0];
  //   luke::gfloat y = data->userdata[1];

  //   // save SI
  //   sim_sensors_SI_.wrist_X_sensor.add(x);
  //   sim_sensors_SI_.wrist_Y_sensor.add(y);

  //   // normalise
  //   x = s_.wrist_sensor_XY.apply_normalisation(x);
  //   y = s_.wrist_sensor_XY.apply_normalisation(y);

  //   // apply noise (can be gaussian based on sensor settings, if std_dev > 0)
  //   x = s_.wrist_sensor_XY.apply_noise(x, uniform_dist, 1);
  //   y = s_.wrist_sensor_XY.apply_noise(y, uniform_dist, 2);

  //   // save
  //   sim_sensors_.wrist_X_sensor.add(x);
  //   sim_sensors_.wrist_Y_sensor.add(y);
    
  //   // record time
  //   wristXY_timestamps.add(data->time);
  // }

  // check the wrist sensor Z force
  if (s_.wrist_sensor_Z.ready_to_read(data->time)) {

    // read
    luke::gfloat z = data->userdata[2];

    // zero the reading (this step is unique to wrist Z sensor)
    z -= s_.wrist_sensor_Z.raw_value_offset;

    // save SI (with reading zero applied!)
    sim_sensors_SI_.wrist_Z_sensor.add(z);

    // normalise
    z = s_.wrist_sensor_Z.apply_normalisation(z);

    // apply noise (can be gaussian based on sensor settings, if std_dev > 0)
    z = s_.wrist_sensor_Z.apply_noise(z, uniform_dist);

    // save
    sim_sensors_.wrist_Z_sensor.add(z);

    // record time
    wristZ_timestamps.add(data->time);
  }
}

void MjClass::sense_gripper_state()
{
  /* save the end target state position of the gripper and base */

  // get position we think each motor should be (NOT luke::get_gripper_state(data)!)
  std::vector<luke::gfloat> state_vec = luke::get_target_state_vector();
  luke::JointStates states = luke::get_target_state();

  // normalise { x, y, z } joint values
  states.gripper_x = normalise_between(
    states.gripper_x, luke::Gripper::xy_min, luke::Gripper::xy_max);
  states.gripper_y = normalise_between(
    states.gripper_y, luke::Gripper::xy_min, luke::Gripper::xy_max);
  states.gripper_z = normalise_between(
    states.gripper_z, luke::Gripper::z_min, luke::Gripper::z_max);
  states.base_x = normalise_between(states.base_x, base_min_[0], base_max_[0]);
  states.base_y = normalise_between(states.base_y, base_min_[1], base_max_[1]);
  states.base_z = normalise_between(states.base_z, base_min_[2], base_max_[2]);
  states.base_yaw = normalise_between(states.base_yaw, base_min_[5], base_max_[5]);

  // apply noise (can be gaussian based on sensor settings, if std_dev > 0)
  states.gripper_x = s_.motor_state_sensor.apply_noise(states.gripper_x, uniform_dist, 1);
  states.gripper_y = s_.motor_state_sensor.apply_noise(states.gripper_y, uniform_dist, 2);
  states.gripper_z = s_.motor_state_sensor.apply_noise(states.gripper_z, uniform_dist, 3);
  states.base_x = s_.base_state_sensor_XY.apply_noise(states.base_x, uniform_dist, 1);
  states.base_y = s_.base_state_sensor_XY.apply_noise(states.base_y, uniform_dist, 2);
  states.base_z = s_.base_state_sensor_Z.apply_noise(states.base_z, uniform_dist);
  states.base_yaw = s_.base_state_sensor_yaw.apply_noise(states.base_yaw, uniform_dist);

  // save reading
  sim_sensors_.x_motor_position.add(states.gripper_x);
  sim_sensors_.y_motor_position.add(states.gripper_y);
  sim_sensors_.z_motor_position.add(states.gripper_z);
  sim_sensors_.x_base_position.add(states.base_x);
  sim_sensors_.y_base_position.add(states.base_y);
  sim_sensors_.z_base_position.add(states.base_z);
  sim_sensors_.yaw_base_rotation.add(states.base_yaw);

  // save the time the reading was made
  step_timestamps.add(data->time);

  // add in cartesian contact point information for MAT
  std::vector<double> finger_forces_SI(3);
  finger_forces_SI[0] = sim_sensors_SI_.finger1_gauge.read_element();
  finger_forces_SI[1] = sim_sensors_SI_.finger2_gauge.read_element();
  finger_forces_SI[2] = sim_sensors_SI_.finger3_gauge.read_element();
  double palm_force_SI = sim_sensors_SI_.palm_sensor.read_element();
  std::vector<luke::Vec3> xyz = luke::get_fingerend_and_palm_xyz(finger_forces_SI);

  // add contact location only if there is a contact force above the threshold
  constexpr double ft = 0.2; // force threshold 0.2N
  sim_sensors_.finger1_x_pos.add(((abs(finger_forces_SI[0])) > ft) ? xyz[0].x : 0.0);
  sim_sensors_.finger1_y_pos.add(((abs(finger_forces_SI[0])) > ft) ? xyz[0].y : 0.0);
  sim_sensors_.finger1_z_pos.add(((abs(finger_forces_SI[0])) > ft) ? xyz[0].z : 0.0);
  sim_sensors_.finger2_x_pos.add(((abs(finger_forces_SI[1])) > ft) ? xyz[1].x : 0.0);
  sim_sensors_.finger2_y_pos.add(((abs(finger_forces_SI[1])) > ft) ? xyz[1].y : 0.0);
  sim_sensors_.finger2_z_pos.add(((abs(finger_forces_SI[1])) > ft) ? xyz[1].z : 0.0);
  sim_sensors_.finger3_x_pos.add(((abs(finger_forces_SI[2])) > ft) ? xyz[2].x : 0.0);
  sim_sensors_.finger3_y_pos.add(((abs(finger_forces_SI[2])) > ft) ? xyz[2].y : 0.0);
  sim_sensors_.finger3_z_pos.add(((abs(finger_forces_SI[2])) > ft) ? xyz[2].z : 0.0);
  sim_sensors_.palm_x_pos.add(((palm_force_SI) > ft) ? xyz[3].x : 0.0);
  sim_sensors_.palm_y_pos.add(((palm_force_SI) > ft) ? xyz[3].y : 0.0);
  sim_sensors_.palm_z_pos.add(((palm_force_SI) > ft) ? xyz[3].z : 0.0);

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
  // env_.obj.qpos 
  std::vector<luke::QPos> qpos_vec = luke::get_object_qpos(model, data);
  std::vector<std::vector<double>> rel_xy = luke::get_object_XY_relative_to_gripper(model, data);

  // how many live objects are in the simulation
  if (env_.obj.size() != qpos_vec.size()) {
    throw std::runtime_error("MjClass.update_env() found unexpected number of live objects");
    env_.obj.clear();
    env_.obj.resize(qpos_vec.size());
  }
  
  int num_obj = env_.obj.size();

  env_.grp.target = luke::get_gripper_target();
  luke::Forces_faster forces = luke::get_object_forces_faster(model, data);

  MjType::Env::ObjValues blank;
  env_.obj_values = blank; // wipe object values

  // loop over the objects and extract information
  for (int i = 0; i < num_obj; i++) {

    env_.obj[i].qpos = qpos_vec[i];
    env_.obj[i].distance_from_gripper = std::sqrt(std::pow(rel_xy[i][0], 2) + std::pow(rel_xy[i][1], 2));

    env_.obj[i].finger1_force = forces.obj[i].finger1_local;
    env_.obj[i].finger2_force = forces.obj[i].finger2_local;
    env_.obj[i].finger3_force = forces.obj[i].finger3_local;
    env_.obj[i].palm_force = forces.obj[i].palm_local;
    env_.obj[i].ground_force = forces.obj[i].ground;
    env_.obj[i].palm_axial_force = forces.obj[i].palm_local[0]; // +ve for compression
    env_.obj[i].finger1_force_mag = env_.obj[i].finger1_force.magnitude3();
    env_.obj[i].finger2_force_mag = env_.obj[i].finger2_force.magnitude3();
    env_.obj[i].finger3_force_mag = env_.obj[i].finger3_force.magnitude3();
    env_.obj[i].palm_force_mag = env_.obj[i].palm_force.magnitude3();
    env_.obj[i].ground_force_mag = env_.obj[i].ground_force.magnitude3();
    env_.obj[i].lift_height = env_.obj[i].qpos.z - env_.obj[i].start_qpos.z;

    env_.obj[i].avg_finger_force = 0.33333 * (
      env_.obj[i].finger1_force_mag 
      + env_.obj[i].finger2_force_mag
      + env_.obj[i].finger3_force_mag);

    // get highest outwards lateral force on finger
    env_.obj[i].peak_finger_lateral_force = -1 * std::min({
      forces.obj[i].finger1_local[1], 
      forces.obj[i].finger2_local[1], 
      forces.obj[i].finger3_local[1]
    });

    // see if these values are maximums
    // if (finger1_force_mag > env_.obj_values.finger1_force_mag) {
    //   env_.obj_values.finger1_force_mag = finger1_force_mag;
    // }
    // if (finger2_force_mag > env_.obj_values.finger2_force_mag) {
    //   env_.obj_values.finger2_force_mag = finger2_force_mag;
    // }
    // if (finger3_force_mag > env_.obj_values.finger3_force_mag) {
    //   env_.obj_values.finger3_force_mag = finger3_force_mag;
    // }
    // if (palm_force_mag > env_.obj_values.palm_force_mag) {
    //   env_.obj_values.palm_force_mag = palm_force_mag;
    // }

    if (env_.obj[i].avg_finger_force > env_.obj_values.avg_finger_force) {
      env_.obj_values.avg_finger_force = env_.obj[i].avg_finger_force;
    }
    if (env_.obj[i].palm_axial_force > env_.obj_values.palm_axial_force) {
      env_.obj_values.palm_axial_force = env_.obj[i].palm_axial_force;
    }
    if (env_.obj[i].peak_finger_lateral_force > env_.grp.peak_finger_lateral_force) {
      env_.grp.peak_finger_lateral_force = env_.obj[i].peak_finger_lateral_force;
    }
    if (env_.obj[i].lift_height > env_.obj_values.highest_lift) {
      env_.obj_values.highest_lift = env_.obj[i].lift_height;
    }
  }

  // extract the gripper height (don't use object height for lifting as fingers tilt)
  luke::JointStates states = luke::get_target_state();
  float gripper_z_height = -1 * states.base_z;
  
  // // for testing
  // forces.print();
  // // save the forces on the object in local frames (from gripper perspective)
  // env_.obj.finger1_force = forces.obj.finger1_local;
  // env_.obj.finger2_force = forces.obj.finger2_local;
  // env_.obj.finger3_force = forces.obj.finger3_local;
  // env_.obj.palm_force = forces.obj.palm_local;
  // env_.obj.ground_force = forces.obj.ground;

  // save forces on the gripper fingers (include all contacts, not just object)
  env_.grp.finger1_force = forces.all.finger1_local;
  env_.grp.finger2_force = forces.all.finger2_local;
  env_.grp.finger3_force = forces.all.finger3_local;

  // calculate finger and palm force magnitudes on the object
  // float finger1_force_mag = env_.obj.finger1_force.magnitude3();
  // float finger2_force_mag = env_.obj.finger2_force.magnitude3();
  // float finger3_force_mag = env_.obj.finger3_force.magnitude3();
  // float palm_force_mag = env_.obj.palm_force.magnitude3();
  // float ground_force_mag = env_.obj.ground_force.magnitude3();

  // // get palm force on object (x = axial in local frame, +ve for compression)
  // env_.obj.palm_axial_force = +1 * env_.obj.palm_force[0];

  // // get average finger force on object
  // env_.obj.avg_finger_force = 0.333 * (finger1_force_mag + finger2_force_mag
  //   + finger3_force_mag);

  // get the highest axial finger force (x = axial in local frame, -ve for comp.)
  env_.grp.peak_finger_axial_force = -1 * std::min({ 
    forces.gnd.finger1_local[0], forces.gnd.finger2_local[0], forces.gnd.finger3_local[0]
  });

  // // get highest outwards lateral force on finger
  // env_.grp.peak_finger_lateral_force = -1 * std::min({
  //   forces.obj.finger1_local[1], forces.obj.finger2_local[1], forces.obj.finger3_local[1]
  // });

  // get information directly from the sensors
  luke::gfloat last_g1 = sim_sensors_SI_.read_finger1_gauge();
  luke::gfloat last_g2 = sim_sensors_SI_.read_finger2_gauge();
  luke::gfloat last_g3 = sim_sensors_SI_.read_finger3_gauge();
  luke::gfloat last_palm_N = sim_sensors_SI_.read_palm_sensor();
  luke::gfloat last_wrist_N = sim_sensors_SI_.read_wrist_Z_sensor();
  luke::gfloat max_gauge_force = std::max(last_g1, last_g2);
  max_gauge_force = std::max(max_gauge_force, last_g3);
  luke::gfloat avg_gauge_force = (1.0/3.0) * (last_g1 + last_g2 + last_g3);

  // determine if a MAT reopen was recently triggered
  if (recent_MAT_reopen) {

    // how far have the joints moved
    double xmove = abs(states.gripper_x - luke::Gripper::xy_home);
    double ymove = abs(states.gripper_y - luke::Gripper::xy_home);
    double zmove = abs(states.gripper_z - luke::Gripper::z_home);
    double maxmove = std::max(xmove, ymove);
    maxmove = std::max(maxmove, zmove);

    if (maxmove > MAT_reopen_XYZ_distance) {
      recent_MAT_reopen = false;
      if (s_.debug)
        std::cout << "MAT reopen no longer recent, furthest joint has moved > "
          << MAT_reopen_XYZ_distance * 1000 << " mm\n";
    }
  }

  /* ----- detect state of binary events (EDIT here to add a binary event) ----- */

  // another step has been made
  env_.cnt.step_num.value = true;

  // initially guess the first object is closest to the gripper
  double closest_object_distance = env_.obj[0].distance_from_gripper;

  for (int i = 0; i < num_obj; i++) {

    env_.obj[i].lifted = false;
    env_.obj[i].oob = false;
    env_.obj[i].lifted_to_height = false;
    env_.obj[i].target_height = false;
    env_.obj[i].contact = false;
    env_.obj[i].stable = false;
    env_.obj[i].stable_height = false;

    // lifted is true if both the object and gripper have zero axial force from the ground
    if (env_.obj[i].ground_force_mag < ftol and 
        env_.obj[i].peak_finger_axial_force < ftol) {
      env_.cnt.lifted.value = true;
      env_.obj[i].lifted = true;
    }

    // check if the object has gone out of bounds
    if (env_.obj[i].qpos.x > s_.oob_distance or env_.obj[i].qpos.x < -s_.oob_distance or
        env_.obj[i].qpos.y > s_.oob_distance or env_.obj[i].qpos.y < -s_.oob_distance) {
      env_.cnt.oob.value = true;
      env_.obj[i].oob = true;
    }

    // object lifted past minimum required and not oob (env_.cnt.lifted and env_.cnt.oob must be set)
    if (env_.obj_values.highest_lift > s_.lift_height - ftol and
        env_.obj[i].lifted and not env_.obj[i].oob) {
      env_.cnt.lifted_to_height.value = true;
      env_.obj[i].lifted_to_height = true;
    }

    // object is lifted and the gripper has reached the target height
    if (env_.obj[i].lifted_to_height and
        gripper_z_height > s_.gripper_target_height - ftol) {
      env_.cnt.target_height.value = true;
      env_.obj[i].target_height = true;
    }

    // detect any gripper contact with the object
    if (env_.obj[i].finger1_force_mag > ftol or
        env_.obj[i].finger2_force_mag > ftol or
        env_.obj[i].finger3_force_mag > ftol or
        env_.obj[i].palm_force.magnitude3() > ftol) {
      env_.cnt.object_contact.value = true;
      env_.obj[i].contact = true;
    }

    // check if object is stable (must also be lifted and env_.cnt.lifted set)
    if (env_.obj[i].finger1_force_mag > s_.stable_finger_force and
        env_.obj[i].finger2_force_mag > s_.stable_finger_force and
        env_.obj[i].finger3_force_mag > s_.stable_finger_force and
        env_.obj[i].finger1_force_mag < s_.stable_finger_force_lim and
        env_.obj[i].finger2_force_mag < s_.stable_finger_force_lim and
        env_.obj[i].finger3_force_mag < s_.stable_finger_force_lim and
        env_.obj[i].palm_force_mag > s_.stable_palm_force and
        env_.obj[i].palm_force_mag < s_.stable_palm_force_lim and 
        env_.cnt.lifted.value) {
      env_.cnt.object_stable.value = true;
      env_.obj[i].stable = true;
    }

    // if stable and lifted to target (need env_.cnt.object_stable and target_height set)
    if (env_.obj[i].stable and env_.obj[i].target_height) {
      env_.cnt.stable_height.value = true;
      env_.obj[i].stable_height = true;
    }

    // if a termination signal has been sen
    if (termination_signal_sent) {
      // do we require stable or lifted termination
      if (s_.lifted_termination.done) {
        // check that the object is lifted to the desired height
        if (env_.obj[i].lifted_to_height) {
          env_.cnt.lifted_termination.value = true;
          env_.obj[i].lifted_termination = true;
        }
        else {
          env_.cnt.failed_termination.value = true;
        }
      }
      else {
        // check that the object is stably grasped
        if (env_.cnt.stable_height.value) {
          env_.cnt.stable_termination.value = true;
          env_.obj[i].stable_termination = true;
        }
        else {
          env_.cnt.failed_termination.value = true;
        }
      }
    }

    // are we within a certain threshold distance from an object
    if (env_.obj[i].distance_from_gripper < s_.XY_distance_threshold) {
      env_.cnt.within_XY_distance.value = true;
      env_.obj[i].within_XY_distance = true;
    }

    // determine the closest object to the gripper
    if (env_.obj[i].distance_from_gripper < closest_object_distance) {
      closest_object_distance = env_.obj[i].distance_from_gripper;
    }
  }

  // check if the object has been dropped (env_.cnt.lifted must already be set)
  env_.cnt.dropped.value = 
    // if lastdropped==false, newlifted==false, and lastlifted==true, set dropped=1
    ((not env_.cnt.dropped.row * not env_.cnt.lifted.value * env_.cnt.lifted.row) ? 1
    // else if lifted==true -> set dropped=0
    : (env_.cnt.lifted.value ? 0 
    // else if lastdropped==true +=1 to it, otherwise -> set dropped=0
    : (env_.cnt.dropped.row ? env_.cnt.dropped.row + 1 : 0)));


  // for testing
  // std::cout << "Highest object lift is: " << env_.obj_values.highest_lift << '\n';
  // std::cout << "Gripper z height is: " << gripper_z_height << '\n';
  // std::cout << "Lifted is: " << env_.cnt.lifted.value << '\n';
  // std::cout << "Target height is: " << env_.cnt.target_height.value << '\n';

  /* ----- input state value of linear events (EDIT here to add a linear event) ----- */

  env_.cnt.exceed_axial.value = env_.grp.peak_finger_axial_force;
  env_.cnt.exceed_lateral.value = env_.grp.peak_finger_lateral_force;
  env_.cnt.palm_force.value = env_.obj_values.palm_axial_force * env_.cnt.lifted.value; // must be lifted
  env_.cnt.exceed_palm.value = env_.obj_values.palm_axial_force;
  env_.cnt.finger_force.value = env_.obj_values.avg_finger_force;
  
  // testing: track info for linear goals
  env_.cnt.finger1_force.value = env_.obj[0].finger1_force_mag;
  env_.cnt.finger2_force.value = env_.obj[0].finger2_force_mag;
  env_.cnt.finger3_force.value = env_.obj[0].finger3_force_mag;
  env_.cnt.ground_force.value = env_.obj[0].ground_force_mag;

  // new: direct sensor rewards, based on measured sensor values from SIMULATED sensors
  env_.cnt.good_bend_sensor.value = avg_gauge_force;
  env_.cnt.exceed_bend_sensor.value = max_gauge_force;
  env_.cnt.dangerous_bend_sensor.value = max_gauge_force;
  env_.cnt.good_palm_sensor.value = last_palm_N;
  env_.cnt.exceed_palm_sensor.value = last_palm_N;
  env_.cnt.dangerous_palm_sensor.value = last_palm_N;
  env_.cnt.exceed_wrist_sensor.value = last_wrist_N;
  env_.cnt.dangerous_wrist_sensor.value = last_wrist_N;

  // scale the action penalty based on the number of actions (not counting termination action)
  env_.cnt.action_penalty_lin.value /= float(n_actions - s_.use_termination_action);
  env_.cnt.action_penalty_sq.value /= float(n_actions - s_.use_termination_action);

  // input closest object (use negative values so decreasing distance increases reward)
  env_.cnt.object_XY_distance.value = -closest_object_distance;

  /* ----- determine the reported success rate (as a proxy, should not have associated reward) ----- */

  /* check for a 'successful' trial, used as a metric only, should have NO associated
  reward. The criteria to determine if a trial was successful is:
    1) an event value is True which
    2) is a binary reward and
    2) gives a reward of +1
    3) and sets done = True
    4) and which is now triggered (ie cnt.row + 1 >= trigger)
  */
  #define BR(NAME, DONTUSE1, DONTUSE2, DONTUSE3)                                     \
            if (env_.cnt.NAME.value) {                                               \
              if (s_.NAME.reward >= (1.0 - 1e-5)) {                                  \
                if (s_.NAME.done) {                                                  \
                  if (env_.cnt.NAME.row + 1 >= s_.NAME.trigger) {                    \
                    env_.cnt.successful_grasp.value = true;                          \
                    if (s_.debug)                                                    \
                      std::printf("successful_grasp=true because %s is triggered\n", \
                        #NAME);                                                      \
                  }                                                                  \
                }                                                                    \
              }                                                                      \
            }
          
    // run the macro to create the code
    LUKE_MJSETTINGS_BINARY_REWARD

  #undef BR

  /* ----- resolve linear events and update counts of all events (no editing needed) ----- */

  update_events(env_.cnt, s_);

  if (s_.debug) env_.cnt.print();

  if (s_.debug) {
    env_.print_objects();
  }

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

bool MjClass::set_new_base_XY(double x, double y)
{
  /* instantly move the base to a new XY position, this position will be 
  tracked by the base XY sensors */

  bool within_limits = luke::set_base_to_XY_position(data, x, y);

  return within_limits;
}

bool MjClass::set_new_base_yaw(double yaw)
{
  /* set a randomised base yaw based off the limits */

  return luke::set_base_to_yaw(data, yaw);
}

double MjClass::random_base_yaw(double size)
{
  /* set a randomised base yaw based off the limits */

  std::uniform_real_distribution<double> distribution(-size, size);
  double new_yaw = distribution(*MjType::generator);

  luke::set_base_to_yaw(data, new_yaw);

  return new_yaw;
}

double MjClass::random_base_Z_movement(double size)
{
  /* Perform a random movement of the base taken from a uniform distribution of
  [-size, size] in metres */

  std::uniform_real_distribution<double> distribution(-size, size);
  double z_move = distribution(*MjType::generator);

  luke::set_base_to_Z_position(data, z_move);

  return z_move;
}

std::vector<float> MjClass::MAT_reopen(double new_angle)
{
  /* perform the reopen maneuovre for MAT. This means:
      - reset gripper joints all back to start
      - move XY of gripper to tactile centre, average of contacts
      - move yaw of gripper to the new angle
  */

  constexpr bool debug = false;

  // find the tactile centre
  double new_x = 0.25 * (
    sim_sensors_.finger1_x_pos.read_element()
    + sim_sensors_.finger2_x_pos.read_element()
    + sim_sensors_.finger3_x_pos.read_element()
    + sim_sensors_.palm_x_pos.read_element()
  );
  double new_y = 0.25 * (
    sim_sensors_.finger1_y_pos.read_element()
    + sim_sensors_.finger2_y_pos.read_element()
    + sim_sensors_.finger3_y_pos.read_element()
    + sim_sensors_.palm_y_pos.read_element()
  );

  // open up the gripper fingers and reset the joints (moves base too)
  luke::set_gripper_and_base_to_reset_position(model, data);

  // now set the gripper base to the tactile centre
  if (debug)
    std::cout << "Tactile centre is at: " << new_x << ", " << new_y << "\n";
  luke::set_base_to_XY_position(data, new_x, new_y);

  // now apply the rotation, current is ABSOLUTE rather than relative
  luke::set_base_to_yaw(data, new_angle);

  // if we reopen too much, we apply a penalty
  if (recent_MAT_reopen) {
    apply_MAT_reopen_penalty = true;
  }

  recent_MAT_reopen = true;

  return luke::get_target_state_vector();
}

/* ----- learning functions ----- */

void MjClass::action_step()
{
  /* step until the simulation settles */

  if (s_.debug) tick();

  for (int i = 0; i < s_.sim_steps_per_action; i++) {
    step();
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

std::vector<float> MjClass::set_discrete_action(int action)
{
  /* call for discrete actions, only the action code */

  return set_action(action, 0); // continous_fraction unused for discrete actions
}

std::vector<float> MjClass::set_continous_action(int action, float magnitude_fraction)
{
  /* call for discrete actions, only the action code */

  // check that the magnitude_fraction is in bounds [-1, +1]
  if (magnitude_fraction < -1.0) magnitude_fraction = -1.0;
  else if (magnitude_fraction > 1.0) magnitude_fraction = 1.0;

  return set_action(action, magnitude_fraction);
}

std::vector<float> MjClass::set_action(int action, float continous_fraction)
{
  /* sets an action in the simulation, but does not step at all. Returns the
  new state vector */

  bool wl = true; // within limits
  termination_signal_sent = false; // reset whether we received a termination action

  if (action < 0 or action >= n_actions) {
    std::cout << "MjClass::set_action() received action = " << action << " which is out of bounds.\n"
      << "This function is now returning without doing anything. The possible actions are:\n";
    print_actions();
    std::vector<float> empty;
    return empty;
  }

  int action_code = action_options[action];

  // // for testing
  // luke::print_vec(action_options, "action options");
  // std::cout << "The action code is " << action_code << '\n';

  if (s_.debug) std::cout << "Action number " << action << " received, name = ";

  switch (action_code) {

    // define action behaviour for positive/negative/continous
    // any new actions should be further defined in ActionSettings::update_action_function()
    #define AA(NAME, USED, VALUE, SIGN)              \
      case MjType::Action::TOKEN_CONCAT(NAME, POSITIVE_TOKEN):  \
        if (s_.debug) std::cout << s_.NAME.name + "_positive";  \
        wl = s_.NAME.call_action_function(s_.NAME.value);       \
        break;                                                  \
      case MjType::Action::TOKEN_CONCAT(NAME, NEGATIVE_TOKEN):  \
        if (s_.debug) std::cout << s_.NAME.name + "_negative";  \
        wl = s_.NAME.call_action_function(-1 * s_.NAME.value);  \
        break;                                                  \
      case MjType::Action::TOKEN_CONCAT(NAME, CONTINOUS_TOKEN): \
        if (s_.debug) {                                         \
          std::cout << s_.NAME.name + "_continous";             \
          std::cout << ", fraction = " << continous_fraction;   \
        }                                                       \
        wl = s_.NAME.call_action_function(s_.NAME.value * continous_fraction);         \
        env_.cnt.action_penalty_lin.value += abs(continous_fraction);                  \
        env_.cnt.action_penalty_sq.value += (continous_fraction * continous_fraction); \
        break;                                                  \

      // run the macro to create the code
      LUKE_MJSETTINGS_ACTION

    #undef AA

    case MjType::Action::termination_signal: {
      float value = 0.0;
      if (s_.continous_actions) value = continous_fraction;
      else value = 1.0;
      bool triggered = (value > s_.termination_threshold);
      if (s_.debug) {
        std::cout << "termination_signal, value = " << value <<", threshold = "
          << s_.termination_threshold << ", hence triggered is " <<
          ((triggered) ? "true" : "false");
      }
      if (triggered) {
        termination_signal_sent = true;
        if (s_.lift_on_termination) {
          // set base target to max height
          luke::lift_base_to_height(base_max_[2]);
          // step the simulation so we lift to the max height
          // hand calibrated, 1x sim_steps_per_actions is enough to get up 30mm
          for (int i = 0; i < s_.sim_steps_per_action * 2; i++) {
            step();
          }
        }
      }
      wl = true;
      break;
    }

    default:
      std::cout << "Action value received is " << action_code << '\n';
      std::cout << "Number of actions is " << n_actions << '\n';
      throw std::runtime_error("MjClass::set_action() received out of bounds int");

  }

  // special check for whether the fingertips are outside safe limits
  if (luke::get_fingertip_z_height() < s_.fingertip_min_mm * 1e-3) {
    wl = false;
  }

  if (s_.debug) std::cout << ", within_limits = " << wl << '\n';

  // // uncomment for debugging
  // std::cout << "fingertip z height is " << luke::get_fingertip_z_height() * 1e3
  //     << ", minimum allowed is " << s_.fingertip_min_mm << "\n";

  // update whether this action was within limits or not
  // use += so True is latching when using continous actions
  env_.cnt.exceed_limits.value += not wl;

  // get the target state and return it as a vector
  return luke::get_target_state_vector();
}

bool MjClass::is_done()
{
  /* determine if an episode should end */

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
    LUKE_MJSETTINGS_BINARY_REWARD
    LUKE_MJSETTINGS_LINEAR_REWARD

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

  /* termination rewards should set done=true,
      -> stable_termination
      -> failed_termination
  */
  // // if a termination signal has been sent
  // if (termination_signal_sent) {
  //   if (s_.debug) std::cout << "is_done = true, termination signal sent\n";
  //   return true;
  // }

  // if the cumulative reward drops below a given threshold
  if (s_.cap_reward and s_.quit_if_cap_exceeded) {
    if (env_.cumulative_reward - ftol < s_.reward_cap_lower_bound) {
      if (s_.debug) std::printf("Reward dropped below limit of %.3f, is_done() = true\n",
        s_.reward_cap_lower_bound);
      return true;
    }
    if (env_.cumulative_reward + ftol > s_.reward_cap_upper_bound) {
      if (s_.debug) std::printf("Reward exceeded upper limit of %.3f, is_done() = true\n",
        s_.reward_cap_upper_bound);
      return true;
    }
  }

  return false;
}

std::vector<luke::gfloat> MjClass::get_observation()
{
  /* get an observation from the simulation sensors */

  return get_observation(sim_sensors_);
}

std::vector<luke::gfloat> MjClass::get_observation(MjType::SensorData sensors)
{
  /* get an observation from a provided set of sensors */

  // use for printing detailed observation debug information
  constexpr bool debug_obs = false;
  constexpr bool debug_data = false;

  std::vector<luke::gfloat> observation;

  if (debug_obs) {
    std::cout << "Observation information:\n";
  }

  // get bending strain gauge sensor output
  if (s_.bending_gauge.in_use) {

    // sample data
    std::vector<luke::gfloat> f1 = 
      (s_.bending_gauge.*sampleFcnPtr)(sensors.finger1_gauge);
    std::vector<luke::gfloat> f2 = 
      (s_.bending_gauge.*sampleFcnPtr)(sensors.finger2_gauge);
    std::vector<luke::gfloat> f3 = 
      (s_.bending_gauge.*sampleFcnPtr)(sensors.finger3_gauge);

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
      (s_.axial_gauge.*sampleFcnPtr)(sensors.finger1_axial_gauge);
    std::vector<luke::gfloat> a2 = 
      (s_.axial_gauge.*sampleFcnPtr)(sensors.finger2_axial_gauge);
    std::vector<luke::gfloat> a3 = 
      (s_.axial_gauge.*sampleFcnPtr)(sensors.finger3_axial_gauge);

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
      (s_.palm_sensor.*sampleFcnPtr)(sensors.palm_sensor);

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
      (s_.wrist_sensor_XY.*sampleFcnPtr)(sensors.wrist_X_sensor);
    std::vector<luke::gfloat> wY =
      (s_.wrist_sensor_XY.*sampleFcnPtr)(sensors.wrist_Y_sensor);

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
      (s_.wrist_sensor_XY.*sampleFcnPtr)(sensors.wrist_Z_sensor);

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
      (s_.motor_state_sensor.*stateFcnPtr)(sensors.x_motor_position);
    std::vector<luke::gfloat> s2 = 
      (s_.motor_state_sensor.*stateFcnPtr)(sensors.y_motor_position);
    std::vector<luke::gfloat> s3 = 
      (s_.motor_state_sensor.*stateFcnPtr)(sensors.z_motor_position);

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

  // get base XY state
  if (s_.base_state_sensor_XY.in_use) {

    // sample data
    std::vector<luke::gfloat> bX = 
      (s_.base_state_sensor_XY.*stateFcnPtr)(sensors.x_base_position);
    std::vector<luke::gfloat> bY = 
      (s_.base_state_sensor_XY.*stateFcnPtr)(sensors.y_base_position);

    // insert data into observation output
    observation.insert(observation.end(), bX.begin(), bX.end());
    observation.insert(observation.end(), bY.begin(), bY.end());

    if (debug_obs) {
      luke::print_vec(bX, "Base state X");
      luke::print_vec(bY, "Base state Y");
    }
  }

  // get base Z state
  if (s_.base_state_sensor_Z.in_use) {

    // sample data
    std::vector<luke::gfloat> bZ = 
      (s_.base_state_sensor_Z.*stateFcnPtr)(sensors.z_base_position);

    // insert data into observation output
    observation.insert(observation.end(), bZ.begin(), bZ.end());

    if (debug_obs) {
      luke::print_vec(bZ, "Base state Z");
    }
  }

  // get base Z yaw
  if (s_.base_state_sensor_yaw.in_use) {

    // sample data
    std::vector<luke::gfloat> byaw = 
      (s_.base_state_sensor_yaw.*stateFcnPtr)(sensors.yaw_base_rotation);

    // insert data into observation output
    observation.insert(observation.end(), byaw.begin(), byaw.end());

    if (debug_obs) {
      luke::print_vec(byaw, "Base yaw rotation");
    }
  }

  // get cartesian contact positions for MAT
  if (s_.cartesian_contacts_XYZ.in_use) {

    // sample data - ONLY USE CHANGE SAMPLE, ignores the simsettings for state sampling
    std::vector<luke::gfloat> (MjType::Sensor::*cartesianFcnPtr)
      (luke::SlidingWindow<luke::gfloat>);
    cartesianFcnPtr = &MjType::Sensor::change_sample;
    std::vector<luke::gfloat> f1x = (s_.cartesian_contacts_XYZ.*cartesianFcnPtr)(sensors.finger1_x_pos);
    std::vector<luke::gfloat> f1y = (s_.cartesian_contacts_XYZ.*cartesianFcnPtr)(sensors.finger1_y_pos);
    std::vector<luke::gfloat> f1z = (s_.cartesian_contacts_XYZ.*cartesianFcnPtr)(sensors.finger1_z_pos);
    std::vector<luke::gfloat> f2x = (s_.cartesian_contacts_XYZ.*cartesianFcnPtr)(sensors.finger2_x_pos);
    std::vector<luke::gfloat> f2y = (s_.cartesian_contacts_XYZ.*cartesianFcnPtr)(sensors.finger2_y_pos);
    std::vector<luke::gfloat> f2z = (s_.cartesian_contacts_XYZ.*cartesianFcnPtr)(sensors.finger2_z_pos);
    std::vector<luke::gfloat> f3x = (s_.cartesian_contacts_XYZ.*cartesianFcnPtr)(sensors.finger3_x_pos);
    std::vector<luke::gfloat> f3y = (s_.cartesian_contacts_XYZ.*cartesianFcnPtr)(sensors.finger3_y_pos);
    std::vector<luke::gfloat> f3z = (s_.cartesian_contacts_XYZ.*cartesianFcnPtr)(sensors.finger3_z_pos);
    std::vector<luke::gfloat> px = (s_.cartesian_contacts_XYZ.*cartesianFcnPtr)(sensors.palm_x_pos);
    std::vector<luke::gfloat> py = (s_.cartesian_contacts_XYZ.*cartesianFcnPtr)(sensors.palm_y_pos);
    std::vector<luke::gfloat> pz = (s_.cartesian_contacts_XYZ.*cartesianFcnPtr)(sensors.palm_z_pos);

    // insert data into observation output
    observation.insert(observation.end(), f1x.begin(), f1x.end());
    observation.insert(observation.end(), f1y.begin(), f1y.end());
    observation.insert(observation.end(), f1z.begin(), f1z.end());
    observation.insert(observation.end(), f2x.begin(), f2x.end());
    observation.insert(observation.end(), f2y.begin(), f2y.end());
    observation.insert(observation.end(), f2z.begin(), f2z.end());
    observation.insert(observation.end(), f3x.begin(), f3x.end());
    observation.insert(observation.end(), f3y.begin(), f3y.end());
    observation.insert(observation.end(), f3z.begin(), f3z.end());
    observation.insert(observation.end(), px.begin(), px.end());
    observation.insert(observation.end(), py.begin(), py.end());
    observation.insert(observation.end(), pz.begin(), pz.end());

    if (debug_obs) {
      luke::print_vec(f1x, "finger1_x_pos");
      luke::print_vec(f1y, "finger1_y_pos");
      luke::print_vec(f1z, "finger1_z_pos");
      luke::print_vec(f2x, "finger2_x_pos");
      luke::print_vec(f2y, "finger2_y_pos");
      luke::print_vec(f2z, "finger2_z_pos");
      luke::print_vec(f3x, "finger3_x_pos");
      luke::print_vec(f3y, "finger3_y_pos");
      luke::print_vec(f3z, "finger3_z_pos");
      luke::print_vec(px, "palm_x_pos");
      luke::print_vec(py, "palm_y_pos");
      luke::print_vec(pz, "palm_z_pos");
    }
  }

  if (debug_obs) {
    std::cout << "End of observation (n_obs = " << observation.size() << ")\n";
  }

  if (debug_data) {
      int data_num = 20;
      std::cout << "Raw values:\n";
      std::cout << "X Motor: "; sensors.x_motor_position.print(data_num);
      std::cout << "Y Motor: "; sensors.y_motor_position.print(data_num);
      std::cout << "Z Motor: "; sensors.z_motor_position.print(data_num);
      std::cout << "Z Base: "; sensors.z_base_position.print(data_num);
      std::cout << "Z Yaw: "; sensors.yaw_base_rotation.print(data_num);
      std::cout << "SI real data values:\n";
      std::cout << "X Motor: "; real_sensors_.SI.x_motor_position.print(data_num);
      std::cout << "Y Motor: "; real_sensors_.SI.y_motor_position.print(data_num);
      std::cout << "Z Motor: "; real_sensors_.SI.z_motor_position.print(data_num);
      std::cout << "Z Base: "; real_sensors_.SI.z_base_position.print(data_num);
      std::cout << "Z Yaw: "; real_sensors_.SI.yaw_base_rotation.print(data_num);
    }
  
  return observation;
}

std::string MjClass::debug_observation(std::vector<luke::gfloat> observation, bool printout = false)
{
  /* get an observation from a provided set of sensors */

  // use for printing detailed observation debug information
  bool debug_obs = printout;

  std::vector<luke::gfloat> real_obs = get_observation();

  std::string info;

  if (real_obs.size() != observation.size()) {
    std::cout << "WARNING from MjClass::debug_observation()\n";
    std::cout << "Observation given to function has length = " << observation.size()
      << ", while MjClass::get_observation returns a size = " << real_obs.size()
      << ", therefore settings do not match and the information from this function will be WRONG\n";
    std::cout << "WARNING DEBUG_OBSERVATION WILL NOT WORK PROPERLY\n";

    info += "INVALID MjClass::debug_observation() due to non-matching sizes\n";

    if (real_obs.size() > observation.size()) {
      std::cout << "The given observation is not long enough, MjClass::debug_observation()"
        << " is returning to avoid a seg fault\n";
      return info;
    }
  }

  // use the sensor data to generate vectors of correct length
  MjType::SensorData sensors = sim_sensors_;
  int sidx = 0; // index of the state vector

  if (debug_obs) {
    std::cout << "Observation information:\n";
  }

  // get bending strain gauge sensor output
  if (s_.bending_gauge.in_use) {

    // sample data
    std::vector<luke::gfloat> f1 = 
      (s_.bending_gauge.*sampleFcnPtr)(sensors.finger1_gauge);
    std::vector<luke::gfloat> f2 = 
      (s_.bending_gauge.*sampleFcnPtr)(sensors.finger2_gauge);
    std::vector<luke::gfloat> f3 = 
      (s_.bending_gauge.*sampleFcnPtr)(sensors.finger3_gauge);

    // copy our observation into these vectors instead
    std::copy(observation.begin() + sidx, observation.begin() + sidx + f1.size(), f1.begin()); sidx += f1.size();
    std::copy(observation.begin() + sidx, observation.begin() + sidx + f2.size(), f2.begin()); sidx += f2.size();
    std::copy(observation.begin() + sidx, observation.begin() + sidx + f3.size(), f3.begin()); sidx += f3.size();

    if (debug_obs) {
      luke::print_vec(f1, "Bending gauge 1");
      luke::print_vec(f2, "Bending gauge 2");
      luke::print_vec(f3, "Bending gauge 3");
    }

    std::string n = std::to_string(f1.size());
    info += "Bend gauge 1 = " + n + " | Bend gauge 2 = " + n + " | Bend gauge 3 = " + n + " | ";
  }

  // get axial strain gauge sensor output
  if (s_.axial_gauge.in_use) {

    // sample data
    std::vector<luke::gfloat> a1 = 
      (s_.axial_gauge.*sampleFcnPtr)(sensors.finger1_axial_gauge);
    std::vector<luke::gfloat> a2 = 
      (s_.axial_gauge.*sampleFcnPtr)(sensors.finger2_axial_gauge);
    std::vector<luke::gfloat> a3 = 
      (s_.axial_gauge.*sampleFcnPtr)(sensors.finger3_axial_gauge);

    // copy our observation into these vectors instead
    std::copy(observation.begin() + sidx, observation.begin() + sidx + a1.size(), a1.begin()); sidx += a1.size();
    std::copy(observation.begin() + sidx, observation.begin() + sidx + a2.size(), a2.begin()); sidx += a2.size();
    std::copy(observation.begin() + sidx, observation.begin() + sidx + a3.size(), a3.begin()); sidx += a3.size();

    if (debug_obs) {
      luke::print_vec(a1, "Axial gauge 1");
      luke::print_vec(a2, "Axial gauge 2");
      luke::print_vec(a3, "Axial gauge 3");
    }

    std::string n = std::to_string(a1.size());
    info += "Axial gauge 1 = " + n + " | Axial gauge 2 = " + n + " | Axial gauge 3 = " + n + " | ";
  }

  // get palm sensor output
  if (s_.palm_sensor.in_use) {

    // sample data
    std::vector<luke::gfloat> p1 = 
      (s_.palm_sensor.*sampleFcnPtr)(sensors.palm_sensor);

    // copy our observation into these vectors instead
    std::copy(observation.begin() + sidx, observation.begin() + sidx + p1.size(), p1.begin()); sidx += p1.size();

    if (debug_obs) {
      luke::print_vec(p1, "Palm gauge");
    }

    std::string n = std::to_string(p1.size());
    info += "Palm gauge = " + n + " | ";
  }

  // get wrist sensor XY output
  if (s_.wrist_sensor_XY.in_use) {

    // sample data
    std::vector<luke::gfloat> wX =
      (s_.wrist_sensor_XY.*sampleFcnPtr)(sensors.wrist_X_sensor);
    std::vector<luke::gfloat> wY =
      (s_.wrist_sensor_XY.*sampleFcnPtr)(sensors.wrist_Y_sensor);

    // copy our observation into these vectors instead
    std::copy(observation.begin() + sidx, observation.begin() + sidx + wX.size(), wX.begin()); sidx += wX.size();
    std::copy(observation.begin() + sidx, observation.begin() + sidx + wY.size(), wY.begin()); sidx += wY.size();

    if (debug_obs) {
      luke::print_vec(wX, "Wrist X");
      luke::print_vec(wY, "Wrist Y");
    }

    std::string n = std::to_string(wX.size());
    info += "Wrist X = " + n + " | Wrist Y = " + n + " | ";
  }

  // get wrist sensor Z output
  if (s_.wrist_sensor_Z.in_use) {
    
    // sample data
    std::vector<luke::gfloat> wZ =
      (s_.wrist_sensor_XY.*sampleFcnPtr)(sensors.wrist_Z_sensor);

    // copy our observation into these vectors instead
    std::copy(observation.begin() + sidx, observation.begin() + sidx + wZ.size(), wZ.begin()); sidx += wZ.size();

    if (debug_obs) {
      luke::print_vec(wZ, "Wrist Z");
    }

    std::string n = std::to_string(wZ.size());
    info += "Wrist Z = " + n + " | ";
  }

  // get motor state output
  if (s_.motor_state_sensor.in_use) {

    // sample data
    std::vector<luke::gfloat> s1 = 
      (s_.motor_state_sensor.*stateFcnPtr)(sensors.x_motor_position);
    std::vector<luke::gfloat> s2 = 
      (s_.motor_state_sensor.*stateFcnPtr)(sensors.y_motor_position);
    std::vector<luke::gfloat> s3 = 
      (s_.motor_state_sensor.*stateFcnPtr)(sensors.z_motor_position);

    // copy our observation into these vectors instead
    std::copy(observation.begin() + sidx, observation.begin() + sidx + s1.size(), s1.begin()); sidx += s1.size();
    std::copy(observation.begin() + sidx, observation.begin() + sidx + s2.size(), s2.begin()); sidx += s2.size();
    std::copy(observation.begin() + sidx, observation.begin() + sidx + s3.size(), s3.begin()); sidx += s3.size();
    
    if (debug_obs) {
      luke::print_vec(s1, "Motor state X");
      luke::print_vec(s2, "Motor state Y");
      luke::print_vec(s3, "Motor state Z");
    }

    std::string n = std::to_string(s1.size());
    info += "Motor state X = " + n + " | Motor state Y = " + n + " | Motor state Z = " + n + " | ";
  }

  // get base XY state
  if (s_.base_state_sensor_XY.in_use) {

    // sample data
    std::vector<luke::gfloat> bX = 
      (s_.base_state_sensor_XY.*stateFcnPtr)(sensors.x_base_position);
    std::vector<luke::gfloat> bY = 
      (s_.base_state_sensor_XY.*stateFcnPtr)(sensors.y_base_position);

    // copy our observation into these vectors instead
    std::copy(observation.begin() + sidx, observation.begin() + sidx + bX.size(), bX.begin()); sidx += bX.size();
    std::copy(observation.begin() + sidx, observation.begin() + sidx + bY.size(), bY.begin()); sidx += bY.size();

    if (debug_obs) {
      luke::print_vec(bX, "Base state X");
      luke::print_vec(bY, "Base state Y");
    }

    std::string n = std::to_string(bX.size());
    info += "Base state X = " + n + " | Base state Y = " + n + " | ";
  }

  // get base Z state
  if (s_.base_state_sensor_Z.in_use) {

    // sample data
    std::vector<luke::gfloat> bZ = 
      (s_.base_state_sensor_Z.*stateFcnPtr)(sensors.z_base_position);

    // copy our observation into these vectors instead
    std::copy(observation.begin() + sidx, observation.begin() + sidx + bZ.size(), bZ.begin()); sidx += bZ.size();

    if (debug_obs) {
      luke::print_vec(bZ, "Base state Z");
    }

    std::string n = std::to_string(bZ.size());
    info += "Base state Z = " + n + " | ";
  }

  // get base Z yaw
  if (s_.base_state_sensor_yaw.in_use) {

    // sample data
    std::vector<luke::gfloat> byaw = 
      (s_.base_state_sensor_yaw.*stateFcnPtr)(sensors.yaw_base_rotation);

    // copy our observation into these vectors instead
    std::copy(observation.begin() + sidx, observation.begin() + sidx + byaw.size(), byaw.begin()); sidx += byaw.size();

    if (debug_obs) {
      luke::print_vec(byaw, "Base Z yaw");
    }

    std::string n = std::to_string(byaw.size());
    info += "Base state yaw = " + n + " | ";
  }

  // get cartesian contact positions for MAT
  if (s_.cartesian_contacts_XYZ.in_use) {

    // sample data - ONLY USE CHANGE SAMPLE, ignores the simsettings for state sampling
    std::vector<luke::gfloat> (MjType::Sensor::*cartesianFcnPtr)
      (luke::SlidingWindow<luke::gfloat>);
    cartesianFcnPtr = &MjType::Sensor::change_sample;
    std::vector<luke::gfloat> f1x = (s_.cartesian_contacts_XYZ.*cartesianFcnPtr)(sensors.finger1_x_pos);
    std::vector<luke::gfloat> f1y = (s_.cartesian_contacts_XYZ.*cartesianFcnPtr)(sensors.finger1_y_pos);
    std::vector<luke::gfloat> f1z = (s_.cartesian_contacts_XYZ.*cartesianFcnPtr)(sensors.finger1_z_pos);
    std::vector<luke::gfloat> f2x = (s_.cartesian_contacts_XYZ.*cartesianFcnPtr)(sensors.finger2_x_pos);
    std::vector<luke::gfloat> f2y = (s_.cartesian_contacts_XYZ.*cartesianFcnPtr)(sensors.finger2_y_pos);
    std::vector<luke::gfloat> f2z = (s_.cartesian_contacts_XYZ.*cartesianFcnPtr)(sensors.finger2_z_pos);
    std::vector<luke::gfloat> f3x = (s_.cartesian_contacts_XYZ.*cartesianFcnPtr)(sensors.finger3_x_pos);
    std::vector<luke::gfloat> f3y = (s_.cartesian_contacts_XYZ.*cartesianFcnPtr)(sensors.finger3_y_pos);
    std::vector<luke::gfloat> f3z = (s_.cartesian_contacts_XYZ.*cartesianFcnPtr)(sensors.finger3_z_pos);
    std::vector<luke::gfloat> px = (s_.cartesian_contacts_XYZ.*cartesianFcnPtr)(sensors.palm_x_pos);
    std::vector<luke::gfloat> py = (s_.cartesian_contacts_XYZ.*cartesianFcnPtr)(sensors.palm_y_pos);
    std::vector<luke::gfloat> pz = (s_.cartesian_contacts_XYZ.*cartesianFcnPtr)(sensors.palm_z_pos);

    // insert data into observation output
    observation.insert(observation.end(), f1x.begin(), f1x.end());
    observation.insert(observation.end(), f1y.begin(), f1y.end());
    observation.insert(observation.end(), f1z.begin(), f1z.end());
    observation.insert(observation.end(), f2x.begin(), f2x.end());
    observation.insert(observation.end(), f2y.begin(), f2y.end());
    observation.insert(observation.end(), f2z.begin(), f2z.end());
    observation.insert(observation.end(), f3x.begin(), f3x.end());
    observation.insert(observation.end(), f3y.begin(), f3y.end());
    observation.insert(observation.end(), f3z.begin(), f3z.end());
    observation.insert(observation.end(), px.begin(), px.end());
    observation.insert(observation.end(), py.begin(), py.end());
    observation.insert(observation.end(), pz.begin(), pz.end());

    if (debug_obs) {
      luke::print_vec(f1x, "finger1_x_pos");
      luke::print_vec(f1y, "finger1_y_pos");
      luke::print_vec(f1z, "finger1_z_pos");
      luke::print_vec(f2x, "finger2_x_pos");
      luke::print_vec(f2y, "finger2_y_pos");
      luke::print_vec(f2z, "finger2_z_pos");
      luke::print_vec(f3x, "finger3_x_pos");
      luke::print_vec(f3y, "finger3_y_pos");
      luke::print_vec(f3z, "finger3_z_pos");
      luke::print_vec(px, "palm_x_pos");
      luke::print_vec(py, "palm_y_pos");
      luke::print_vec(pz, "palm_z_pos");
    }

    std::string n = std::to_string(f1x.size());
    info += "finger1 x pos = " + n + " | ";
    n = std::to_string(f1y.size());
    info += "finger1 y pos = " + n + " | ";
    n = std::to_string(f1z.size());
    info += "finger1 z pos = " + n + " | ";
    n = std::to_string(f2x.size());
    info += "finger2 x pos = " + n + " | ";
    n = std::to_string(f2y.size());
    info += "finger2 y pos = " + n + " | ";
    n = std::to_string(f2z.size());
    info += "finger2 z pos = " + n + " | ";
    n = std::to_string(f3x.size());
    info += "finger3 x pos = " + n + " | ";
    n = std::to_string(f3y.size());
    info += "finger3 y pos = " + n + " | ";
    n = std::to_string(f3z.size());
    info += "finger3 z pos = " + n + " | ";
    n = std::to_string(px.size());
    info += "palm x pos = " + n + " | ";
    n = std::to_string(py.size());
    info += "palm y pos = " + n + " | ";
    n = std::to_string(pz.size());
    info += "palm z pos = " + n + " | ";
  }

  if (debug_obs) {
    std::cout << "End of observation (n_obs = " << observation.size() << ")\n";
  }

  info += " Observation length = " + std::to_string(observation.size()) + "\n";
  
  return info;
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

  // remove our record of any objects
  env_.obj.clear();
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
  /* spawn an object at a give (x, y, rotation) */

  MjType::Env::SpawnObj to_spawn;
  to_spawn.index = index;
  to_spawn.x_centre = xpos;
  to_spawn.y_centre = ypos;
  to_spawn.z_rotation = zrot;

  luke::Vec3 bound = luke::get_object_xyz_bounding_box(index);
  to_spawn.model_x = bound.x;
  to_spawn.model_y = bound.y;
  to_spawn.model_z = bound.z;

  spawn_object(to_spawn);
}

void MjClass::spawn_object(MjType::Env::SpawnObj to_spawn)
{
  /* spawn an object beneath the gripper at (xpos, ypos) with a given rotation
  zrot in radians about the vertical axis */

  if (to_spawn.index < 0 or to_spawn.index >= env_.object_names.size()) {
    throw std::runtime_error("bad index to spawn_object()");
  }

  int objvec_idx = -1;

  // check if this object is already live
  if (luke::is_object_live(to_spawn.index)) {
    
    // we want to find the existing entry and update it
    for (int i = 0; i < env_.obj.size(); i++) {
      if (env_.obj[i].name == env_.object_names[to_spawn.index]) {
        objvec_idx = i;
        break;
      }
    }
    if (objvec_idx == -1) {
      std::cout << "New object id = " << to_spawn.index << ", name = " << env_.object_names[to_spawn.index] << '\n';
      std::cout << "Existing objects in the simulation:\n";
      for (int i = 0; i < env_.obj.size(); i++) {
        std::cout << "Object name = " << env_.obj[i].name << '\n';
      }
      throw std::runtime_error("MjClass::spawn_object() has is_object_live=true, but cannot find that object");
    }
  }
  else {

    // create and add a new object entry in the environment variable
    MjType::Env::Obj new_obj;
    env_.obj.push_back(new_obj);
    objvec_idx = env_.obj.size() - 1;

  }

  // set the position to be spawned
  luke::QPos spawn_pos;
  spawn_pos.x = to_spawn.x_centre;
  spawn_pos.y = to_spawn.y_centre;
  spawn_pos.z = -1;     // will automatically be set to keyframe value

  // set the rotation to be spawned
  double x1 = spawn_pos.qx;
  double y1 = spawn_pos.qy;
  double z1 = spawn_pos.qz;
  double w1 = spawn_pos.qw;
  double x2 = 1 * 1 * sin(-to_spawn.z_rotation / 2.0) - 0 * 0 * cos(-to_spawn.z_rotation / 2.0);
  double y2 = 0 * 1 * cos(-to_spawn.z_rotation / 2.0) - 1 * 0 * sin(-to_spawn.z_rotation / 2.0);
  double z2 = 1 * 0 * cos(-to_spawn.z_rotation / 2.0) + 0 * 1 * sin(-to_spawn.z_rotation / 2.0);
  double w2 = 1 * 1 * cos(-to_spawn.z_rotation / 2.0) + 0 * 0 * sin(-to_spawn.z_rotation / 2.0);
  spawn_pos.qw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
  spawn_pos.qx = w1 * x2 + x1 * w2 - y1 * z2 + z1 * y2;
  spawn_pos.qy = w1 * y2 + x1 * z2 + y1 * w2 - z1 * x2;
  spawn_pos.qz = w1 * z2 - x1 * y2 + y1 * x2 + z1 * w2;

  // spawn the object and save its start position
  env_.obj[objvec_idx].start_qpos = luke::spawn_object(model, data, to_spawn.index, spawn_pos);

  // save info on object to be spawned
  env_.obj[objvec_idx].name = env_.object_names[to_spawn.index];
  env_.obj[objvec_idx].spawn_info = to_spawn;

  // update everything for rendering
  forward();
}

bool MjClass::spawn_into_scene(int index)
{
  /* spawn object at index with all other parameters default */

  MjType::SpawnParams p;
  p = default_spawn_params;
  p.index = index;
  return spawn_into_scene(p);
}

bool MjClass::spawn_into_scene(int index, double xpos, double ypos) 
{

  /* spawn an object into a scene, given (x, y) and no range on these values,
  return false if cannot spawn without a collision */

  MjType::SpawnParams p;
  p = default_spawn_params;
  p.index = index;
  p.x = xpos;
  p.y = ypos;
  return spawn_into_scene(p);
}

bool MjClass::spawn_into_scene(int index, double xpos, double ypos, double zrot) 
{
  /* spawn an object into a scene, given (x, y, rot) and no range on these values.
  Return false if cannot spawn without a collision */

  MjType::SpawnParams p;
  p = default_spawn_params;
  p.index = index;
  p.x = xpos;
  p.y = ypos;
  p.zrot = zrot;
  return spawn_into_scene(p);
}

bool MjClass::spawn_into_scene(int index, double xpos, double ypos, double zrot,
  double xrange, double yrange, double rotrange)
{
  MjType::SpawnParams p;
  p = default_spawn_params;
  p.index = index;
  p.x = xpos;
  p.y = ypos;
  p.zrot = zrot;
  p.xrange = xrange;
  p.yrange = yrange;
  p.rotrange = rotrange;
  return spawn_into_scene(p);
}

bool MjClass::spawn_into_scene(MjType::SpawnParams params)
{
  /* spawn an object into a scene, given (x, y, rot) and a range for these values.
  Return false if cannot spawn without a collision */

  // extract object spawning information
  int index = params.index;
  double xpos = params.x;
  double ypos = params.y;
  double zrot = params.zrot;
  double xrange = params.xrange;
  double yrange = params.yrange;
  double rotrange = params.rotrange;
  double xmin = params.xmin;
  double xmax = params.xmax;
  double ymin = params.ymin;
  double ymax = params.ymax;
  double smallest_gap = params.smallest_gap;
  double xy_increment = params.xy_increment;
  double rot_increment = params.rot_increment;

  // for debugging, 0=off, 1=key info, 2=most info, 3=excessive info
  constexpr int debug_level = 0;

  // create a grid of possible points to spawn in an object
  int num_x = ((2 * xrange) / xy_increment) + 1;
  int num_y = ((2 * yrange) / xy_increment) + 1;
  std::vector<std::array<double, 2>> xy_points(num_x * num_y);
  for (int ix = 0; ix < num_x; ix++) {
    for (int iy = 0; iy < num_y; iy++) {

      // does our range exceed our increment
      if (num_x > 1) xy_points[ix * num_y + iy][0] = -xrange + ix * xy_increment + xpos;
      else xy_points[ix * num_y + iy][0] = xpos;

      if (num_y > 1) xy_points[ix * num_y + iy][1] = -yrange + iy * xy_increment + ypos;
      else xy_points[ix * num_y + iy][1] = ypos;
    }
  }

  if (debug_level > 2) {
    for (int i = 0; i < xy_points.size(); i++) {
      std::cout << "raw (x, y) >> ("
        << xy_points[i][0] << ", " << xy_points[i][1] << ")\n";
    }
  }

  // create a vector of possible rotatations
  int num_r = ((2 * rotrange) / rot_increment) + 1;
  std::vector<double> rot_points(num_r);
  for (int ir = 0; ir < num_r; ir++) {

    // does our range exceed our increment
    if (num_r > 1) rot_points[ir] = -rotrange + ir * rot_increment + zrot;
    else rot_points[ir] = zrot;
  }

  // now shuffle the points into a random order (if we have multiple options)
  if (xy_points.size() > 1)
    std::shuffle(std::begin(xy_points), std::end(xy_points), *MjType::generator);
  if (rot_points.size() > 1)
    std::shuffle(std::begin(rot_points), std::end(rot_points), *MjType::generator);

  if (debug_level > 0) {
    std::cout << "There are " << xy_points.size() << " xy points, and "
      << rot_points.size() << " rotation points\n";
    std::cout << "There are " << env_.obj.size() << " existing objects in the scene\n";
    if (debug_level > 1) {
      for (int i = 0; i < env_.obj.size(); i++) {
        std::cout << "index = " << env_.obj[i].spawn_info.index
          << ", name = " << env_.obj[i].name << "\n";
      }
    }
  }

  // determine how much space the new object to spawn needs
  MjType::Env::SpawnObj to_spawn;
  luke::Vec3 obj_xyz = luke::get_object_xyz_bounding_box(index);
  to_spawn.index = index;
  to_spawn.model_x = obj_xyz.x;
  to_spawn.model_y = obj_xyz.y;
  to_spawn.model_z = obj_xyz.z;

  if (debug_level > 0)
    std::cout << "Adding object " << to_spawn.index << ", with (x,y,z) bounding >> ("
      << to_spawn.model_x << ", " << to_spawn.model_y
      << ", " << to_spawn.model_z << ")\n";

  // loop over our points and rotations and try to spawn
  int i_xy = -1;
  int i_rot = -1;
  int total_tries = std::max(xy_points.size(), rot_points.size());
  bool good_spawn_point = false;

  for (int i = 0; i < total_tries; i++) {

    i_xy += 1;
    i_rot += 1;

    if (i_xy >= xy_points.size()) i_xy = 0;
    if (i_rot >= rot_points.size()) i_rot = 0;

    if (debug_level > 1)
      std::cout << "Point " << i << " (x, y, rot) >> ("
        << xy_points[i_xy][0] << ", "
        << xy_points[i_xy][1] << ", "
        << rot_points[i_rot] << ")"
        << " - i_xy is " << i_xy << " and i_rot is " << i_rot
        << "\n";

    good_spawn_point = true;

    // make a box for our object at this point
    luke::Box2d ourBox;
    ourBox.initCentre(xy_points[i_xy][0], xy_points[i_xy][1], to_spawn.model_x,
      to_spawn.model_y);
    ourBox.rotate(rot_points[i_rot]);

    // are we inbounds at the point
    if (not ourBox.inbounds(xmin, ymin, xmax, ymax)) {
      if (debug_level > 1) std::cout << "Can't spawn here, exceed outer bounds\n";
      good_spawn_point = false;
      continue;
    }

    // loop over existing objects and see if we collide with them
    for (int i_obj = 0; i_obj < env_.obj.size(); i_obj++) {

      // check to ensure this existing object is not us
      if (env_.obj[i_obj].spawn_info.index == index) continue;

      // determine the space taken up by the object
      luke::Box2d spawnBox;
      spawnBox.initCentre(env_.obj[i_obj].spawn_info.x_centre, env_.obj[i_obj].spawn_info.y_centre, 
        env_.obj[i_obj].spawn_info.model_x, env_.obj[i_obj].spawn_info.model_y);
      spawnBox.rotate(env_.obj[i_obj].spawn_info.z_rotation);

      if (ourBox.overlapsWith(spawnBox, smallest_gap)) {
        good_spawn_point = false;
        if (debug_level > 1) {
          std::cout << "Can't spawn here, collides with existing object\n";
          std::cout << "This object has (x,y) >> ("
            << env_.obj[i_obj].spawn_info.x_centre << ", " << env_.obj[i_obj].spawn_info.y_centre
            << ") and size (mx, my) >> ("
            << env_.obj[i_obj].spawn_info.model_x << ", " << env_.obj[i_obj].spawn_info.model_y
            << ")\n";
        }
        break;
      }
    }

    if (not good_spawn_point) continue;

    // see if this spawn point clashes with the gripper fingers
    for (luke::Box2d& finger_box : env_.init_fingertip_boxes) {
      if (ourBox.overlapsWith(finger_box, smallest_gap)) {
        good_spawn_point = false;
        if (debug_level > 1) std::cout << "Can't spawn here, hits gripper fingers\n";
        break;
      }
    }

    // we can proceed to spawn the object
    if (good_spawn_point) {
      to_spawn.x_centre = xy_points[i_xy][0];
      to_spawn.y_centre = xy_points[i_xy][1];
      to_spawn.z_rotation = rot_points[i_rot];
      if (debug_level > 0) std::cout << "Found a good spawn point\n";
      break;
    }
  }

  // did we find a point suitable for spawning?
  if (not good_spawn_point) return false;

  // spawn the object
  spawn_object(to_spawn.index, to_spawn.x_centre, to_spawn.y_centre, to_spawn.z_rotation);
  
  return true;
}

int MjClass::spawn_scene(int num_objects, double xrange, double yrange,
  double smallest_gap)
{
  /* spawn a scene of objects in the simulation in the given range. Returns the
  number of spawned objects in case some cannot be spawned */

  // // remove any live objects
  // reset_object();

  // 0=off, 1=key info, 2=all info
  constexpr int debug_level = 0;

  double origin_x = 0.0;
  double origin_y = 0.0;

  std::uniform_real_distribution<double> rotation_dist(0.0, M_PI);

  // determine the spawning bounds
  double xmin = origin_x - xrange;
  double xmax = origin_x + xrange;
  double ymin = origin_y - yrange;
  double ymax = origin_y + yrange;

  // determine the gripper fingertip positions
  std::vector<luke::Vec3> tip_pos = luke::get_finger_hook_locations();
  std::vector<luke::Box2d> tip_boxes;
  for (int i = 0; i < 3; i++) {
    luke::Box2d tip;
    tip.initCentre(tip_pos[i].x, tip_pos[i].y, tip_pos[3 + i].y, tip_pos[3 + i].x);
    tip.rotate(-tip_pos[3 + i].z);
    tip_boxes.push_back(tip);

    if (debug_level > 1)
      std::cout << "Gripper fingertip " << i << " has (x, y) >> ("
        << tip_pos[i].x << ", " << tip_pos[i].y
        << "; width = " << tip_pos[3 + i].x
        << " and height = " << tip_pos[3 + i].y
        << "; rotation = " << tip_pos[3 + i].z
        << "\n";
  }

  // create a grid of possible points to spawn in an object
  constexpr double xy_increment = 2e-3; // 2mm
  int num_x = (2 * xrange) / xy_increment;
  int num_y = (2 * yrange) / xy_increment;
  std::vector<std::array<double, 2>> xy_points(num_x * num_y);

  // loop through and set the xy values
  for (int ix = 0; ix < num_x; ix++) {
    for (int iy = 0; iy < num_y; iy++) {
      xy_points[ix * num_x + iy][0] = -xrange + ix * xy_increment + origin_x;
      xy_points[ix * num_x + iy][1] = -yrange + iy * xy_increment + origin_y;
    }
  }

  // now shuffle the points into a random order
  std::shuffle(std::begin(xy_points), std::end(xy_points), *MjType::generator);

  // now generate a random order of objects
  std::vector<int> obj_idx(env_.object_names.size());
  for (int i = 0; i < obj_idx.size(); i++) {
    obj_idx[i] = i;
  }
  std::shuffle(std::begin(obj_idx), std::end(obj_idx), *MjType::generator);

  // now choose the objects and form them into a list
  std::vector<MjType::Env::SpawnObj> objects(num_objects);

  for (int i = 0; i < num_objects; i++) {
    
    objects[i].index = obj_idx[i];

    // get object size from the centre point
    luke::Vec3 obj_xyz = luke::get_object_xyz_bounding_box(obj_idx[i]);
    objects[i].model_x = obj_xyz.x;
    objects[i].model_y = obj_xyz.y;
    objects[i].model_z = obj_xyz.z;

    if (debug_level > 0)
      std::cout << "Adding object " << objects[i].index << ", with (x,y,z) bounding >> ("
        << objects[i].model_x << ", " << objects[i].model_y
        << ", " << objects[i].model_z << ")\n";
  }

  // now loop over our scene and add objects where we can
  std::vector<MjType::Env::SpawnObj> spawned_objects;
  int spawn_idx = 0;
  for (int i = 0; i < xy_points.size(); i++) {
    
    if (debug_level > 1)
      std::cout << "Now at point (x, y) >> ("
        << xy_points[i][0] << ", " << xy_points[i][1] << ")\n";

    bool good_spawn_point = true;

    // determine the space that we need
    luke::Box2d ourBox;
    ourBox.initCentre(xy_points[i][0], xy_points[i][1], objects[spawn_idx].model_x,
      objects[spawn_idx].model_y);

    // rotate by a random amount
    double rand_rot = rotation_dist(*MjType::generator);
    ourBox.rotate(rand_rot);

    if (not ourBox.inbounds(xmin, ymin, xmax, ymax)) {
        
      if (debug_level > 1) std::cout << "Can't spawn here, exceed outer bounds\n";
      continue;
    }

    // see if we can spawn an object at this point
    for (MjType::Env::SpawnObj& spawned : spawned_objects) {

      // determine the space taken up by the object
      luke::Box2d spawnBox;
      spawnBox.initCentre(spawned.x_centre, spawned.y_centre, spawned.model_x,
        spawned.model_y);
      spawnBox.rotate(spawned.z_rotation);

      if (ourBox.overlapsWith(spawnBox, smallest_gap)) {
        good_spawn_point = false;
        if (debug_level > 1) std::cout << "Can't spawn here, exceed object bounds\n";
        break;
      }
    }

    if (not good_spawn_point) continue;

    // see if this spawn point clashes with the gripper fingers
    for (luke::Box2d& finger_box : tip_boxes) {
      if (ourBox.overlapsWith(finger_box, smallest_gap)) {
        good_spawn_point = false;
        if (debug_level > 1) std::cout << "Can't spawn here, hits gripper fingers\n";
        break;
      }
    }

    if (not good_spawn_point) continue;

    // this is a good place to spawn, record this
    MjType::Env::SpawnObj new_spawn = objects[spawn_idx];
    new_spawn.x_centre = xy_points[i][0];
    new_spawn.y_centre = xy_points[i][1];
    new_spawn.z_rotation = rand_rot;
    spawned_objects.push_back(new_spawn);

    spawn_idx += 1;

    if (spawn_idx == num_objects) break;
  }

  // finally, we can loop over the objects and spawn them
  if (debug_level > 0) std::cout << "Spawning the following:\n";
  for (MjType::Env::SpawnObj& to_spawn : spawned_objects) {

    if (debug_level > 0)
      std::cout << "Spawning object " << to_spawn.index 
        << " called " << env_.object_names[to_spawn.index]
        << " at (x, y) >> (" << to_spawn.x_centre << ", " << to_spawn.y_centre << ")"
        << " with rotation " << to_spawn.z_rotation
        << "; this object has (x, y, z) sizes of (" << to_spawn.model_x
        << ", " << to_spawn.model_y << ", " << to_spawn.model_z << ")"
        << "\n";

    // spawn the object
    spawn_object(to_spawn.index, to_spawn.x_centre, to_spawn.y_centre, to_spawn.z_rotation);
  }

  return spawned_objects.size();
}

void MjClass::set_scene_grasp_target(int num_objects)
{
  /* set a target to grasp a certain number of objects */

  if (num_objects > env_.obj.size()) {
    std::cout << "MjClass::set_scene_grasp_target() warning: num_objects "
      << num_objects << " is greater than number of objects in the scene "
      << env_.obj.size() << ", capping num_objects\n";
  }

  scene_grasp_target = num_objects;
  current_grasp_num = 0;
}

void MjClass::randomise_every_colour()
{
  /* makes every single item in the simulation a random colour */

  randomise_object_colour(true);
  randomise_finger_colours(false);
  randomise_ground_colour();
}

void MjClass::randomise_object_colour(bool all_objects)
{
  /* randomise the colour of the object */

  if (all_objects) {
    luke::randomise_all_object_colours(model, MjType::generator);
  }
  else {

    std::vector<float> rgb(3);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    rgb[0] = distribution(*MjType::generator);
    rgb[1] = distribution(*MjType::generator);
    rgb[2] = distribution(*MjType::generator);

    luke::set_object_colour(model, rgb);
  }
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

void MjClass::randomise_finger_colours(bool all_same)
{
  /* give all the fingers the same random colour */

  if (all_same) {

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
  else {

    // loop over fingers (1-3) and palm (4)
    for (int i = 1; i <=4; i++) {

      std::vector<float> rgb(3);
      std::uniform_real_distribution<double> distribution(0.0, 1.0);
      rgb[0] = distribution(*MjType::generator);
      rgb[1] = distribution(*MjType::generator);
      rgb[2] = distribution(*MjType::generator);

      // set each finger in turn to be different colours
      luke::set_finger_colour(model, rgb, i);
    }
  }
}

std::vector<int> MjClass::convert_segmentation_array(std::vector<int>& array)
{
  /* convert a segmented array into the seperate gripper parts */

  return luke::convert_segmentation_array(array);
}

void MjClass::set_neat_colours()
{
  /* set nice colours for the items/objects in the scene */

  float x = 1.0 / 255.0;

  std::vector<float> object_colour  {50*x,  205*x, 50*x};
  std::vector<float> gripper_colour {220*x, 220*x, 220*x};
  std::vector<float> finger_colour  {255*x, 140*x, 0*x};
  std::vector<float> ground_colour  {100*x, 100*x, 100*x};

  luke::set_ground_colour(model, ground_colour);
  luke::set_all_objects_colour(model, object_colour);
  luke::set_main_body_colour(model, gripper_colour);

  for (int i = 1; i < 5; i++) // 4 means palm
    luke::set_finger_colour(model, finger_colour, i);

}

void MjClass::create_object_mask()
{
  /* mask out only all the objects in the scene */

  // turn everything black first
  std::vector<float> black { 0, 0, 0, 0 };
  luke::set_everything_colour(model, black);

  // now turn all the objects white
  std::vector<float> white { 1, 1, 1, 1 };
  luke::set_all_objects_colour(model, white);
}

void MjClass::create_gripper_mask()
{
  /* mask out the gripper parts in the scene */

  // turn everything black first
  std::vector<float> black { 0, 0, 0, 0 };
  luke::set_everything_colour(model, black);

  // now turn all the gripper parts white
  std::vector<float> white { 1, 1, 1, 1 };
  luke::set_finger_colour(model, white, 1);
  luke::set_finger_colour(model, white, 2);
  luke::set_finger_colour(model, white, 3);
  luke::set_finger_colour(model, white, 4); // 4 means palm
}

void MjClass::create_finger_mask(int num)
{
  /* mask out only one gripper finger (1,2,3) or the palm (4)*/

  // turn everything black first
  std::vector<float> black { 0, 0, 0, 0 };
  luke::set_everything_colour(model, black);

  // now turn all the gripper parts white
  std::vector<float> white { 1, 1, 1, 1 };
  luke::set_finger_colour(model, white, num);
}

void MjClass::create_ground_mask()
{
  /* show up only the ground in view */

  // turn everything black first
  std::vector<float> black { 0, 0, 0, 1 };
  luke::set_everything_colour(model, black);

  // now turn the ground white
  std::vector<float> white { 1, 1, 1, 1 };
  luke::set_ground_colour(model, white);
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

  // if we are using MAT and a reopen was triggered too recently
  if (apply_MAT_reopen_penalty) {
    if (s_.debug) 
      std::cout << "MAT reopen triggered too recently, reward -= 0.05\n";
    transition_reward -= 0.05;
    apply_MAT_reopen_penalty = false;
  }
   
  // this value is not used in python
  env_.cumulative_reward += transition_reward;

  // if we are capping the maximum cumulative negative reward
  if (env_.cumulative_reward < s_.reward_cap_lower_bound) {
    if (s_.cap_reward) {
      // reduce the reward to not put us below the cap
      transition_reward += s_.reward_cap_lower_bound - env_.cumulative_reward;
      env_.cumulative_reward = s_.reward_cap_lower_bound;
    }
  }

  // if we are capping the maximum cumulative positive reward
  if (env_.cumulative_reward > s_.reward_cap_upper_bound) {
    if (s_.cap_reward) {
      // reduce the reward to not put us above the cap
      transition_reward += s_.reward_cap_upper_bound - env_.cumulative_reward;
      env_.cumulative_reward = s_.reward_cap_upper_bound;
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

/* ----- sensor functions ----- */

std::vector<luke::gfloat> MjClass::get_finger_forces(bool realworld)
{
  /* return a vector [g1, g2, g3] of the three bend gauges last reading */

  /* WARNING: in simulation raw unnormalised bend gauge readings are NOT SI */

  std::vector<luke::gfloat> readings(3);

  if (realworld) {
    // return calibrated values in SI units
    readings[0] = real_sensors_.SI.read_finger1_gauge();
    readings[1] = real_sensors_.SI.read_finger2_gauge();
    readings[2] = real_sensors_.SI.read_finger3_gauge();
  }
  else {
    // map back from [-1, +1] to [min, max] in SI units
    readings[0] = sim_sensors_.read_finger1_gauge() * s_.bending_gauge.normalise;
    readings[1] = sim_sensors_.read_finger2_gauge() * s_.bending_gauge.normalise;
    readings[2] = sim_sensors_.read_finger3_gauge() * s_.bending_gauge.normalise;
  }

  return readings;
}

luke::gfloat MjClass::get_palm_force(bool realworld)
{
  /* get the last palm reading */

  luke::gfloat reading;
  
  if (realworld) {
    reading = real_sensors_.SI.read_palm_sensor();
  }
  else {
    // map back from [-1, +1] to [min, max] in SI units
    reading = sim_sensors_.read_palm_sensor() * s_.palm_sensor.normalise;
  }

  return reading;
}

luke::gfloat MjClass::get_wrist_force(bool realworld)
{
  /* get the last wrist Z reading */

  luke::gfloat reading;
  
  if (realworld) {
    reading = real_sensors_.SI.read_wrist_Z_sensor();
  }
  else {
    // map back from [-1, +1] to [min, max] in SI units
    reading = sim_sensors_.read_wrist_Z_sensor() * s_.wrist_sensor_Z.normalise;
  }

  return reading;
}

std::vector<luke::gfloat> MjClass::get_state_metres(bool realworld)
{
  /* get a vector [gripperx, grippery, gripperz, basez] or if using base
  xyz it will be [gripperx, grippery, gripperz, basex, basey, basez] */

  // motor state vector elements
  int n = 4;

  // if using base xyz we add two elements for xy
  bool base_xy = s_.base_state_sensor_XY.in_use;
  if (base_xy) n += 2;

  std::vector<luke::gfloat> readings(n);

  if (realworld) {
    readings[0] = real_sensors_.SI.read_x_motor_position();
    readings[1] = real_sensors_.SI.read_y_motor_position();
    readings[2] = real_sensors_.SI.read_z_motor_position();
    if (base_xy) {
      readings[3] = real_sensors_.SI.read_x_base_position();
      readings[4] = real_sensors_.SI.read_y_base_position();
      readings[5] = real_sensors_.SI.read_z_base_position();
    }
    else {
      readings[3] = real_sensors_.SI.read_z_base_position();
    }
  }
  else {
    // map back from [-1, +1] to [min, max] in SI units
    readings[0] = unnormalise_from(
      sim_sensors_.read_x_motor_position(), luke::Gripper::xy_min, luke::Gripper::xy_max); 
    readings[1] = unnormalise_from(
      sim_sensors_.read_y_motor_position(), luke::Gripper::xy_min, luke::Gripper::xy_max);
    readings[2] = unnormalise_from(
      sim_sensors_.read_z_motor_position(), luke::Gripper::z_min, luke::Gripper::z_max);
    if (base_xy) {
      readings[3] = unnormalise_from(
        sim_sensors_.read_x_base_position(), base_min_[0], base_max_[0]);
      readings[4] = unnormalise_from(
        sim_sensors_.read_y_base_position(), base_min_[1], base_max_[1]);
      readings[5] = unnormalise_from(
        sim_sensors_.read_z_base_position(), base_min_[2], base_max_[2]);
    }
    else {
      readings[3] = unnormalise_from(
        sim_sensors_.read_z_base_position(), base_min_[2], base_max_[2]);
    }
  }

  return readings;
}

luke::gfloat MjClass::get_finger_angle()
{
  /* return in radians the finger angle */

  return luke::get_target_finger_angle();
}

/* ----- real gripper functions ----- */

void MjClass::calibrate_real_sensors()
{
  /* configure the calibration of real sensors. Running this function will wipe
  and zero-ing done at runtime, therefore this function sets a flag to reset offsets */

  // initialise a clean set of calibrations
  MjType::RealCalibrations real_calibrations;

  // select calibrations suitable for the fingers in use
  double width = luke::get_finger_width();
  double thickness = luke::get_finger_thickness();
  real_sensors_.g1 = real_calibrations.get_gauge_calibration(1, thickness, width);
  real_sensors_.g2 = real_calibrations.get_gauge_calibration(2, thickness, width);
  real_sensors_.g3 = real_calibrations.get_gauge_calibration(3, thickness, width);

  // get wrist and palm calibrations
  real_sensors_.palm = real_calibrations.palm;
  real_sensors_.wrist_Z = real_calibrations.wrist_Z;

  // apply normalisation to all sensors
  /* we CANT use s_.bending_gauge.normalise as this is NOT an SI value. For the other
  sensors this is a force in newtons, but for the simulated gauges this is actually
  an arbritrary scaling factor to go from our 'strain' measurement to SI. This value
  is calibrated based on a known SI force in 'calibrate_simulated_sensors()'. Instead
  we should use the value of this known SI force - as this is the actual saturation/
  normalisation value. The expression for this value is:
    s_.saturation_yield_factor * yield_load() */
  real_sensors_.g1.norm = s_.saturation_yield_factor * yield_load();
  real_sensors_.g2.norm = s_.saturation_yield_factor * yield_load();
  real_sensors_.g3.norm = s_.saturation_yield_factor * yield_load();
  real_sensors_.palm.norm = s_.palm_sensor.normalise;
  real_sensors_.wrist_Z.norm = s_.wrist_sensor_Z.normalise;

  // re-zero all sensors, recalculate their offsets, wipe all saved real data
  real_sensors_.reset();
}

std::vector<float> MjClass::input_real_data(std::vector<float> state_data, 
  std::vector<float> sensor_data)
{
  /* insert real data. Certain quantities (motors/gauges/palm) must always be input and saved */

  // safety check to ensure we configure and calibrate before running with real data
  static bool first_call = true;
  if (first_call) {
    configure_settings();
    calibrate_real_sensors();
    first_call = false;
  }

  // what states should be input (must be in this order!)
  bool motors = true; // always input, see rl_grasping_node.py - s_.motor_state_sensor.in_use;
  bool base_xy = s_.base_state_sensor_XY.in_use;
  bool base_z = s_.base_state_sensor_Z.in_use;
  bool base_z_rot = s_.base_state_sensor_yaw.in_use;
  int state_expected_length = 3*motors + 2*base_xy + base_z + base_z_rot; // hardcoded amounts of data for each state
  if (state_data.size() != state_expected_length) {
    throw std::runtime_error("MjClass::input_real_data() error: state_data has expected length = " + std::to_string(state_expected_length) + ", but actual length is " + std::to_string(state_data.size()));
  }

  // what sensors should be input (must be in this order!)
  bool gauges = true; // always input, see rl_grasping_node.py - s_.bending_gauge.in_use;
  bool palm = true; // always input, see rl_grasping_node.py - s_.palm_sensor.in_use;
  bool wrist_Z = true; // always input, see rl_grasping_node.py - s_.wrist_sensor_Z.in_use;
  bool wrist_XY = false; // not yet added to this function - s_.wrist_sensor_XY.in_use; 
  bool cart_XYZ = s_.cartesian_contacts_XYZ.in_use;
  int sensor_expected_length = 3*gauges + palm + 2*wrist_XY + wrist_Z + 12*cart_XYZ; // hardcoded amounts of data for each sensor
  if (sensor_data.size() != sensor_expected_length) {
    throw std::runtime_error("MjClass::input_real_data() error: sensor_data has expected length = " + std::to_string(sensor_expected_length) + ", but actual length is " + std::to_string(sensor_data.size()));
  }

  // vector which outputs all the freshly normalised values
  std::vector<float> output;

  // count data inputs
  samples_since_last_obs += 1;

  // add state data
  int i = 0;

  // uncomment for debugging
  // std::cout << "Adding state noise of " << s_.motor_state_sensor.noise_std << '\n';
  // std::cout << "Adding sensor noise of " << s_.bending_gauge.noise_std << '\n';

  if (motors) {

    // normalise and save state data
    real_sensors_.raw.x_motor_position.add(state_data[i]);
    real_sensors_.SI.x_motor_position.add(state_data[i]);
    state_data[i] = normalise_between(
      state_data[i], luke::Gripper::xy_min, luke::Gripper::xy_max);
    state_data[i] = s_.motor_state_sensor.apply_noise(state_data[i], uniform_dist, 1);
    real_sensors_.normalised.x_motor_position.add(state_data[i]);
    output.push_back(state_data[i]);
    ++i; 

    real_sensors_.raw.y_motor_position.add(state_data[i]);
    real_sensors_.SI.y_motor_position.add(state_data[i]);
    state_data[i] = normalise_between(
      state_data[i], luke::Gripper::xy_min, luke::Gripper::xy_max);
    state_data[i] = s_.motor_state_sensor.apply_noise(state_data[i], uniform_dist, 2);
    real_sensors_.normalised.y_motor_position.add(state_data[i]);
    output.push_back(state_data[i]);
    ++i; 

    real_sensors_.raw.z_motor_position.add(state_data[i]);
    real_sensors_.SI.z_motor_position.add(state_data[i]);
    state_data[i] = normalise_between(
      state_data[i], luke::Gripper::z_min, luke::Gripper::z_max);
    state_data[i] = s_.motor_state_sensor.apply_noise(state_data[i], uniform_dist, 3);
    real_sensors_.normalised.z_motor_position.add(state_data[i]);
    output.push_back(state_data[i]);
    ++i; 
  }

  if (base_xy) {

    real_sensors_.raw.x_base_position.add(state_data[i]);
    real_sensors_.SI.x_base_position.add(state_data[i]);
    state_data[i] = normalise_between(state_data[i], base_min_[0], base_max_[0]);
    state_data[i] = s_.base_state_sensor_XY.apply_noise(state_data[i], uniform_dist, 1);
    real_sensors_.normalised.x_base_position.add(state_data[i]);
    output.push_back(state_data[i]);
    ++i; 

    real_sensors_.raw.y_base_position.add(state_data[i]);
    real_sensors_.SI.y_base_position.add(state_data[i]);
    state_data[i] = normalise_between(state_data[i], base_min_[1], base_max_[1]);
    state_data[i] = s_.base_state_sensor_XY.apply_noise(state_data[i], uniform_dist, 2);
    real_sensors_.normalised.y_base_position.add(state_data[i]);
    output.push_back(state_data[i]);
    ++i;

  }
  
  if (base_z) {

    real_sensors_.raw.z_base_position.add(state_data[i]);
    real_sensors_.SI.z_base_position.add(state_data[i]);
    state_data[i] = normalise_between(state_data[i], base_min_[2], base_max_[2]);
    state_data[i] = s_.base_state_sensor_Z.apply_noise(state_data[i], uniform_dist);
    real_sensors_.normalised.z_base_position.add(state_data[i]);
    output.push_back(state_data[i]);
    ++i; 

  }

  if (base_z_rot) {

    real_sensors_.raw.yaw_base_rotation.add(state_data[i]);
    real_sensors_.SI.yaw_base_rotation.add(state_data[i]);
    state_data[i] = normalise_between(state_data[i], base_min_[5], base_max_[5]);
    state_data[i] = s_.base_state_sensor_yaw.apply_noise(state_data[i], uniform_dist);
    real_sensors_.normalised.yaw_base_rotation.add(state_data[i]);
    output.push_back(state_data[i]);
    ++i; 

  }

  // add sensor data - pay attention to order! Input vector must be the same
  int j = 0;

  if (gauges) {

    // calibrate the finger 1 sensor
    if (real_sensors_.f1_calibration.size() < real_sensors_.calibration_samples) {
      real_sensors_.f1_calibration.push_back(sensor_data[j]);
      real_sensors_.g1.offset = 0;
      for (int k = 0; k < real_sensors_.f1_calibration.size(); k++) {
        real_sensors_.g1.offset += real_sensors_.f1_calibration[k];
      }
      real_sensors_.g1.offset /= (float) real_sensors_.f1_calibration.size();
    }

    // scale, normalise, and save gauge data
    real_sensors_.raw.finger1_gauge.add(sensor_data[j]);
    sensor_data[j] = real_sensors_.g1.apply_calibration(sensor_data[j]);
    real_sensors_.SI.finger1_gauge.add(sensor_data[j]);
    sensor_data[j] = real_sensors_.g1.apply_normalisation(sensor_data[j]);
    sensor_data[j] = s_.bending_gauge.apply_noise(sensor_data[j], uniform_dist, 1);
    real_sensors_.normalised.finger1_gauge.add(sensor_data[j]);
    output.push_back(sensor_data[j]);
    ++j;

    // calibrate the finger 2 sensor
    if (real_sensors_.f2_calibration.size() < real_sensors_.calibration_samples) {
      real_sensors_.f2_calibration.push_back(sensor_data[j]);
      real_sensors_.g2.offset = 0;
      for (int k = 0; k < real_sensors_.f2_calibration.size(); k++) {
        real_sensors_.g2.offset += real_sensors_.f2_calibration[k];
      }
      real_sensors_.g2.offset /= (float) real_sensors_.f2_calibration.size();
    }

    real_sensors_.raw.finger2_gauge.add(sensor_data[j]);
    sensor_data[j] = real_sensors_.g2.apply_calibration(sensor_data[j]);
    real_sensors_.SI.finger2_gauge.add(sensor_data[j]);
    sensor_data[j] = real_sensors_.g2.apply_normalisation(sensor_data[j]);
    sensor_data[j] = s_.bending_gauge.apply_noise(sensor_data[j], uniform_dist, 2);
    real_sensors_.normalised.finger2_gauge.add(sensor_data[j]);
    output.push_back(sensor_data[j]);
    ++j;

    // calibrate the finger 3 sensor
    if (real_sensors_.f3_calibration.size() < real_sensors_.calibration_samples) {
      real_sensors_.f3_calibration.push_back(sensor_data[j]);
      real_sensors_.g3.offset = 0;
      for (int k = 0; k < real_sensors_.f3_calibration.size(); k++) {
        real_sensors_.g3.offset += real_sensors_.f3_calibration[k];
      }
      real_sensors_.g3.offset /= (float) real_sensors_.f3_calibration.size();
    }
  
    real_sensors_.raw.finger3_gauge.add(sensor_data[j]);
    sensor_data[j] = real_sensors_.g3.apply_calibration(sensor_data[j]);
    real_sensors_.SI.finger3_gauge.add(sensor_data[j]);
    sensor_data[j] = real_sensors_.g3.apply_normalisation(sensor_data[j]);
    sensor_data[j] = s_.bending_gauge.apply_noise(sensor_data[j], uniform_dist, 3);
    real_sensors_.normalised.finger3_gauge.add(sensor_data[j]);
    output.push_back(sensor_data[j]);
    ++j;

  }

  if (palm) {

    // calibrate the palm sensor
    if (real_sensors_.palm_calibration.size() < real_sensors_.calibration_samples) {
      real_sensors_.palm_calibration.push_back(sensor_data[j]);
      real_sensors_.palm.offset = 0;
      for (int k = 0; k < real_sensors_.palm_calibration.size(); k++) {
        real_sensors_.palm.offset += real_sensors_.palm_calibration[k];
      }
      real_sensors_.palm.offset /= (float) real_sensors_.palm_calibration.size();
    }

    // scale, normalise, and save gauge data
    real_sensors_.raw.palm_sensor.add(sensor_data[j]);
    sensor_data[j] = real_sensors_.palm.apply_calibration(sensor_data[j]);
    real_sensors_.SI.palm_sensor.add(sensor_data[j]);
    sensor_data[j] = real_sensors_.palm.apply_normalisation(sensor_data[j]);
    sensor_data[j] = s_.palm_sensor.apply_noise(sensor_data[j], uniform_dist);
    real_sensors_.normalised.palm_sensor.add(sensor_data[j]); 
    output.push_back(sensor_data[j]);
    ++j;
    
  }

  if (wrist_Z) {

    constexpr bool debug_wrist_Z = false;
    float pre_cal = sensor_data[j]; // for debugging only

    // hardcoded from mujoco: wrist sensor starts at -0.832, *28=23.3
    float target_wrist_value = 0; // was 23.3

    // calibrate the wrist sensor
    if (real_sensors_.wrist_Z_calibration.size() < real_sensors_.calibration_samples) {

      // known error: wrist sensor initially gives out (0,0,0,0,0,0)
      constexpr float tol = 1e-5;
      if (abs(sensor_data[j]) > tol) {

        // add the data to calibration vector, tally up and calculate
        real_sensors_.wrist_Z_calibration.push_back(sensor_data[j]);
        real_sensors_.wrist_Z.offset = 0;
        for (int k = 0; k < real_sensors_.wrist_Z_calibration.size(); k++) {
          real_sensors_.wrist_Z.offset += real_sensors_.wrist_Z_calibration[k]  - target_wrist_value;
        }
        real_sensors_.wrist_Z.offset /= (float) real_sensors_.wrist_Z_calibration.size();
      }
    }

    // scale, normalise, and save wrist data
    real_sensors_.raw.wrist_Z_sensor.add(sensor_data[j]);
    sensor_data[j] = real_sensors_.wrist_Z.apply_calibration(sensor_data[j]);
    real_sensors_.SI.wrist_Z_sensor.add(sensor_data[j]);
    sensor_data[j] = real_sensors_.wrist_Z.apply_normalisation(sensor_data[j]);
    sensor_data[j] = s_.wrist_sensor_Z.apply_noise(sensor_data[j], uniform_dist);
    real_sensors_.normalised.wrist_Z_sensor.add(sensor_data[j]); 
    output.push_back(sensor_data[j]);
    ++j;

    if (debug_wrist_Z) {
      float post_cal = (pre_cal - real_sensors_.wrist_Z.offset) * real_sensors_.wrist_Z.scale;
      std::cout << "Wrist sensor data raw " << pre_cal << ", after scaling "
        << post_cal << ", normalised " << sensor_data[j-1] << '\n';
    }
  }

  if (cart_XYZ) {

    // add cartesian contacts, no calibration, scaling or normalisation necessary
    real_sensors_.raw.finger1_x_pos.add(sensor_data[j]);
    real_sensors_.SI.finger1_x_pos.add(sensor_data[j]);
    real_sensors_.normalised.finger1_x_pos.add(sensor_data[j]);
    output.push_back(sensor_data[j]);
    ++j;

    real_sensors_.raw.finger1_y_pos.add(sensor_data[j]);
    real_sensors_.SI.finger1_y_pos.add(sensor_data[j]);
    real_sensors_.normalised.finger1_y_pos.add(sensor_data[j]);
    output.push_back(sensor_data[j]);
    ++j;

    real_sensors_.raw.finger1_z_pos.add(sensor_data[j]);
    real_sensors_.SI.finger1_z_pos.add(sensor_data[j]);
    real_sensors_.normalised.finger1_z_pos.add(sensor_data[j]);
    output.push_back(sensor_data[j]);
    ++j;

    real_sensors_.raw.finger2_x_pos.add(sensor_data[j]);
    real_sensors_.SI.finger2_x_pos.add(sensor_data[j]);
    real_sensors_.normalised.finger2_x_pos.add(sensor_data[j]);
    output.push_back(sensor_data[j]);
    ++j;

    real_sensors_.raw.finger2_y_pos.add(sensor_data[j]);
    real_sensors_.SI.finger2_y_pos.add(sensor_data[j]);
    real_sensors_.normalised.finger2_y_pos.add(sensor_data[j]);
    output.push_back(sensor_data[j]);
    ++j;

    real_sensors_.raw.finger2_z_pos.add(sensor_data[j]);
    real_sensors_.SI.finger2_z_pos.add(sensor_data[j]);
    real_sensors_.normalised.finger2_z_pos.add(sensor_data[j]);
    output.push_back(sensor_data[j]);
    ++j;

    real_sensors_.raw.finger3_x_pos.add(sensor_data[j]);
    real_sensors_.SI.finger3_x_pos.add(sensor_data[j]);
    real_sensors_.normalised.finger3_x_pos.add(sensor_data[j]);
    output.push_back(sensor_data[j]);
    ++j;

    real_sensors_.raw.finger3_y_pos.add(sensor_data[j]);
    real_sensors_.SI.finger3_y_pos.add(sensor_data[j]);
    real_sensors_.normalised.finger3_y_pos.add(sensor_data[j]);
    output.push_back(sensor_data[j]);
    ++j;

    real_sensors_.raw.finger3_z_pos.add(sensor_data[j]);
    real_sensors_.SI.finger3_z_pos.add(sensor_data[j]);
    real_sensors_.normalised.finger3_z_pos.add(sensor_data[j]);
    output.push_back(sensor_data[j]);
    ++j;

    real_sensors_.raw.palm_x_pos.add(sensor_data[j]);
    real_sensors_.SI.palm_x_pos.add(sensor_data[j]);
    real_sensors_.normalised.palm_x_pos.add(sensor_data[j]);
    output.push_back(sensor_data[j]);
    ++j;

    real_sensors_.raw.palm_y_pos.add(sensor_data[j]);
    real_sensors_.SI.palm_y_pos.add(sensor_data[j]);
    real_sensors_.normalised.palm_y_pos.add(sensor_data[j]);
    output.push_back(sensor_data[j]);
    ++j;

    real_sensors_.raw.palm_z_pos.add(sensor_data[j]);
    real_sensors_.SI.palm_z_pos.add(sensor_data[j]);
    real_sensors_.normalised.palm_z_pos.add(sensor_data[j]);
    output.push_back(sensor_data[j]);
    ++j;
  }

  // // add timestamp data - not used currently
  // gauge_timestamps.add(timestamp);
  // palm_timestamps.add(timestamp);

  return output;
}

std::vector<float> MjClass::get_real_observation()
{
  /* get an observation of real data, samples_since_last_obs should NOT be inclusive, 
  so if before we had [0,1,2] and now we have [0,1,2,3,4,5] then n=3 */

  // manually set reading settings to ensure correctness
  s_.motor_state_sensor.update_n_readings(samples_since_last_obs, s_.state_n_prev_steps);
  s_.base_state_sensor_XY.update_n_readings(samples_since_last_obs, s_.state_n_prev_steps);
  s_.base_state_sensor_Z.update_n_readings(samples_since_last_obs, s_.state_n_prev_steps);
  s_.base_state_sensor_yaw.update_n_readings(samples_since_last_obs, s_.state_n_prev_steps);
  s_.cartesian_contacts_XYZ.update_n_readings(samples_since_last_obs, s_.state_n_prev_steps);
  s_.bending_gauge.update_n_readings(samples_since_last_obs, s_.sensor_n_prev_steps);
  s_.palm_sensor.update_n_readings(samples_since_last_obs, s_.sensor_n_prev_steps);
  s_.wrist_sensor_Z.update_n_readings(samples_since_last_obs, s_.sensor_n_prev_steps);

  // reset as we are about to return an observation
  samples_since_last_obs = 0;

  return get_observation(real_sensors_.normalised);
}

std::vector<float> MjClass::get_simple_state_vector(MjType::SensorData sensor)
{
  /* return a simple state vector - that is with only the most recent reading
  from each sensor. So with 5 sensors and 4 state then the simple state is 9 elements.
  Make sure the ordering of the vector in this function is the same as in
  get_observation() */

  // use for printing detailed observation debug information
  constexpr bool debug_obs = false;
  constexpr bool return_all = true;

  // what states should be returned (check its the same as input_real_data())
  bool motors = true; // always input, see rl_grasping_node.py - s_.motor_state_sensor.in_use;
  bool base_xy = s_.base_state_sensor_XY.in_use;
  bool base_z = s_.base_state_sensor_Z.in_use;
  bool base_z_rot = s_.base_state_sensor_yaw.in_use;

  // what sensors should be returned (check its the same as input_real_data())
  bool gauges = true; // always input, see rl_grasping_node.py - s_.bending_gauge.in_use;
  bool palm = true; // always input, see rl_grasping_node.py - s_.palm_sensor.in_use;
  bool wrist_Z = true; // always input, see rl_grasping_node.py - s_.wrist_sensor_Z.in_use;
  bool wrist_XY = false; // not yet added to this function - s_.wrist_sensor_XY.in_use; 
  bool cart_XYZ = false; // not interested in this - s_.cartesian_contacts_XYZ.in_use;

  std::vector<luke::gfloat> simple_state;

  // get bending strain gauge sensor output
  if (gauges) {

    simple_state.push_back(sensor.read_finger1_gauge());
    simple_state.push_back(sensor.read_finger2_gauge());
    simple_state.push_back(sensor.read_finger3_gauge());
  }

  // get palm sensor output
  if (palm) {

    simple_state.push_back(sensor.read_palm_sensor());
  }

  // get wrist sensor XY output
  if (wrist_XY) {

    simple_state.push_back(sensor.read_wrist_X_sensor());
    simple_state.push_back(sensor.read_wrist_Y_sensor());
  }

  // get wrist sensor Z output
  if (wrist_Z) {
    
    simple_state.push_back(sensor.read_wrist_Z_sensor());
  }

  // get motor state output
  if (motors) {

    simple_state.push_back(sensor.read_x_motor_position());
    simple_state.push_back(sensor.read_y_motor_position());
    simple_state.push_back(sensor.read_z_motor_position());
  }

  // get base XY state
  if (base_xy) {

    simple_state.push_back(sensor.read_x_base_position());
    simple_state.push_back(sensor.read_y_base_position());
  }

  // get base Z state
  if (base_z) {

    simple_state.push_back(sensor.read_z_base_position());
  }

  // get base Z yaw
  if (base_z_rot) {

    simple_state.push_back(sensor.read_yaw_base_rotation());
  }

  return simple_state;
}

std::vector<float> MjClass::get_SI_gauge_forces(std::vector<float> raw_gauges)
{
  /* special function to return the SI force values for a raw gauge input, without
  saving it. Takes a 4 element input (g1, g2, g3, palm) */

  std::vector<float> output(4);

  if (raw_gauges.size() != 4)
    throw std::runtime_error("MjClass::get_SI_gauge_forces() error: raw_gauge input values did not have length 4");

  // calibrate raw readings to SI values in Newtons (assumes offset calibrated from 20 calls to 'input_real_data()')
  output[0] = real_sensors_.g1.apply_calibration(raw_gauges[0]);
  output[1] = real_sensors_.g2.apply_calibration(raw_gauges[1]);
  output[2] = real_sensors_.g3.apply_calibration(raw_gauges[2]);
  output[3] = real_sensors_.palm.apply_calibration(raw_gauges[3]);

  return output;
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

std::string MjClass::print_actions()
{
  /* print out (and return a string) for each possible action */

  std::string str;

  for (int i = 0; i < n_actions; i++) {

    int action_code = action_options[i];

    str += "Action number " + std::to_string(i) +  ", name = ";

    switch (action_code) {

      // define action behaviour for positive/negative/continous
      // any new actions should be further defined in ActionSettings::update_action_function()
      #define AA(NAME, USED, VALUE, SIGN)              \
        case MjType::Action::TOKEN_CONCAT(NAME, POSITIVE_TOKEN):  \
          str += s_.NAME.name + "_positive\n";  \
          break;                                                  \
        case MjType::Action::TOKEN_CONCAT(NAME, NEGATIVE_TOKEN):  \
          str += s_.NAME.name + "_negative\n";  \
          break;                                                  \
        case MjType::Action::TOKEN_CONCAT(NAME, CONTINOUS_TOKEN): \
          str += s_.NAME.name + "_continous\n"; \
          break;                                                  \

        // run the macro to create the code
        LUKE_MJSETTINGS_ACTION

      #undef AA

      case MjType::Action::termination_signal:
        str += "termination_action\n";
        break;

      default:
        throw std::runtime_error("MjClass::print_actions() received out of bounds int");

    }
  }

  std::cout << str;

  return str;
}

float MjClass::get_fingertip_z_height()
{
  /* return the z height of the fingertips, given that the starting position
  is 0.0, so if the fingers start 10mm above the ground, a value of -10mm or
  less indicates ground contact */

  return luke::get_fingertip_z_height();
}

std::vector<std::vector<double>> MjClass::get_object_XY_relative_to_gripper()
{
  /* get the relative XY position of all the objects in the scene, relative
  to where the gripper target currently is */

  return luke::get_object_XY_relative_to_gripper(model, data);
}

std::vector<std::vector<double>> MjClass::get_object_bounding_boxes()
{
  /* get the bounding boxes of all objects in the scene. Note that these objects
  are spawned but may not be in view of the camera */

  return luke::get_live_object_bounding_boxes();
}

MjType::TestReport MjClass::get_test_report()
{
  /* fills out and returns the test report */

  testReport_.object_name = env_.obj[0].name;
  testReport_.cumulative_reward = env_.cumulative_reward;
  testReport_.cnt = env_.cnt;

  return testReport_;
}

void MjClass::set_finger_thickness(double thickness)
{
  /* set a new finger thickness for the gripper. This does not affect the URDF model,
  but updates the finger stiffness behaviour. It will also throw off the gauge
  calibration so if auto-calibration is on then it recalibrates */

  // change the finger thickness, but full changes occur on call to reset()
  bool changed = luke::change_finger_thickness(thickness);

  // changes are finished upon next call to reset()
  if (changed) resetFlags.finger_EI_changed = true;
}

void MjClass::set_finger_width(double width)
{
  /* set a new finger width for the gripper. For the actual width to change a new
  URDF should have been or about to be loaded. Since EI has changed we need new
  finger stiffnesses. It will also throw off the gauge calibration so if auto-calibration
  is on then it recalibrates */

  bool changed = luke::change_finger_width(width);

  // changes are finished upon next call to reset()
  if (changed) resetFlags.finger_EI_changed = true;
}

void MjClass::set_finger_modulus(double E)
{
  /* set the youngs modulus for the finger */

  bool changed = luke::change_youngs_modulus(E);

  // changes are finished upon next call to reset()
  if (changed) resetFlags.finger_EI_changed = true;
}

void MjClass::set_base_XYZ_limits(double x, double y, double z)
{
  /* set the base XYZ limits */

  luke::set_base_XYZ_limits(x, y, z);
}

void MjClass::set_base_yaw_limit(double yaw)
{
  /* set the base yaw rotation limit (rotation about z axis) */

  luke::set_base_yaw_limit(yaw);
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

int MjClass::get_N()
{
  /* return the number of free segments, N */

  return luke::get_N();
}

double MjClass::get_finger_thickness()
{
  /* get the current saved finger thickness */

  return luke::get_finger_thickness();
}

double MjClass::get_finger_width()
{
  /* get the current saved finger width */

  return luke::get_finger_width();
}

double MjClass::get_finger_modulus()
{
  /* get the current saved finger youngs modulus */

  return luke::get_youngs_modulus();
}

double MjClass::get_finger_rigidity()
{
  /* get the current saved value of finger rigidity */

  return luke::get_finger_rigidity();
}

double MjClass::get_finger_length()
{
  /* get the current saved finger length */

  return luke::get_finger_length();
}

double MjClass::get_finger_hook_length()
{
  /* return the gripper finger hook length in mm */

  return luke::get_finger_hook_length();
}

double MjClass::get_finger_hook_angle_degrees()
{
  /* get the current saved finger hook angle, only valid if fixed */

  return luke::get_finger_hook_angle_degrees();
}

bool MjClass::is_finger_hook_fixed()
{
  /* get whether the finger hook is fixed */

  return luke::is_finger_hook_fixed();
}

double MjClass::get_fingertip_clearance()
{
  /* get the current saved fingertip clearance */

  return luke::get_fingertip_clearance();
}

bool MjClass::using_xyz_base_actions()
{
  /* get whether xy base actions are in use */

  return luke::use_base_xyz();
}

std::vector<luke::gfloat> MjClass::get_finger_stiffnesses()
{
  /* return a vector of the joint stiffnesses */

  return luke::get_stiffnesses();
}

std::vector<double> MjClass::get_object_xyz_bounding_box()
{
  /* return the x, y, z bounding box of the object */

  int live_id = env_.obj[0].spawn_info.index;
  luke::Vec3 xyz = luke::get_object_xyz_bounding_box(live_id);
  std::vector<double> vec = { xyz.x, xyz.y, xyz.z };
  return vec;
}

MjType::CurveFitData::PoseData MjClass::validate_curve(int force_style)
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
    s_.tip_force_applied, force_style);
  luke::verify_small_angle_model(data, 1, pose.f2.joints, 
    pose.f2.pred_j, pose.f2.pred_x, pose.f2.pred_y, pose.f2.theory_y,
    pose.f2.theory_x_curve, pose.f2.theory_y_curve,
    s_.tip_force_applied, force_style);
  luke::verify_small_angle_model(data, 2, pose.f3.joints, 
    pose.f3.pred_j, pose.f3.pred_x, pose.f3.pred_y, pose.f3.theory_y,
    pose.f3.theory_x_curve, pose.f3.theory_y_curve,
    s_.tip_force_applied, force_style);

  // TESTING: replace theory with new points (more accurate)
  pose.f1.theory_y = luke::discretise_curve(pose.f1.x, pose.f1.theory_x_curve,
    pose.f1.theory_y_curve);
  pose.f2.theory_y = luke::discretise_curve(pose.f1.x, pose.f2.theory_x_curve,
    pose.f2.theory_y_curve);
  pose.f3.theory_y = luke::discretise_curve(pose.f1.x, pose.f3.theory_x_curve,
    pose.f3.theory_y_curve);

  // calculate errors
  pose.calc_error();

  // save
  curve_validation_data_.entries.push_back(pose);

  return pose;
}

MjType::CurveFitData::PoseData MjClass::validate_curve_under_force(float force, int force_style)
{
  /* determine the cubic fit error and displacement of the fingers under
  a given force
  
  force_style:    0 = tip force
                  1 = UDL
                  2 = tip moment
  */

  bool dynamic_timestep_adjustment = true;
  int dynamic_repeats_allowed = 5;
  int dynamic_repeats_done = 1;

  // turn on curve validation mode and set the tip force to apply
  s_.curve_validation = true;
  s_.tip_force_applied = force;

  // step the simulation to allow the forces to settle
  float time_to_settle = 50; // 30; // have tried up to 100
  int steps_to_make = time_to_settle / s_.mujoco_timestep;
  // std::cout << "Stepping for " << steps_to_make << " steps to allow settling\n";

  while (true) {

    // what is the loading condition
    if (force_style == 0) luke::apply_tip_force(force);
    else if (force_style == 1) luke::apply_UDL(force);
    else if (force_style == 2) luke::apply_tip_moment(force);
    else {
      std::cout << "force_style = " << force_style << '\n';
      throw std::runtime_error("force style was not valid in validate_curve_under_force(...)");
    }

    for (int i = 0; i < steps_to_make; i++) {

      step();

      // do we catch instability? If so reduce timestep
      if (dynamic_timestep_adjustment) {
        if (luke::is_sim_unstable(model, data)) {
          s_.mujoco_timestep *= 0.8;
          reset();
          dynamic_repeats_done += 1;
          if (dynamic_repeats_done > dynamic_repeats_allowed)
            throw std::runtime_error("curve validation unstable");
          std::cout << "Curve validation unstable, trying again with timestep "
            << s_.mujoco_timestep * 1000 << " milliseconds (retry number " << dynamic_repeats_done << ")\n";
          continue; // try again
        }
      }
    }

    break;
  }

  // evaluate the finger pose
  MjType::CurveFitData::PoseData pose;
  pose = validate_curve(force_style);
  pose.tag_string = "Force is " + std::to_string(force) + " N";

  // turn off curve validation mode
  s_.curve_validation = false;
  luke::wipe_segment_forces();
  
  return pose;
}

MjType::CurveFitData MjClass::curve_validation_regime(bool print, int force_style)
{
  /* peform test battery to validate finger bending, print is false by default */

  MjType::CurveFitData curvedata;

  bool debug_state = s_.debug;

  s_.debug = false;

  // move base to maximum to ensure finger hooks could never touch the ground
  if (force_style == 2) {
    // this has only been observed an issue with end moments applied
    luke::set_base_to_max_height(data);
  }
  
  // NOTE: not forces in newtons, these are 100/200/300/400g (ie 0.981*1/2/3/4)
  std::vector<float> forces { 1 * 0.981, 2 * 0.981, 3 * 0.981, 4 * 0.981 };

  // scale forces to get equal theoretical tip deflection
  float L = luke::get_finger_length();
  if (force_style == 1) {
    for (int i = 0; i < forces.size(); i++) {
      // W = 8F / 3L
      // forces[i] *= 11.3475;
      forces[i] *= 8.0 / (3*L);
    }
  }
  else if (force_style == 2) {
    for (int i = 0; i < forces.size(); i++) {
      // M = 2LF / 3
      // forces[i] *= 0.15667;
      forces[i] *= (2*L) / 3.0;
    }
  }

  for (float f : forces) {
    
    MjType::CurveFitData::PoseData pose;
    pose = validate_curve_under_force(f, force_style);
    if (print) pose.print();
    curvedata.entries.push_back(pose);

    // // for testing
    // readings[i - 1] = luke::read_armadillo_gauge(data, 0);
    // norms[i - 1] = sim_sensors_.finger1_gauge.read_element();
  }

  // // for testing
  // luke::print_vec(readings, "Gauge readings for 1-5N");
  // luke::print_vec(norms, "normalised readings");

  s_.debug = debug_state;

  // overwrite the internal curve validation data
  curve_validation_data_ = curvedata;

  return curvedata;
}

std::string MjClass::numerical_stiffness_converge(float force, float target_accuracy)
{
  /* converge on basic theory */

  std::vector<float> theory_X;
  std::vector<float> theory_Y;
  int theory_N = 100;

  // create theoretical curve at this force
  luke::fill_theory_curve(theory_X, theory_Y, force, theory_N);

  return numerical_stiffness_converge(force, target_accuracy, theory_X, theory_Y);
}

std::string MjClass::numerical_stiffness_converge(float force, float target_accuracy,
  std::vector<float> X, std::vector<float> Y)
{
  /* converge on a given X,Y profile with repeated numerical solving of the mujoco
  finger profile in the case of point end loading 
  
  target_accuracy -> 0.5e-3 gives good agreement, 2e-3 gives decent
  */

  bool print = true;            // print out only the final result
  bool print_minimal = true;    // also print out error every 50 loops
  bool print_detailed = false;  // also print out all possible information every loop

  // use default stiffnesses as initial guess
  std::vector<luke::gfloat> stiffnesses = luke::get_stiffnesses();
  int N = stiffnesses.size();

  int loops = 0;
  int max_loops = 200;
  float max_stiffness = 800;
  float min_stiffness = 0.5;

  float avg_error = 0;
  int good_error_count = 0;
  int required_good_error = 3;
  float partial_threshold = 2e-3;
  float initial_momentum = 4;
  float final_momentum = 2;

  // float error_threshold = 0.5e-3; // 0.5e-3 gives excellent agreement, 2e-3 gives decent

  // how large are the step changes allowed to be (default 4)
  float momentum = initial_momentum;

  while (loops < max_loops) {

    if (print_detailed) {
      std::cout << "Loop " << loops << '\n';
      luke::print_vec(stiffnesses, "stiffnesses");
    }
    else if (print_minimal and loops == 0) {
      std::cout << "Starting convergence, loop 1, target error is " << target_accuracy * 100 << "%\n";
    }

    // begin by preparing
    reset();
    loops++;
    float sum_error_ratio = 0;

    // set the stiffness
    luke::set_finger_stiffness(model, stiffnesses);

    // now evaluate the profile and error
    MjType::CurveFitData::PoseData curvedata = validate_curve_under_force(force);
    bool relative_error = false;
    std::vector<float> error = profile_error(curvedata.f1.x, curvedata.f1.y, X, Y, relative_error);

    // update to new stiffnesses (i==0 is fixed end, always zero error)
    for (int i = 1; i < N + 1; i++) {

      float alpha = momentum * (N + 1 - i);
      float max_error = 1.0;            // ie 100% error, cap this to prevent huge jumps
      float error_ratio = error[i] / curvedata.f1.x[i];

      if (error_ratio > max_error) error_ratio = max_error;
      if (error_ratio < -max_error) error_ratio = -max_error;

      sum_error_ratio += abs(error_ratio);

      float new_stiffness = stiffnesses[i - 1] + error_ratio * alpha;

      if (new_stiffness < min_stiffness) new_stiffness = min_stiffness;
      if (new_stiffness > max_stiffness) new_stiffness = max_stiffness;

      if (new_stiffness != new_stiffness) {
        std::cout << "found nan in index " << i << ", setting to 8\n";
        new_stiffness = 8;
      }

      stiffnesses[i - 1] = new_stiffness;
    }

    avg_error = sum_error_ratio / (float) N;

    if (print_detailed) {
      // move errors up to millimeters
      for (int x = 0; x < error.size(); x++) error[x] *= 1000;
      luke::print_vec(error, "error (mm)");
      std::cout << "avg sum error ratio is " << avg_error * 100 << "%\n";
    }
    else if (print_minimal) {
      if (loops % 50 == 0) {
        std::cout << "Loop " << loops << ", ";
        std::cout << "avg sum error ratio is " << avg_error * 100 << "%, ";
        luke::print_vec(stiffnesses, "c");
      }
    }

    // do we adjust momentum
    if (avg_error > partial_threshold) 
      momentum = initial_momentum;
    else
      momentum = final_momentum;

    // does this error fall below our required value
    if (avg_error < target_accuracy) {
      good_error_count += 1;
    }
    else good_error_count = 0;

    // have we had enough good errors in a row
    if (good_error_count >= required_good_error) break;
  }

  if (print) {
    std::cout << "Stiffness for N=" << N << " are: "; luke::print_vec(stiffnesses, "c");
    std::cout << "There were " << loops << " loops and final avg error is "
      << avg_error * 100 << " %\n";
  }

  // return key information as a string
  std::string output;
  output = "Loops = " + std::to_string(loops) + ", %error = " + std::to_string(avg_error * 100);
  return output;
}

std::string MjClass::numerical_stiffness_converge_2(float target_accuracy)
{
  /* converge on a given X,Y profile with repeated numerical solving of the mujoco
  finger profile in the case of point end loading 
  
  target_accuracy -> 0.5e-3 gives good agreement, 2e-3 gives decent
  */

  constexpr int theory_N = 100;

  std::vector<float> forces { 1 * 0.981, 2 * 0.981, 3 * 0.981, 4 * 0.981 };

  std::vector<float> theory_X_1;
  std::vector<float> theory_Y_1;
  std::vector<float> theory_X_2;
  std::vector<float> theory_Y_2;
  std::vector<float> theory_X_3;
  std::vector<float> theory_Y_3;
  std::vector<float> theory_X_4;
  std::vector<float> theory_Y_4;

  std::vector<std::vector<float>*> theory_X { &theory_X_1, &theory_X_2, &theory_X_3, &theory_X_4 };
  std::vector<std::vector<float>*> theory_Y { &theory_Y_1, &theory_Y_2, &theory_Y_3, &theory_Y_4 };

  std::vector<float> force_errors(4);

  std::vector<std::vector<int>> k_regieme {
    { 0 },
    { 0, 1 },
       { 1 },
       { 1, 2 },
          { 2 },
          { 2, 3 },
             { 3 },
             { 3, 0 },
  };

  // std::vector<std::vector<int>> k_regieme {
  //   { 0, 1, 2 },
  //   { 1, 2, 3 },
  //   { 2, 3, 0 },
  //   { 3, 0, 1 }
  // };
  
  // create theoretical curve at each force
  for (int i = 0; i < forces.size(); i++) {
    
    std::vector<float> theory_X_vec;
    std::vector<float> theory_Y_vec;
    luke::fill_theory_curve(*theory_X[i], *theory_Y[i], forces[i], theory_N);
  }

  bool print = true;            // print out only the final result
  bool print_minimal = true;    // also print out error every 50 loops
  bool print_detailed = false;  // also print out all possible information every loop

  // use default stiffnesses as initial guess
  std::vector<luke::gfloat> stiffnesses = luke::get_stiffnesses();
  int N = stiffnesses.size();

  // for uniform random numbers from 0-1
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  int loops = 0;
  int max_loops = 750;
  float max_error = 1.0; // ie 100%, caps biggest jump
  float max_stiffness = 800;
  float min_stiffness = 0.5;
  bool stochastic_alpha = true; // does stiffness step have a +0 to +100% random value

  float avg_error = 0;
  float cumulative_error = 0;
  float overall_avg = 0;
  float best_overall_avg = 1e3;
  int best_loop = 0;
  std::vector<float> error(N + 1);
  std::vector<float> error_sum_vec(N + 1);
  std::vector<luke::gfloat> best_stiffnesses(N);
  int good_error_count = 0;
  int required_good_error = 4; // one per force
  float partial_threshold = 20e-3; // 10e-3; // 1%
  // float initial_momentum = 0.25;
  // float final_momentum = 0.15;

  float initial_momentum = 2; // was 0.5
  float final_momentum = 1; // was 0.1

  // float error_threshold = 0.5e-3; // 0.5e-3 gives excellent agreement, 2e-3 gives decent

  // how large are the step changes allowed to be (default 4)
  float momentum = initial_momentum;
  
  // int k = 3;
  int j = 0;
  // int wait_num = 5;
  // required_good_error *= wait_num - 1;

  while (loops < max_loops) {

    j++;
    if (j >= k_regieme.size()) j = 0;

    if (print_detailed) {
      std::cout << "Loop " << loops << '\n';
      luke::print_vec(stiffnesses, "stiffnesses");
    }
    else if (print_minimal and loops == 0) {
      std::cout << "Starting convergence, loop 1, target error is " << target_accuracy * 100 << "%\n";
    }

    // begin by preparing
    loops++;
    // std::fill(error_sum_vec.begin(), error_sum_vec.end(), 0.0); // zero initialisation
    for (int b = 0; b < error_sum_vec.size(); b++) error_sum_vec[b] = 0;
    reset();
    luke::set_finger_stiffness(model, stiffnesses); // set from previous loop

    // select k based on k_regieme
    for (int k : k_regieme[j]) {

    /* set k and now errors are found for that force */

    // now evaluate the profile and error
    MjType::CurveFitData::PoseData curvedata = validate_curve_under_force(forces[k]);
    bool relative_error = true;
    error = profile_error(curvedata.f1.x, curvedata.f1.y, *theory_X[k], *theory_Y[k], relative_error);

    // tally the errors
    avg_error = 0;
    cumulative_error = 0;
    for (int m = 0; m < error_sum_vec.size(); m++) {
      // error[m] /= curvedata.f1.x[m]; // old, profile error already gives relative
      if (error[m] > max_error) error[m] = max_error;
      if (error[m] < -max_error) error[m] = -max_error;
      error_sum_vec[m] += error[m];
      cumulative_error += abs(error[m]);
    }
    avg_error = cumulative_error / (float) N;
    force_errors[k] = avg_error;

    }

    /* stiffnesses are updated based on the error vector magnitudes */

    // update to new stiffnesses (i==0 is fixed end, always zero error)
    for (int i = 1; i < N + 1; i++) {

      // add stochastic value to momentum and weight stronger towards fixed end
      float alpha = momentum * ((N + 1 - i) / (float) N);
      if (stochastic_alpha) alpha *= (1 + distribution(*MjType::generator));

      // calculate and apply the new stiffness, capping min/max
      float new_stiffness = stiffnesses[i - 1] + error[i] * alpha;
      if (new_stiffness < min_stiffness) new_stiffness = min_stiffness;
      if (new_stiffness > max_stiffness) new_stiffness = max_stiffness;
      stiffnesses[i - 1] = new_stiffness;
    }

    /* finished update, now check termination and print information */

    // not enough data to calculate overall average
    if (loops < k_regieme.size()) continue;

    // get average error per force
    overall_avg = 0;
    for (int g = 0; g < 4; g++) {
      overall_avg += force_errors[g];
    }
    overall_avg /= 4.0;

    // check if this is the best
    if (overall_avg < best_overall_avg) {
      for (int d = 0; d < stiffnesses.size(); d++) {
        best_stiffnesses[d] = stiffnesses[d];
      }
      best_overall_avg = overall_avg;
      best_loop = loops;
    }

    if (print_detailed) {
      // move errors up to millimeters
      for (int x = 0; x < error_sum_vec.size(); x++) error_sum_vec[x] *= 1000;
      luke::print_vec(error_sum_vec, "error sum vec (mm)");
      luke::print_vec(force_errors, "force errors overall");
      std::cout << "overall_avg error ratio is " << overall_avg * 100 << "%\n";
    }
    else if (print_minimal) {
      if (loops % 50 == 0) {
        std::cout << "Loop " << loops << ", ";
        std::cout << "overall_avg error ratio is " << overall_avg * 100 << "%, ";
        luke::print_vec(stiffnesses, "c");
      }
    }

    // does this error fall below our required value
    if (overall_avg < target_accuracy) {
      good_error_count += 1;
    }
    else good_error_count = 0;

    // have we had enough good errors in a row
    if (good_error_count >= required_good_error) break;
  }

  // if (print) {
  //   std::cout << "Stiffness for N=" << N << " are: "; luke::print_vec(stiffnesses, "c");
  //   std::cout << "There were " << loops << " loops and final overall_avg error is "
  //     << overall_avg * 100 << " %\n";
  // }

  if (print) {
    std::cout << "Best stiffness for N=" << N << " are: "; luke::print_vec(best_stiffnesses, "c");
    std::cout << "There were " << loops << " loops and best overall_avg error is "
      << best_overall_avg * 100 << "% which occurred at loop " << best_loop << "\n";
  }

  // set the best stiffnesess
  luke::set_finger_stiffness(model, best_stiffnesses);

  // return best information as a string
  std::string output;
  output = "Loops = " + std::to_string(best_loop) + ", %error = " + std::to_string(best_overall_avg * 100);
  return output;
}

std::vector<float> MjClass::profile_error(std::vector<float> profile_X, std::vector<float> profile_Y,
  std::vector<float> truth_X, std::vector<float> truth_Y, bool relative)
{
  /* get a vector of errors on a discrete profile vs a (more) continous truth */

  if (profile_X.size() != profile_Y.size())
    throw std::runtime_error("profile X and Y lengths are different in profile_error(...)");

  if (truth_X.size() != truth_Y.size())
    throw std::runtime_error("ground truth X and Y lengths are different in profile_error(...)");
  
  if (profile_X[0] > profile_X[profile_X.size() - 1])
    throw std::runtime_error("profile X must increase from the first value to the last");
  if (truth_X[0] > truth_X[truth_X.size() - 1])
    throw std::runtime_error("truth X must increase from the first value to the last");

  int n_profile = profile_X.size();
  int n_truth = truth_X.size();

  // if (profile_X.size() > truth_X.size()) {
  //   std::cout << "Profile has " << profile_X.size() << " points\n";
  //   std::cout << "Ground truth has " << truth_X.size() << " points\n";
  //   throw std::runtime_error("profile has more points than ground truth in profile_error(...)");
  // }

  std::vector<float> errors;
  float error_X = 0;
  float error_Y = 0;

  int last = 0;
  bool found = false;

  for (int i = 0; i < n_profile; i++) {

    // find the closest X point in the 'truth'
    for (int j = last; j < n_truth; j++) {

      if (truth_X[j] > profile_X[i]) {
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
      float a = (truth_X[last + 1] - profile_X[i]) / interval;
      float b = 1 - a;
      truth_X_val = (a * truth_X[last] + b * truth_X[last + 1]);
      truth_Y_val = (a * truth_Y[last] + b * truth_Y[last + 1]);
    }

    // calculate the Y error and ignore the X error
    error_X = profile_X[i] - truth_X_val;
    error_Y = profile_Y[i] - truth_Y_val;

    // if we are calculating relative error (ie percentage but not *100)
    if (relative) {
      if (truth_Y_val < 1e-5) truth_Y_val = 1e-5;
      error_Y /= truth_Y_val;
    }

    // std::cout << "(Y, truth) is (" << profile_Y[i] << ", " << truth_Y_val << ")\t error is "
    //   <<  profile_Y[i] - truth_Y_val << "\t relative error is " << error_Y << '\n';

    // std::cout << "error_Y is " << error_Y * truth_Y_val << ", relative: " << error_Y << ", Y is " << profile_Y[i] << '\n';

    errors.push_back(error_Y);

    found = false;
  }

  return errors;
}

float MjClass::curve_area(std::vector<float> X, std::vector<float> Y)
{
  /* find the area under a curve using the trapezium rule */

  if (X.size() != Y.size()) {
    throw std::runtime_error("MjClass::curve_area() vectors X and Y must have the same size");
  }

  float area = 0.0;
  for (int i = 1; i < X.size(); i++) {
    float h = X[i] - X[i - 1];       // Width of the trapezium
    float sumOfY = Y[i] + Y[i - 1];  // Sum of the y-coordinates of the endpoints
    area += 0.5 * h * sumOfY;         // Area of the trapezium
  }

  return area;
}

void MjClass::calibrate_simulated_sensors(float bend_gauge_normalise)
{
  /* run a calibration scheme to normalise gauge outputs for a set force and
  to zero the wrist Z sensor */

  // disable noise and normalisation for wrist Z sensor
  bool original_norm = s_.wrist_sensor_Z.use_normalisation;
  bool original_noise = s_.wrist_sensor_Z.use_noise;
  s_.wrist_sensor_Z.use_normalisation = false;
  s_.wrist_sensor_Z.use_noise = false;
  s_.wrist_sensor_Z.raw_value_offset = 0;

  // let the simulation settle and calibrate wrist Z sensor, restore settings
  float settle_time = 0.3; // must exceed read rate (0.1s) to ensure a sensor reading
  int steps_for_settle = settle_time / s_.mujoco_timestep;
  for (int i = 0; i < steps_for_settle; i++) step();
  s_.wrist_sensor_Z.raw_value_offset = sim_sensors_.wrist_Z_sensor.read_element();
  s_.wrist_sensor_Z.use_normalisation = original_norm;
  s_.wrist_sensor_Z.use_noise = original_noise;

  // now calibrate the finger bending gauges
  if (luke::use_segments()) {
    validate_curve_under_force(bend_gauge_normalise);
    luke::gfloat max_gauge_reading = luke::read_armadillo_gauge(data, 0);
    s_.bending_gauge.normalise = max_gauge_reading;
  }
  else {
    std::cout << "MjClass::calibrate_simulated_sensors() warning: segments not in use, using SI force values for finger bending with identity normalisation\n";
    s_.bending_gauge.normalise = bend_gauge_normalise;
  }

  // turn off curve validation mode
  s_.curve_validation = false;
}

float MjClass::yield_load()
{
  /* return the yield force (end applied) for the current finger thickness */

  return luke::calc_yield_point_load();
}

float MjClass::yield_load(float thickness, float width)
{
  /* return the yield force (end applied) for a given thickness and width */

  return luke::calc_yield_point_load(thickness, width);
}

MjType::EventTrack MjClass::add_events(MjType::EventTrack& e1, MjType::EventTrack& e2)
{
  /* add the absolute count and activity of two events, all else is ignored */

  MjType::EventTrack out;

  #define BR(NAME, REWARD, DONE, TRIGGER)                                      \
            out.NAME.abs = e1.NAME.abs + e2.NAME.abs;                          \
            out.NAME.active_sum = e1.NAME.active_sum + e2.NAME.active_sum;

  #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT)                 \
            out.NAME.abs = e1.NAME.abs + e2.NAME.abs;                          \
            out.NAME.active_sum = e1.NAME.active_sum + e2.NAME.active_sum;     \
            out.NAME.last_value = e1.NAME.last_value + e2.NAME.last_value;    

    // run the macro to create the code
    LUKE_MJSETTINGS_BINARY_REWARD
    LUKE_MJSETTINGS_LINEAR_REWARD
    
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

  #define BR(NAME, REWARD, DONE, TRIGGER)                                      \
            if (goal_.NAME.involved) { s_.NAME.trigger = default_trigger; }    

  #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT)                 \
            if (goal_.NAME.involved) { s_.NAME.trigger = default_trigger; }    

    // run the macro to create the code
    LUKE_MJSETTINGS_BINARY_REWARD
    LUKE_MJSETTINGS_LINEAR_REWARD
    
  #undef BR
  #undef LR
}

float MjClass::find_highest_stable_timestep()
{
  /* find the highest timestep where the simulation is stable after a set amount of seconds */

  constexpr bool debug = true;

  float coarse_increment = 0.5e-3;          // 1 millisecond
  float fine_increment = 50e-6;             // 0.05 milliseconds
  float start_value = 1.0e-3;               // 1 millseconds
  float test_time = 1.0;                    // 1.0 seconds
  float max_allowable_timestep = 20.0e-3;   // 50 milliseconds

  float tune_param = 1.0;           // should be 1.0, reduce to make timestep shorter

  float next_timestep = start_value;
  bool unstable = false;

  // are we making coarse or fine increments
  bool coarse_pass = true;

  // find stable timestep
  while (true) {

    // prepare and set the new timestep
    int num_steps = (test_time / next_timestep) + 1;
    s_.mujoco_timestep = next_timestep;
    reset();

    // loop and determine if the simulation becomes unstable
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

    // if there is instability, propose a new timestep, otherwise break
    if (unstable) {

      if (coarse_pass) coarse_pass = false;

      // comb down in fine steps
      next_timestep -= fine_increment;
      unstable = false;

    }
    else {

      if (coarse_pass) {
        // search up in coarse steps
        next_timestep += coarse_increment;
      }
      else break; // we are finished

    }

    // check that the next timestep does not violate end conditions
    if (next_timestep < fine_increment) {
      throw std::runtime_error("no stable timestep found for the simulation\n");
    }
    if (next_timestep > max_allowable_timestep) {
      next_timestep = max_allowable_timestep;
      coarse_pass = false;
    }
  }

  if (debug) {
    std::printf("Timestep of %.3f milliseconds remained stable for %.2f seconds\n",
            next_timestep * 1000, test_time);
  }

  // hand tuned conversions as 1.0 is not long enough to be sure of stability and accuracy
  float factor;
  if (next_timestep <= 3.0e-3) {
    factor = tune_param * 0.8;
  }
  else if (next_timestep < 5.0e-3) {
    factor = tune_param * 0.75;
  }
  else if (next_timestep < 10.0e-3) {
    factor = tune_param * 0.65;
  }
  else {
    factor = tune_param * 0.65;
  }

  // for safety, reduce timestep by hand calibrated factors
  // std::cout << "factor is " << factor << "\n";
  float final_timestep = next_timestep * factor;

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

void MjClass::set_sensor_noise_and_normalisation_to(bool set_as)
{
  /* setter to overwrtie all sensor settings for noise and normalisation */

  s_.set_use_normalisation(set_as);
  s_.set_use_noise(set_as);
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

float unnormalise_from(float val, float min, float max)
{
  /* undo a normalisation into interval -1 +1 */

  if (val > 1.0) return max;
  else if (val < -1.0) return min;

  return (0.5 * (val + 1) * (max - min)) + min;
}

std::string make_detail_row(std::vector<std::string> x, int type_chars,
  int name_chars, int val_chars, int float_bonus, std::vector<bool> float_bonus_vec)
{
  /* Helper function for MjType::Settings::get_settings
  makes the top row detail string */

  if (x.size() < 3) {
    throw std::runtime_error("make_detail_row() got x with size < 3");
  }

  std::string str;
  std::string pad;
  std::string type_str;
  std::string name_str;
  std::string val_str;

  /* type first */
  type_str = x[0];
  pad.resize(type_chars - type_str.size(), ' ');
  str += "\n" + type_str + pad;
  /* name next */
  name_str = x[1];
  pad.clear(); 
  pad.resize(name_chars - name_str.size(), ' ');
  str += name_str + pad;
  /* value last */
  for (int i = 2; i < x.size(); i++) {
    val_str.clear();
    val_str = x[i];
    if (i == 2) {
      str += "{";
    }
    pad.clear();
    if (float_bonus_vec[i - 2]) {
      pad.resize(val_chars + float_bonus - val_str.size(), ' ');
    }
    else {
      pad.resize(val_chars - val_str.size(), ' ');
    }
    str += pad + val_str;
    if (i == x.size() - 1) {
      str += " }\n";
    }
    else {
      str += ", ";
    }
  }
  
  return str;
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
  constexpr int val_chars = 11;//5;
  constexpr int type_chars = 14;
  constexpr int float_bonus = 0;

  // create the column headers, type first
  type_str = "Type";
  pad.resize(type_chars - type_str.size(), ' ');
  str += type_str + pad;
  // next name
  name_str = "Name";
  pad.clear(); pad.resize(name_chars - name_str.size(), ' ');
  str += name_str + pad;
  // now values
  val_str = "Value";
  str += val_str + "\n";
  // now add headers to output
  output_str += str;

  std::vector<std::string> SS_vec { "Type", "Name", "Used", "Normalise", "Readrate", "Noise mu", "Noise std", "N override" };
  std::vector<bool> SS_float_vec { false, true, true, true, true, false };
  std::string SS_str = make_detail_row(SS_vec, type_chars,
    name_chars, val_chars, float_bonus, SS_float_vec);

  std::vector<std::string> AA_vec { "Type", "Name", "Used", "Continous", "Value", "Sign" };
  std::vector<bool> AA_float_vec { false, false, true, false };
  std::string AA_str = make_detail_row(AA_vec, type_chars,
    name_chars, val_chars, float_bonus, AA_float_vec);

  std::vector<std::string> BR_vec { "Type", "Name", "Reward", "Done", "Trigger" };
  std::vector<bool> BR_float_vec { true, false, false };
  std::string BR_str = make_detail_row(BR_vec, type_chars,
    name_chars, val_chars, float_bonus, BR_float_vec);

  std::vector<std::string> LR_vec { "Type", "Name", "Reward", "Done", "Trigger", "Min", "Max", "Overshoot" };
  std::vector<bool> LR_float_vec { true, false, false, true, true, true };
  std::string LR_str = make_detail_row(LR_vec, type_chars,
    name_chars, val_chars, float_bonus, LR_float_vec);

  /* be aware when using macro fields other than name as it will pull values 
     from simsettings.h not s_, instead of using TRIGGER we need to use 
     s_.NAME.trigger */

  // we will use our macros to build up strings
  #define XX(NAME, TYPE, DONTUSE1) \
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

  #define SS(NAME, DONTUSE1, DONTUSE2, DONTUSE3) \
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
            str += pad + val_str + ", ";\
            val_str.clear(); val_str += std::to_string((float)NAME.noise_mu * NAME.use_noise);\
            pad.clear(); pad.resize(val_chars + float_bonus - val_str.size(), ' ');\
            str += pad + val_str + ", ";\
            val_str.clear(); val_str += std::to_string((float)NAME.noise_std * NAME.use_noise);\
            pad.clear(); pad.resize(val_chars + float_bonus - val_str.size(), ' ');\
            str += pad + val_str + ", ";\
            val_str.clear(); val_str += std::to_string(NAME.noise_overriden);\
            pad.clear(); pad.resize(val_chars - val_str.size(), ' ');\
            str += pad + val_str + " }\n";\
            /* add to output */\
            output_str += str;

  #define AA(NAME, DONTUSE1, DONTUSE2, DONTUSE3) \
            str.clear();\
            /* type first */\
            type_str.clear(); type_str += "Action";\
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
            val_str.clear(); val_str += std::to_string((bool)NAME.continous);\
            pad.clear(); pad.resize(val_chars - val_str.size(), ' ');\
            str += pad + val_str + ", ";\
            val_str.clear(); val_str += std::to_string((float)NAME.value);\
            pad.clear(); pad.resize(val_chars + float_bonus - val_str.size(), ' ');\
            str += pad + val_str + ", ";\
            val_str.clear(); val_str += std::to_string((int)NAME.sign);\
            pad.clear(); pad.resize(val_chars - val_str.size(), ' ');\
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

    LUKE_MJSETTINGS_GENERAL

    output_str += SS_str;

    LUKE_MJSETTINGS_SENSOR

    output_str += AA_str;
    
    LUKE_MJSETTINGS_ACTION

    output_str += BR_str;

    LUKE_MJSETTINGS_BINARY_REWARD

    output_str += LR_str;

    LUKE_MJSETTINGS_LINEAR_REWARD

  #undef XX
  #undef SS
  #undef AA
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
    LUKE_MJSETTINGS_BINARY_REWARD
    LUKE_MJSETTINGS_LINEAR_REWARD
  
  #undef BR
  #undef LR
}

void MjType::Settings::disable_sensors()
{
  /* do NOT use other fields than name as it will pull values from simsettings.h not s_,
     eg instead of using TRIGGER we need to use s_.NAME.trigger */

  #define SS(NAME, IN_USE, NORM, READRATE) NAME.in_use = false;
  
    // run the macro and disable all the sensors
    LUKE_MJSETTINGS_SENSOR
  
  #undef SS
}

void MjType::Settings::set_sensor_prev_steps_to(int prev_steps)
{
  /* do NOT use other fields than name as it will pull values from simsettings.h not s_,
     eg instead of using TRIGGER we need to use s_.NAME.trigger */

  #define SS(NAME, IN_USE, NORM, READRATE) NAME.prev_steps = prev_steps;
  
    // run the macro and disable all the sensors
    LUKE_MJSETTINGS_SENSOR
  
  #undef SS
}

void MjType::Settings::update_sensor_settings(double time_since_last_sample)
{
  /* updates the number of readings each sensor is taking based on time between
  samples and read rate */

  // set the number of previous steps to sample back for all sensors
  set_sensor_prev_steps_to(sensor_n_prev_steps);

  // manually override state sensors
  int state_sensor_n_readings_per_step = 1;
  motor_state_sensor.update_n_readings(state_sensor_n_readings_per_step, state_n_prev_steps);
  base_state_sensor_XY.update_n_readings(state_sensor_n_readings_per_step, state_n_prev_steps);
  base_state_sensor_Z.update_n_readings(state_sensor_n_readings_per_step, state_n_prev_steps);
  base_state_sensor_yaw.update_n_readings(state_sensor_n_readings_per_step, state_n_prev_steps);
  cartesian_contacts_XYZ.update_n_readings(state_sensor_n_readings_per_step, state_n_prev_steps);
  
  // update n_readings for all except state sensors
  #define SS(NAME, IN_USE, NORM, READRATE)                       \
            if (#NAME != "motor_state_sensor" and                \
                #NAME != "base_state_sensor_XY" and              \
                #NAME != "base_state_sensor_Z" and               \
                #NAME != "base_state_sensor_yaw" and             \
                #NAME != "cartesian_contacts_XYZ")               \
              NAME.update_n_readings(time_since_last_sample);
  
    // run the macro and update all the sensors
    LUKE_MJSETTINGS_SENSOR

  #undef SS
}

void MjType::Settings::set_use_normalisation(bool set_as)
{
  /* do NOT use other fields than name as it will pull values from simsettings.h not s_,
     eg instead of using TRIGGER we need to use s_.NAME.trigger */

  #define SS(NAME, IN_USE, NORM, READRATE) NAME.use_normalisation = set_as;
  
    // run the macro and disable all the sensors
    LUKE_MJSETTINGS_SENSOR
  
  #undef SS
}

void MjType::Settings::set_use_noise(bool set_as)
{
  /* do NOT use other fields than name as it will pull values from simsettings.h not s_,
     eg instead of using TRIGGER we need to use s_.NAME.trigger */

  #define SS(NAME, IN_USE, NORM, READRATE) NAME.use_noise = set_as;
  
    // run the macro and disable all the sensors
    LUKE_MJSETTINGS_SENSOR
  
  #undef SS
}

void MjType::Settings::apply_noise_params(std::uniform_real_distribution<float>& uniform_dist)
{
  /* do NOT use other fields than name as it will pull values from simsettings.h not s_,
     eg instead of using TRIGGER we need to use s_.NAME.trigger */

  // set the noise to default UNLESS it has been overriden for ALL sensors
  // mu is randomly chosen between [-noise_mu, noise_mu]
  #define SS(NAME, DONTUSE1, DONTUSE2, DONTUSE3)   \
            if (not NAME.noise_overriden) {        \
              NAME.noise_mag = sensor_noise_mag;   \
              NAME.noise_std = sensor_noise_std;   \
              NAME.noise_mu = sensor_noise_mu;     \
            }                                      \
            NAME.randomise_mu(uniform_dist);                 
  
    // run the macro and disable all the sensors
    LUKE_MJSETTINGS_SENSOR
  
  #undef SS

  // then manually override the state sensors (we want state_noise not sensor_noise)
  if (not motor_state_sensor.noise_overriden) {
    motor_state_sensor.noise_mag = state_noise_mag;
    motor_state_sensor.noise_mu = state_noise_mu;
    motor_state_sensor.noise_std = state_noise_std;
  }
  if (not base_state_sensor_XY.noise_overriden) {
    base_state_sensor_XY.noise_mag = state_noise_mag;
    base_state_sensor_XY.noise_mu = state_noise_mu;
    base_state_sensor_XY.noise_std = state_noise_std;
  }
  if (not base_state_sensor_Z.noise_overriden) {
    base_state_sensor_Z.noise_mag = state_noise_mag;
    base_state_sensor_Z.noise_mu = state_noise_mu;
    base_state_sensor_Z.noise_std = state_noise_std;
  }
  if (not base_state_sensor_yaw.noise_overriden) {
    base_state_sensor_yaw.noise_mag = state_noise_mag;
    base_state_sensor_yaw.noise_mu = state_noise_mu;
    base_state_sensor_yaw.noise_std = state_noise_std;
  }
  if (not cartesian_contacts_XYZ.noise_overriden) {
    cartesian_contacts_XYZ.noise_mag = state_noise_mag;
    cartesian_contacts_XYZ.noise_mu = state_noise_mu;
    cartesian_contacts_XYZ.noise_std = state_noise_std;
  }

  // randomise seed for state
  motor_state_sensor.randomise_mu(uniform_dist);
  base_state_sensor_XY.randomise_mu(uniform_dist);
  base_state_sensor_Z.randomise_mu(uniform_dist);
  base_state_sensor_yaw.randomise_mu(uniform_dist);
  cartesian_contacts_XYZ.randomise_mu(uniform_dist);
}

void MjType::Settings::scale_rewards(float scale)
{
  /* scale all of the rewards by a given value */

  /* do NOT use other fields than name as it will pull values from simsettings.h not s_,
     eg instead of using TRIGGER we need to use s_.NAME.trigger */
  #define BR(NAME, DONTUSE1, DONTUSE2, DONTUSE3) NAME.reward *= scale;
  #define LR(NAME, DONTUSE1, DONTUSE2, DONTUSE3, DONTUSE4, DONTUSE5, DONTUSE6) \
            NAME.reward *= scale;
  
    // run the macro and scale the rewards
    LUKE_MJSETTINGS_BINARY_REWARD
    LUKE_MJSETTINGS_LINEAR_REWARD
  
  #undef BR
  #undef LR
}

void MjType::Settings::set_all_action_use(bool set_as)
{
  /* set all the actions to 'set_as' */

  #define AA(NAME, DONT1, DONT2, DONT3) NAME.in_use = set_as;

    LUKE_MJSETTINGS_ACTION

  #undef AA
}

void MjType::Settings::set_all_action_continous(bool set_as)
{
  /* set all the actions to 'set_as' */

  #define AA(NAME, DONT1, DONT2, DONT3) NAME.continous = set_as;

    LUKE_MJSETTINGS_ACTION

  #undef AA
}

void MjType::Settings::set_all_action_value(float set_as)
{
  /* set all the actions to 'set_as' */

  #define AA(NAME, DONT1, DONT2, DONT3) NAME.value = set_as;

    LUKE_MJSETTINGS_ACTION

  #undef AA
}

void MjType::Settings::set_all_action_sign(int set_as)
{
  /* set all the actions to 'set_as' */

  if (set_as != -1 and set_as != 1) {
    throw std::runtime_error("MjType::Settings::set_all_action_sign recieved a value not either +1 or -1");
  }

  #define AA(NAME, DONT1, DONT2, DONT3) NAME.sign = set_as;

    LUKE_MJSETTINGS_ACTION

  #undef AA
}

void MjType::EventTrack::print()
{
  /* print out the event track information */

  calculate_percentage();

  std::cout << "EventTrack = row (abs); "

  #define BR(NAME, REWARD, DONE, TRIGGER)                               \
            << #NAME << " = " << NAME.row << " (" << NAME.abs << "); "
  #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT)          \
            << #NAME << " = " << NAME.row << " (" << NAME.abs << ", " << NAME.last_value << "); "

    // run the macro to create the code
    LUKE_MJSETTINGS_BINARY_REWARD
    LUKE_MJSETTINGS_LINEAR_REWARD 
    << "\n";

  #undef BR
  #undef LR

}

void update_events(MjType::EventTrack& events, MjType::Settings& settings)
{
  /* update the count of each event and reset recent event information */

  bool active = false; // is an event active

  #define BR(NAME, REWARD, DONE, TRIGGER)                                    \
            events.NAME.row = events.NAME.row *                              \
                                  events.NAME.value + events.NAME.value;     \
            events.NAME.abs += events.NAME.value;                            \
            events.NAME.last_value = events.NAME.value;                      \
            events.NAME.active_sum = bool(events.NAME.row);                  \
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
            events.NAME.active_sum = active;                                 \
            events.NAME.value = 0.0; // reset for next step

    // run the macro to create the code
    LUKE_MJSETTINGS_BINARY_REWARD
    LUKE_MJSETTINGS_LINEAR_REWARD

  #undef BR
  #undef LR
}

float calc_rewards(MjType::EventTrack& events, MjType::Settings& settings)
{
  /* calculate the reward of one transition based on the simulation events */

  float reward = 0;

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
                  #NAME, events.NAME.last_value, scaled_reward);                \
              reward += scaled_reward;                                          \
            }
            
    // run the macro to create the code
    LUKE_MJSETTINGS_BINARY_REWARD
    LUKE_MJSETTINGS_LINEAR_REWARD

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
    LUKE_MJSETTINGS_BINARY_REWARD
    LUKE_MJSETTINGS_LINEAR_REWARD

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

  #define BR(NAME, REWARD, DONE, TRIGGER)                                      \
            out.push_back(NAME.row);

  #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT)                 \
            out.push_back(NAME.row);                                           \
            out.push_back(NAME.last_value);  

    // run the macro to create the code
    LUKE_MJSETTINGS_BINARY_REWARD
    LUKE_MJSETTINGS_LINEAR_REWARD
    
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

  #define BR(NAME, REWARD, DONE, TRIGGER)                                      \
            NAME.row = in[i] + 0.5; /* casts float -> int */                   \
            i++;

  #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT)                 \
            NAME.row = in[i] + 0.5; /* casts float -> int */                   \
            i++;                                                               \
            NAME.last_value = in[i];                                           \
            i++;                                                               

    // run the macro to create the code
    LUKE_MJSETTINGS_BINARY_REWARD
    LUKE_MJSETTINGS_LINEAR_REWARD
    
  #undef BR
  #undef LR

}

std::vector<float> MjType::Goal::vectorise() const
{
  /* return a vector of the goal state, which must map from [-1,+1]*/

  std::vector<float> out;

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
    LUKE_MJSETTINGS_BINARY_REWARD
    LUKE_MJSETTINGS_LINEAR_REWARD
    
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
    LUKE_MJSETTINGS_BINARY_REWARD
    LUKE_MJSETTINGS_LINEAR_REWARD
    
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
    LUKE_MJSETTINGS_BINARY_REWARD
    LUKE_MJSETTINGS_LINEAR_REWARD

  #undef BR
  #undef LR

  return new_goal;
}

std::vector<std::string> MjType::Goal::goal_names()
{
  /* get the names of the currently active goals */

  std::vector<std::string> goal_names;

  #define BR(NAME, REWARD, DONE, TRIGGER)                                      \
            if (NAME.involved) { goal_names.push_back(#NAME); }

  #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT)                 \
            if (NAME.involved) { goal_names.push_back(#NAME); }

    // run the macro to create the code
    LUKE_MJSETTINGS_BINARY_REWARD
    LUKE_MJSETTINGS_LINEAR_REWARD

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
    LUKE_MJSETTINGS_BINARY_REWARD
    LUKE_MJSETTINGS_LINEAR_REWARD

  #undef BR
  #undef LR

  goal_info += " }\n";

  return goal_info;
}

// end