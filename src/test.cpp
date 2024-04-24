#include <string>
#include <vector>
#include <iostream>

#include "mjclass.h"
#include "myfunctions.h"

int num_xml_tasks = 4; // intentionally low number to accomodate different object sets
bool print_step = false;
bool print_ep = true;
bool debug_print = false;
bool allow_done = false; // allow early termination

bool learning_step(MjClass& mj)
{
  /* complete whole step of learning, but do nothing with the data. Return if
  is_done() = true */

  // set a random action
  int randact = rand() % mj.n_actions;
  mj.set_discrete_action(randact);

  // complete this action in the simulatin
  mj.action_step();

  // get an observation
  std::vector<luke::gfloat> obs = mj.get_observation();

  double actreward = mj.reward();
  bool done = mj.is_done();

  // if printing
  if (print_step) {
    std::cout << "Action taken: " << randact << '\n';
    luke::print_vec(obs, "Observation");
    std::cout << "Reward is " << actreward << '\n';
    std::cout << "Cumulative reward is " << mj.env_.cumulative_reward << '\n';
    std::cout << "is_done = " << (done ? "true\n" : "false\n");
    std::cout << '\n';
  }

  return done;
}

void reset_sim(MjClass& mj)
{
  /* reset the simulation */

  mj.reset();

  // spawn a random object
  std::vector<std::string> objects = luke::get_objects();
  int randobj = rand() % objects.size();
  mj.spawn_object(randobj);

  // reset normally returns the next observation
  mj.get_observation();

  return;
}

void load_task(MjClass& mj, int task_num, std::string object_set = "default")
{
  /* load a task xml file */

  std::vector<std::string> arguments = {
    "--task", std::to_string(task_num),
    "--width", "28",
    "--segments", "8",
    "--thickness", "1.0",
    "--path", "/home/luke/mujoco-devel/mjcf/"
  };

  if (object_set != "default") {
    arguments.push_back("--object-set");
    arguments.push_back(object_set);
  }

  std::vector<char*> argv;
  for (const auto& arg : arguments)
      argv.push_back((char*)arg.data());
  argv.push_back(nullptr);

  std::string file = mj.file_from_from_command_line(argv.size() - 1, argv.data());
  mj.load(file);
}

void run_test(int num_episodes, int step_cap, int reload_rate)
{
  /* run a test as if we were learning, no data generated */

  // user options
  bool randxml = false;
  bool use_default_object_set = true;
  std::string object_set = "set9_fullset";
  bool use_random_seed = true;
  int random_seed = 13572;

  if (print_ep) std::cout << "Test started\n";

  // create initial MjClass
  MjClass mj;
  mj.s_.debug = debug_print;

  // which object set should we use
  if (use_default_object_set) {
    object_set = "default";
  }

  // do we have a fixed random seed
  if (use_random_seed) {
    mj.s_.random_seed = random_seed;
  }

  // auto-detect stable timestep
  mj.s_.auto_set_timestep = true;
  // mj.s_.mujoco_timestep = 3.187e-3;

  // to determine length of simulation time
  int action_step_counter = 0;

  // begin timer
  mj.tick();

  // load an xml model file
  int xml_num = 0;
  if (randxml) xml_num = rand() % num_xml_tasks;
  load_task(mj, randxml);

  reset_sim(mj);

  for (int i = 1; i < num_episodes + 1; i++) {

    if (print_ep and not print_step) std::cout << "Episode " << i << "\n";

    for (int j = 0; j < step_cap; j++) {

      if (print_step) std::cout << "Episode " << i << " step " << j << '\n';

      bool done = learning_step(mj);
      action_step_counter += 1;

      if (done and allow_done) break;
    }

    if (i % reload_rate == 0) {

      if (print_ep) std::cout << "Reloading simulation\n";

      // get the xml file name
      if (randxml) xml_num = rand() % num_xml_tasks;
      else {
        xml_num += 1;
        if (randxml >= num_xml_tasks) randxml = 0;
      }

      // load a new task
      load_task(mj, xml_num);
    }

    reset_sim(mj);
  }

  if (print_ep) std::cout << "Test finished\n";

  // finish timing
  float total_time = mj.tock();

  // determine how long was simulated
  double sim_time = action_step_counter * mj.s_.time_for_action;
  double realtime_factor = sim_time / total_time;

  std::cout << "The total time was " << total_time 
    << " seconds. " << sim_time << " seconds were simulated, hence the realtime speed up is "
    << realtime_factor << '\n';

  return;
}

int main(int argc, char** argv)
{
  /* ----- run a test of 10 learning steps ----- */

  // precompiled settings
  /* settings of 20, 200, 20 -> initial time taken 52.6s, newest 42.6s (both laptop times, newest PC is 45.0s */
  /* settings of 20, 200, 20 -> mujoco-2.1.5 takes 12.261/12.466/12.872 seconds */
  /* settings of 20, 200, 20 -> mujoco-2.2.0 takes 12.307/12.888/12.729 seconds */
  int num_episodes = 20;
  int step_cap = 200;
  int reload_rate = 21;

  run_test(num_episodes, step_cap, reload_rate);

  return 0;

  /* ----- load the gripper, generic testing ----- */

  // std::string relpath = "gripper_N10/gripper_task_0.xml";
  // mjObj.load_relative(relpath);

  MjClass mjObj;
  std::string filepath = mjObj.file_from_from_command_line(argc, argv);
  mjObj.load(filepath);

  // hardcode stable timestep
  mjObj.s_.auto_set_timestep = false;
  mjObj.s_.mujoco_timestep = 3.187e-3;

  // // change settings
  // mjObj.s_.mujoco_timestep = 1.8e-3;
  // mjObj.s_.sensor_n_prev_steps = 2;
  // mjObj.s_.sensor_sample_mode = MjType::Sample::change;

  // apply changes and begin simulating
  mjObj.reset();
  mjObj.spawn_object(0);

  int num = 100;

  mjObj.tick();

  for (int i = 0; i < num; i++) {
    luke::RGBD rgbd_image = mjObj.get_RGBD();
  }

  std::cout << "Time taken was " << mjObj.tock() << " seconds\n";

  return 0;

  // old
  mjObj.set_step_target(6000, 7000, 0);

  // disable or enable sensors
  mjObj.s_.motor_state_sensor.in_use = false;
  mjObj.s_.base_state_sensor_XY.in_use = false;
  mjObj.s_.base_state_sensor_Z.in_use = false;
  mjObj.s_.bending_gauge.in_use = true;
  mjObj.s_.axial_gauge.in_use = false;
  mjObj.s_.palm_sensor.in_use = false;
  mjObj.s_.wrist_sensor_XY.in_use = false;

  // turn on noise so values vary
  mjObj.s_.wrist_sensor_Z.use_noise = true;
  mjObj.s_.wrist_sensor_Z.noise_std = 0.5;
  mjObj.s_.wrist_sensor_Z.noise_mu = 0.8;

  double start_time = mjObj.data->time;

  for (int i = 0; i < 10; i++) {

    while (true) {
      char c;
      std::cin >> c;

      if (c == 'f') {
        for (int i = 0; i < 10; i++) mjObj.step();
      }
      else if (c == 'q') {
        for (int i = 0; i < 100; i++) mjObj.step();
      }
      else if (c == 's') {
        for (int i = 0; i < mjObj.s_.sim_steps_per_action; i++) mjObj.step();
      }
      else if (c == 'p') {
        luke::print_vec(mjObj.get_observation(), "Observation");
        continue;
      }
      else if (c == 'm') {
        mjObj.move_step_target(500, 1000, 0);
        mjObj.action_step();
        std::cout << "moved (x,y,z) by (500,1000,0) steps\n";
      }

      // test new features
      else if (c == 't') {
        
      }

      mjObj.step();

      std::printf("Time is: %.1f ms, ", (mjObj.data->time - start_time ) * 1000);
      std::cout << "printing finger 1 sensor: ";
      // mjObj.wrist_Z_sensor.print(10);
      mjObj.sim_sensors_.finger1_gauge.print(10);

      
    }

    mjObj.step();
    // mjObj.render(); // does nothing
  }

  MjType::Sensor mysensor(true, 1, 1);
  mysensor.use_noise = false;
  mysensor.use_normalisation = false;
  mysensor.prev_steps = 3;
  mysensor.readings_per_step = 1;
  mysensor.update_n_readings();

  luke::SlidingWindow<float> vec(10);

  vec.add(1);
  vec.add(2);
  vec.add(3);
  vec.add(4);
  vec.add(5);
  vec.add(6);

  std::vector<float> sample = mysensor.change_sample(vec);

  luke::print_vec(sample, "Sample is");
  vec.print();

  return 0;
  // std::cout << mjObj.reward(goal) << '\n';

  return 0;
}

