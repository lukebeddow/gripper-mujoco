#include <string>
#include <vector>
#include <iostream>

#include "mjclass.h"
#include "myfunctions.h"

int num_xml_tasks = 4; // intentionally low number to accomodate different object sets
bool print_step = false;
bool print_ep = true;
bool debug_print = false;

bool learning_step(MjClass& mj)
{
  /* complete whole step of learning, but do nothing with the data. Return if
  is_done() = true */

  // set a random action
  int randact = rand() % mj.n_actions;
  mj.set_action(randact);

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

void run_test(int num_episodes, int step_cap, int reload_rate)
{
  /* run a test as if we were learning, no data generated */

  if (print_ep) std::cout << "Test started\n";

  // create initial MjClass
  MjClass mj;
  mj.s_.debug = debug_print;

  // begin timer
  mj.tick();

  // get the xml file name
  int randxml = rand() % num_xml_tasks;
  std::string xmltask = "/task/gripper_task_" + std::to_string(randxml) + ".xml";
  mj.load_relative(xmltask);

  reset_sim(mj);

  for (int i = 1; i < num_episodes + 1; i++) {

    if (print_ep and not print_step) std::cout << "Episode " << i << "\n";

    for (int j = 0; j < step_cap; j++) {

      if (print_step) std::cout << "Episode " << i << " step " << j << '\n';

      bool done = learning_step(mj);
      if (done) break;
    }

    if (i % reload_rate == 0) {

      if (print_ep) std::cout << "Reloading simulation\n";

      // get the xml file name
      randxml = rand() % num_xml_tasks;
      std::string xmltask = "/task/gripper_task_" + std::to_string(randxml) + ".xml";

      // load a new task
      mj.load_relative(xmltask);
    }

    reset_sim(mj);
  }

  if (print_ep) std::cout << "Test finished\n";

  // finish timing
  float total_time = mj.tock();

  std::cout << "The total time was " << total_time << '\n';

  return;
}

int main(int argc, char** argv)
{
  // MjClass testmj;

  // std::vector<float> profile_X { 0, 1, 2, 3 };
  // std::vector<float> profile_Y { 0, 1.25, 3.5, 9.5 };

  // std::vector<float> truth_X;
  // std::vector<float> truth_Y;

  // for (int i = 0; i < 35; i++) {

  //   float X = i * 0.1 + 0.01;
  //   float Y = X * X;

  //   truth_X.push_back(X);
  //   truth_Y.push_back(Y);

  // }

  // bool relative_error = false;
  // std::vector<float> errors = testmj.profile_error(profile_X, profile_Y, truth_X, truth_Y, relative_error);

  // luke::print_vec(errors, "errors");

  // return 0;

  /* ----- run a test of 10 learning steps ----- */

  // // precompiled settings
  // /* settings of 20, 200, 20 -> initial time taken 52.6s, newest 42.6s (both laptop times, newest PC is 45.0s */
  // int num_episodes = 20;
  // int step_cap = 200;
  // int reload_rate = 20;

  // run_test(num_episodes, step_cap, reload_rate);

  // return 0;

  /* ----- load the gripper, generic testing ----- */

  // std::string relpath = "gripper_N10/gripper_task_0.xml";
  // mjObj.load_relative(relpath);

  MjClass mjObj;
  std::string filepath = mjObj.file_from_from_command_line(argc, argv);
  mjObj.load(filepath);

  // change settings
  mjObj.s_.mujoco_timestep = 1.8e-3;
  mjObj.s_.sensor_n_prev_steps = 2;
  mjObj.s_.sensor_sample_mode = MjType::Sample::change;

  // apply changes and begin simulating
  mjObj.reset();
  mjObj.spawn_object(0);
  mjObj.set_step_target(6000, 7000, 0);

  // disable or enable sensors
  mjObj.s_.motor_state_sensor.in_use = false;
  mjObj.s_.base_state_sensor.in_use = false;
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

