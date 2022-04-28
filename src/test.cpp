#include <string>
#include <vector>
#include <iostream>

#include "mjclass.h"
#include "myfunctions.h"

int num_gauge_readings_if_raw = 7;
int num_xml_tasks = 37;
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
  std::vector<luke::gfloat> obs = mj.get_observation(num_gauge_readings_if_raw);

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
  mj.get_observation(num_gauge_readings_if_raw);

  return;
}

void run_test(int num_episodes, int step_cap, int reload_rate)
{
  /* run a test as if we were learning, no data generated */

  if (print_ep) std::cout << "Test started\n";

  // create initial MjClass
  MjClass mj;
  mj.s_.debug = debug_print;

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

  return;
}

int main(int argc, char** argv)
{
  // testing area

  luke::SlidingWindow<int> testwindow(10);

  testwindow.add(0);
  testwindow.add(1);
  testwindow.add(2);
  testwindow.add(3);
  testwindow.add(4);
  testwindow.add(5);
  testwindow.add(6);

  std::cout << "testwindow is "; testwindow.print();
  std::cout << "read_element(4) gives " << testwindow.read_element(4) << "\n";
  std::cout << "read(4) gives: "; testwindow.print(4);

  return 0;

  // precompiled settings
  int num_episodes = 10;
  int step_cap = 30;
  int reload_rate = 2;

  run_test(num_episodes, step_cap, reload_rate);

  return 0;

  std::string path = "/home/luke/gripper_repo_ws/src/gripper_v2/"
    "gripper_description/urdf/mujoco/";
  std::string gripper_file = "gripper_mujoco.xml";
  std::string panda_file = "panda_mujoco.xml";
  std::string both_file = "panda_and_gripper_mujoco.xml";
  std::string task_file = "gripper_task.xml";

  std::string filepath = path + "task/gripper_task_0.xml";

  // if we receive command line arguments
  if (argc > 1) {
    if (not strcmp(argv[1], "gripper")) {
    filepath = path + gripper_file;
    }
    else if (not strcmp(argv[1], "panda")) {
        filepath = path + panda_file;
    }
    else if (not strcmp(argv[1], "both")) {
        filepath = path + both_file;
    }
    else if (not strcmp(argv[1], "task")) {
        filepath = path + task_file;
        if (argc > 2) {
            filepath = path + "/task/gripper_task_" + argv[2] + ".xml";
        }
    }
    else {
        printf("Command line argument not valid, ignored\n");
    }
  }
  else {
    printf("No command line arguments detected, using default model\n");
  }

  MjClass mjObj(filepath);

  mjObj.spawn_object(0);

  for (int i = 0; i < 10; i++) {
    mjObj.step();
    // mjObj.render();
  }

  return 0;
}

