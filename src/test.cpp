#include <string>
#include <vector>
#include <iostream>

#include "mjclass.h"
#include "myfunctions.h"

int num_xml_tasks = 4; // intentionally low number to allow different object sets
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

  /* ----- run a test of 10 learning steps ----- */

  // // precompiled settings
  // /* settings of 20, 200, 20 -> initial time taken 52.6s, newest 42.6s (both laptop times, newest PC is 45.0s */
  // int num_episodes = 20;
  // int step_cap = 200;
  // int reload_rate = 20;

  // run_test(num_episodes, step_cap, reload_rate);

  // return 0;

  /* ----- load the gripper, generic testing ----- */

  std::string relpath = "task/gripper_task_0.xml";

  MjClass mjObj;
  mjObj.load_relative(relpath);

  mjObj.spawn_object(0);

  for (int i = 0; i < 10; i++) {
    mjObj.step();
    // mjObj.render();
  }

  MjType::Goal goal;
  MjClass mj;

  goal.step_num.involved = true;
  goal.lifted.involved = true;
  goal.stable_height.involved = true;
  goal.finger_force.involved = true;
  goal.oob.involved = true;

  goal.step_num.state = true;
  goal.oob.state = true;
  goal.finger_force.state = true;

  goal.print();

  // std::cout << mjObj.reward(goal) << '\n';

  std::vector<float> test { -1, 1, 1, -1, -1 };
  goal.unvectorise(test);

  goal.print();

  // std::cout << mjObj.reward(goal) << '\n';

  return 0;
}

