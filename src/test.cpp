#include "mjclass.h"
#include <string>
#include <vector>
#include <iostream>

int main(int argc, char** argv)
{
  // testing area

  std::string path = "/home/luke/gripper_repo_ws/src/gripper_v2/"
    "gripper_description/urdf/mujoco/";
  std::string gripper_file = "gripper_mujoco.xml";
  std::string panda_file = "panda_mujoco.xml";
  std::string both_file = "panda_and_gripper_mujoco.xml";
  std::string task_file = "gripper_task.xml";

  std::string filepath = path + task_file;

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

