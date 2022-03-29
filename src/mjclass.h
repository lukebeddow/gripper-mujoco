#ifndef MJCLASS_H_
#define MJCLASS_H_

#include "mjxmacro.h"
#include "stdio.h"
#include "string.h"
#include "mujoco.h"

#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>

#include "simsettings.h"
#include "myfunctions.h"
#include "customtypes.h"

// if we are on the cluster, the render function must immediately return false
#if defined(LUKE_CLUSTER)
  #define LUKE_FILE_ROOT "/home/lbeddow/mjcf/"
#else
  #include "rendering.h"
  #define LUKE_FILE_ROOT "/home/luke/gripper_repo_ws/src/gripper_v2/gripper_description/urdf/mujoco/"
#endif

namespace MjType
{
  /* types used inside the MjClass, including data structures */

  // what are the possible actions (order matters - see configure_settings())
  struct Action {
    enum {
      x_motor_positive = 0,
      x_motor_negative,
      prismatic_positive,
      prismatic_negative,
      y_motor_positive,
      y_motor_negative,
      z_motor_positive,
      z_motor_negative,
      height_positive,
      height_negative,
      count             // how many possible actions, leave this last
    };
  };

  // what key events will we keep track of in the simulation
  struct EventTrack {

    // we keep track of every event related to a reward
    #define X(name, type, value)
    #define BR(name, reward, done, trigger) int name { false };
    #define LR(name, reward, done, trigger, min, max, overshoot) int name { false };
      // run the macro to create the code
      LUKE_MJSETTINGS
    #undef X
    #undef BR
    #undef LR

    void print();
  };

  struct BinaryReward {

    float reward;
    int done;
    int trigger;

    BinaryReward(float reward, int done, int trigger)
      : reward(reward), done(done), trigger(trigger) {}

    void set(float reward_, int done_, int trigger_) {
      reward = reward_; done = done_; trigger = trigger_;
    }

  };

  struct LinearReward {

    float reward;
    int done;
    int trigger;
    float min;
    float max;
    float overshoot;

    LinearReward(float reward, int done, int trigger, float min, float max, float overshoot)
      : reward(reward), done(done), trigger(trigger), min(min), max(max), overshoot(overshoot) {}

    void set(float reward_, int done_, int trigger_, float min_, float max_, float overshoot_) {
      reward = reward_; done = done_; trigger = trigger_;
      min = min_; max = max_; overshoot_ = overshoot_;
    }

  };

  // settings structure fully read/write exposed to python
  struct Settings {

    // see simsettings.h

    // define the assignment code we want for X, BR, LR
    #define X(name, type, value) type name { value };
    #define BR(name, reward, done, trigger) BinaryReward name { reward, done, trigger };
    #define LR(name, reward, done, trigger, min, max, overshoot) \
              LinearReward name { reward, done, trigger, min, max, overshoot };
      // run the macro to create the code
      LUKE_MJSETTINGS
    #undef X
    #undef BR
    #undef LR

    // never used, added here only for convenience in bind.cpp
    bool dummy = false;

    std::string get_settings();
    void wipe_rewards();

  };

  // data on the objects and environment
  struct Env {
    
    // data that is not reset
    std::vector<std::string> object_names;

    // data that is reset
    float cumulative_reward = 0;
    int num_action_steps = 0;
    luke::QPos start_qpos;

    EventTrack cnt;         // track events in a row
    EventTrack abs_cnt;     // absolute count, never reset to zero

    // track the state of the object at this step
    struct Obj {
      std::string name;
      luke::QPos qpos;
      luke::myNum finger1_force;
      luke::myNum finger2_force;
      luke::myNum finger3_force;
      luke::myNum palm_force;
      luke::myNum ground_force;
      float palm_axial_force;
      float avg_finger_force;
    } obj;

    // track the state of the gripper
    struct Grp {
      luke::Gripper target;
      luke::myNum finger1_force;
      luke::myNum finger2_force;
      luke::myNum finger3_force;
      float peak_finger_axial_force;
      float peak_finger_lateral_force;
    } grp;

    void reset() {
      // reinitialise defaults
      EventTrack blank_cnt;
      Obj blank_obj;
      Grp blank_grp;
      // override current
      cnt = blank_cnt;
      abs_cnt = blank_cnt;
      obj = blank_obj;
      grp = blank_grp;
      // reset data
      cumulative_reward = 0;
      num_action_steps = 0;
      start_qpos.reset();
    }

  };

  // info to give to python about simlulation
  struct TestReport {

    // details
    std::string object_name;
    float cumulative_reward = 0;
    int num_steps = 0;

    // count everytime an event occurs in the whole test
    EventTrack abs_cnt;

    // snapshot of what events have occured at the end of the test
    EventTrack final_cnt;

    // take note of forces
    float final_palm_force = 0;
    float final_finger_force = 0;

  };
}

class MjClass
{
public:

  /* wrapper class to be exposed to python that manipulates an instance of a
  mujoco model and data */

  /* ----- unchanging constants ----- */

  // for measuring timings
  typedef std::chrono::high_resolution_clock time_;

  // parameters set at compile time
  static constexpr bool debug = false;           // are we in debug mode
  static constexpr double ftol = 1e-5;          // floating point tolerance
  static constexpr int gauge_buffer_size = 50;  // buffer to store gauge data 
  static constexpr int state_buffer_size = 50;  // buffer to store state data

  /* ----- parameters that are unchanged with reset() ----- */

  mjModel* model;
  mjData* data;

  MjType::Settings s_;                          // simulation settings
  std::string current_load_path;                // xml path of currently loaded model
  std::chrono::time_point<time_> start_time_;   // time from tick() call

  /* ----- variables that are reset ----- */

  // standard class variables
  bool render_init = false;           // have we initialised the render window
  int timeout_count = 0;              // number of action_step() timeouts in a row
  double last_read_time = 0;          // last gauge read time in seconds
  std::vector<int> action_options;    // possible action codes
  int n_actions;                      // number of possible actions
  
  // create storage containers for strain gauge data
  luke::SlidingWindow<luke::gfloat> finger1_gauge { gauge_buffer_size };
  luke::SlidingWindow<luke::gfloat> finger2_gauge { gauge_buffer_size };
  luke::SlidingWindow<luke::gfloat> finger3_gauge { gauge_buffer_size };
  luke::SlidingWindow<luke::gfloat> palm_sensor { gauge_buffer_size };
  luke::SlidingWindow<float> gauge_timestamps { gauge_buffer_size };

  // data structures
  MjType::Env env_;
  MjType::TestReport testReport_;

  /* ----- member functions ----- */

  // constructors
  MjClass();
  ~MjClass();
  MjClass(std::string file_path);
  MjClass(mjModel* m, mjData* d);
  MjClass(MjType::Settings settings_to_use);
  void init();
  void init(mjModel* m, mjData* d);
  void configure_settings();

  // core functionality
  void load(std::string file_path);
  void load_relative(std::string file_path);
  void reset();
  void step();
  bool render();

  // sensing
  bool monitor_gauges();
  std::vector<luke::gfloat> read_gauges();
  luke::gfloat read_palm();
  std::vector<luke::gfloat> get_gripper_state();
  bool is_target_reached();
  bool is_settled();
  void update_env();

  // control
  bool set_joint_target(double x, double th, double z);
  bool set_motor_target(double x, double y, double z);
  bool set_step_target(int x, int y, int z);
  bool move_motor_target(double x, double y, double z);
  bool move_joint_target(double x, double th, double z);
  bool move_step_target(int x, int y, int z);

  // learning functions
  bool action_step();
  void set_action(int action);
  void reset_object();
  void spawn_object(int index);
  void spawn_object(int index, double xpos, double ypos);
  bool is_done();
  std::vector<luke::gfloat> get_observation();
  std::vector<luke::gfloat> get_observation(int n);
  float reward();
  int get_n_actions();
  int get_n_obs();

  // misc
  void forward() { mj_forward(model, data); }
  int get_number_of_objects() { return env_.object_names.size(); }
  std::string get_current_object_name() { return env_.obj.name; }
  MjType::TestReport get_test_report();
  void tick();
  float tock();

}; // class MjClass

// utility functions
float linear_reward(float val, float min, float max, float overshoot);
float normalise_between(float val, float min, float max);

#endif // MJCLASS_H_