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

class MjClass
{
public:

  /* wrapper class to be exposed to python that manipulates an instance of a
  mujoco model and data */

  /* class variables */

  mjModel* model;
  mjData* data;

  // for measuring timings
  typedef std::chrono::high_resolution_clock time_;
  std::chrono::time_point<time_> start_time_;

  // parameters set at compile time
  static constexpr bool debug = false;           // are we in debug mode
  static constexpr double ftol = 1e-5;          // floating point tolerance
  static constexpr int gauge_buffer_size = 50;  // buffer to store gauge data 
  static constexpr int state_buffer_size = 50;  // buffer to store state data

  // standard class variables
  bool render_init = false;           // have we initialised the render window
  int timeout_count = 0;              // number of action_step() timeouts in a row
  double last_read_time = 0;          // last gauge read time in seconds
  std::vector<int> action_options;    // possible action codes

  // create storage containers for strain gauge data
  luke::SlidingWindow<luke::gfloat> finger1_gauge {gauge_buffer_size};
  luke::SlidingWindow<luke::gfloat> finger2_gauge {gauge_buffer_size};
  luke::SlidingWindow<luke::gfloat> finger3_gauge {gauge_buffer_size};
  luke::SlidingWindow<float> gauge_timestamps {gauge_buffer_size};

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

    int step_num = false;                 // count the number of steps
    int lifted = false;                   // object lifted off the ground
    int oob = false;                      // object out of bounds
    int dropped = false;                  // object dropped
    int target_height = false;            // reached target height above ground
    int exceed_limits = false;            // gripper limits exceeded
    int exceed_axial = false;             // too much axial finger compression force
    int exceed_lateral = false;           // too much lateral outwards finger force
    int object_contact = false;           // gripper contact with object
    int object_stable = false;            // object is stably grasped
    int palm_force = false;               // palm applying force to object
    int exceed_palm = false;              // palm applying too much force to object

    void print() {
      std::cout << "track = " << "step: " << step_num
        << ", lifted: " << lifted << ", oob: " << oob << ", dropped: " << dropped
        << ", t_height: " << target_height << ", ex.lims: " << exceed_limits
        << ", ex.axial: " << exceed_axial << ", ex.lateral: " << exceed_lateral
        << ", contact: " << object_contact << ", stable: " << object_stable 
        << ", p.force: " << palm_force << ", ex.palm: " << exceed_palm << '\n';
    }
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

    std::string fetch_string();

  } s_; // settings

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

  } env_; // environment

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

  } testReport_;

  /* member functions */

  // constructors
  MjClass();
  ~MjClass();
  MjClass(std::string file_path);
  MjClass(mjModel* m, mjData* d);
  MjClass(Settings settings_to_use);
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
  std::vector<luke::gfloat> get_observation(int n);
  float reward();

  // misc
  void forward() { mj_forward(model, data); }
  int get_number_of_objects() { return env_.object_names.size(); }
  std::string get_current_object_name() { return env_.obj.name; }
  MjClass::TestReport get_test_report();
  void tick();
  float tock();

}; // class MjClass

// utility functions
float linear_reward(float val, float min, float max, float overshoot);

#endif // MJCLASS_H_