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

// if we are on the cluster, we must not include the rendering libraries
#if !defined(LUKE_CLUSTER)
  #include "rendering.h"
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

  // what are the possible sampling methods to get observation data
  struct Sample {
    enum {
      raw = 0,
      change,
      average
    };
  };

  // sensor type for sensing in simulation
  struct Sensor {

    bool in_use;            // is this sensor currently in use
    float normalise;        // value with which to normalise readings to [-1,1]
    float read_rate;        // rate in Hz which this sensor is read
    double last_read_time;  // time in seconds sensor was last read

    bool use_normalisation = true; // are we using normalisation

    Sensor(bool in_use, float normalise, float read_rate)
      : in_use(in_use), normalise(normalise), read_rate(read_rate)
    {
      last_read_time = 0.0;
    }

    void set(bool in_use_, float normalise_, float read_rate_) {
      in_use = in_use_; normalise = normalise_; read_rate = read_rate_;
    }

    void reset() {
      last_read_time = 0.0;
    }

    float apply_normalisation(float value) 
    {
      /* normalise the given value from [-1 to +1] */

      if (not use_normalisation) return value;

      if (in_use) {
        // bumper sensor only
        if (normalise <= 0) {
          if (value < 0) return -1;
          else return 1;
        }
        // regular normalisation
        else if (value > normalise) {
        return 1.0;
        }
        if (value < -normalise) {
          return -1.0;
        }
        return value / normalise;
      }
      // not in use
      else {
        return 0.0;
      }
    }

    bool ready_to_read(double current_time_seconds) 
    {
      /* return true if the sensor is ready to read, also saves last read time
         as the current time */

      if (not in_use) return false;

      double time_between_reads = 1 / read_rate;

      if (current_time_seconds > last_read_time + time_between_reads) {
        // assume a read will be made, save last read time as now
        last_read_time = current_time_seconds;
        return true;
      }
      else return false;
    }

    int get_n_readings(float time_since_last_sample)
    {
      /* get the number of readings since the last sample, with overlap, so the
      final reading in the last sample will be the first reading in this one */

      int n_readings;

      // how many readings since last sample, with overlap
      if (read_rate > 0) {
        double readings_since_step = time_since_last_sample * read_rate;
        n_readings = std::ceil(readings_since_step);
      }
      else {
        n_readings = -1 * read_rate;
      }

      return n_readings;
    }

    std::vector<luke::gfloat> raw_sample(luke::SlidingWindow<luke::gfloat> data, 
      float time_since_last_sample)
    {
      /* sample some data from a given time interval in seconds */

      int n_readings = get_n_readings(time_since_last_sample);

      return data.read(n_readings);
    }

    std::vector<luke::gfloat> change_sample(luke::SlidingWindow<luke::gfloat> data,
      float time_since_last_sample)
    {
      /* sample the first and last reading as well as the change [x0, dx, x1] */

      int n_readings = get_n_readings(time_since_last_sample);

      std::vector<luke::gfloat> result(3);
      result[0] = data.read_element(n_readings - 1); // read_element 0 indexed
      result[2] = data.read_element();
      result[1] = result[2] - result[0];

      return result;
    }

    std::vector<luke::gfloat> average_sample(luke::SlidingWindow<luke::gfloat> data,
      float time_since_last_sample)
    {
      /* sample the first and last reading as well as the average [x0, xbar, x1] */

      int n_readings = get_n_readings(time_since_last_sample);

      std::vector<luke::gfloat> all = data.read(n_readings);

      luke::gfloat mean = 0.0;

      for (uint i = 0; i < all.size(); i++) {
        mean += all[i];
      }

      mean /= (luke::gfloat)all.size();

      std::vector<luke::gfloat> result {
        all[0], mean, all[all.size() - 1]
      };

      return result;
    }
  };

  // what key events will we keep track of in the simulation
  struct EventTrack {

    struct BinaryEvent {
      bool value { false };
      int last_value { false }; // can represent sum of booleans (eg 2 = true twice)
      int row { 0 };
      int abs { 0 };
      float percent { 0.0 };

      void reset() { value = 0; last_value = 0; row = 0; abs = 0; percent = 0; }
    };

    struct LinearEvent {
      float value { 0.0 };
      float last_value { 0.0 };
      int row { 0 };
      int abs { 0 };
      float percent { 0.0 };

      void reset() { value = 0; last_value = 0; row = 0; abs = 0; percent = 0; }
    };

    // create an event for each reward, binary->binary, linear->linear
    #define XX(NAME, TYPE, VALUE)
    #define SS(NAME, USED, NORMALISE, READ_RATE)
    #define BR(NAME, REWARD, DONE, TRIGGER) BinaryEvent NAME;
    #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT) LinearEvent NAME;
      // run the macro to create the code
      LUKE_MJSETTINGS
    #undef XX
    #undef SS
    #undef BR
    #undef LR

    void print();
    std::vector<float> vectorise();
    void unvectorise(std::vector<float> in);

    void reset()
    {
      /* run the reset function for all members of the class */

      #define XX(NAME, TYPE, VALUE)
      #define SS(NAME, USED, NORMALISE, READ_RATE)
      #define BR(NAME, REWARD, DONE, TRIGGER) NAME.reset();
      #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT) NAME.reset();

        // run the macro to create the code
        LUKE_MJSETTINGS

      #undef XX
      #undef SS
      #undef BR
      #undef LR
    }

    void calculate_percentage()
    {
      /* calculate the percentage of steps where this event occured */

      #define XX(NAME, TYPE, VALUE)
      #define SS(NAME, USED, NORMALISE, READ_RATE)

      #define BR(NAME, REWARD, DONE, TRIGGER)                             \
                NAME.percent = (100.0 * NAME.abs) / (float) step_num.abs;

      #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT)        \
                NAME.percent = (100.0 * NAME.abs) / (float) step_num.abs;
                
        // run the macro to create the code
        LUKE_MJSETTINGS

      #undef XX
      #undef SS
      #undef BR
      #undef LR
    }

  };

  struct BinaryReward {

    float reward;            // numerical reward value
    int done;                // instances of this behaviour in a row to terminate sim
    int trigger;             // instances of this behaviour in a row to trigger reward
    bool goal = false;       // is this reward a goal in HER

    BinaryReward(float reward, int done, int trigger)
      : reward(reward), done(done), trigger(trigger) {}

    void set(float reward_, int done_, int trigger_) {
      reward = reward_; done = done_; trigger = trigger_;
    }

  };

  struct LinearReward {

    float reward;            // numerical reward value
    int done;                // instances of this behaviour in a row to terminate sim
    int trigger;             // instances of this behaviour in a row to trigger reward
    float min;               // min value for this behaviour to be measured (0)
    float max;               // max value for the behaviour (1)
    float overshoot;         // -1 = saturate above max, >max = linear decay from max-overshoot
    float value = 0;         // container to store the value which triggers reward
    bool goal = false;       // is this reward a goal in HER

    LinearReward(float reward, int done, int trigger, float min, float max, float overshoot)
      : reward(reward), done(done), trigger(trigger), min(min), max(max), overshoot(overshoot) {}

    void set(float reward_, int done_, int trigger_, float min_, float max_, float overshoot_) {
      reward = reward_; done = done_; trigger = trigger_;
      min = min_; max = max_; overshoot_ = overshoot_;
    }

  };

  // goals for the simulation for hindsight experience replay (HER)
  struct Goal {

    struct Event {
      // are events registered as goals and if so what is their state
      bool involved = false;
      float state = false; // should be from [-1 to +1], <0=false, >0=true
    };

    // create an event for each reward, default none involved
    #define XX(NAME, TYPE, VALUE)
    #define SS(NAME, USED, NORMALISE, READ_RATE)
    #define BR(NAME, REWARD, DONE, TRIGGER) Event NAME;
    #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT) Event NAME;
      // run the macro to create the code
      LUKE_MJSETTINGS
    #undef XX
    #undef SS
    #undef BR
    #undef LR

    // functions
    std::vector<float> vectorise() const;
    void unvectorise(std::vector<float> vec);
    void print();
    void print_verbose();
    std::vector<std::string> goal_names();
    std::string get_goal_info();

    void reset(bool reset_involved = false)
    {
      /* reset the goal */

      #define XX(NAME, TYPE, VALUE)
      #define SS(NAME, USED, NORMALISE, READ_RATE)
      #define BR(NAME, REWARD, DONE, TRIGGER)                                  \
                NAME.state = -1.0;                                             \
                if (reset_involved) { NAME.involved = false; }                    

      #define LR(NAME, REWARD, DONE, TRIGGER, MIN, MAX, OVERSHOOT)             \
                NAME.state = -1.0;                                             \
                if (reset_involved) { NAME.involved = false; }                    

        // run the macro to create the code
        LUKE_MJSETTINGS

      #undef XX
      #undef SS
      #undef BR
      #undef LR
    }
  };

  // settings structure fully read/write exposed to python
  struct Settings {

    // see simsettings.h

    // define the assignment code we want for X, BR, LR
    #define XX(name, type, value) type name { value };
    #define SS(name, use, norm, readrate) Sensor name { use, norm, readrate };
    #define BR(name, reward, done, trigger) BinaryReward name { reward, done, trigger };
    #define LR(name, reward, done, trigger, min, max, overshoot) \
              LinearReward name { reward, done, trigger, min, max, overshoot };
      // run the macro to create the code
      LUKE_MJSETTINGS
    #undef XX
    #undef SS
    #undef BR
    #undef LR

    // never used, added here only for convenience in bind.cpp
    bool dummy = false;

    // function definitions
    std::string get_settings();
    void wipe_rewards();
    void disable_sensors();
    void scale_rewards(float scale); 
    void set_use_normalisation(bool set_as);

  };

  // data on the simulated objects and environment
  struct Env {
    
    // data that is not reset
    std::vector<std::string> object_names;

    // data that is reset
    float cumulative_reward = 0;
    int num_action_steps = 0;
    luke::QPos start_qpos;

    // track important events in the environment
    EventTrack cnt;

    // track the state of the object at this step
    struct Obj {
      std::string name;
      luke::QPos qpos;
      luke::rawNum finger1_force;
      luke::rawNum finger2_force;
      luke::rawNum finger3_force;
      luke::rawNum palm_force;
      luke::rawNum ground_force;
      float palm_axial_force;
      float avg_finger_force;
    } obj;

    // track the state of the gripper
    struct Grp {
      luke::Gripper target;
      luke::rawNum finger1_force;
      luke::rawNum finger2_force;
      luke::rawNum finger3_force;
      float peak_finger_axial_force;
      float peak_finger_lateral_force;
    } grp;

    void reset() {
      // reset to initialised values
      Obj blank_obj;
      Grp blank_grp;
      obj = blank_obj;
      grp = blank_grp;
      // reset data
      cumulative_reward = 0;
      num_action_steps = 0;
      start_qpos.reset();
      cnt.reset();
    }

  };

  // data structure for real gripper data
  struct Real {
    std::vector<float> gauge1_since_last_read;
    std::vector<float> gauge2_since_last_read;
    std::vector<float> gauge3_since_last_read;
  };

  // info to give to python about simlulation
  struct TestReport {

    // details
    std::string object_name;
    float cumulative_reward = 0;

    // save the breakdown of events
    EventTrack cnt;

  };

  // data to validate the curve fitting
  struct CurveFitData {
    struct PoseData {
      struct FingerData {

        std::vector<float> x;         // segment end x positions
        std::vector<float> y;         // segment end y positions
        std::vector<float> coeff;     // curve fit coefficients
        std::vector<float> errors;    // curve fit point prediction errors

        void print_vec(std::vector<float> v, std::string name) {
          std::cout << name << ":\n";
          if (v.size() == 0) {
            std::cout << "empty\n";
            return;
          }
          for (unsigned int i = 0; i < v.size() - 1; i++) {
            std::cout << v[i] << "\n";
          }
          std::cout << v[v.size() - 1] << "\n\n";
        }
        void print() {
          print_vec(x, "x positions");
          print_vec(y, "y positions");
          print_vec(coeff, "coefficients");
          print_vec(errors, "errors");
          std::cout << "-----end-----\n\n";
        }
      };

      FingerData f1;
      FingerData f2;
      FingerData f3;

    };

    // save a series of data points
    std::vector<PoseData> entries;

    void print() {
      for (unsigned int i = 0; i < entries.size(); i++) {
        std::cout << "ENTRY " << i << "\n";
        std::cout << "Finger 1\n";
        entries[i].f1.print();
        std::cout << "Finger 2\n";
        entries[i].f2.print();
        std::cout << "Finger 3\n";
        entries[i].f3.print();
        std::cout << "\n\n";
      }
    }

    void print_errors() {
      float cum_error = 0.0;
      for (unsigned int i = 0; i < entries.size(); i++) {
        for (unsigned int j = 0; j < entries[i].f1.errors.size(); j++) {
          cum_error += abs(entries[i].f1.errors[j]);
          cum_error += abs(entries[i].f2.errors[j]);
          cum_error += abs(entries[i].f3.errors[j]);
        }
      }
      float avg_error = cum_error / (3 * entries.size() * entries[0].f1.errors.size());

      std::cout << "The average error from " << entries.size()
        << " entries was " << avg_error << '\n';
    }

  };

  // calibration constants for gauge data
  struct RealGaugeCalibrations {
    /* applied as follows: g_out = (g_raw + offset) * scale */
    struct RealSensors { float g1 {}, g2 {}, g3 {}, palm {}; };

    RealSensors offset;
    RealSensors scale;
    RealSensors norm;

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
  static constexpr double ftol = 1e-5;             // floating point tolerance
  static constexpr int gauge_buffer_size = 50;     // buffer to store gauge data 

  /* ----- parameters that are unchanged with reset() ----- */

  // mujoco model and data pointers
  mjModel* model;
  mjData* data;

  MjType::Settings s_;                          // simulation settings
  std::chrono::time_point<time_> start_time_;   // time from tick() call
  bool render_init = false;                     // have we initialised the render window

  // function pointers for sampling functions
  std::vector<luke::gfloat> (MjType::Sensor::*sampleFcnPtr)
    (luke::SlidingWindow<luke::gfloat>, float);
  std::vector<luke::gfloat> (MjType::Sensor::*stateFcnPtr)
    (luke::SlidingWindow<luke::gfloat>, float);

  // path information for loading models, have we defined defaults?
  #if defined(LUKE_MJCF_PATH)
    std::string model_folder_path = LUKE_MJCF_PATH;
  #else
    std::string model_folder_path = "";
  #endif

  #if defined(LUKE_DEFAULTOBJECTS)
    std::string object_set_name = LUKE_DEFAULTOBJECTS;
  #else
    std::string object_set_name = "";
  #endif

  #if defined(LUKE_MACHINE)
    std::string machine = LUKE_MACHINE;
  #else
    std::string machine = "machine_not_defined";
  #endif

  std::string current_load_path;                // xml path of currently loaded model
  
  // reward goal (if using)
  MjType::Goal goal_;

  // real gripper parameters
  MjType::RealGaugeCalibrations calibrate_; 

  /* ----- variables that are reset ----- */

  // standard class variables
  int n_actions;                      // number of possible actions
  std::vector<int> action_options;    // possible action codes

  // storage containers for state data
  luke::SlidingWindow<luke::gfloat> x_motor_position { gauge_buffer_size };
  luke::SlidingWindow<luke::gfloat> y_motor_position { gauge_buffer_size };
  luke::SlidingWindow<luke::gfloat> z_motor_position { gauge_buffer_size };
  luke::SlidingWindow<luke::gfloat> z_base_position { gauge_buffer_size };
  
  // create storage containers for sensor data
  luke::SlidingWindow<luke::gfloat> finger1_gauge { gauge_buffer_size };
  luke::SlidingWindow<luke::gfloat> finger2_gauge { gauge_buffer_size };
  luke::SlidingWindow<luke::gfloat> finger3_gauge { gauge_buffer_size };
  luke::SlidingWindow<luke::gfloat> palm_sensor { gauge_buffer_size };
  luke::SlidingWindow<luke::gfloat> finger1_axial_gauge { gauge_buffer_size };
  luke::SlidingWindow<luke::gfloat> finger2_axial_gauge { gauge_buffer_size };
  luke::SlidingWindow<luke::gfloat> finger3_axial_gauge { gauge_buffer_size };
  luke::SlidingWindow<luke::gfloat> wrist_X_sensor { gauge_buffer_size };
  luke::SlidingWindow<luke::gfloat> wrist_Y_sensor { gauge_buffer_size };
  luke::SlidingWindow<luke::gfloat> wrist_Z_sensor { gauge_buffer_size };

  // track the timestamps of sensor updates, this is for plotting in mysimlulate.cpp
  luke::SlidingWindow<float> gauge_timestamps { gauge_buffer_size };
  luke::SlidingWindow<float> axial_timestamps { gauge_buffer_size };
  luke::SlidingWindow<float> palm_timestamps { gauge_buffer_size };
  luke::SlidingWindow<float> wristXY_timestamps { gauge_buffer_size };
  luke::SlidingWindow<float> wristZ_timestamps { gauge_buffer_size };

  // data structures
  MjType::Env env_;
  MjType::TestReport testReport_;
  MjType::CurveFitData curve_validation_data_;

  // for using the real gripper
  int samples_since_last_obs = 0;

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
  std::vector<float> get_finger_gauge_data();
  void monitor_sensors();
  void sense_gripper_state();
  void update_env();

  // control
  bool set_joint_target(double x, double th, double z);
  bool set_motor_target(double x, double y, double z);
  bool set_step_target(int x, int y, int z);
  bool move_motor_target(double x, double y, double z);
  bool move_joint_target(double x, double th, double z);
  bool move_step_target(int x, int y, int z);

  // learning functions
  void action_step();
  std::vector<float> set_action(int action);
  void reset_object();
  void spawn_object(int index);
  void spawn_object(int index, double xpos, double ypos, double zrot);
  bool is_done();
  std::vector<luke::gfloat> get_observation();
  std::vector<float> get_event_state();
  std::vector<float> get_goal();
  std::vector<float> assess_goal();
  std::vector<float> assess_goal(std::vector<float> event_vec);
  float reward();
  float reward(std::vector<float> goal_vec, std::vector<float> event_vec);
  int get_n_actions();
  int get_n_obs();

  // real world gripper functions
  void input_real_data(std::vector<float> state_data, std::vector<float> sensor_data, 
    float timestamp);
  std::vector<float> get_real_observation();

  // misc
  void forward() { mj_forward(model, data); }
  int get_number_of_objects() { return env_.object_names.size(); }
  std::string get_current_object_name() { return env_.obj.name; }
  MjType::TestReport get_test_report();
  void validate_curve();
  void tick();
  float tock();
  MjType::EventTrack add_events(MjType::EventTrack& e1, MjType::EventTrack& e2);
  void reset_goal();
  void print(std::string s) { std::printf("%s\n", s.c_str()); }
  void default_goal_event_triggering();

}; // class MjClass

// utility functions
float linear_reward(float val, float min, float max, float overshoot);
float normalise_between(float val, float min, float max);
void update_events(MjType::EventTrack& events, MjType::Settings& settings);
float calc_rewards(MjType::EventTrack& events, MjType::Settings& settings);
float goal_rewards(MjType::EventTrack& events, MjType::Settings& settings,
  MjType::Goal goal);
MjType::Goal score_goal(MjType::Goal const goal, std::vector<float> event_vec, 
  MjType::Settings settings);
MjType::Goal score_goal(MjType::Goal const goal, MjType::EventTrack event, 
  MjType::Settings settings);

#endif // MJCLASS_H_