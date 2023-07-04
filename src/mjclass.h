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
#include <random>
#include <memory>

#include "simsettings.h"
#include "myfunctions.h"
#include "customtypes.h"

// if we are on the cluster, we must not include the rendering libraries
#if !defined(LUKE_CLUSTER)
  #include "rendering.h"
#endif

// utility functions with no MjType dependency
float linear_reward(float val, float min, float max, float overshoot);
float normalise_between(float val, float min, float max);
float unnormalise_from(float val, float min, float max);

namespace MjType
{
  /* types used inside the MjClass, including data structures */

  // random generator pointer, seeded in MjClass::configure_settings()
  extern std::shared_ptr<std::default_random_engine> generator;

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
      raw = 0,    // return raw values
      change,     // add the change between values ie (a, b-a, b)
      average,    // add the average between values ie (a, (a+b)/2, b)
      median,     // add the median between values ie (a, med(a...b), b)
      sign        // add the sign of the change ie (a, sign(b-a), b) where sign() outputs -1,0,+1
    };
  };

  // sensor type for sensing in simulation
  struct Sensor {

    // initialised settings from simsettings.h
    bool in_use = false;                // is this sensor currently in use
    float normalise = 1;                // value with which to normalise readings to [-1,1]
    float read_rate = 1;                // rate in Hz which this sensor is read

    // user options that can be overriden
    bool use_normalisation = true;      // are we using normalisation
    bool use_noise = true;              // are we adding synthetic noise
    float raw_value_offset = 0.0;       // value to subract from the sensor raw value (only used for wrist Z sensor)
    float noise_mag = 0;                // magnitude of added noise
    float noise_mu = 0;                 // mean of noise (>0 uniformly randomly set)
    float noise_std = -1;               // std deviation of noise (< 0 means flat)

    // internal variables set via functions, do not touch
    double last_read_time = 0;          // time in seconds sensor was last read
    int prev_steps = 1;                 // back how many previous steps do we read
    int readings_per_step = 1;          // how many readings during action execution
    int total_readings = 1;             // how many samples back do we start reading
    bool noise_overriden = false;       // noise settings have been manually set, should not be changed
    
    // random mu values for use in multiple sensing streams
    float rand_mu_1 = 0;
    float rand_mu_2 = 0;
    float rand_mu_3 = 0;

    Sensor(bool in_use, float normalise, float read_rate)
      : in_use(in_use), normalise(normalise), read_rate(read_rate)
    {
      last_read_time = 0.0;
    }

    void set(bool in_use_, float normalise_, float read_rate_) {
      in_use = in_use_; normalise = normalise_; read_rate = read_rate_;
    }

    void set_gaussian_noise(float mu, float stddev) {
      noise_mag = 0;
      noise_mu = mu;
      noise_std = stddev;
      noise_overriden = true;
    }

    void set_uniform_noise(float mag) {
      noise_mag = mag;
      noise_mu = 0;
      noise_std = -1;
      noise_overriden = true;
    }

    void randomise_mu(std::uniform_real_distribution<float>& uniform_dist) {
      // if mu is non-zero, select a uniform random mu
      rand_mu_1 = noise_mu * (2 * uniform_dist(*MjType::generator) - 1);
      rand_mu_2 = noise_mu * (2 * uniform_dist(*MjType::generator) - 1);
      rand_mu_3 = noise_mu * (2 * uniform_dist(*MjType::generator) - 1);
    }

    void wipe_noise() {
      noise_mag = 0;
      noise_mu = 0;
      noise_std = -1;
      noise_overriden = false;
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

    float apply_noise(float value, std::uniform_real_distribution<float>& uniform_dist, int i = 1)
    {
      /* add noise to a reading */

      if (not use_noise) return value;

      constexpr float two_pi = 2.0 * M_PI;
      constexpr float epsilon = std::numeric_limits<float>::epsilon();

      // which noise mean to choose, 1,2,3 can be used for gauge1,2,3 etc
      float mu_to_use;
      if (i == 1) mu_to_use = rand_mu_1;
      else if (i == 2) mu_to_use = rand_mu_2;
      else if (i == 3) mu_to_use = rand_mu_3;
      else {
        throw std::runtime_error("Sensor::apply_noise() got i not equal 1,2,3");
      }

      // calculate a uniform random noise
      if (noise_std < epsilon) {
        float noise = mu_to_use + noise_mag * (2 * uniform_dist(*MjType::generator) - 1);
        value += noise;
      }
      // use the box mueller transform to calculate normal noise
      else {

        // get random values, ensuring u1 isn't 0 (division by 0 error)
        float u1, u2;
        do {
          u1 = uniform_dist(*MjType::generator);
        }
        while (u1 <= epsilon);
        u2 = uniform_dist(*MjType::generator);

        // compute z0 and z1
        float mag = noise_std * std::sqrt(-2.0 * std::log(u1));
        float z0 = mag * std::cos(two_pi * u2) + mu_to_use;
        // float z1 = mag * std::sin(two_pi * u2) + mu_to_use; // not needed
  
        value += z0;
      }

      // ensure we remain in bounds
      if (value > 1) value = 1;
      else if (value < -1) value = -1;

      return value;
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

    void update_n_readings()
    {
      /* use class settings to update the number of sensor readings expected at
      the next sample. Note that this value is only used for sensor sampling and
      also beware that for 'raw_sample()' this value will be +1 compared to the
      actual number of readings */

      total_readings = 1 + readings_per_step * prev_steps;
    }

    void update_n_readings(int readings_since_last_step, int set_prev_steps_to)
    {
      /* update reading information but specifying how many readings have arrived
      SINCE the last sample (NOT inclusive of the last sample), as well as overriding
      the number of previous steps */

      // set function inputs to class variables first
      prev_steps = set_prev_steps_to;
      readings_per_step = readings_since_last_step;

      update_n_readings();
    }

    void update_n_readings(float time_since_last_sample)
    {
      /* get the number of readings SINCE the last sample without overlap based
      on the read rate (Hz) of the sensor and the time elapsed */

      // how many readings since last sample
      double readings_since_step = time_since_last_sample * read_rate;
      readings_per_step = std::floor(readings_since_step);

      update_n_readings();
    }

    std::vector<luke::gfloat> raw_sample(luke::SlidingWindow<luke::gfloat> data)
    {
      /* sample some data from a given time interval in seconds. The function
      update_n_readings() works perfectly for all other sample modes, but gives
      an answer +1 to what is needed by this function, so we read but -1 first.
      
      The reason for this is that raw_sample is different from other sampling
      methods. One reading for raw_sample is a single number, but one reading
      for change_sample is three numbers, the old reading, the change, and the
      new reading. Since this includes the old reading and the new reading, from
      raw_samples perspective this is actually two readings, not one. */

      return data.read(total_readings - 1);
    }

    std::vector<luke::gfloat> change_sample(luke::SlidingWindow<luke::gfloat> data)
    {
      /* sample the first and last reading as well as the change [x0, dx, x1] */

      // make the return vector, first element is furthest back reading
      std::vector<luke::gfloat> result(2 * prev_steps + 1);
      result[0] = data.read_element(total_readings - 1); // read_element is 0 indexed

      // loop through steps to add in elements
      for (int i = 0; i < prev_steps; i++) {
        int first_sample = total_readings - 1 - i * readings_per_step;
        result[i * 2 + 2] = data.read_element(first_sample - readings_per_step);
        result[i * 2 + 1] = result[i * 2 + 2] - result[i * 2];
      }

      return result;
    }

    std::vector<luke::gfloat> average_sample(luke::SlidingWindow<luke::gfloat> data)
    {
      /* sample the first and last reading as well as the average [x0, xbar, x1] */

      // make the return vector, first element is furthest back reading
      std::vector<luke::gfloat> result(2 * prev_steps + 1);
      result[0] = data.read_element(total_readings - 1);

      // loop through steps to add in elements
      for (int i = 0; i < prev_steps; i++) {
        int first_sample = total_readings - 1 - i * readings_per_step;
        result[i * 2 + 2] = data.read_element(first_sample - readings_per_step);

        // sum intermediate values and get the mean
        result[i * 2 + 1] = 0;
        for (int j = 0; j < readings_per_step + 1; j++) {
          result[i * 2 + 1] += data.read_element(first_sample - j);
        }
        result[i * 2 + 1] /= readings_per_step + 1;
      }

      return result;
    }

    std::vector<luke::gfloat> median_sample(luke::SlidingWindow<luke::gfloat> data)
    {
      /* sample the first and last reading as well as the median [x0, xbar, x1] */

      // make the return vector, first element is furthest back reading
      std::vector<luke::gfloat> result(2 * prev_steps + 1);
      result[0] = data.read_element(total_readings - 1);

      // loop through steps to add in elements
      for (int i = 0; i < prev_steps; i++) {
        int first_sample = total_readings - 1 - i * readings_per_step;
        result[i * 2 + 2] = data.read_element(first_sample - readings_per_step);

        // put the data values in a new vector for this step
        std::vector<luke::gfloat> values;
        for (int j = 0; j < readings_per_step + 1; j++) {
          values.push_back(data.read_element(first_sample - j));
        }

        // should never be empty (readings_per_step = 0)
        if (values.empty()) {
          result[i * 2 + 1] = 0.0;
          continue;
        }

        // get the centre of the vector and find the median
        int n = values.size() / 2;
        nth_element(values.begin(), values.begin() + n, values.end());
        luke::gfloat med = values[n];

        // if median of even number of values, get the largest value up to n
        if (!(values.size() & 1)) { //If the set size is even
          auto max_it = max_element(values.begin(), values.begin() + n);
          // the median is the aveage of these two adjacent values
          med = (*max_it + med) / 2.0;
        }

        // save the result
        result[i * 2 + 1] = med;
      }

      return result;
    }

    std::vector<luke::gfloat> sign_sample(luke::SlidingWindow<luke::gfloat> data)
    {
      /* sample the first and last reading as well as the sign of the change 
      [x0, sign(dx), x1]. Sign outputs either +1, -1, or 0 */

      static constexpr luke::gfloat tol = 1e-6;

      // make the return vector, first element is furthest back reading
      std::vector<luke::gfloat> result(2 * prev_steps + 1);
      result[0] = data.read_element(total_readings - 1); // read_element is 0 indexed

      // loop through steps to add in elements
      for (int i = 0; i < prev_steps; i++) {
        int first_sample = total_readings - 1 - i * readings_per_step;
        result[i * 2 + 2] = data.read_element(first_sample - readings_per_step);
        
        luke::gfloat change = result[i * 2 + 2] - result[i * 2];
        if (change > tol) {
          result[i * 2 + 1] = 1.0;
        }
        else if (change < -tol) {
          result[i * 2 + 1] = -1.0;
        }
        else {
          result[i * 2 + 1] = 0.0;
        }
      }

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
    #define SS(NAME, IN_USE, NORM, READRATE)
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
      #define SS(NAME, IN_USE, NORM, READRATE)
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
      #define SS(NAME, IN_USE, NORM, READRATE)

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
    #define SS(NAME, IN_USE, NORM, READRATE)
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
      #define SS(NAME, IN_USE, NORM, READRATE)
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
    #define SS(name, in_use, norm, readrate) Sensor name { in_use, norm, readrate };
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
    void set_use_noise(bool set_as);
    void set_sensor_prev_steps_to(int prev_steps);
    void update_sensor_settings(double time_since_last_sample);
    void apply_noise_params(std::uniform_real_distribution<float>& uniform_dist);
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

        std::vector<float> joints;    // joint values
        std::vector<float> pred_j;    // errors vs simple joint calculation
        std::vector<float> pred_x;    // predicted x in simple model
        std::vector<float> pred_y;    // predicted y in simple model
        std::vector<float> theory_y;  // theory predicted y deflection at joint positions
        std::vector<float> theory_x_curve;  // smooth theory curve
        std::vector<float> theory_y_curve;  // smooth theory curve

        struct Error {
          float x_wrt_pred_x = 0;
          float x_wrt_pred_x_percent = 0;
          float x_wrt_pred_x_tipratio = 0;

          float y_wrt_pred_y = 0;
          float y_wrt_pred_y_percent = 0;
          float y_wrt_pred_y_tipratio = 0;

          float y_wrt_theory_y = 0;
          float y_wrt_theory_y_percent = 0;
          float y_wrt_theory_y_tipratio = 0;

          float y_pred_wrt_theory_y = 0;
          float y_pred_wrt_theory_y_percent = 0;
          float y_pred_wrt_theory_y_tipratio = 0;

          float j_wrt_pred_j = 0;
          float j_wrt_pred_j_percent = 0;

          float x_tip_wrt_pred_x = 0;
          float y_tip_wrt_pred_y = 0;
          float y_tip_wrt_theory_y = 0;
          float y_pred_tip_wrt_theory_y = 0;

          float x_tip_wrt_pred_x_percent = 0;
          float y_tip_wrt_pred_y_percent = 0;
          float y_tip_wrt_theory_y_percent = 0;
          float y_pred_tip_wrt_theory_y_percent = 0;
          
          float std_x_wrt_pred_x = 0;
          float std_y_wrt_pred_y = 0;
          float std_y_wrt_theory_y = 0;
          float std_y_pred_wrt_theory_y = 0;
          float std_j_wrt_pred_j = 0;

          void print() {
            std::printf("Avg. per joint:     \t error   \t %%error \t tip err \t %%tip err \t std dev\n");
            std::printf("x wrt pred_x        \t %.3f mm \t %.3f%% \t %.3f mm \t %.3f%% \t %.1f mm\n",        x_wrt_pred_x * 1000,        x_wrt_pred_x_percent * 100,        x_tip_wrt_pred_x * 1000,        x_tip_wrt_pred_x_percent * 100,        std_x_wrt_pred_x * 1000);
            std::printf("y wrt pred_y        \t %.3f mm \t %.3f%% \t %.3f mm \t %.3f%% \t %.1f mm\n",        y_wrt_pred_y * 1000,        y_wrt_pred_y_percent * 100,        y_tip_wrt_pred_y * 1000,        y_tip_wrt_pred_y_percent * 100,        std_y_wrt_pred_y * 1000);
            std::printf("y wrt theory_y      \t %.3f mm \t %.3f%% \t %.3f mm \t %.3f%% \t %.1f mm\n",      y_wrt_theory_y * 1000,      y_wrt_theory_y_percent * 100,      y_tip_wrt_theory_y * 1000,      y_tip_wrt_theory_y_percent * 100,      std_y_wrt_theory_y * 1000);
            std::printf("y_pred wrt theory_y \t %.3f mm \t %.3f%% \t %.3f mm \t %.3f%% \t %.1f mm\n", y_pred_wrt_theory_y * 1000, y_pred_wrt_theory_y_percent * 100, y_pred_tip_wrt_theory_y * 1000, y_pred_tip_wrt_theory_y_percent * 100, std_y_pred_wrt_theory_y * 1000);
            std::printf("j wrt pred_j        \t %.3f deg \t %.3f%% \t        \t        \t %.1f deg\n", j_wrt_pred_j * (180 / 3.1415926536), j_wrt_pred_j_percent * 100, std_j_wrt_pred_j * (180 / 3.1415926536));
            // std::printf("Free end tip error:\n");
            // std::printf("x tip wrt pred_x   \t %.3f mm\n", x_tip_wrt_pred_x * 1000);
            // std::printf("y tip wrt pred_y   \t %.3f mm\n", y_tip_wrt_pred_y * 1000);
            // std::printf("y tip wrt theory_y \t %.3f mm\n", y_tip_wrt_theory_y * 1000);
          }
        } error;

        void calc_error() {

          // reset all error counts
          Error blank_error;
          error = blank_error;

          // skip first entries as that is the fixed end
          uint skip = 1;

          // get length of vectors (this is N+1)
          uint Np1 = y.size();

          float x_cum = 0;
          float y_cum = 0;
          float pred_y_cum = 0;
          float theory_y_cum = 0;

          // loop and calculate abs and percentage errors sums
          for (uint i = skip; i < Np1; i++) {

            error.x_wrt_pred_x += abs(x[i] - pred_x[i]);
            error.x_wrt_pred_x_percent += -1 * (x[i] - pred_x[i]) / pred_x[i];
            error.y_wrt_pred_y += abs(y[i] - pred_y[i]);
            error.y_wrt_pred_y_percent += -1 * (y[i] - pred_y[i]) / pred_y[i];
            error.y_wrt_theory_y += abs(y[i] - theory_y[i]);
            error.y_wrt_theory_y_percent += -1 * (y[i] - theory_y[i]) / theory_y[i];
            error.y_pred_wrt_theory_y += abs(pred_y[i] - theory_y[i]);
            error.y_pred_wrt_theory_y_percent += -1 * (pred_y[i] - theory_y[i]) / theory_y[i];
            error.j_wrt_pred_j += abs(joints[i - 1] - pred_j[i - 1]);
            error.j_wrt_pred_j_percent += -1 * (joints[i - 1] - pred_j[i - 1]) / pred_j[i - 1];

            // calculate the total deflection values (for average tipratio)
            x_cum += x[i]; 
            y_cum += y[i];
            pred_y_cum += pred_y[i];
            theory_y_cum += theory_y[i];
          }

          // divide to get average
          float N = Np1 - 1.0;
          error.x_wrt_pred_x /= N;
          error.x_wrt_pred_x_percent /= N;
          error.y_wrt_pred_y /= N;
          error.y_wrt_pred_y_percent /= N;
          error.y_wrt_theory_y /= N;
          error.y_wrt_theory_y_percent /= N;
          error.y_pred_wrt_theory_y /= N;
          error.y_pred_wrt_theory_y_percent /= N;
          error.j_wrt_pred_j /= N;
          error.j_wrt_pred_j_percent /= N;

          // get tip ratios from average error over max value
          error.x_wrt_pred_x_tipratio = error.x_wrt_pred_x / x[Np1 - 1];
          error.y_wrt_pred_y_tipratio = error.y_wrt_pred_y / y[Np1 - 1];
          error.y_wrt_theory_y_tipratio = error.y_wrt_theory_y / y[Np1 - 1];
          error.y_pred_wrt_theory_y_tipratio = error.y_pred_wrt_theory_y / pred_y[Np1 - 1];

          // get tip differences
          error.x_tip_wrt_pred_x = x[Np1 - 1] - pred_x[Np1 - 1];
          error.y_tip_wrt_pred_y = y[Np1 - 1] - pred_y[Np1 - 1];
          error.y_tip_wrt_theory_y = y[Np1 - 1] - theory_y[Np1 - 1];
          error.y_pred_tip_wrt_theory_y = pred_y[Np1 - 1] - theory_y[Np1 - 1];

          // get tip difference percentages
          error.x_tip_wrt_pred_x_percent = (error.x_tip_wrt_pred_x) / pred_x[Np1 - 1];
          error.y_tip_wrt_pred_y_percent = (error.y_tip_wrt_pred_y) / pred_y[Np1 - 1];
          error.y_tip_wrt_theory_y_percent = (error.y_tip_wrt_theory_y) / theory_y[Np1 - 1];
          error.y_pred_tip_wrt_theory_y_percent = (error.y_pred_tip_wrt_theory_y) / theory_y[Np1 - 1];

          // loop through to get standard deviation
          for (uint i = skip; i < N; i++) {
            error.std_x_wrt_pred_x += std::pow(x[i] - pred_x[i] - error.x_wrt_pred_x, 2);
            error.std_y_wrt_pred_y += std::pow(y[i] - pred_y[i] - error.y_wrt_pred_y, 2);
            error.std_y_wrt_theory_y += std::pow(y[i] - theory_y[i] - error.y_wrt_theory_y, 2);
            error.std_y_pred_wrt_theory_y += std::pow(pred_y[i] - theory_y[i] - error.y_pred_wrt_theory_y, 2);
            error.std_j_wrt_pred_j += std::pow(joints[i] - pred_j[i] - error.j_wrt_pred_j, 2);
          }

          // average and square root
          error.std_x_wrt_pred_x = std::sqrt(error.std_x_wrt_pred_x / N);
          error.std_y_wrt_pred_y = std::sqrt(error.std_y_wrt_pred_y / N);
          error.std_y_wrt_theory_y = std::sqrt(error.std_y_wrt_theory_y / N);
          error.std_y_pred_wrt_theory_y = std::sqrt(error.std_y_pred_wrt_theory_y / N);
          error.std_j_wrt_pred_j = std::sqrt(error.std_j_wrt_pred_j / N);
        }

        void print_table() {
          float to_deg = (180.0 / 3.1415926535897);
          std::cout << "n \t x \t x_p \t E_x \t y_t \t y \t y_p \t E_y \t j \t j_p \t E_j (units mm/deg)\n";
          for (uint i = 0; i < x.size(); i++) {
            std::printf("%i \t %.1f \t %.1f \t %.1f \t %.1f \t %.1f \t %.1f \t %.1f \t %.1f \t %.1f \t %.1f\n",
              i, x[i] * 1000, pred_x[i] * 1000, (x[i] - pred_x[i]) * 1000,
              theory_y[i] * 1000, y[i] * 1000, pred_y[i] * 1000, (y[i] - pred_y[i]) * 1000,
              joints[i] * to_deg, pred_j[i] * to_deg, (joints[i] - pred_j[i]) * to_deg);
          }
        }
      };

      FingerData f1;
      FingerData f2;
      FingerData f3;
      FingerData::Error avg_error;
      std::string tag_string;

      void calc_error()
      {
        f1.calc_error();
        f2.calc_error();
        f3.calc_error();

        avg_error.x_wrt_pred_x = (f1.error.x_wrt_pred_x + f2.error.x_wrt_pred_x + f3.error.x_wrt_pred_x) / 3.0;
        avg_error.x_wrt_pred_x_percent = (f1.error.x_wrt_pred_x_percent + f2.error.x_wrt_pred_x_percent + f3.error.x_wrt_pred_x_percent) / 3.0;
        avg_error.y_wrt_pred_y = (f1.error.y_wrt_pred_y + f2.error.y_wrt_pred_y + f3.error.y_wrt_pred_y) / 3.0;
        avg_error.y_wrt_pred_y_percent = (f1.error.y_wrt_pred_y_percent + f2.error.y_wrt_pred_y_percent + f3.error.y_wrt_pred_y_percent) / 3.0;
        avg_error.y_wrt_theory_y = (f1.error.y_wrt_theory_y + f2.error.y_wrt_theory_y + f3.error.y_wrt_theory_y) / 3.0;
        avg_error.y_wrt_theory_y_percent = (f1.error.y_wrt_theory_y_percent + f2.error.y_wrt_theory_y_percent + f3.error.y_wrt_theory_y_percent) / 3.0;
        avg_error.y_pred_wrt_theory_y = (f1.error.y_pred_wrt_theory_y + f2.error.y_pred_wrt_theory_y + f3.error.y_pred_wrt_theory_y) / 3.0;
        avg_error.y_pred_wrt_theory_y_percent = (f1.error.y_pred_wrt_theory_y_percent + f2.error.y_pred_wrt_theory_y_percent + f3.error.y_pred_wrt_theory_y_percent) / 3.0;
        avg_error.j_wrt_pred_j = (f1.error.j_wrt_pred_j + f2.error.j_wrt_pred_j + f3.error.j_wrt_pred_j) / 3.0;
        avg_error.j_wrt_pred_j_percent = (f1.error.j_wrt_pred_j_percent + f2.error.j_wrt_pred_j_percent + f3.error.j_wrt_pred_j_percent) / 3.0;

        avg_error.x_wrt_pred_x_tipratio = (f1.error.x_wrt_pred_x_tipratio + f2.error.x_wrt_pred_x_tipratio + f3.error.x_wrt_pred_x_tipratio) / 3.0;
        avg_error.y_wrt_pred_y_tipratio = (f1.error.y_wrt_pred_y_tipratio + f2.error.y_wrt_pred_y_tipratio + f3.error.y_wrt_pred_y_tipratio) / 3.0;
        avg_error.y_wrt_theory_y_tipratio = (f1.error.y_wrt_theory_y_tipratio + f2.error.y_wrt_theory_y_tipratio + f3.error.y_wrt_theory_y_tipratio) / 3.0;
        avg_error.y_pred_wrt_theory_y_tipratio = (f1.error.y_pred_wrt_theory_y_tipratio + f2.error.y_pred_wrt_theory_y_tipratio + f3.error.y_pred_wrt_theory_y_tipratio) / 3.0;

        avg_error.x_tip_wrt_pred_x = (f1.error.x_tip_wrt_pred_x + f2.error.x_tip_wrt_pred_x + f3.error.x_tip_wrt_pred_x) / 3.0;
        avg_error.y_tip_wrt_pred_y = (f1.error.y_tip_wrt_pred_y + f2.error.y_tip_wrt_pred_y + f3.error.y_tip_wrt_pred_y) / 3.0;
        avg_error.y_tip_wrt_theory_y = (f1.error.y_tip_wrt_theory_y + f2.error.y_tip_wrt_theory_y + f3.error.y_tip_wrt_theory_y) / 3.0;
        avg_error.y_pred_tip_wrt_theory_y = (f1.error.y_pred_tip_wrt_theory_y + f2.error.y_pred_tip_wrt_theory_y + f3.error.y_pred_tip_wrt_theory_y) / 3.0;

        avg_error.x_tip_wrt_pred_x_percent = (f1.error.x_tip_wrt_pred_x_percent + f2.error.x_tip_wrt_pred_x_percent + f3.error.x_tip_wrt_pred_x_percent) / 3.0;
        avg_error.y_tip_wrt_pred_y_percent = (f1.error.y_tip_wrt_pred_y_percent + f2.error.y_tip_wrt_pred_y_percent + f3.error.y_tip_wrt_pred_y_percent) / 3.0;
        avg_error.y_tip_wrt_theory_y_percent = (f1.error.y_tip_wrt_theory_y_percent + f2.error.y_tip_wrt_theory_y_percent + f3.error.y_tip_wrt_theory_y_percent) / 3.0;
        avg_error.y_pred_tip_wrt_theory_y_percent = (f1.error.y_pred_tip_wrt_theory_y_percent + f2.error.y_pred_tip_wrt_theory_y_percent + f3.error.y_pred_tip_wrt_theory_y_percent) / 3.0;

        avg_error.std_x_wrt_pred_x = (f1.error.std_x_wrt_pred_x + f2.error.std_x_wrt_pred_x + f3.error.std_x_wrt_pred_x) / 3.0;
        avg_error.std_y_wrt_pred_y = (f1.error.std_y_wrt_pred_y + f2.error.std_y_wrt_pred_y + f3.error.std_y_wrt_pred_y) / 3.0;
        avg_error.std_y_wrt_theory_y = (f1.error.std_y_wrt_theory_y + f2.error.std_y_wrt_theory_y + f3.error.std_y_wrt_theory_y) / 3.0;
        avg_error.std_y_pred_wrt_theory_y = (f1.error.std_y_pred_wrt_theory_y + f2.error.std_y_pred_wrt_theory_y + f3.error.std_y_pred_wrt_theory_y) / 3.0;
        avg_error.std_j_wrt_pred_j = (f1.error.std_j_wrt_pred_j + f2.error.std_j_wrt_pred_j + f3.error.std_j_wrt_pred_j) / 3.0;
      }

      void print() {

        // print identifying information
        std::cout << "Finger pose data: \t" << tag_string << '\n';

        // print table data only for finger 2
        f2.print_table();
        
        // update and print error calculations for all three fingers
        calc_error();
        avg_error.print();

        std::cout << "\n\n";
      }
    };

    // save a series of data points
    std::vector<PoseData> entries;

    void update() {
      for (uint i = 0; i < entries.size(); i++) entries[i].calc_error();
    }

    void print() {
      for (unsigned int i = 0; i < entries.size(); i++) {
        std::cout << "ENTRY " << i << "\n";
        entries[i].print();
        std::cout << "\n\n";
      }
    }

    void reset() {
      std::vector<PoseData>().swap(entries);
    }
  };

  // // calibration constants for gauge data
  // struct RealGaugeCalibrations {

  //   /* applied as follows: g_out = (g_raw + offset) * scale */
  //   struct RealSensors { float g1 {}, g2 {}, g3 {}, palm {}, wrist_Z {}; };

  //   RealSensors offset;
  //   RealSensors scale;
  //   RealSensors norm;

  //   // when true, automatically detect the offset
  //   bool recalibrate_offset_flag = false;

  // };

  // data containers for all of the possible sensors
  struct SensorData {

    static constexpr int buffer_size = 50;

    // storage containers for state data
    luke::SlidingWindow<luke::gfloat> x_motor_position { buffer_size };
    luke::SlidingWindow<luke::gfloat> y_motor_position { buffer_size };
    luke::SlidingWindow<luke::gfloat> z_motor_position { buffer_size };
    luke::SlidingWindow<luke::gfloat> x_base_position { buffer_size };
    luke::SlidingWindow<luke::gfloat> y_base_position { buffer_size };
    luke::SlidingWindow<luke::gfloat> z_base_position { buffer_size };
    luke::SlidingWindow<luke::gfloat> roll_base_rotation { buffer_size };
    luke::SlidingWindow<luke::gfloat> pitch_base_rotation { buffer_size };
    luke::SlidingWindow<luke::gfloat> yaw_base_rotation { buffer_size };

    // create storage containers for sensor data
    luke::SlidingWindow<luke::gfloat> finger1_gauge { buffer_size };
    luke::SlidingWindow<luke::gfloat> finger2_gauge { buffer_size };
    luke::SlidingWindow<luke::gfloat> finger3_gauge { buffer_size };
    luke::SlidingWindow<luke::gfloat> palm_sensor { buffer_size };
    luke::SlidingWindow<luke::gfloat> finger1_axial_gauge { buffer_size };
    luke::SlidingWindow<luke::gfloat> finger2_axial_gauge { buffer_size };
    luke::SlidingWindow<luke::gfloat> finger3_axial_gauge { buffer_size };
    luke::SlidingWindow<luke::gfloat> wrist_X_sensor { buffer_size };
    luke::SlidingWindow<luke::gfloat> wrist_Y_sensor { buffer_size };
    luke::SlidingWindow<luke::gfloat> wrist_Z_sensor { buffer_size };

    void reset() {
      x_motor_position.reset();
      y_motor_position.reset();
      z_motor_position.reset();
      x_base_position.reset();
      y_base_position.reset();
      z_base_position.reset();
      roll_base_rotation.reset();
      pitch_base_rotation.reset();
      yaw_base_rotation.reset();
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
    }

    // getters to read the most recent element for each sensor, exposed to python
    luke::gfloat read_x_motor_position() { return x_motor_position.read_element(); }
    luke::gfloat read_y_motor_position() { return y_motor_position.read_element(); }
    luke::gfloat read_z_motor_position() { return z_motor_position.read_element(); }
    luke::gfloat read_x_base_position() { return x_base_position.read_element(); }
    luke::gfloat read_y_base_position() { return y_base_position.read_element(); }
    luke::gfloat read_z_base_position() { return z_base_position.read_element(); }
    luke::gfloat read_roll_base_rotation() { return roll_base_rotation.read_element(); }
    luke::gfloat read_pitch_base_rotation() { return pitch_base_rotation.read_element(); }
    luke::gfloat read_yaw_base_rotation() { return yaw_base_rotation.read_element(); }
    luke::gfloat read_finger1_gauge() { return finger1_gauge.read_element(); }
    luke::gfloat read_finger2_gauge() { return finger2_gauge.read_element(); }
    luke::gfloat read_finger3_gauge() { return finger3_gauge.read_element(); }
    luke::gfloat read_palm_sensor() { return palm_sensor.read_element(); }
    luke::gfloat read_finger1_axial_gauge() { return finger1_axial_gauge.read_element(); }
    luke::gfloat read_finger2_axial_gauge() { return finger2_axial_gauge.read_element(); }
    luke::gfloat read_finger3_axial_gauge() { return finger3_axial_gauge.read_element(); }
    luke::gfloat read_wrist_X_sensor() { return wrist_X_sensor.read_element(); }
    luke::gfloat read_wrist_Y_sensor() { return wrist_Y_sensor.read_element(); }
    luke::gfloat read_wrist_Z_sensor() { return wrist_Z_sensor.read_element(); }
  };

  // real sensor calibration data structure
  struct RealCalibrations {

    struct Calibration {

      double scale = 0;
      double offset = 0;
      double norm = 0;

      Calibration() {}
      Calibration(double scale, double offset)
        : scale(scale), offset(offset), norm(0) {}

      double apply_calibration(double value) {
        return (value - offset) * scale;
      }

      double apply_normalisation(double value) {
        return normalise_between(value, -norm, norm);
      }

    };

    // gauge calibrations  scale        offset (note ALL offsets are wiped at runtime for zero-ing)
    Calibration g1_0p9_28 {2.7996e-6,   260576};
    Calibration g2_0p9_28 {2.7440e-6,   911830};
    Calibration g3_0p9_28 {2.7303e-6,   1000000};
    Calibration g1_1p0_24 {2.6866e-6,   196784};
    Calibration g2_1p0_24 {2.7391e-6,   615652};
    Calibration g3_1p0_24 {2.7146e-6,   869492};
    Calibration g1_1p0_28 {3.1321e-6,   96596};
    Calibration g2_1p0_28 {3.1306e-6,   7583};
    Calibration g3_1p0_28 {3.1390e-6,   66733};
    Calibration palm      {7.4626e-5,   0};         // offset set at runtime
    Calibration wrist_Z   {-1,          0};         // offset set at runtime

    // get the correct calibration for the fingers
    Calibration get_gauge_calibration(int gauge_num, double thickness, double width)
    {
      /* return the correct calibration given finger dimensions */

      if (thickness > 2e-3) {
        throw std::runtime_error("get_gauge_calibration(...) got finger thickness over 2mm, check SI units!");
      }
      if (width > 50e-3) {
        throw std::runtime_error("get_gauge_calibration(...) got finger width over 50mm, check SI units!");
      }

      double tol = 1e-5;

      if (abs(thickness - 0.9e-3) < tol) {

        if (abs(width - 28e-3) < tol) {
          switch (gauge_num) {
            case 1: return g1_0p9_28;
            case 2: return g2_0p9_28;
            case 3: return g3_0p9_28;
          }
        }
        else {
          throw std::runtime_error("get_gauge_calibration(...) does not have a calibration for 0.9mm finger with this finger width");
        }

      }
      else if (abs(thickness - 1.0e-3) < tol) {

        if (abs(width - 24e-3) < tol) {
          switch (gauge_num) {
            case 1: return g1_1p0_24;
            case 2: return g2_1p0_24;
            case 3: return g3_1p0_24;
          }
        }
        else if (abs(width - 28e-3) < tol) {
          switch (gauge_num) {
            case 1: return g1_1p0_28;
            case 2: return g2_1p0_28;
            case 3: return g3_1p0_28;
          }
        }
        else {
          throw std::runtime_error("get_gauge_calibration(...) does not have a calibration for 1.0mm finger with this finger width");
        }

      }
      else {
        throw std::runtime_error("get_gauge_calibration(...) does not have a calibration for this finger thickness");
      }
    }

  };

  // data structure for storing real sensor data
  struct RealSensorData {

    MjType::RealCalibrations::Calibration g1;      // PCB gauge 1
    MjType::RealCalibrations::Calibration g2;      // PCB gauge 2
    MjType::RealCalibrations::Calibration g3;      // PCB gauge 3
    MjType::RealCalibrations::Calibration palm;    // PCB gauge 4
    MjType::RealCalibrations::Calibration wrist_Z; // force/torque wrist sensor

    // autocalibration vectors
    int calibration_samples = 20; // 10Hz so 20=2sec calibration time
    std::vector<float> f1_calibration;
    std::vector<float> f2_calibration;
    std::vector<float> f3_calibration;
    std::vector<float> palm_calibration;
    std::vector<float> wrist_Z_calibration;

    MjType::SensorData raw;          // raw sensor values
    MjType::SensorData SI;           // calibrated and scaled to SI units
    MjType::SensorData normalised;   // normalised [-1,+1] for neural network

    // flag which indicates if we should re-zero sensors
    bool recalibrate_offset_flag = false;

    void reset() {

      recalibrate_offset_flag = true;

      // wipe calibration, causes sensors to re-zero
      f1_calibration.clear();
      f2_calibration.clear();
      f3_calibration.clear();
      palm_calibration.clear();
      wrist_Z_calibration.clear();

      // wipe all saved real data
      raw.reset();
      SI.reset();
      normalised.reset();
    }

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

  // for uniform random numbers [0.0, 1.0]
  std::uniform_real_distribution<float> uniform_dist {0.0, 1.0};

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
  bool render_reload = false;                   // have we reloaded and need to update rendering
  double sim_gauge_raw_to_N_factor = 1.0;       // calibration factor, set by auto_calibrate

  // function pointers for sampling functions
  std::vector<luke::gfloat> (MjType::Sensor::*sampleFcnPtr)
    (luke::SlidingWindow<luke::gfloat>);
  std::vector<luke::gfloat> (MjType::Sensor::*stateFcnPtr)
    (luke::SlidingWindow<luke::gfloat>);

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

  // these flags are only reset with hard_reset()
  struct ResetFlags {

    // have we initialised the flags to starting values
    bool flags_init = false;

    // flag to indicate which auto function are selected
    bool auto_timestep = false;
    bool auto_calibrate = false;
    bool auto_simsteps = false;
    bool auto_exceed_lateral_lim = false;

    bool finger_EI_changed = false;

  } resetFlags;

  /* ----- variables that are reset ----- */

  // standard class variables
  int n_actions;                      // number of possible actions
  std::vector<int> action_options;    // possible action codes

  // track the timestamps of sensor updates, this is for plotting in mysimlulate.cpp
  luke::SlidingWindow<float> step_timestamps { MjType::SensorData::buffer_size };
  luke::SlidingWindow<float> gauge_timestamps { MjType::SensorData::buffer_size };
  luke::SlidingWindow<float> axial_timestamps { MjType::SensorData::buffer_size };
  luke::SlidingWindow<float> palm_timestamps { MjType::SensorData::buffer_size };
  luke::SlidingWindow<float> wristXY_timestamps { MjType::SensorData::buffer_size };
  luke::SlidingWindow<float> wristZ_timestamps { MjType::SensorData::buffer_size };

  // operating limits for the base joints
  std::vector<double> base_min_;
  std::vector<double> base_max_;

  // data structures
  MjType::SensorData sim_sensors_;
  MjType::SensorData sim_sensors_SI_;
  MjType::RealSensorData real_sensors_;
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
  std::string file_from_from_command_line(int argc, char **argv);

  // core functionality
  void load(std::string file_path);
  void load_relative(std::string file_path);
  void reset();
  void hard_reset();
  void step();
  bool render();
  void close_render();

  // sensing
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
  void randomise_object_colour();
  void randomise_ground_colour();
  void randomise_finger_colours();
  void set_neat_colours();
  bool is_done();
  std::vector<luke::gfloat> get_observation();
  std::vector<luke::gfloat> get_observation(MjType::SensorData sensors);
  std::vector<float> get_event_state();
  std::vector<float> get_goal();
  std::vector<float> assess_goal();
  std::vector<float> assess_goal(std::vector<float> event_vec);
  float reward();
  float reward(std::vector<float> goal_vec, std::vector<float> event_vec);
  int get_n_actions();
  int get_n_obs();
  int get_N();
  float get_finger_thickness();
  std::vector<luke::gfloat> get_finger_stiffnesses();

  // sensor getters
  // std::vector<luke::gfloat> get_bend_gauge_readings(bool unnormalise);
  // luke::gfloat get_palm_reading(bool unnormalise);
  // luke::gfloat get_wrist_reading(bool unnormalise);
  // std::vector<luke::gfloat> get_state_readings(bool unnormalise);
  std::vector<luke::gfloat> get_finger_forces(bool realworld);
  luke::gfloat get_palm_force(bool realworld);
  luke::gfloat get_wrist_force(bool realworld);
  std::vector<luke::gfloat> get_state_metres(bool realworld);
  luke::gfloat get_finger_angle();

  // real world gripper functions
  void calibrate_real_sensors();
  std::vector<float> input_real_data(std::vector<float> state_data, 
    std::vector<float> sensor_data);
  std::vector<float> get_real_observation();
  std::vector<float> get_simple_state_vector(MjType::SensorData sensor);

  // misc
  void forward() { mj_forward(model, data); }
  int get_number_of_objects() { return env_.object_names.size(); }
  std::string get_current_object_name() { return env_.obj.name; }
  MjType::TestReport get_test_report();
  MjType::CurveFitData::PoseData validate_curve(int force_style = 0);
  MjType::CurveFitData::PoseData validate_curve_under_force(float force, int force_style = 0);
  MjType::CurveFitData curve_validation_regime(bool print = false, int force_style = 0);
  std::string numerical_stiffness_converge(float force, float target_accuracy);
  std::string numerical_stiffness_converge(float force, float target_accuracy, 
    std::vector<float> X, std::vector<float> Y);
  std::string numerical_stiffness_converge_2(float target_accuracy);
  std::vector<float> profile_error(std::vector<float> profile_X, std::vector<float> profile_Y,
    std::vector<float> truth_X, std::vector<float> truth_Y, bool relative);
  float curve_area(std::vector<float> X, std::vector<float> Y);
  void calibrate_simulated_sensors(float bend_gauge_normalise);
  void set_finger_thickness(float thickness);
  void set_finger_width(float width);
  void set_finger_modulus(float E);
  float yield_load();
  float yield_load(float thickness, float width);
  void tick();
  float tock();
  MjType::EventTrack add_events(MjType::EventTrack& e1, MjType::EventTrack& e2);
  void reset_goal();
  void print(std::string s) { std::printf("%s\n", s.c_str()); }
  void default_goal_event_triggering();
  bool last_action_gripper();
  bool last_action_panda();
  float get_fingertip_z_height();
  float find_highest_stable_timestep();
  void set_sensor_noise_and_normalisation_to(bool set_as);

}; // class MjClass

// helper functions
void update_events(MjType::EventTrack& events, MjType::Settings& settings);
float calc_rewards(MjType::EventTrack& events, MjType::Settings& settings);
float goal_rewards(MjType::EventTrack& events, MjType::Settings& settings,
  MjType::Goal goal);
MjType::Goal score_goal(MjType::Goal const goal, std::vector<float> event_vec, 
  MjType::Settings settings);
MjType::Goal score_goal(MjType::Goal const goal, MjType::EventTrack event, 
  MjType::Settings settings);

// simple command line argument parsing class for mysimulate and test binaries
// from: https://stackoverflow.com/questions/865668/parsing-command-line-arguments-in-c
class MjClassInputParser {
  public:
    MjClassInputParser (int &argc, char **argv){
      for (int i=1; i < argc; ++i)
        this->tokens.push_back(std::string(argv[i]));
    }
    /// @author iain
    std::string getCmdOption(std::string option) {
      std::vector<std::string>::const_iterator itr;
      itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
      if (itr != this->tokens.end() && ++itr != this->tokens.end()) {
        return *itr;
      }
      std::string empty_string("");
      return empty_string;
    }
    /// @author iain
    bool cmdOptionExists(std::string option) {
      return std::find(this->tokens.begin(), this->tokens.end(), option)
        != this->tokens.end();
    }
    /// @author luke
    std::string getCmdFromList(std::string a, std::string b) {
      std::vector<std::string> v { a, b };
      return getCmdFromList(v);
    }
    std::string getCmdFromList(std::vector<std::string> options) {
      std::string output;
      for (std::string s : options) {
        output = getCmdOption(s);
        if (not output.empty()) {
          return output;
        }
      }
      return output;
    }


  private:
    std::vector <std::string> tokens;
};

#endif // MJCLASS_H_