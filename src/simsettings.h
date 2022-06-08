#ifndef SIMSETTINGS_H_
#define SIMSETTINGS_H_

/* The LUKE_MJSETTINGS  macro defines the settings for the simulation.

   The types of simulation settings are:
      XX - standard setting using a built-in type
      SS - sensor information for getting and normalising data
      BR - binary reward, reward given when the trigger is reached
      LR - linear reward, reward given linearly between min and max

   This macro makes use the X Macro style: https://www.geeksforgeeks.org/x-macros-in-c/

   Essentially we save each variable name (first arg) with some values. Then,
   later in the code, by calling LUKE_MJSETTINGS, we get access to every variable
   along with its values. We can then use the name and values as substitution
   arguments for code snippets, and so build up repetative code for every variable.
*/

// start of user defined simulation settings, all lines except last must end in '\'
// all settings defined here will be read/write exposed to python under MjClass.set

#define LUKE_MJSETTINGS \
  /*

  1. Regular settings
                                type      value
  general */\
  XX(  debug,                   bool,     true)     /* print debug info to terminal */\
  XX(  mujoco_timestep,         float,    0.002)    /* sim timestep in seconds - default 0.002 */\
  XX(  curve_validation,        bool,     false)    /* save finger curve data for testing */\
  /*
  HER settings */\
  XX(  use_HER,                 bool,     false)    /* use hindsight experience replay (HER) */\
  XX(  goal_reward,             float,    1.0)      /* reward for achieving a goal */\
  XX(  divide_goal_reward,      bool,     true)     /* if multiple goals, do we split goal reward betwen them */\
  XX(  reward_on_end_only,      bool,     true)     /* give goal reward only on episode end */\
  XX(  binary_goal_vector,      bool,     false)    /* do we use only binary goals */\
  /*
  get_observation() settings    (NB: sample modes: 0=raw, 1=change, 2=average) */\
  XX(  sensor_sample_mode,      int,      1)        /* how to sample sensor observations, see MjType::Sample*/\
  XX(  state_sample_mode,       int,      0)        /* how to sample motor state, see MjType::Sample*/\
  /* 
  update_env() settings */\
  XX(  oob_distance,            double,   75e-3)    /* distance to consider object out of bounds */\
  XX(  done_height,             double,   25e-3)    /* grasp done if object lifted to this height */\
  XX(  stable_finger_force,     double,   0.4)      /* finger force (N) on object to consider stable */\
  XX(  stable_palm_force,       double,   1.0)      /* palm force (N) on object to consider stable */\
  /* 
  is_done() settings */\
  XX(  quit_on_reward_below,    float,    -1.01)    /* done=true if reward drops below this value */\
  XX(  quit_reward_capped,      bool,     true)     /* cap reward at quit_on_reward_below */\
  /* 
  set_action() settings */\
  XX(  action_motor_steps,      int,      100)      /* stepper motor steps per action */\
  XX(  action_base_translation, double,   2e-3)     /* base translation per action */\
  XX(  sim_steps_per_action,    int,      200)      /* sim steps in one action */\
  XX(  paired_motor_X_step,     bool,     true)     /* run both X and Y motors for X step */\
  XX(  use_palm_action,         bool,     true)     /* moving palm is a possible action */\
  XX(  use_height_action,       bool,     true)     /* moving base height is possible action */\
  /* 
  render() settings */\
  XX(  render_on_step,          bool,     false)    /* render on every single sim step */\
  XX(  use_render_delay,        bool,     false)    /* pause when rendering */\
  XX(  render_delay,            double,   0.5)      /* how long to pause when rendering */\
  /* 

  2. Sensors
      name                      used      normalise read-rate   (NB: -ve read-rate gives -n_readings) */\
  SS(  motor_state_sensor,      true,     0,        -2)     /* xyz motor states, normalise is ignored */\
  SS(  base_state_sensor,       true,     0,        -2)     /* base position state, normalise is ignored)*/\
  SS(  bending_gauge,           true,     100.0,    10)     /* strain gauge to measure finger bending */\
  SS(  axial_gauge,             true,     3.0,      10)     /* strain gauge to measure axial finger strain */\
  SS(  palm_sensor,             true,     8.0,      10)     /* palm force sensor */\
  SS(  wrist_sensor_XY,         true,     5.00,     10)     /* force wrist sensor X and Y forces */\
  SS(  wrist_sensor_Z,          true,     28.0,     10)     /* force wrist sensor Z force */\
  /* 

  3. Binary rewards
      name                      reward    done      trigger */\
  BR(  step_num,                -0.01,    false,    1)      /* when a step is made */\
  BR(  lifted,                  0.005,    false,    1)      /* object leaves the ground */\
  BR(  oob,                     0.0,      1,        1)      /* object out of bounds */\
  BR(  dropped,                 0.0,      false,    1000)   /* object lifted and then touches gnd */\
  BR(  target_height,           1.0,      false,    1000)   /* object lifted to done_height */\
  BR(  exceed_limits,           -0.1,     false,    1)      /* gripper motor limits exceeded */\
  BR(  object_contact,          0.005,    false,    1)      /* fingers or palm touches object */\
  BR(  object_stable,           1.0,      false,    1)      /* fingers and palm apply min force */\
  BR(  stable_height,           0.0,      1,        1)      /* object stable and at height target */\
  /*

  4. Linear rewards
      name                      reward    done   trigger  min   max   overshoot */\
  LR(  finger_force,            0.0,      false,    1,    1.0,  2.0,  -1)     /* avg. force from fingers */\
  LR(  palm_force,              0.05,     false,    1,    1.0,  3.0,  6.0)    /* palm force applied */\
  LR(  exceed_axial,            -0.05,    false,    1,    2.0,  6.0,  -1)     /* exceed axial finger force limit */\
  LR(  exceed_lateral,          -0.05,    false,    1,    4.0,  6.0,  -1)     /* exceed lateral finger force limit */\
  LR(  exceed_palm,             -0.05,    false,    1,    6.0,  10.0, -1)     /* exceed palm force limit */\
  /* testing extras for goals */\
  LR(  finger1_force,           0.0,      false,    1,    0.0,  2.0, 6.0)     /* finger 1 force */\
  LR(  finger2_force,           0.0,      false,    1,    0.0,  2.0, 6.0)     /* finger 2 force */\
  LR(  finger3_force,           0.0,      false,    1,    0.0,  2.0, 6.0)     /* finger 3 force */\
  LR(  ground_force,            0.0,      false,    1,    0.0,   2.0, -1)     /* ground force on object */\
  LR(  grasp_metric,            0.0,      false,    1,    0.0,  10.0, -1)     /* grasping metric score */\

// end of user defined simulation settings

#endif