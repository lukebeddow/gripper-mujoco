#ifndef SIMSETTINGS_H_
#define SIMSETTINGS_H_

/* The LUKE_MJSETTINGS  macro defines the settings for the simulation.

   The types of simulation settings are:
      X  - standard setting using a built-in type
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
  XX(  curve_validation,        bool,     false)    /* save finger curve data for testing */\
  /*
  step() and gauge settings */\
  XX(  gauge_read_rate_hz,      double,   10.0)     /* frequency of gauge readings */\
  XX(  normalising_force,       double,   100.0)    /* max reading to normalise gauges to -1, 1*/\
  XX(  use_palm_sensor,         bool,     true)     /* use a sensor on the palm */\
  XX(  palm_force_normalise,    double,   8.0)      /* <0:bumper sensor, >0:normalising value*/\
  XX(  obs_raw_data,            bool,     false)    /* use raw sensor data for get_observation()*/\
  XX(  use_wrist_sensor_XY,     bool,     false)    /* get X and Y force information from wrist sensor*/\
  XX(  use_wrist_sensor_Z,      bool,     false)    /* get Z force information from wrist sensor*/\
  XX(  use_axial_strain_gauge,  bool,     false)    /* get axial finger strain information*/\
  /* 
  update_env() settings */\
  XX(  lift_distance,           double,   1e-3)     /* distance to consider object lifted */\
  XX(  oob_distance,            double,   75e-3)    /* distance to consider object out of bounds */\
  XX(  height_target,           double,   30e-3)    /* target height to raise the object by */\
  XX(  stable_finger_force,     double,   0.4)      /* finger force on object to consider stable */\
  XX(  stable_palm_force,       double,   1.0)      /* palm force on object to consider stable */\
  /* 
  is_done() settings */\
  XX(  max_timeouts,            int,      10)       /* done=true if unsettled this times in a row */\
  XX(  quit_on_reward_below,    float,    -1.01)    /* done=true if reward drops below this value */\
  XX(  quit_reward_capped,      bool,     true)     /* cap reward at quit_on_reward_below */\
  /* 
  set_action() settings */\
  XX(  action_motor_steps,      int,      100)      /* stepper motor steps per action */\
  XX(  action_base_translation, double,   2e-3)     /* base translation per action */\
  XX(  max_action_steps,        int,      200)      /* max sim steps in one action */\
  XX(  paired_motor_X_step,     bool,     true)     /* run both X and Y motors for X step */\
  XX(  use_palm_action,         bool,     true)     /* moving palm is a possible action */\
  XX(  use_height_action,       bool,     true)     /* moving base height is possible action */\
  /* 
  action_step() settings */\
  XX(  render_on_step,          bool,     false)    /* render on every single sim step */\
  XX(  use_settling,            bool,     false)    /* quit early if sim settled (buggy atm) */\
  /* 
  render() settings */\
  XX(  use_render_delay,        bool,     false)    /* pause when rendering */\
  XX(  render_delay,            double,   0.5)      /* how long to pause when rendering */\
  /* 

  2. Sensors
      name                      used      normalise read-rate */\
  SS(  bending_gauge,           true,     100.0,    10)       /* strain gauge to measure finger bending*/\
  SS(  axial_gauge,             false,    1,        1)        /* strain gauge to measure axial finger strain*/\
  SS(  palm_sensor,             false,    1,        1)        /* palm force sensor */\
  SS(  wrist_sensor_XY,         false,    1,        1)        /* F/T wrist sensor X and Y forces*/\
  SS(  wrist_sensor_Z,          false,    1,        1)        /* F/T wrist sensor Z force*/\
  /* 

  3. Binary rewards
      name                      reward    done      trigger */\
  BR( step_num,                 -0.01,    false,    1)      /* when a step is made */\
  BR( lifted,                   0.005,    false,    1)      /* object leaves the ground */\
  BR( oob,                      0.0,      1,        1)      /* object out of bounds */\
  BR( dropped,                  0.0,      false,    1000)   /* object lifted and then touches gnd */\
  BR( target_height,            1.0,      false,    1000)   /* object lifted to height target */\
  BR( exceed_limits,            -0.1,     false,    1)      /* gripper motor limits exceeded */\
  BR( object_contact,           0.005,    false,    1)      /* fingers or palm touches object */\
  BR( object_stable,            1.0,      false,    1)      /* fingers and palm apply min force */\
  BR( stable_height,            0.0,      1,        1)      /* object stable and at height target */\
  /*

  4. Linear rewards
      name                      reward    done   trigger  min   max   overshoot */\
  LR( finger_force,             0.0,      false,    1,    1.0,  2.0,  -1)     /* avg. force from fingers */\
  LR( palm_force,               0.05,     false,    1,    1.0,  3.0,  6.0)    /* palm force applied */\
  LR( exceed_axial,             -0.05,    false,    1,    2.0,  6.0,  -1)     /* exceed axial finger force limit */\
  LR( exceed_lateral,           -0.05,    false,    1,    4.0,  6.0,  -1)     /* exceed lateral finger force limit */\
  LR( exceed_palm,              -0.05,    false,    1,    6.0,  10.0, -1)     /* exceed palm force limit */
  

// end of user defined simulation settings

#endif