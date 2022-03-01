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
  X(  debug,                    bool,     true)     /* print debug info to terminal*/\
  /*
  step() settings */\
  X(  gauge_read_rate_hz,       double,   10.0)     /* frequency of gauge readings*/\
  /* 
  update_env() settings */\
  X(  lift_distance,            double,   1e-3)     /* distance to consider object lifted*/\
  X(  oob_distance,             double,   75e-3)    /* distance to consider object out of bounds*/\
  X(  height_target,            double,   25e-3)    /* target height to raise the object by*/\
  X(  stable_finger_force,      double,   1.0)      /* finger force on object to consider stable*/\
  X(  stable_palm_force,        double,   2.0)      /* palm force on object to consider stable*/\
  /* 
  is_done() settings */\
  X(  max_timeouts,             int,      10)       /* done=true if unsettled this times in a row*/\
  /* 
  set_action() settings */\
  X(  action_motor_steps,       int,      100)      /* motor steps per action*/\
  X(  action_base_translation,  double,   2e-3)     /* base translation per action*/\
  X(  max_action_steps,         int,      200)      /* max sim steps in one action*/\
  X(  paired_motor_X_step,      bool,     true)     /* run both X and Y motors for X step*/\
  X(  use_palm_action,          bool,     true)     /* moving palm is a possible action*/\
  X(  use_height_action,        bool,     true)     /* moving base height is possible action*/\
  /* 
  action_step() settings */\
  X(  render_on_step,           bool,     false)    /* render on every single sim step*/\
  X(  use_settling,             bool,     false)    /* quit early if sim settled (buggy atm)*/\
  /* 
  render() settings */\
  X(  use_render_delay,         bool,     false)    /* pause when rendering */\
  X(  render_delay,             double,   0.5)      /* how long to pause when rendering*/\
  /* 

  2. Binary rewards
      name                      reward    done      trigger */\
  BR( step_num,                 -0.01,    false,    1)    \
  BR( lifted,                   0.005,    false,    1)    \
  BR( oob,                      0.0,      1,        1)    \
  BR( dropped,                  0.0,      false,    1000) \
  BR( target_height,            1.0,      false,    1000) \
  BR( exceed_limits,            -0.1,     false,    1)    \
  BR( object_contact,           0.005,    false,    1)    \
  BR( object_stable,            1.0,      1,        1)    \
  /*

  3. Linear rewards
      name                      reward    done   trigger  min   max   overshoot*/\
  LR( exceed_axial,             -0.05,    false,    1,    2.0,  6.0,  -1)   \
  LR( exceed_lateral,           -0.05,    false,    1,    4.0,  6.0,  -1)   \
  LR( exceed_palm,              -0.05,    false,    1,    6.0,  10.0, -1)   \
  LR( palm_force,               0.05,     false,    1,    1.0,  3.0,  6.0) 

// end of user defined simulation settings

#endif