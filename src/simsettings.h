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

#define LUKE_MJSETTINGS_GENERAL \
  /*
  
  1. Regular settings
                                type      value */\
  XX(  debug,                   bool,     true)     /* print debug info to terminal */\
  XX(  mujoco_timestep,         double,   0.001)    /* sim timestep in seconds - default 0.002 */\
  XX(  curve_validation,        bool,     false)    /* are we in curve validation mode, if yes collect curve data */\
  XX(  tip_force_applied,       double,   0.0)      /* apply a tip force, only possible in curve validation mode */\
  XX(  random_seed,             uint,     0)        /* random seed */\
  XX(  randomise_colours,       bool,     false)    /* randomise the colours of the objects */\
  /*
  automatic settings value detection, and parameters for guiding this */\
  XX(  auto_set_timestep,       bool,     true)     /* find the highest stable timestep, overrides mujoco_timestep */\
  XX(  auto_calibrate_gauges,   bool,     true)     /* normalise gauges between +-5N, overrides bending_gauge.normalise */\
  XX(  auto_sim_steps,          bool,     true)     /* automatically find the sim steps per action, overrides sim_steps_per_action */\
  XX(  auto_exceed_lateral_lim, bool,     false)     /* calculate safe finger bending automaticalled based on yield load */\
  XX(  time_for_action,         double,    0.2)      /* time in seconds to give for each action to complete, only used if auto_sim_steps=true */\
  XX(  saturation_yield_factor, float,    1.0)      /* saturate bend sensors at what factor times yield load */\
  XX(  exceed_lat_min_factor,   float,    0.75)     /* minimum factor of yield load to consider lateral force exceeded */\
  XX(  exceed_lat_max_factor,   float,    1.5)      /* maximum factor of yield load to consider lateral force exceeded (saturates beyond)*/\
  /*    
  HER settings */\
  XX(  use_HER,                 bool,     false)    /* use hindsight experience replay (HER) */\
  XX(  goal_reward,             float,    1.0)      /* reward for achieving a goal */\
  XX(  divide_goal_reward,      bool,     true)     /* if multiple goals, do we split goal reward betwen them */\
  XX(  reward_on_end_only,      bool,     true)     /* give goal reward only on episode end */\
  XX(  binary_goal_vector,      bool,     false)    /* do we use only binary goals */\
  /*
  get_observation() settings    (NB: sample modes: 0=raw, 1=change, 2=average, 3=median, 4=binary marker) */\
  XX(  sensor_sample_mode,      int,      2)        /* how to sample sensor observations, see MjType::Sample*/\
  XX(  state_sample_mode,       int,      4)        /* how to sample motor state, see MjType::Sample*/\
  XX(  sensor_n_prev_steps,     int,      3)        /* how many steps back do we sample with sensors */\
  XX(  state_n_prev_steps,      int,      3)        /* how many steps back do we sample with state sensors */\
  XX(  sensor_noise_mag,        double,   0.0)      /* noise magnitude if using uniform distribution (std <= 0) */\
  XX(  sensor_noise_mu,         double,   0.05)     /* abs range of sensor mean shift */\
  XX(  sensor_noise_std,        double,   0.025)    /* std deviation of noise, <= 0 means uniform */\
  XX(  state_noise_mag,         double,   0.0)      /* noise magnitude if using uniform distribution (std <= 0)*/\
  XX(  state_noise_mu,          double,   0.025)    /* abs range of state sensor mean shift*/\
  XX(  state_noise_std,         double,   0.0)      /* std deviation of noise, <= 0 means uniform*/\
  /* 
  update_env() settings */\
  XX(  oob_distance,            double,   75e-3)    /* distance to consider object out of bounds */\
  XX(  done_height,             double,   15e-3)    /* the object AND the gripper must go up by this height from starting positions */\
  XX(  stable_finger_force,     double,   1.0)      /* finger force (N) on object to consider stable */\
  XX(  stable_palm_force,       double,   1.0)      /* palm force (N) on object to consider stable */\
  XX(  stable_finger_force_lim, double,   100.0)    /* finger force (N) limit on the object to stop considering stable */\
  XX(  stable_palm_force_lim,   double,   100.0)    /* palm force (N) limit on the object to stop considering stable*/\
  /* 
  is_done() settings */\
  XX(  use_quit_on_reward,      bool,     true)     /* cap reward at quit_on_reward_below */\
  XX(  quit_on_reward_above,    float,    1.01)     /* done=true if reward rises above this value */\
  XX(  quit_on_reward_below,    float,    -1.01)    /* done=true if reward drops below this value */\
  /* 
  set_action() settings */\
  XX(  continous_actions,       bool,     false)    /* are actions continous or discrete */\
  XX(  use_termination_action,  bool,     true)    /* include an action for termination signalling to end grasp */\
  XX(  termination_threshold,   float,    0.9)      /* threshold for termination action to trigger (only relevant for continous actions) */\
  XX(  sim_steps_per_action,    int,      200)      /* number of sim steps performed to complete one action */\
  XX(  fingertip_min_mm,        double,   -12.5)    /* minimum allowable fingertip depth below start position before within_limits=false */\
  /*
  render() settings */\
  XX(  render_on_step,          bool,     false)    /* render on every single sim step */\
  XX(  use_render_delay,        bool,     false)    /* pause when rendering */\
  XX(  render_delay,            double,   0.5)      /* how long to pause when rendering */\
  

  
#define LUKE_MJSETTINGS_SENSOR \
  /* 

  2. Sensors
      name                      used      normalise read-rate (NB ignore read rate for state sensors) */\
  SS(  motor_state_sensor,      true,     0,        -1)  /* xyz motor states, normalise is ignored */\
  SS(  base_state_sensor_Z,     true,     0,        -1)  /* base position state, normalise is ignored)*/\
  SS(  base_state_sensor_XY,    true,     0,        -1)  /* base position state, normalise is ignored)*/\
  SS(  bending_gauge,           true,     20,       10)  /* strain gauge to measure finger bending */\
  SS(  axial_gauge,             false,    3.0,      10)  /* strain gauge to measure axial finger strain */\
  SS(  palm_sensor,             true,     10.0,     10)  /* palm force sensor */\
  SS(  wrist_sensor_XY,         false,    5.0,      10)  /* force wrist sensor X and Y forces */\
  SS(  wrist_sensor_Z,          true,     10.0,     10)  /* force wrist sensor Z force */\
  
  
  
#define LUKE_MJSETTINGS_ACTION \
  /* 

  3. Actions
      name                      used      value   sign */\
  AA(  gripper_X,               false,    1.0e-3,  -1)        /* move gripper X motor by m */\
  AA(  gripper_prismatic_X,     true,     1.0e-3,  -1)        /* move gripper X and Y motors to move prismatically by m */\
  AA(  gripper_Y,               false,    1.0e-3,  -1)        /* move gripper Y motor by m */\
  AA(  gripper_revolute_Y,      true,     0.01,    -1)        /* move gripper Y motor with angular motions/targets in radians */\
  AA(  gripper_Z,               true,     2.0e-3,   1)        /* move gripper Z motor by m */\
  AA(  base_X,                  true,     2.0e-3,   1)        /* move gripper base X by m */\
  AA(  base_Y,                  true,     2.0e-3,   1)        /* move gripper base Y by m */\
  AA(  base_Z,                  true,     2.0e-3,   1)        /* move gripper base Z by m */\
  AA(  base_roll,               false,    0.01,     1)        /* rotate gripper base about X in radians */\
  AA(  base_pitch,              false,    0.01,     1)        /* rotate gripper base about Y in radians */\
  AA(  base_yaw,                false,    0.01,     1)        /* rotate gripper base about Z in radians */\
  
  
  
#define LUKE_MJSETTINGS_BINARY_REWARD \
  /* 

  4. Binary rewards
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
  BR(  stable_termination,      1.0,      1,        1)      /* object stable and termination signal sent */\
  BR(  failed_termination,      -1.0,     1,        1)      /* termination signal sent but object not stable */\
  BR(  successful_grasp,        0.0,      1,        1)      /* metric to indicate a grasp is stable, shouldn't have associated reward */
  
  
#define LUKE_MJSETTINGS_LINEAR_REWARD \
  /*

  5. Linear rewards
      name                      reward    done   trigger  min   max   overshoot */\
  LR(  finger_force,            0.0,      false,    1,    1.0,  2.0,  -1)     /* avg. force from fingers */\
  LR(  palm_force,              0.05,     false,    1,    1.0,  3.0,  6.0)    /* palm force applied */\
  LR(  exceed_axial,            -0.05,    false,    1,    2.0,  6.0,  -1)     /* exceed axial finger force limit */\
  LR(  exceed_lateral,          -0.05,    false,    1,    4.0,  6.0,  -1)     /* exceed lateral finger force limit */\
  LR(  exceed_palm,             -0.05,    false,    1,    6.0,  10.0, -1)     /* exceed palm force limit */\
  /* new rewards based on direct sensor data */\
  LR(  good_bend_sensor,        0.0,      false,    1,    0.2,  1.0,  -1)     /* encourage bending force, direct sensor reward */\
  LR(  good_palm_sensor,        0.0,      false,    1,    0.2,  1.0,  -1)     /* encourage palm force, direct sensor reward */\
  LR(  exceed_bend_sensor,      0.0,      false,    1,    5.0,  10.0, -1)     /* exceed bending force, direct sensor limit */\
  LR(  exceed_wrist_sensor,     0.0,      false,    1,    5.0,  10.0, -1)     /* exceed wrist force, direct sensor limit */\
  LR(  exceed_palm_sensor,      0.0,      false,    1,    10.0, 20.0, -1)     /* exceed palm force, direct sensor limit */\
  LR(  dangerous_bend_sensor,   0.0,      true,     1,    10.0, 11.0, -1)     /* dangerous bending force, direct sensor limit */\
  LR(  dangerous_wrist_sensor,  0.0,      true,     1,    12.0, 13.0, -1)     /* dangerous wrist force, direct sensor limit */\
  LR(  dangerous_palm_sensor,   0.0,      true,     1,    20.0, 21.0, -1)     /* dangerous palm force, direct sensor limit */\
  /* testing extras for goals */\
  LR(  finger1_force,           0.0,      false,    1,    0.0,  2.0, 6.0)     /* finger 1 force */\
  LR(  finger2_force,           0.0,      false,    1,    0.0,  2.0, 6.0)     /* finger 2 force */\
  LR(  finger3_force,           0.0,      false,    1,    0.0,  2.0, 6.0)     /* finger 3 force */\
  LR(  ground_force,            0.0,      false,    1,    0.0,  2.0,  -1)     /* ground force on object */\
  LR(  grasp_metric,            0.0,      false,    1,    0.0,  10.0, -1)     /* grasping metric score */\

// end of user defined simulation settings

#endif