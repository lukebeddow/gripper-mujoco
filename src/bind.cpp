#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "mjclass.h"

constexpr bool debug_bind = false;

namespace py = pybind11;

// create a python module, called bind (must be saved in bind.so)
PYBIND11_MODULE(bind, m) {

  m.doc() = "A module to wrap mujoco into python"; // module docstring
  
  // module functions
  m.def("calc_rewards", &calc_rewards);
  m.def("goal_rewards", &goal_rewards);
  m.def("score_goal", static_cast<MjType::Goal (*)(MjType::Goal, std::vector<float>, MjType::Settings)>(&score_goal));
  m.def("score_goal", static_cast<MjType::Goal (*)(MjType::Goal, MjType::EventTrack, MjType::Settings)>(&score_goal));

  // main module class
  {py::class_<MjClass>(m, "MjClass")

    // constructors
    .def(py::init<>())
    .def(py::init<std::string>())
    .def(py::init<MjType::Settings>())

    // core functionality
    .def("load", &MjClass::load)
    .def("load_relative", &MjClass::load_relative)
    .def("reset", &MjClass::reset)
    .def("hard_reset", &MjClass::hard_reset)
    .def("step", &MjClass::step)
    .def("render", &MjClass::render)

    // sensing
    // none atm

    // control
    .def("set_joint_target", &MjClass::set_joint_target)
    .def("set_motor_target", &MjClass::set_motor_target)
    .def("set_step_target", &MjClass::set_step_target)
    .def("move_motor_target", &MjClass::move_motor_target)
    .def("move_joint_target", &MjClass::move_joint_target)
    .def("move_step_target", &MjClass::move_step_target)

    // learning functions
    .def("action_step", &MjClass::action_step)
    .def("set_action", &MjClass::set_action)
    .def("reset_object", &MjClass::reset_object)
    .def("spawn_object", static_cast<void (MjClass::*)(int)>(&MjClass::spawn_object)) /* see bottom */
    .def("spawn_object", static_cast<void (MjClass::*)(int, double, double, double)>(&MjClass::spawn_object))
    .def("randomise_object_colour", &MjClass::randomise_object_colour)
    .def("randomise_ground_colour", &MjClass::randomise_ground_colour)
    .def("randomise_finger_colours", &MjClass::randomise_finger_colours)
    .def("is_done", &MjClass::is_done)
    .def("get_observation", &MjClass::get_observation)
    .def("get_event_state", &MjClass::get_event_state)
    .def("get_goal", &MjClass::get_goal)
    .def("assess_goal", static_cast<std::vector<float> (MjClass::*)()>(&MjClass::assess_goal))
    .def("assess_goal", static_cast<std::vector<float> (MjClass::*)(std::vector<float>)>(&MjClass::assess_goal))
    .def("reward", static_cast<float (MjClass::*)()>(&MjClass::reward))
    .def("reward", static_cast<float (MjClass::*)(std::vector<float>, std::vector<float>)>(&MjClass::reward))
    .def("get_n_actions", &MjClass::get_n_actions)
    .def("get_n_obs", &MjClass::get_n_obs)
    .def("get_N", &MjClass::get_N)
    .def("set_finger_thickness", &MjClass::set_finger_thickness)
    .def("set_finger_width", &MjClass::set_finger_width)
    .def("get_finger_thickness", &MjClass::get_finger_thickness)
    .def("get_finger_stiffnesses", &MjClass::get_finger_stiffnesses)

    // sensor getters (set a default argument for 'unnormalise' to be false)
    .def("get_bend_gauge_readings", &MjClass::get_bend_gauge_readings, py::arg("unnormalise") = false)
    .def("get_palm_reading", &MjClass::get_palm_reading, py::arg("unnormalise") = false)
    .def("get_wrist_reading", &MjClass::get_wrist_reading, py::arg("unnormalise") = false)
    .def("get_state_readings", &MjClass::get_state_readings, py::arg("unnormalise") = false)
    .def("get_finger_angle", &MjClass::get_finger_angle)

    // real life gripper functions
    .def("get_finger_gauge_data", &MjClass::get_finger_gauge_data)
    .def("input_real_data", &MjClass::input_real_data)
    .def("get_real_observation", &MjClass::get_real_observation)

    // misc
    .def("tick", &MjClass::tick)
    .def("tock", &MjClass::tock)
    .def("forward", &MjClass::forward)
    .def("get_number_of_objects", &MjClass::get_number_of_objects)
    .def("get_current_object_name", &MjClass::get_current_object_name)
    .def("get_test_report", &MjClass::get_test_report)
    .def("add_events", &MjClass::add_events)
    .def("reset_goal", &MjClass::reset_goal)
    .def("print", &MjClass::print)
    .def("default_goal_event_triggering", &MjClass::default_goal_event_triggering)
    .def("validate_under_force", &MjClass::validate_curve_under_force)
    .def("curve_validation_regime", &MjClass::curve_validation_regime, py::arg("print") = false, py::arg("force_style") = 0)
    .def("last_action_gripper", &MjClass::last_action_gripper)
    .def("last_action_panda", &MjClass::last_action_panda)
    .def("get_fingertip_z_height", &MjClass::get_fingertip_z_height)
    .def("profile_error", &MjClass::profile_error)
    .def("numerical_stiffness_converge", static_cast<std::string (MjClass::*)(float, float)>(&MjClass::numerical_stiffness_converge))
    .def("numerical_stiffness_converge", static_cast<std::string (MjClass::*)(float, float, std::vector<float>, std::vector<float>)>(&MjClass::numerical_stiffness_converge))
    .def("numerical_stiffness_converge_2", &MjClass::numerical_stiffness_converge_2)
    .def("set_sensor_noise_and_normalisation_to", &MjClass::set_sensor_noise_and_normalisation_to)

    // exposed variables
    .def_readwrite("set", &MjClass::s_)
    .def_readwrite("goal", &MjClass::goal_)
    .def_readwrite("model_folder_path", &MjClass::model_folder_path)
    .def_readwrite("object_set_name", &MjClass::object_set_name)
    .def_readonly("machine", &MjClass::machine)
    .def_readonly("current_load_path", &MjClass::current_load_path)
    .def_readwrite("curve_validation_data", &MjClass::curve_validation_data_)

    // pickle support
    .def(py::pickle(
      [](const MjClass &mjobj) { // __getstate___
        /* return a tuple that fully encodes the state of the object */
        return py::make_tuple(
          mjobj.s_,                   // simulation settings struct
          mjobj.current_load_path,    // path of currently loaded model
          mjobj.model_folder_path,    // path to the mjcf models folder
          mjobj.object_set_name,      // name of object set in use
          mjobj.machine,              // machine library is compiled for
          mjobj.goal_                 // event goal, if using HER
        );
      },
      [](py::tuple t) { // __setstate__

        if (t.size() != 5 and t.size() != 6) // 5 is OLD version, delete later
          throw std::runtime_error("mjclass py::pickle got invalid state (tuple size wrong)");

        // create new c++ instance with old settings
        MjClass mjobj(t[0].cast<MjType::Settings>());

        // set the variables (must be same order as tuple above)
        if (t.size() >= 2) mjobj.current_load_path = t[1].cast<std::string>();
        // if (t.size() >= 3) mjobj.model_folder_path = t[2].cast<std::string>();
        if (t.size() >= 4) mjobj.object_set_name = t[3].cast<std::string>();
        // if (t.size() >= 5) mjobj.machine = t[4].cast<std::string>();
        if (t.size() >= 6) mjobj.goal_ = t[5].cast<MjType::Goal>();

        // disable automatic setting calibration as loading implies this is already done
        mjobj.resetFlags.flags_init = true;

        return mjobj;
      }
    ))
    ;
  }

  // internal simulation settings class which gets entirely pickled
  {py::class_<MjType::Settings>(m, "set")

    .def(py::init<>())
    .def("get_settings", &MjType::Settings::get_settings)
    .def("wipe_rewards", &MjType::Settings::wipe_rewards)
    .def("disable_sensors", &MjType::Settings::disable_sensors)
    .def("scale_rewards", &MjType::Settings::scale_rewards)
    .def("set_use_normalisation", &MjType::Settings::set_use_normalisation)
    .def("set_sensor_prev_steps_to", &MjType::Settings::set_sensor_prev_steps_to)

    // use a macro to create code snippets for all of the settings
    #define XX(name, type, value) .def_readwrite(#name, &MjType::Settings::name)
    #define SS(name, in_use, norm, readrate) .def_readwrite(#name, &MjType::Settings::name)
    #define BR(name, reward, done, trigger) .def_readwrite(#name, &MjType::Settings::name)
    #define LR(name, reward, done, trigger, min, max, overshoot) \
              .def_readwrite(#name, &MjType::Settings::name)
      // run the macro to create the code
      LUKE_MJSETTINGS
    #undef XX
    #undef SS
    #undef BR
    #undef LR

    // example snippets produced by the above macro
    // .def_readwrite("step_num", &MjType::Settings::step_num)
    // .def_readwrite("lifted", &MjType::Settings::lifted)
    // .def_readwrite("gauge_read_rate_hz", &MjType::Settings::gauge_read_rate_hz)

    // pickle support
    .def(py::pickle(
      [](const MjType::Settings s) { // __getstate___
        /* return a tuple that fully encodes the state of the object */
        return py::make_tuple(

          // expand settings into list of variable names for tuple
          #define XX(name, type, value) s.name,
          #define SS(name, in_use, norm, readrate) s.name,
          #define BR(name, reward, done, trigger) s.name,
          #define LR(name, reward, done, trigger, min, max, overshoot) s.name,
            // run the macro to create the code
            LUKE_MJSETTINGS
          #undef XX
          #undef SS
          #undef BR
          #undef LR

          // example snippets produced by the above macro
          // s.step_num,
          // s.lifted,
          // s.gauge_read_rate_hz,

          // include dummy last, all above snippets have trailing commas
          s.dummy
        );
      },
      [](py::tuple t) { // __setstate__
        
        if (debug_bind)
          std::cout << "unpickling MjType::Settings now\n";

        // create new c++ instance
        MjType::Settings out;

        // fill in with the old data
        int i = 0;

        // expand the tuple elements and type cast them with a macro
        #define XX(name, type, value) out.name = t[i].cast<type>(); ++i;
        #define SS(name, in_use, norm, readrate) \
                  out.name = t[i].cast<MjType::Sensor>(); ++i;
        #define BR(name, reward, done, trigger) \
                  out.name = t[i].cast<MjType::BinaryReward>(); ++i;
        #define LR(name, reward, done, trigger, min, max, overshoot) \
                  out.name = t[i].cast<MjType::LinearReward>(); ++i;
          // run the macro to create the code
          LUKE_MJSETTINGS
        #undef XX
        #undef SS
        #undef BR
        #undef LR

        // example snippet using dummy
        out.dummy = t[i].cast<bool>(); ++i;

        if (debug_bind)
          std::cout << "unpickling MjType::Settings finished, i is " << i
            << ", size of tuple is " << t.size() << '\n';

        return out;
      }
    ))
    ;
  }

  {py::class_<MjType::EventTrack::BinaryEvent>(m, "BinaryEvent")

    .def_readonly("value", &MjType::EventTrack::BinaryEvent::value)
    .def_readonly("last_value", &MjType::EventTrack::BinaryEvent::last_value)
    .def_readonly("row", &MjType::EventTrack::BinaryEvent::row)
    .def_readonly("abs", &MjType::EventTrack::BinaryEvent::abs)
    .def_readonly("percent", &MjType::EventTrack::BinaryEvent::percent)

    // pickle support
    .def(py::pickle(
      [](const MjType::EventTrack::BinaryEvent s) { // __getstate___
        /* return a tuple that fully encodes the state of the object */
        return py::make_tuple(
          s.value,
          s.last_value,
          s.row,
          s.abs,
          s.percent
        );
      },
      [](py::tuple t) { // __setstate__
        
        if (debug_bind)
          std::cout << "unpickling MjType::EventTrack::BinaryEvent now\n";

        // create new c++ instance
        MjType::EventTrack::BinaryEvent out;
        out.value = t[0].cast<bool>();
        out.last_value = t[1].cast<int>();
        out.row = t[2].cast<int>();
        out.abs = t[3].cast<int>();
        out.percent = t[4].cast<float>();

        if (debug_bind)
          std::cout << "unpickling MjType::EventTrack::BinaryEvent finished\n";

        return out;
      }
    ))
    ;
  }

  {py::class_<MjType::EventTrack::LinearEvent>(m, "LinearEvent")

    .def_readonly("value", &MjType::EventTrack::LinearEvent::value)
    .def_readonly("last_value", &MjType::EventTrack::LinearEvent::last_value)
    .def_readonly("row", &MjType::EventTrack::LinearEvent::row)
    .def_readonly("abs", &MjType::EventTrack::LinearEvent::abs)
    .def_readonly("percent", &MjType::EventTrack::LinearEvent::percent)

    // pickle support
    .def(py::pickle(
      [](const MjType::EventTrack::LinearEvent s) { // __getstate___
        /* return a tuple that fully encodes the state of the object */
        return py::make_tuple(
          s.value,
          s.last_value,
          s.row,
          s.abs,
          s.percent
        );
      },
      [](py::tuple t) { // __setstate__
        
        if (debug_bind)
          std::cout << "unpickling MjType::EventTrack::LinearEvent now\n";

        // create new c++ instance
        MjType::EventTrack::LinearEvent out;
        out.value = t[0].cast<bool>();
        out.last_value = t[1].cast<float>();
        out.row = t[2].cast<int>();
        out.abs = t[3].cast<int>();
        out.percent = t[4].cast<float>();

        if (debug_bind)
          std::cout << "unpickling MjType::EventTrack::LinearEvent finished\n";

        return out;
      }
    ))
    ;
  }

  // tracking of important events in the simulation
  {py::class_<MjType::EventTrack>(m, "EventTrack")

    .def(py::init<>())
    .def("print", &MjType::EventTrack::print)
    .def("reset", &MjType::EventTrack::reset)
    .def("calculate_percentage", &MjType::EventTrack::calculate_percentage)

    #define XX(name, type, value)
    #define SS(name, in_use, norm, readrate)
    #define BR(name, reward, done, trigger) \
              .def_readonly(#name, &MjType::EventTrack::name)

    #define LR(name, reward, done, trigger, min, max, overshoot) \
              .def_readonly(#name, &MjType::EventTrack::name)

      // run the macro to create the binding code
      LUKE_MJSETTINGS

    #undef XX
    #undef SS
    #undef BR
    #undef LR

    // pickle support
    .def(py::pickle(
      [](const MjType::EventTrack &et) { // __getstate___
        /* return a tuple that fully encodes the state of the object */
        return py::make_tuple(
          
          #define XX(name, type, value)
          #define SS(name, in_use, norm, readrate)
          #define BR(name, reward, done, trigger) et.name,
          #define LR(name, reward, done, trigger, min, max, overshoot) et.name,
            // run the macro to create the binding code
            LUKE_MJSETTINGS
          #undef XX
          #undef SS
          #undef BR
          #undef LR

          // dummy value as macro above always uses trailing comma
          0.0
        );
      },
      [](py::tuple t) { // __setstate__

        // create new c++ instance with old settings
        MjType::EventTrack et;

        int i = 0;

        // expand the tuple elements and type cast them with a macro
        #define XX(name, type, value)
        #define SS(name, in_use, norm, readrate)
        #define BR(name, reward, done, trigger) \
                  et.name = t[i].cast<MjType::EventTrack::BinaryEvent>(); ++i;
        #define LR(name, reward, done, trigger, min, max, overshoot) \
                  et.name = t[i].cast<MjType::EventTrack::LinearEvent>(); ++i;
          // run the macro to create the code
          LUKE_MJSETTINGS
        #undef XX
        #undef SS
        #undef BR
        #undef LR

        return et;
      }
    ))
    ;
  }

  // class for outputing test results and event tracking
  {py::class_<MjType::TestReport>(m, "TestReport")

    .def(py::init<>())
    .def_readonly("object_name", &MjType::TestReport::object_name)
    .def_readonly("cumulative_reward", &MjType::TestReport::cumulative_reward)
    .def_readonly("cnt", &MjType::TestReport::cnt)
    ;
  }

  // set up sensor type so python can interact and change them
  {py::class_<MjType::Sensor>(m, "Sensor")

    .def(py::init<bool, float, float>())
    .def("set", &MjType::Sensor::set)

    .def_readwrite("in_use", &MjType::Sensor::in_use)
    .def_readwrite("normalise", &MjType::Sensor::normalise)
    .def_readwrite("read_rate", &MjType::Sensor::read_rate)
    .def_readwrite("prev_steps", &MjType::Sensor::prev_steps)
    .def_readwrite("use_normalisation", &MjType::Sensor::use_normalisation)
    .def_readwrite("use_noise", &MjType::Sensor::use_noise)
    .def_readwrite("raw_value_offset", &MjType::Sensor::raw_value_offset)
    .def_readwrite("noise_mag", &MjType::Sensor::noise_mag)
    .def_readwrite("noise_mu", &MjType::Sensor::noise_mu)
    .def_readwrite("noise_std", &MjType::Sensor::noise_std)

    // pickle support
    .def(py::pickle(
      [](const MjType::Sensor r) { // __getstate___
        /* return a tuple that fully encodes the state of the object */
        return py::make_tuple(
          r.in_use, 
          r.normalise, 
          r.read_rate,
          r.use_normalisation,
          r.use_noise,
          r.raw_value_offset,
          r.noise_mag,
          r.noise_mu,
          r.noise_std
        );
      },
      [](py::tuple t) { // __setstate__

        // size == 3 is old and can be later deleted
        if (t.size() != 3 and t.size() != 9)
          throw std::runtime_error("MjType::Sensor py::pickle got invalid state");

        // create new c++ instance with old data
        MjType::Sensor out(t[0].cast<bool>(), t[1].cast<float>(), t[2].cast<float>());

        if (t.size() == 9) {
          out.use_normalisation = t[3].cast<bool>();
          out.use_noise = t[4].cast<bool>();
          out.raw_value_offset = t[5].cast<float>();
          out.noise_mag = t[6].cast<float>();
          out.noise_mu = t[7].cast<float>();
          out.noise_std = t[8].cast<float>();
        }
        
        return out;
      }
    ))
    ;
  }

  // set up binary reward type so python can interact and change them
  {py::class_<MjType::BinaryReward>(m, "BinaryReward")

    .def(py::init<float, bool, int>())
    .def("set", &MjType::BinaryReward::set)

    .def_readwrite("reward", &MjType::BinaryReward::reward)
    .def_readwrite("done", &MjType::BinaryReward::done)
    .def_readwrite("trigger", &MjType::BinaryReward::trigger)

    // pickle support
    .def(py::pickle(
      [](const MjType::BinaryReward r) { // __getstate___
        /* return a tuple that fully encodes the state of the object */
        return py::make_tuple(r.reward, r.done, r.trigger);
      },
      [](py::tuple t) { // __setstate__

        if (t.size() != 3)
          throw std::runtime_error("MjType::BinaryReward py::pickle got invalid state");

        // create new c++ instance with old data
        MjType::BinaryReward out(t[0].cast<float>(), t[1].cast<int>(), t[2].cast<int>());

        return out;
      }
    ))
    ;
  }

  // set up linear reward type so python can edit them
  {py::class_<MjType::LinearReward>(m, "LinearReward")

    .def(py::init<float, bool, int, float, float, float>())
    .def("set", &MjType::LinearReward::set)

    .def_readwrite("reward", &MjType::LinearReward::reward)
    .def_readwrite("done", &MjType::LinearReward::done)
    .def_readwrite("trigger", &MjType::LinearReward::trigger)
    .def_readwrite("min", &MjType::LinearReward::min)
    .def_readwrite("max", &MjType::LinearReward::max)
    .def_readwrite("overshoot", &MjType::LinearReward::overshoot)

    // pickle support
    .def(py::pickle(
      [](const MjType::LinearReward r) { // __getstate___
        /* return a tuple that fully encodes the state of the object */
        return py::make_tuple(r.reward, r.done, r.trigger, r.min, r.max, r.overshoot);
      },
      [](py::tuple t) { // __setstate__

        if (t.size() != 6)
          throw std::runtime_error("MjType::LinearReward py::pickle got invalid state");

        // create new c++ instance with old data
        MjType::LinearReward out(t[0].cast<float>(), t[1].cast<int>(), t[2].cast<int>(),
          t[3].cast<float>(), t[4].cast<float>(), t[5].cast<float>());

        return out;
      }
    ))
    ;
  }

  // setting up goals events
  {py::class_<MjType::Goal::Event>(m, "Event")
    .def(py::init<>())
    .def_readwrite("involved", &MjType::Goal::Event::involved)
    .def_readwrite("state", &MjType::Goal::Event::state)

    // pickle support
    .def(py::pickle(
      [](const MjType::Goal::Event e) { // __getstate___
        /* return a tuple that fully encodes the state of the object */
        return py::make_tuple(e.involved, e.state);
      },
      [](py::tuple t) { // __setstate__

        if (t.size() != 2)
          throw std::runtime_error("MjType::Goal::Event py::pickle got invalid state");

        // create new c++ instance with old data
        MjType::Goal::Event out;

        out.involved = t[0].cast<bool>();
        out.state = t[1].cast<bool>();

        return out;
      }
    ))

    ;
  }

  {py::class_<MjType::Goal>(m, "Goal")
    .def(py::init<>())
    .def("print", &MjType::Goal::print)
    .def("print_verbose", &MjType::Goal::print_verbose)
    .def("get_goal_info", &MjType::Goal::get_goal_info)

    #define XX(name, type, value)
    #define SS(name, in_use, norm, readrate)
    #define BR(name, reward, done, trigger) \
              .def_readwrite(#name, &MjType::Goal::name)

    #define LR(name, reward, done, trigger, min, max, overshoot) \
              .def_readwrite(#name, &MjType::Goal::name)

      // run the macro to create the binding code
      LUKE_MJSETTINGS

    #undef XX
    #undef SS
    #undef BR
    #undef LR

    // pickle support
    .def(py::pickle(
      [](const MjType::Goal &g) { // __getstate___
        /* return a tuple that fully encodes the state of the object */
        return py::make_tuple(
          
          #define XX(name, type, value)
          #define SS(name, in_use, norm, readrate)
          #define BR(name, reward, done, trigger) g.name,
          #define LR(name, reward, done, trigger, min, max, overshoot) g.name,
            // run the macro to create the binding code
            LUKE_MJSETTINGS
          #undef XX
          #undef SS
          #undef BR
          #undef LR

          // dummy value as macro above always uses trailing comma
          0.0
        );
      },
      [](py::tuple t) { // __setstate__

        // create new c++ instance with old settings
        MjType::Goal g;

        int i = 0;

        // expand the tuple elements and type cast them with a macro
        #define XX(name, type, value)
        #define SS(name, in_use, norm, readrate)
        #define BR(name, reward, done, trigger) \
                  g.name = t[i].cast<MjType::Goal::Event>(); ++i;
        #define LR(name, reward, done, trigger, min, max, overshoot) \
                  g.name = t[i].cast<MjType::Goal::Event>(); ++i;
          // run the macro to create the code
          LUKE_MJSETTINGS
        #undef XX
        #undef SS
        #undef BR
        #undef LR

        return g;
      }
    ))

    ;
  }

  // four classes to extract detailed curve fit data from the simulation
  {py::class_<MjType::CurveFitData::PoseData::FingerData::Error>(m, "Error")
    .def(py::init<>())
    .def_readonly("x_wrt_pred_x", &MjType::CurveFitData::PoseData::FingerData::Error::x_wrt_pred_x)
    .def_readonly("x_wrt_pred_x_percent", &MjType::CurveFitData::PoseData::FingerData::Error::x_wrt_pred_x_percent)
    .def_readonly("x_wrt_pred_x_tipratio", &MjType::CurveFitData::PoseData::FingerData::Error::x_wrt_pred_x_tipratio)
    .def_readonly("y_wrt_pred_y", &MjType::CurveFitData::PoseData::FingerData::Error::y_wrt_pred_y)
    .def_readonly("y_wrt_pred_y_percent", &MjType::CurveFitData::PoseData::FingerData::Error::y_wrt_pred_y_percent)
    .def_readonly("y_wrt_pred_y_tipratio", &MjType::CurveFitData::PoseData::FingerData::Error::y_wrt_pred_y_tipratio)
    .def_readonly("y_wrt_theory_y", &MjType::CurveFitData::PoseData::FingerData::Error::y_wrt_theory_y)
    .def_readonly("y_wrt_theory_y_percent", &MjType::CurveFitData::PoseData::FingerData::Error::y_wrt_theory_y_percent)
    .def_readonly("y_wrt_theory_y_tipratio", &MjType::CurveFitData::PoseData::FingerData::Error::y_wrt_theory_y_tipratio)
    .def_readonly("y_pred_wrt_theory_y", &MjType::CurveFitData::PoseData::FingerData::Error::y_pred_wrt_theory_y)
    .def_readonly("y_pred_wrt_theory_y_percent", &MjType::CurveFitData::PoseData::FingerData::Error::y_pred_wrt_theory_y_percent)
    .def_readonly("y_pred_wrt_theory_y_tipratio", &MjType::CurveFitData::PoseData::FingerData::Error::y_pred_wrt_theory_y_tipratio)
    .def_readonly("j_wrt_pred_j", &MjType::CurveFitData::PoseData::FingerData::Error::j_wrt_pred_j)
    .def_readonly("j_wrt_pred_j_percent", &MjType::CurveFitData::PoseData::FingerData::Error::j_wrt_pred_j_percent)
    .def_readonly("x_tip_wrt_pred_x", &MjType::CurveFitData::PoseData::FingerData::Error::x_tip_wrt_pred_x)
    .def_readonly("y_tip_wrt_pred_y", &MjType::CurveFitData::PoseData::FingerData::Error::y_tip_wrt_pred_y)
    .def_readonly("y_tip_wrt_theory_y", &MjType::CurveFitData::PoseData::FingerData::Error::y_tip_wrt_theory_y)
    .def_readonly("y_pred_tip_wrt_theory_y", &MjType::CurveFitData::PoseData::FingerData::Error::y_pred_tip_wrt_theory_y)
    .def_readonly("x_tip_wrt_pred_x_percent", &MjType::CurveFitData::PoseData::FingerData::Error::x_tip_wrt_pred_x_percent)
    .def_readonly("y_tip_wrt_pred_y_percent", &MjType::CurveFitData::PoseData::FingerData::Error::y_tip_wrt_pred_y_percent)
    .def_readonly("y_tip_wrt_theory_y_percent", &MjType::CurveFitData::PoseData::FingerData::Error::y_tip_wrt_theory_y_percent)
    .def_readonly("y_pred_tip_wrt_theory_y_percent", &MjType::CurveFitData::PoseData::FingerData::Error::y_pred_tip_wrt_theory_y_percent)
    .def_readonly("std_x_wrt_pred_x", &MjType::CurveFitData::PoseData::FingerData::Error::std_x_wrt_pred_x)
    .def_readonly("std_y_wrt_pred_y", &MjType::CurveFitData::PoseData::FingerData::Error::std_y_wrt_pred_y)
    .def_readonly("std_y_wrt_theory_y", &MjType::CurveFitData::PoseData::FingerData::Error::std_y_wrt_theory_y)
    .def_readonly("std_y_pred_wrt_theory_y", &MjType::CurveFitData::PoseData::FingerData::Error::std_y_pred_wrt_theory_y)
    .def_readonly("std_j_wrt_pred_j", &MjType::CurveFitData::PoseData::FingerData::Error::std_j_wrt_pred_j)

    // pickle support
    .def(py::pickle(
      [](const MjType::CurveFitData::PoseData::FingerData::Error e) { // __getstate___
        /* return a tuple that fully encodes the state of the object */

        return py::make_tuple(
          e.x_wrt_pred_x,
          e.x_wrt_pred_x_percent,
          e.x_wrt_pred_x_tipratio,
          e.y_wrt_pred_y,
          e.y_wrt_pred_y_percent,
          e.y_wrt_pred_y_tipratio,
          e.y_wrt_theory_y,
          e.y_wrt_theory_y_percent,
          e.y_wrt_theory_y_tipratio,
          e.y_pred_wrt_theory_y,
          e.y_pred_wrt_theory_y_percent,
          e.y_pred_wrt_theory_y_tipratio,
          e.j_wrt_pred_j,
          e.j_wrt_pred_j_percent,
          e.x_tip_wrt_pred_x,
          e.y_tip_wrt_pred_y,
          e.y_tip_wrt_theory_y,
          e.y_pred_tip_wrt_theory_y,
          e.x_tip_wrt_pred_x_percent,
          e.y_tip_wrt_pred_y_percent,
          e.y_tip_wrt_theory_y_percent,
          e.y_pred_tip_wrt_theory_y_percent,
          e.std_x_wrt_pred_x,
          e.std_y_wrt_pred_y,
          e.std_y_wrt_theory_y,
          e.std_y_pred_wrt_theory_y,
          e.std_j_wrt_pred_j
        );
      },
      [](py::tuple t) { // __setstate__
        
        if (debug_bind)
          std::cout << "unpickling MjType::CurveFitData::PoseData::FingerData::Error now\n";

        // create new c++ instance
        MjType::CurveFitData::PoseData::FingerData::Error out;

        // fill in with the old data
        int i = 0;

        out.x_wrt_pred_x = t[i].cast<float>(); i++;
        out.x_wrt_pred_x_percent = t[i].cast<float>(); i++;
        out.x_wrt_pred_x_tipratio = t[i].cast<float>(); i++;
        out.y_wrt_pred_y = t[i].cast<float>(); i++;
        out.y_wrt_pred_y_percent = t[i].cast<float>(); i++;
        out.y_wrt_pred_y_tipratio = t[i].cast<float>(); i++;
        out.y_wrt_theory_y = t[i].cast<float>(); i++;
        out.y_wrt_theory_y_percent = t[i].cast<float>(); i++;
        out.y_wrt_theory_y_tipratio = t[i].cast<float>(); i++;
        out.y_pred_wrt_theory_y = t[i].cast<float>(); i++;
        out.y_pred_wrt_theory_y_percent = t[i].cast<float>(); i++;
        out.y_pred_wrt_theory_y_tipratio = t[i].cast<float>(); i++;
        out.j_wrt_pred_j = t[i].cast<float>(); i++;
        out.j_wrt_pred_j_percent = t[i].cast<float>(); i++;
        out.x_tip_wrt_pred_x = t[i].cast<float>(); i++;
        out.y_tip_wrt_pred_y = t[i].cast<float>(); i++;
        out.y_tip_wrt_theory_y = t[i].cast<float>(); i++;
        out.y_pred_tip_wrt_theory_y = t[i].cast<float>(); i++;
        out.x_tip_wrt_pred_x_percent = t[i].cast<float>(); i++;
        out.y_tip_wrt_pred_y_percent = t[i].cast<float>(); i++;
        out.y_tip_wrt_theory_y_percent = t[i].cast<float>(); i++;
        out.y_pred_tip_wrt_theory_y_percent = t[i].cast<float>(); i++;
        out.std_x_wrt_pred_x = t[i].cast<float>(); i++;
        out.std_y_wrt_pred_y = t[i].cast<float>(); i++;
        out.std_y_wrt_theory_y = t[i].cast<float>(); i++;
        out.std_y_pred_wrt_theory_y = t[i].cast<float>(); i++;
        out.std_j_wrt_pred_j = t[i].cast<float>(); i++;

        if (debug_bind)
          std::cout << "unpickling MjType::CurveFitData::PoseData::FingerData::Error finished, i is " << i
            << ", size of tuple is " << t.size() << '\n';

        return out;
      }
    ))
    ;
  }

  {py::class_<MjType::CurveFitData::PoseData::FingerData>(m, "FingerData")
    .def(py::init<>())
    .def_readonly("x", &MjType::CurveFitData::PoseData::FingerData::x)
    .def_readonly("y", &MjType::CurveFitData::PoseData::FingerData::y)
    .def_readonly("coeff", &MjType::CurveFitData::PoseData::FingerData::coeff)
    .def_readonly("errors", &MjType::CurveFitData::PoseData::FingerData::errors)
    .def_readonly("joints", &MjType::CurveFitData::PoseData::FingerData::joints)
    .def_readonly("pred_j", &MjType::CurveFitData::PoseData::FingerData::pred_j)
    .def_readonly("pred_x", &MjType::CurveFitData::PoseData::FingerData::pred_x)
    .def_readonly("pred_y", &MjType::CurveFitData::PoseData::FingerData::pred_y)
    .def_readonly("theory_y", &MjType::CurveFitData::PoseData::FingerData::theory_y)
    .def_readonly("theory_x_curve", &MjType::CurveFitData::PoseData::FingerData::theory_x_curve)
    .def_readonly("theory_y_curve", &MjType::CurveFitData::PoseData::FingerData::theory_y_curve)
    .def_readonly("error", &MjType::CurveFitData::PoseData::FingerData::error)

    // pickle support
    .def(py::pickle(
      [](const MjType::CurveFitData::PoseData::FingerData f) { // __getstate___
        /* return a tuple that fully encodes the state of the object */

        return py::make_tuple(
          py::tuple(py::cast(f.x)),
          py::tuple(py::cast(f.y)),
          py::tuple(py::cast(f.coeff)),
          py::tuple(py::cast(f.errors)),
          py::tuple(py::cast(f.joints)),
          py::tuple(py::cast(f.pred_j)),
          py::tuple(py::cast(f.pred_x)),
          py::tuple(py::cast(f.pred_y)),
          py::tuple(py::cast(f.theory_y)),
          py::tuple(py::cast(f.theory_x_curve)),
          py::tuple(py::cast(f.theory_y_curve)),
          f.error
        );
      },
      [](py::tuple t) { // __setstate__
        
        if (debug_bind)
          std::cout << "unpickling MjType::CurveFitData::PoseData::FingerData now\n";

        // create new c++ instance
        MjType::CurveFitData::PoseData::FingerData out;

        // fill in with the old data
        int i = 0;

        out.x = t[i].cast<std::vector<float>>(); i++;
        out.y = t[i].cast<std::vector<float>>(); i++;
        out.coeff = t[i].cast<std::vector<float>>(); i++;
        out.errors = t[i].cast<std::vector<float>>(); i++;
        out.joints = t[i].cast<std::vector<float>>(); i++;
        out.pred_j = t[i].cast<std::vector<float>>(); i++;
        out.pred_x = t[i].cast<std::vector<float>>(); i++;
        out.pred_y = t[i].cast<std::vector<float>>(); i++;
        out.theory_y = t[i].cast<std::vector<float>>(); i++;
        out.theory_x_curve = t[i].cast<std::vector<float>>(); i++;
        out.theory_y_curve = t[i].cast<std::vector<float>>(); i++;
        out.error = t[i].cast<MjType::CurveFitData::PoseData::FingerData::Error>(); i++;

        if (debug_bind)
          std::cout << "unpickling MjType::CurveFitData::PoseData::FingerData finished, i is " << i
            << ", size of tuple is " << t.size() << '\n';

        return out;
      }
    ))
    ;
  }

  {py::class_<MjType::CurveFitData::PoseData>(m, "PoseData")
    .def(py::init<>())
    .def("print", &MjType::CurveFitData::PoseData::print)
    .def_readonly("f1", &MjType::CurveFitData::PoseData::f1)
    .def_readonly("f2", &MjType::CurveFitData::PoseData::f2)
    .def_readonly("f3", &MjType::CurveFitData::PoseData::f3)
    .def_readonly("avg_error", &MjType::CurveFitData::PoseData::avg_error)
    .def_readonly("tag_string", &MjType::CurveFitData::PoseData::tag_string)

    // pickle support
    .def(py::pickle(
      [](const MjType::CurveFitData::PoseData p) { // __getstate___
        /* return a tuple that fully encodes the state of the object */

        return py::make_tuple(
          p.f1, p.f2, p.f3, p.avg_error, p.tag_string
        );
      },
      [](py::tuple t) { // __setstate__
        
        if (debug_bind)
          std::cout << "unpickling MjType::CurveFitData::PoseData now\n";

        // create new c++ instance
        MjType::CurveFitData::PoseData out;

        // fill in with the old data
        int i = 0;

        out.f1 = t[i].cast<MjType::CurveFitData::PoseData::FingerData>(); i++;
        out.f2 = t[i].cast<MjType::CurveFitData::PoseData::FingerData>(); i++;
        out.f3 = t[i].cast<MjType::CurveFitData::PoseData::FingerData>(); i++;
        out.avg_error = t[i].cast<MjType::CurveFitData::PoseData::FingerData::Error>(); i++;
        out.tag_string = t[i].cast<std::string>(); i++;

        if (debug_bind)
          std::cout << "unpickling MjType::CurveFitData::PoseData finished, i is " << i
            << ", size of tuple is " << t.size() << '\n';

        return out;
      }
    ))
    ;
  }

  {py::class_<MjType::CurveFitData>(m, "CurveFitData")
    .def(py::init<>())
    .def_readonly("entries", &MjType::CurveFitData::entries)
    .def("update", &MjType::CurveFitData::update)
    .def("print", &MjType::CurveFitData::print)

    // pickle support
    .def(py::pickle(
      [](const MjType::CurveFitData c) { // __getstate___
        /* return a tuple that fully encodes the state of the object */

        return py::make_tuple(
          py::tuple(py::cast(c.entries))
        );
      },
      [](py::tuple t) { // __setstate__
        
        if (debug_bind)
          std::cout << "unpickling MjType::CurveFitData: now\n";

        // create new c++ instance
        MjType::CurveFitData out;

        // fill in with the old data
        int i = 0;

        // auto entries_vec = t[i].cast<std::vector<MjType::CurveFitData::PoseData>>();
        // out.entries.push_back(entries_vec[0]);

        out.entries = t[i].cast<std::vector<MjType::CurveFitData::PoseData>>(); i++;

        if (debug_bind)
          std::cout << "unpickling MjType::CurveFitData finished, i is " << i
            << ", size of tuple is " << t.size() << '\n';

        return out;
      }
    ))
    ;
  }

  // classes to set gauge calibration
  {py::class_<MjType::RealGaugeCalibrations::RealSensors>(m, "RealSensors")
    .def(py::init<>())
    .def_readwrite("g1", &MjType::RealGaugeCalibrations::RealSensors::g1)
    .def_readwrite("g2", &MjType::RealGaugeCalibrations::RealSensors::g2)
    .def_readwrite("g3", &MjType::RealGaugeCalibrations::RealSensors::g3)
    .def_readwrite("palm", &MjType::RealGaugeCalibrations::RealSensors::palm)
    ;
  }

  {py::class_<MjType::RealGaugeCalibrations>(m, "RealGaugeCalibrations")
    .def_readwrite("offset", &MjType::RealGaugeCalibrations::offset)
    .def_readwrite("scale", &MjType::RealGaugeCalibrations::scale)
    .def_readwrite("norm", &MjType::RealGaugeCalibrations::norm)
    ;
  }

  /* py::overload_cast requires c++14 */

};