#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include "mjclass.h"

constexpr bool debug_bind = false;

namespace py = pybind11;

// numpy conversion helpers
static py::array_t<luke::rgbint> rgbvec_to_array(std::vector<luke::rgbint>& vec)
{
  return py::array(vec.size(), vec.data());
}

static py::array_t<float> depthvec_to_array(std::vector<float>& vec)
{
  return py::array(vec.size(), vec.data());
}

static py::tuple RGBD_struct_to_tuple(luke::RGBD rgb_struct)
{
  return py::make_tuple(rgbvec_to_array(rgb_struct.rgb), depthvec_to_array(rgb_struct.depth));
}

static py::array_t<luke::gfloat> floatvec_to_array(std::vector<luke::gfloat>& vec)
{
  return py::array(vec.size(), vec.data());
}

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

    // rendering
    .def("rendering_enabled", &MjClass::rendering_enabled)
    .def("render", &MjClass::render)
    .def("close_render", &MjClass::close_render)
    .def("init_rgbd", &MjClass::init_rgbd)
    .def("render_RGBD", &MjClass::render_RGBD)
    .def("read_existing_RGBD", &MjClass::read_existing_RGBD)
    .def("set_RGBD_size", &MjClass::set_RGBD_size)
    .def("get_RGBD", &MjClass::get_RGBD)
    .def("get_RGBD_numpy",
      [](MjClass &mj) {
        return RGBD_struct_to_tuple(mj.get_RGBD());
      })

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
    .def("set_continous_action", &MjClass::set_continous_action)
    .def("set_discrete_action", &MjClass::set_discrete_action)
    .def("reset_object", &MjClass::reset_object)
    .def("spawn_object", static_cast<void (MjClass::*)(int)>(&MjClass::spawn_object)) /* see bottom */
    .def("spawn_object", static_cast<void (MjClass::*)(int, double, double, double)>(&MjClass::spawn_object))
    .def("spawn_into_scene", static_cast<bool (MjClass::*)(int)>(&MjClass::spawn_into_scene))
    .def("spawn_into_scene", static_cast<bool (MjClass::*)(int, double, double)>(&MjClass::spawn_into_scene))
    .def("spawn_into_scene", static_cast<bool (MjClass::*)(int, double, double, double)>(&MjClass::spawn_into_scene))
    .def("spawn_into_scene", static_cast<bool (MjClass::*)(int, double, double, double, double, double, double)>(&MjClass::spawn_into_scene))
    .def("spawn_scene", &MjClass::spawn_scene)
    .def("randomise_every_colour", &MjClass::randomise_every_colour)
    .def("randomise_object_colour", &MjClass::randomise_object_colour)
    .def("randomise_ground_colour", &MjClass::randomise_ground_colour)
    .def("randomise_finger_colours", &MjClass::randomise_finger_colours)
    .def("is_done", &MjClass::is_done)
    .def("get_observation", static_cast<std::vector<luke::gfloat> (MjClass::*)()>(&MjClass::get_observation))
    .def("get_observation", static_cast<std::vector<luke::gfloat> (MjClass::*)(MjType::SensorData)>(&MjClass::get_observation))
    .def("debug_observation", &MjClass::debug_observation, py::arg("observation") = std::vector<luke::gfloat> {}, py::arg("printout") = false)
    .def("get_observation_numpy",
      [](MjClass &mj) {
        std::vector<luke::gfloat> obs = mj.get_observation();
        return floatvec_to_array(obs);
      })
    .def("get_event_state", &MjClass::get_event_state)
    .def("get_goal", &MjClass::get_goal)
    .def("assess_goal", static_cast<std::vector<float> (MjClass::*)()>(&MjClass::assess_goal))
    .def("assess_goal", static_cast<std::vector<float> (MjClass::*)(std::vector<float>)>(&MjClass::assess_goal))
    .def("reward", static_cast<float (MjClass::*)()>(&MjClass::reward))
    .def("reward", static_cast<float (MjClass::*)(std::vector<float>, std::vector<float>)>(&MjClass::reward))

    // sensor getters (set a default argument for 'unnormalise' to be false)
    .def("get_finger_forces", &MjClass::get_finger_forces)
    .def("get_palm_force", &MjClass::get_palm_force)
    .def("get_wrist_force", &MjClass::get_wrist_force)
    .def("get_state_metres", &MjClass::get_state_metres)
    .def("get_finger_angle", &MjClass::get_finger_angle)

    // real life gripper functions
    .def("calibrate_real_sensors", &MjClass::calibrate_real_sensors)
    .def("input_real_data", &MjClass::input_real_data)
    .def("get_real_observation", &MjClass::get_real_observation)
    .def("get_simple_state_vector", &MjClass::get_simple_state_vector)

    // misc
    .def("tick", &MjClass::tick)
    .def("tock", &MjClass::tock)
    .def("forward", &MjClass::forward)
    .def("get_number_of_objects", &MjClass::get_number_of_objects)
    .def("get_object_name", &MjClass::get_object_name)
    .def("get_current_object_name", &MjClass::get_current_object_name)
    .def("get_test_report", &MjClass::get_test_report)
    .def("get_n_actions", &MjClass::get_n_actions)
    .def("get_n_obs", &MjClass::get_n_obs)
    .def("get_N", &MjClass::get_N)
    .def("set_finger_thickness", &MjClass::set_finger_thickness)
    .def("set_finger_width", &MjClass::set_finger_width)
    .def("set_finger_modulus", &MjClass::set_finger_modulus)
    .def("get_finger_thickness", &MjClass::get_finger_thickness)
    .def("get_finger_stiffnesses", &MjClass::get_finger_stiffnesses)
    .def("get_finger_width", &MjClass::get_finger_width)
    .def("get_finger_modulus", &MjClass::get_finger_modulus)
    .def("get_finger_rigidity", &MjClass::get_finger_rigidity)
    .def("get_finger_length", &MjClass::get_finger_length)
    .def("get_finger_hook_length", &MjClass::get_finger_hook_length)
    .def("get_finger_hook_angle_degrees", &MjClass::get_finger_hook_angle_degrees)
    .def("get_object_xyz_bounding_box", &MjClass::get_object_xyz_bounding_box)
    .def("is_finger_hook_fixed", &MjClass::is_finger_hook_fixed)
    .def("get_fingertip_clearance", &MjClass::get_fingertip_clearance)
    .def("using_xyz_base_actions", &MjClass::using_xyz_base_actions)
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
    .def("curve_area", &MjClass::curve_area)
    .def("numerical_stiffness_converge", static_cast<std::string (MjClass::*)(float, float)>(&MjClass::numerical_stiffness_converge))
    .def("numerical_stiffness_converge", static_cast<std::string (MjClass::*)(float, float, std::vector<float>, std::vector<float>)>(&MjClass::numerical_stiffness_converge))
    .def("numerical_stiffness_converge_2", &MjClass::numerical_stiffness_converge_2)
    .def("set_sensor_noise_and_normalisation_to", &MjClass::set_sensor_noise_and_normalisation_to)
    .def("yield_load", static_cast<float (MjClass::*)()>(&MjClass::yield_load))
    .def("yield_load", static_cast<float (MjClass::*)(float, float)>(&MjClass::yield_load))

    // exposed variables
    .def_readwrite("set", &MjClass::s_)
    .def_readwrite("goal", &MjClass::goal_)
    .def_readwrite("model_folder_path", &MjClass::model_folder_path)
    .def_readwrite("object_set_name", &MjClass::object_set_name)
    .def_readonly("machine", &MjClass::machine)
    .def_readonly("current_load_path", &MjClass::current_load_path)
    .def_readwrite("curve_validation_data", &MjClass::curve_validation_data_)
    .def_readwrite("real_sensors", &MjClass::real_sensors_)
    .def_readwrite("sim_sensors", &MjClass::sim_sensors_)
    .def_readwrite("default_spawn_params", &MjClass::default_spawn_params)

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
          mjobj.goal_,                // event goal, if using HER
          mjobj.default_spawn_params  // object spawn parameters
        );
      },
      [](py::tuple t) { // __setstate__

        if (t.size() != 6 and t.size() != 7) // 6 is OLD version, delete later
          throw std::runtime_error("mjclass py::pickle got invalid state (tuple size wrong)");

        // create new c++ instance with old settings
        MjClass mjobj(t[0].cast<MjType::Settings>());

        // set the variables (must be same order as tuple above)
        if (t.size() >= 2) mjobj.current_load_path = t[1].cast<std::string>();
        // if (t.size() >= 3) mjobj.model_folder_path = t[2].cast<std::string>();
        if (t.size() >= 4) mjobj.object_set_name = t[3].cast<std::string>();
        // if (t.size() >= 5) mjobj.machine = t[4].cast<std::string>();
        if (t.size() >= 6) mjobj.goal_ = t[5].cast<MjType::Goal>();
        if (t.size() >= 7) mjobj.default_spawn_params = t[6].cast<MjType::SpawnParams>();

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
    .def("set_use_noise", &MjType::Settings::set_use_noise)
    .def("set_all_action_use", &MjType::Settings::set_all_action_use)
    .def("set_all_action_continous", &MjType::Settings::set_all_action_continous)
    .def("set_all_action_value", &MjType::Settings::set_all_action_value)
    .def("set_all_action_sign", &MjType::Settings::set_all_action_sign)

    // use a macro to create code snippets for all of the settings
    #define XX(name, type, value) .def_readwrite(#name, &MjType::Settings::name)
    #define SS(name, in_use, norm, readrate) .def_readwrite(#name, &MjType::Settings::name)
    #define AA(name, in_use, value, sign) .def_readwrite(#name, &MjType::Settings::name)
    #define BR(name, reward, done, trigger) .def_readwrite(#name, &MjType::Settings::name)
    #define LR(name, reward, done, trigger, min, max, overshoot) \
              .def_readwrite(#name, &MjType::Settings::name)
      // run the macro to create the code
      LUKE_MJSETTINGS_GENERAL
      LUKE_MJSETTINGS_SENSOR
      LUKE_MJSETTINGS_ACTION
      LUKE_MJSETTINGS_BINARY_REWARD
      LUKE_MJSETTINGS_LINEAR_REWARD

    #undef XX
    #undef SS
    #undef AA
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
          #define AA(name, in_use, value, sign) s.name,
          #define BR(name, reward, done, trigger) s.name,
          #define LR(name, reward, done, trigger, min, max, overshoot) s.name,
            // run the macro to create the code
            LUKE_MJSETTINGS_GENERAL
            LUKE_MJSETTINGS_SENSOR
            LUKE_MJSETTINGS_ACTION
            LUKE_MJSETTINGS_BINARY_REWARD
            LUKE_MJSETTINGS_LINEAR_REWARD
          #undef XX
          #undef SS
          #undef AA
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
        #define AA(name, used, value, sign) \
                  out.name = t[i].cast<MjType::ActionSetting>(); ++i;
        #define BR(name, reward, done, trigger) \
                  out.name = t[i].cast<MjType::BinaryReward>(); ++i;
        #define LR(name, reward, done, trigger, min, max, overshoot) \
                  out.name = t[i].cast<MjType::LinearReward>(); ++i;

          // run the macro to create the code
          LUKE_MJSETTINGS_GENERAL
          LUKE_MJSETTINGS_SENSOR
          LUKE_MJSETTINGS_ACTION
          LUKE_MJSETTINGS_BINARY_REWARD
          LUKE_MJSETTINGS_LINEAR_REWARD

        #undef XX
        #undef SS
        #undef AA
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
    .def_readonly("active_sum", &MjType::EventTrack::BinaryEvent::active_sum)
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

  // set up sensor type so python can interact and change them
  {py::class_<MjType::SpawnParams>(m, "SpawnParams")

    .def(py::init<>())
    .def_readwrite("index", &MjType::SpawnParams::index)
    .def_readwrite("x", &MjType::SpawnParams::x)
    .def_readwrite("y", &MjType::SpawnParams::y)
    .def_readwrite("zrot", &MjType::SpawnParams::zrot)
    .def_readwrite("xrange", &MjType::SpawnParams::xrange)
    .def_readwrite("yrange", &MjType::SpawnParams::yrange)
    .def_readwrite("rotrange", &MjType::SpawnParams::rotrange)
    .def_readwrite("xmin", &MjType::SpawnParams::xmin)
    .def_readwrite("xmax", &MjType::SpawnParams::xmax)
    .def_readwrite("ymin", &MjType::SpawnParams::ymin)
    .def_readwrite("ymax", &MjType::SpawnParams::ymax)
    .def_readwrite("smallest_gap", &MjType::SpawnParams::smallest_gap)
    .def_readwrite("xy_increment", &MjType::SpawnParams::xy_increment)
    .def_readwrite("rot_increment", &MjType::SpawnParams::rot_increment)

    // pickle support
    .def(py::pickle(
      [](const MjType::SpawnParams r) { // __getstate___
        /* return a tuple that fully encodes the state of the object */
        return py::make_tuple(
          r.index, 
          r.x, 
          r.y,
          r.zrot,
          r.xrange,
          r.yrange,
          r.rotrange,
          r.xmin,
          r.xmax,
          r.ymin,
          r.ymax,
          r.smallest_gap,
          r.xy_increment,
          r.rot_increment
        );
      },
      [](py::tuple t) { // __setstate__

        if (t.size() != 14)
          throw std::runtime_error("MjType::SpawnParams py::pickle got invalid state");

        // create new c++ instance with old data
        MjType::SpawnParams out;

        out.index = t[0].cast<int>();
        out.x = t[1].cast<double>();
        out.y = t[2].cast<double>();
        out.zrot = t[3].cast<double>();
        out.xrange = t[4].cast<double>();
        out.yrange = t[5].cast<double>();
        out.rotrange = t[6].cast<double>();
        out.xmin = t[7].cast<double>();
        out.xmax = t[8].cast<double>();
        out.ymin = t[9].cast<double>();
        out.ymax = t[10].cast<double>();
        out.smallest_gap = t[11].cast<double>();
        out.xy_increment = t[12].cast<double>();
        out.rot_increment = t[13].cast<double>();
        
        return out;
      }
    ))
    ;
  }

  {py::class_<MjType::EventTrack::LinearEvent>(m, "LinearEvent")

    .def_readonly("value", &MjType::EventTrack::LinearEvent::value)
    .def_readonly("last_value", &MjType::EventTrack::LinearEvent::last_value)
    .def_readonly("active_sum", &MjType::EventTrack::LinearEvent::active_sum)
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

        // out.active_sum = bool(out.row);

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

    #define BR(name, reward, done, trigger) \
              .def_readonly(#name, &MjType::EventTrack::name)

    #define LR(name, reward, done, trigger, min, max, overshoot) \
              .def_readonly(#name, &MjType::EventTrack::name)

      // run the macro to create the binding code
      LUKE_MJSETTINGS_BINARY_REWARD
      LUKE_MJSETTINGS_LINEAR_REWARD

    #undef BR
    #undef LR

    // pickle support
    .def(py::pickle(
      [](const MjType::EventTrack &et) { // __getstate___
        /* return a tuple that fully encodes the state of the object */
        return py::make_tuple(
          
          #define BR(name, reward, done, trigger) et.name,
          #define LR(name, reward, done, trigger, min, max, overshoot) et.name,
            
            // run the macro to create the binding code
            LUKE_MJSETTINGS_BINARY_REWARD
            LUKE_MJSETTINGS_LINEAR_REWARD
          
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
        #define BR(name, reward, done, trigger) \
                  et.name = t[i].cast<MjType::EventTrack::BinaryEvent>(); ++i;
        #define LR(name, reward, done, trigger, min, max, overshoot) \
                  et.name = t[i].cast<MjType::EventTrack::LinearEvent>(); ++i;
          
          // run the macro to create the code
          LUKE_MJSETTINGS_BINARY_REWARD
          LUKE_MJSETTINGS_LINEAR_REWARD

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
    .def("set_gaussian_noise", &MjType::Sensor::set_gaussian_noise)
    .def("set_uniform_noise", &MjType::Sensor::set_uniform_noise)

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
          r.noise_std,
          r.noise_overriden
        );
      },
      [](py::tuple t) { // __setstate__

        if (t.size() != 10)
          throw std::runtime_error("MjType::Sensor py::pickle got invalid state");

        // create new c++ instance with old data
        MjType::Sensor out(t[0].cast<bool>(), t[1].cast<float>(), t[2].cast<float>());

        out.use_normalisation = t[3].cast<bool>();
        out.use_noise = t[4].cast<bool>();
        out.raw_value_offset = t[5].cast<float>();
        out.noise_mag = t[6].cast<float>();
        out.noise_mu = t[7].cast<float>();
        out.noise_std = t[8].cast<float>();
        out.noise_overriden = t[9].cast<bool>();
        
        return out;
      }
    ))
    ;
  }

  // set up action setting type so python can interact and change them
  {py::class_<MjType::ActionSetting>(m, "ActionSetting")

    .def(py::init<std::string, bool, double, int>())
    .def_readwrite("name", &MjType::ActionSetting::name)
    .def_readwrite("in_use", &MjType::ActionSetting::in_use)
    .def_readwrite("value", &MjType::ActionSetting::value)
    .def_readwrite("sign", &MjType::ActionSetting::sign)

    // pickle support
    .def(py::pickle(
      [](const MjType::ActionSetting r) { // __getstate___
        /* return a tuple that fully encodes the state of the object */
        return py::make_tuple(
          r.name,
          r.in_use,
          r.value,
          r.sign
        );
      },
      [](py::tuple t) { // __setstate__

        // if (debug_bind) {
        //   std::cout << "unpickling MjType::ActionSetting ...";
        // }

        if (t.size() != 4)
          throw std::runtime_error("MjType::ActionSetting py::pickle got invalid state");

        // create new c++ instance with old data
        MjType::ActionSetting out(t[0].cast<std::string>(), t[1].cast<bool>(), 
          t[2].cast<double>(), t[3].cast<int>());

        // if (debug_bind) {
        //   std::cout << " finished\n";
        // }
        
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

    #define BR(name, reward, done, trigger) \
              .def_readwrite(#name, &MjType::Goal::name)

    #define LR(name, reward, done, trigger, min, max, overshoot) \
              .def_readwrite(#name, &MjType::Goal::name)

      // run the macro to create the binding code
      LUKE_MJSETTINGS_BINARY_REWARD
      LUKE_MJSETTINGS_LINEAR_REWARD

    #undef BR
    #undef LR

    // pickle support
    .def(py::pickle(
      [](const MjType::Goal &g) { // __getstate___
        /* return a tuple that fully encodes the state of the object */
        return py::make_tuple(
          
          #define BR(name, reward, done, trigger) g.name,
          #define LR(name, reward, done, trigger, min, max, overshoot) g.name,
            
            // run the macro to create the binding code
            LUKE_MJSETTINGS_BINARY_REWARD
            LUKE_MJSETTINGS_LINEAR_REWARD

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
        #define BR(name, reward, done, trigger) \
                  g.name = t[i].cast<MjType::Goal::Event>(); ++i;
        #define LR(name, reward, done, trigger, min, max, overshoot) \
                  g.name = t[i].cast<MjType::Goal::Event>(); ++i;

          // run the macro to create the code
          LUKE_MJSETTINGS_BINARY_REWARD
          LUKE_MJSETTINGS_LINEAR_REWARD

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

  // sensor calibration profile
  {py::class_<MjType::RealCalibrations::Calibration>(m, "Calibration")
    .def_readwrite("offset", &MjType::RealCalibrations::Calibration::offset)
    .def_readwrite("scale", &MjType::RealCalibrations::Calibration::scale)
    .def_readwrite("norm", &MjType::RealCalibrations::Calibration::norm)
    ;
  }

  // sensor data storage class
  {py::class_<MjType::SensorData>(m, "SensorData")
    .def(py::init<>())
    .def("reset", &MjType::SensorData::reset)
    .def("read_x_motor_position", &MjType::SensorData::read_x_motor_position)
    .def("read_y_motor_position", &MjType::SensorData::read_y_motor_position)
    .def("read_z_motor_position", &MjType::SensorData::read_z_motor_position)
    .def("read_z_base_position", &MjType::SensorData::read_z_base_position)
    .def("read_finger1_gauge", &MjType::SensorData::read_finger1_gauge)
    .def("read_finger2_gauge", &MjType::SensorData::read_finger2_gauge)
    .def("read_finger3_gauge", &MjType::SensorData::read_finger3_gauge)
    .def("read_palm_sensor", &MjType::SensorData::read_palm_sensor)
    .def("read_finger1_axial_gauge", &MjType::SensorData::read_finger1_axial_gauge)
    .def("read_finger2_axial_gauge", &MjType::SensorData::read_finger2_axial_gauge)
    .def("read_finger3_axial_gauge", &MjType::SensorData::read_finger3_axial_gauge)
    .def("read_wrist_X_sensor", &MjType::SensorData::read_wrist_X_sensor)
    .def("read_wrist_Y_sensor", &MjType::SensorData::read_wrist_Y_sensor)
    .def("read_wrist_Z_sensor", &MjType::SensorData::read_wrist_Z_sensor)
    ;
  }

  {py::class_<MjType::RealSensorData>(m, "RealSensorData")
    .def("reset", &MjType::RealSensorData::reset)
    .def_readwrite("raw", &MjType::RealSensorData::raw)
    .def_readwrite("SI", &MjType::RealSensorData::SI)
    .def_readwrite("normalised", &MjType::RealSensorData::normalised)
    .def_readwrite("g1", &MjType::RealSensorData::g1)
    .def_readwrite("g2", &MjType::RealSensorData::g2)
    .def_readwrite("g3", &MjType::RealSensorData::g3)
    .def_readwrite("palm", &MjType::RealSensorData::palm)
    .def_readwrite("wrist_Z", &MjType::RealSensorData::wrist_Z)
    ;
  }

  #if !defined(LUKE_CLUSTER)
    {py::class_<luke::RGBD>(m, "RGBD")
      .def(py::init<>())
      .def_readonly("rgb", &luke::RGBD::rgb)
      .def_readonly("depth", &luke::RGBD::depth)
      ;
    }
  #endif

  /* py::overload_cast requires c++14 */

};