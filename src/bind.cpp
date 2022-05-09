#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "mjclass.h"

namespace py = pybind11;

int add(int i, int j) {
  return i + j;
}

// create a python module, called bind (must be saved in bind.so)
PYBIND11_MODULE(bind, m) {

  m.doc() = "A module to wrap mujoco into python"; // module docstring

  m.def("add", &add, "A function which adds two numbers");
  
  // main wrapper
  py::class_<MjClass>(m, "MjClass")

    // constructors
    .def(py::init<>())
    .def(py::init<std::string>())
    .def(py::init<MjType::Settings>())

    // core functionality
    .def("load", &MjClass::load)
    .def("load_relative", &MjClass::load_relative)
    .def("reset", &MjClass::reset)
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
    .def("spawn_object", static_cast<void (MjClass::*)(int, double, double)>(&MjClass::spawn_object))
    .def("is_done", &MjClass::is_done)
    .def("get_observation", &MjClass::get_observation)
    .def("reward", &MjClass::reward)
    .def("get_n_actions", &MjClass::get_n_actions)
    .def("get_n_obs", &MjClass::get_n_obs)

    // misc
    .def("forward", &MjClass::forward)
    .def("get_number_of_objects", &MjClass::get_number_of_objects)
    .def("get_current_object_name", &MjClass::get_current_object_name)
    .def("get_test_report", &MjClass::get_test_report)
    .def_readwrite("set", &MjClass::s_)
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
          mjobj.machine               // machine library is compiled for
        );
      },
      [](py::tuple t) { // __setstate__

        if (t.size() != 5 and t.size() != 2) // 2 is OLD version, delete later
          throw std::runtime_error("mjclass py::pickle got invalid state (tuple size wrong)");

        // create new c++ instance with old settings
        MjClass mjobj(t[0].cast<MjType::Settings>());

        // set the variables (must be same order as tuple above)
        if (t.size() >= 2) mjobj.current_load_path = t[1].cast<std::string>();
        if (t.size() >= 3) mjobj.model_folder_path = t[2].cast<std::string>();
        if (t.size() >= 4) mjobj.object_set_name = t[3].cast<std::string>();
        if (t.size() >= 5) mjobj.machine = t[4].cast<std::string>();

        return mjobj;
      }
    ))
    ;

  // internal simulation settings class which gets entirely pickled
  py::class_<MjType::Settings>(m, "set")

    .def(py::init<>())
    .def("get_settings", &MjType::Settings::get_settings)
    .def("wipe_rewards", &MjType::Settings::wipe_rewards)
    .def("scale_rewards", &MjType::Settings::scale_rewards)

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
        constexpr bool debug = true;
        if (debug)
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

        if (debug)
          std::cout << "unpickling MjType::Settings finished, i is " << i
            << ", size of tuple is " << t.size() << '\n';

        return out;
      }
    ))
    ;

  // tracking of important events in the simulation
  py::class_<MjType::EventTrack>(m, "EventTrack")

    .def(py::init<>())

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

    ; // this semicolon is required to finish the py::class definition

    // example snippets from the macro above
    // .def_readonly("step_num", &MjType::EventTrack::step_num)
    // .def_readonly("lifted", &MjType::EventTrack::lifted)
    // .def_readonly("oob", &MjType::EventTrack::oob)

  // class for outputing test results and event tracking
  py::class_<MjType::TestReport>(m, "TestReport")

    .def(py::init<>())
    .def_readonly("object_name", &MjType::TestReport::object_name)
    .def_readonly("cumulative_reward", &MjType::TestReport::cumulative_reward)
    .def_readonly("num_steps", &MjType::TestReport::num_steps)
    .def_readonly("abs_cnt", &MjType::TestReport::abs_cnt)
    .def_readonly("final_cnt", &MjType::TestReport::final_cnt)
    .def_readonly("final_palm_force", &MjType::TestReport::final_palm_force)
    .def_readonly("final_finger_force", &MjType::TestReport::final_finger_force)
    ;

  // set up sensor type so python can interact and change them
  py::class_<MjType::Sensor>(m, "Sensor")

    .def(py::init<float, bool, int>())
    .def("set", &MjType::Sensor::set)

    .def_readwrite("in_use", &MjType::Sensor::in_use)
    .def_readwrite("normalise", &MjType::Sensor::normalise)
    .def_readwrite("read_rate", &MjType::Sensor::read_rate)

    // pickle support
    .def(py::pickle(
      [](const MjType::Sensor r) { // __getstate___
        /* return a tuple that fully encodes the state of the object */
        return py::make_tuple(r.in_use, r.normalise, r.read_rate);
      },
      [](py::tuple t) { // __setstate__

        if (t.size() != 3)
          throw std::runtime_error("MjType::Sensor py::pickle got invalid state");

        // create new c++ instance with old data
        MjType::Sensor out(t[0].cast<bool>(), t[1].cast<float>(), t[2].cast<float>());

        return out;
      }
    ))
    ;

  // set up binary reward type so python can interact and change them
  py::class_<MjType::BinaryReward>(m, "BinaryReward")

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

  // set up linear reward type so python can edit them
  py::class_<MjType::LinearReward>(m, "LinearReward")

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

  // three classes to extract detailed curve fit data from the simulation
  py::class_<MjType::CurveFitData::PoseData::FingerData>(m, "FingerData")
    .def(py::init<>())
    .def_readwrite("x", &MjType::CurveFitData::PoseData::FingerData::x)
    .def_readwrite("y", &MjType::CurveFitData::PoseData::FingerData::y)
    .def_readwrite("coeff", &MjType::CurveFitData::PoseData::FingerData::coeff)
    .def_readwrite("errors", &MjType::CurveFitData::PoseData::FingerData::errors)
    ;

  py::class_<MjType::CurveFitData::PoseData>(m, "PoseData")
    .def(py::init<>())
    .def_readwrite("f1", &MjType::CurveFitData::PoseData::f1)
    .def_readwrite("f2", &MjType::CurveFitData::PoseData::f2)
    .def_readwrite("f3", &MjType::CurveFitData::PoseData::f3)
    ;

  py::class_<MjType::CurveFitData>(m, "CurveFitData")

    .def(py::init<>())
    .def_readwrite("entries", &MjType::CurveFitData::entries)
    ;


  /* py::overload_cast requires c++14 */

};