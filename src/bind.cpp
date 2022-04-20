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
    .def("read_gauges", &MjClass::read_gauges)
    .def("read_palm", &MjClass::read_palm)
    .def("get_gripper_state", &MjClass::get_gripper_state)
    .def("is_target_reached", &MjClass::is_target_reached)
    .def("is_settled", &MjClass::is_settled)

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
    .def("get_observation", static_cast<std::vector<luke::gfloat> (MjClass::*)()>(&MjClass::get_observation))
    .def("get_observation", static_cast<std::vector<luke::gfloat> (MjClass::*)(int)>(&MjClass::get_observation))
    .def("reward", &MjClass::reward)
    .def("get_n_actions", &MjClass::get_n_actions)
    .def("get_n_obs", &MjClass::get_n_obs)

    // misc
    .def("forward", &MjClass::forward)
    .def("get_number_of_objects", &MjClass::get_number_of_objects)
    .def("get_current_object_name", &MjClass::get_current_object_name)
    .def("get_test_report", &MjClass::get_test_report)
    .def_readwrite("set", &MjClass::s_)
    .def_readonly("current_load_path", &MjClass::current_load_path)
    .def_readwrite("curve_validation_data", &MjClass::curve_validation_data_)

    // pickle support
    .def(py::pickle(
      [](const MjClass &mjobj) { // __getstate___
        /* return a tuple that fully encodes the state of the object */
        return py::make_tuple(mjobj.s_, mjobj.current_load_path);
      },
      [](py::tuple t) { // __setstate__

        if (t.size() != 2)
          throw std::runtime_error("mjclass py::pickle got invalid state (tuple size wrong)");

        // create new c++ instance with old settings
        MjClass mjobj(t[0].cast<MjType::Settings>());

        // save the old load path
        mjobj.current_load_path = t[1].cast<std::string>();

        return mjobj;
      }
    ))
    ;

  // internal simulation settings class which gets entirely pickled
  py::class_<MjType::Settings>(m, "set")

    .def(py::init<>())
    .def("get_settings", &MjType::Settings::get_settings)
    .def("wipe_rewards", &MjType::Settings::wipe_rewards)

    // use a macro to create code snippets for all of the settings
    #define X(name, type, value) .def_readwrite(#name, &MjType::Settings::name)
    #define BR(name, reward, done, trigger) .def_readwrite(#name, &MjType::Settings::name)
    #define LR(name, reward, done, trigger, min, max, overshoot) \
              .def_readwrite(#name, &MjType::Settings::name)
      // run the macro to create the code
      LUKE_MJSETTINGS
    #undef X
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
          #define X(name, type, value) s.name,
          #define BR(name, reward, done, trigger) s.name,
          #define LR(name, reward, done, trigger, min, max, overshoot) s.name,
            // run the macro to create the code
            LUKE_MJSETTINGS
          #undef X
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
        #define X(name, type, value) out.name = t[i].cast<type>(); ++i;
        #define BR(name, reward, done, trigger) \
                  out.name = t[i].cast<MjType::BinaryReward>(); ++i;
        #define LR(name, reward, done, trigger, min, max, overshoot) \
                  out.name = t[i].cast<MjType::LinearReward>(); ++i;
          // run the macro to create the code
          LUKE_MJSETTINGS
        #undef X
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

    #define X(name, type, value)
    #define BR(name, reward, done, trigger) .def_readonly(#name, &MjType::EventTrack::name)
    #define LR(name, reward, done, trigger, min, max, overshoot) \
              .def_readonly(#name, &MjType::EventTrack::name)
      // run the macro to create the binding code
      LUKE_MJSETTINGS
    #undef X
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

  // settings up rewards so python can interact and change them
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