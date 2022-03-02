#include "pybind11/pybind11.h"
#include <pybind11/stl.h>
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
    .def(py::init<MjClass::Settings>())

    // core functionality
    .def("load", &MjClass::load)
    .def("load_relative", &MjClass::load_relative)
    .def("reset", &MjClass::reset)
    .def("step", &MjClass::step)
    .def("render", &MjClass::render)

    // sensing
    .def("read_gauges", &MjClass::read_gauges)
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
    .def("get_observation", &MjClass::get_observation)
    .def("reward", &MjClass::reward)

    // misc
    .def("forward", &MjClass::forward)
    .def("get_number_of_objects", &MjClass::get_number_of_objects)
    .def("get_current_object_name", &MjClass::get_current_object_name)
    .def("get_test_report", &MjClass::get_test_report)
    .def_readwrite("set", &MjClass::s_)

    // pickle support
    .def(py::pickle(
      [](const MjClass &mjobj) { // __getstate___
        /* return a tuple that fully encodes the state of the object */
        return py::make_tuple(mjobj.s_);
      },
      [](py::tuple t) { // __setstate__

        if (t.size() != 1)
          throw std::runtime_error("mjclass py::pickle got invalid state");

        // create new c++ instance with old settings
        MjClass mjobj(t[0].cast<MjClass::Settings>());

        return mjobj;
      }
    ))
    ;

  // internal simulation settings class which gets entirely pickled
  py::class_<MjClass::Settings>(m, "set")

    .def(py::init<>())
    .def("get_settings", &MjClass::Settings::get_settings)
    .def("wipe_rewards", &MjClass::Settings::wipe_rewards)

    // use a macro to create code snippets for all of the settings
    #define X(name, type, value) .def_readwrite(#name, &MjClass::Settings::name)
    #define BR(name, reward, done, trigger) .def_readwrite(#name, &MjClass::Settings::name)
    #define LR(name, reward, done, trigger, min, max, overshoot) \
              .def_readwrite(#name, &MjClass::Settings::name)
      // run the macro to create the code
      LUKE_MJSETTINGS
    #undef X
    #undef BR
    #undef LR

    // example snippets produced by the above macro
    // .def_readwrite("step_num", &MjClass::Settings::step_num)
    // .def_readwrite("lifted", &MjClass::Settings::lifted)
    // .def_readwrite("gauge_read_rate_hz", &MjClass::Settings::gauge_read_rate_hz)

    // pickle support
    .def(py::pickle(
      [](const MjClass::Settings s) { // __getstate___
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
        constexpr bool debug = false;
        if (debug)
          std::cout << "unpickling MjClass::Settings now\n";

        // create new c++ instance
        MjClass::Settings out;

        // fill in with the old data
        int i = 0;

        // expand the tuple elements and type cast them with a macro
        #define X(name, type, value) out.name = t[i].cast<type>(); ++i;
        #define BR(name, reward, done, trigger) \
                  out.name = t[i].cast<MjClass::BinaryReward>(); ++i;
        #define LR(name, reward, done, trigger, min, max, overshoot) \
                  out.name = t[i].cast<MjClass::LinearReward>(); ++i;
          // run the macro to create the code
          LUKE_MJSETTINGS
        #undef X
        #undef BR
        #undef LR

        // example snippet using dummy
        out.dummy = t[i].cast<bool>(); ++i;

        if (debug)
          std::cout << "unpickling MjClass::Settings finished, i is " << i
            << ", size of tuple is " << t.size() << '\n';

        return out;
      }
    ))
    ;

  // tracking of important events in the simulation
  py::class_<MjClass::EventTrack>(m, "EventTrack")

    .def(py::init<>())
    .def_readonly("step_num", &MjClass::EventTrack::step_num)
    .def_readonly("lifted", &MjClass::EventTrack::lifted)
    .def_readonly("oob", &MjClass::EventTrack::oob)
    .def_readonly("dropped", &MjClass::EventTrack::dropped)
    .def_readonly("target_height", &MjClass::EventTrack::target_height)
    .def_readonly("exceed_limits", &MjClass::EventTrack::exceed_limits)
    .def_readonly("exceed_axial", &MjClass::EventTrack::exceed_axial)
    .def_readonly("exceed_lateral", &MjClass::EventTrack::exceed_lateral)
    .def_readonly("object_contact", &MjClass::EventTrack::object_contact)
    .def_readonly("object_stable", &MjClass::EventTrack::object_stable)
    .def_readonly("palm_force", &MjClass::EventTrack::palm_force)
    .def_readonly("exceed_palm", &MjClass::EventTrack::exceed_palm)
    ;

  // class for outputing test results and event tracking
  py::class_<MjClass::TestReport>(m, "TestReport")

    .def(py::init<>())
    .def_readonly("object_name", &MjClass::TestReport::object_name)
    .def_readonly("cumulative_reward", &MjClass::TestReport::cumulative_reward)
    .def_readonly("num_steps", &MjClass::TestReport::num_steps)
    .def_readonly("abs_cnt", &MjClass::TestReport::abs_cnt)
    .def_readonly("final_cnt", &MjClass::TestReport::final_cnt)
    .def_readonly("final_palm_force", &MjClass::TestReport::final_palm_force)
    .def_readonly("final_finger_force", &MjClass::TestReport::final_finger_force)
    ;

  // settings up rewards so python can interact and change them
  py::class_<MjClass::BinaryReward>(m, "BinaryReward")

    .def(py::init<float, bool, int>())
    .def("set", &MjClass::BinaryReward::set)

    .def_readwrite("reward", &MjClass::BinaryReward::reward)
    .def_readwrite("done", &MjClass::BinaryReward::done)
    .def_readwrite("trigger", &MjClass::BinaryReward::trigger)

    // pickle support
    .def(py::pickle(
      [](const MjClass::BinaryReward r) { // __getstate___
        /* return a tuple that fully encodes the state of the object */
        return py::make_tuple(r.reward, r.done, r.trigger);
      },
      [](py::tuple t) { // __setstate__

        if (t.size() != 3)
          throw std::runtime_error("MjClass::BinaryReward py::pickle got invalid state");

        // create new c++ instance with old data
        MjClass::BinaryReward out(t[0].cast<float>(), t[1].cast<int>(), t[2].cast<int>());

        return out;
      }
    ))
    ;

  py::class_<MjClass::LinearReward>(m, "LinearReward")

    .def(py::init<float, bool, int, float, float, float>())
    .def("set", &MjClass::LinearReward::set)

    .def_readwrite("reward", &MjClass::LinearReward::reward)
    .def_readwrite("done", &MjClass::LinearReward::done)
    .def_readwrite("trigger", &MjClass::LinearReward::trigger)
    .def_readwrite("min", &MjClass::LinearReward::min)
    .def_readwrite("max", &MjClass::LinearReward::max)
    .def_readwrite("overshoot", &MjClass::LinearReward::overshoot)

    // pickle support
    .def(py::pickle(
      [](const MjClass::LinearReward r) { // __getstate___
        /* return a tuple that fully encodes the state of the object */
        return py::make_tuple(r.reward, r.done, r.trigger, r.min, r.max, r.overshoot);
      },
      [](py::tuple t) { // __setstate__

        if (t.size() != 6)
          throw std::runtime_error("MjClass::LinearReward py::pickle got invalid state");

        // create new c++ instance with old data
        MjClass::LinearReward out(t[0].cast<float>(), t[1].cast<int>(), t[2].cast<int>(),
          t[3].cast<float>(), t[4].cast<float>(), t[5].cast<float>());

        return out;
      }
    ))
    ;

  /* py::overload_cast requires c++14 */

};