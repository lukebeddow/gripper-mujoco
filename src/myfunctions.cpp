#include "myfunctions.h"

namespace luke
{

/* ----- helper functions ----- */

void print_vec(std::vector<bool> v, std::string name) {
  std::cout << name << ": { ";
  if (v.size() == 0) {
    std::cout << "}\n";
    return;
  }
  for (int i = 0; i < v.size() - 1; i++) {
    std::cout << v[i] << " ";
  }
  std::cout << v[v.size() - 1] << " }\n";
}
void print_vec(std::vector<int> v, std::string name) {
  std::cout << name << ": { ";
  if (v.size() == 0) {
    std::cout << "}\n";
    return;
  }
  for (int i = 0; i < v.size() - 1; i++) {
    std::cout << v[i] << ", ";
  }
  std::cout << v[v.size() - 1] << " }\n";
}
void print_vec(std::vector<mjtNum> v, std::string name) {
  std::cout << name << ": { ";
  if (v.size() == 0) {
    std::cout << "}\n";
    return;
  }
  // cast mjtNum to float32 and show to 3dp
  for (int i = 0; i < v.size() - 1; i++) {
    printf("%.3f, ", float(v[i]));
  }
  printf("%.3f }\n", float(v[v.size() - 1]));
}
void print_vec(std::vector<gfloat> v, std::string name) {
  std::cout << name << ": { ";
  if (v.size() == 0) {
    std::cout << "}\n";
    return;
  }
  // cast mjtNum to float32 and show to 3dp
  for (int i = 0; i < v.size() - 1; i++) {
    printf("%.3f, ", float(v[i]));
  }
  printf("%.3f }\n", float(v[v.size() - 1]));
}
void print_vec(std::vector<QPos> v, std::string name) {
  std::cout << name << ": { ";
  if (v.size() == 0) {
    std::cout << "}\n";
    return;
  }
  // cast mjtNum to float32 and show to 3dp
  for (int i = 0; i < v.size(); i++) {
    printf("\n\txyz = (%.3f, %.3f, %.3f) quat = (%.3f, %.3f, %.3f, %.3f)",
      v[i].x, v[i].y, v[i].z, v[i].qx, v[i].qy, v[i].qz, v[i].qw);
  }
  std::cout << "\n}\n";
}
void print_vec(std::vector<std::string> v, std::string name) {
  std::cout << name << ": { ";
  if (v.size() == 0) {
    std::cout << "}\n";
    return;
  }
  for (int i = 0; i < v.size() - 1; i++) {
    std::cout << v[i] << ", ";
  }
  std::cout << v[v.size() - 1] << " }\n";
}
template <typename T> int sign(T val) {
    return (T(0) < val) - (val < T(0));
}
bool strcmp_w_sub(std::string ref_str, std::string sub_str, int num) {
  /* check if two strings are equal, but one string having a substitued value.
  This value should be indicated by {X}, and will be swapped with integers
  from 1 ... num. */

  if (sub_str.size() < 3) {
    throw std::runtime_error("string compare with substitution failed as input"
      " string has size less than 3, and the substitution value is '{X}'");
  }

  int sub_idx;
  std::string before_sub_str;
  std::string after_sub_str;
  bool found_sub = false;

  // first find the substitution point
  char c1;
  char c2 = sub_str[0];
  char c3 = sub_str[1];

  for (int i = 2; i < sub_str.size(); i++) {

    c1 = c2;
    c2 = c3; 
    c3 = sub_str[i];

    if (c1 == '{' and c2 == 'X' and c3 == '}') {
      before_sub_str = sub_str.substr(0, i - 2);
      if (sub_str.size() == i + 1) {
        after_sub_str = "";
      }
      else {
        after_sub_str = sub_str.substr(i + 1, sub_str.size() - (i + 1));
      }
      found_sub = true;
      break;
    }
  }

  // if the sub string doesn't contain a substitution marker, do normal strcmp
  if (not found_sub) {
    return (ref_str == sub_str);
  }

  // otherwise perform the comparison with substitution
  for (int i = 1; i < num + 1; i++) {

    std::string to_comp = before_sub_str + std::to_string(i) + after_sub_str;

    if (to_comp == ref_str) {
      return true;
    }
  }

  return false;
}

/* ----- Global variables and settings ----- */

// global settings for joints in the model
struct JointSettings {

  /* ----- user input settings ----- */

  // keyframes (poses) defined in the xml files
  std::string initial_keyframe = "initial pose";
  std::string reset_keyframe = "initial pose";

  // joint names, need to be hardcoded in here for gripper and panda
  struct Names {
    std::vector<std::string> panda = {
      "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
      "panda_joint5", "panda_joint6", "panda_joint7"
    };
    std::vector<std::string> gripper = {
      "finger_1_prismatic_joint", "finger_1_revolute_joint",
      "finger_2_prismatic_joint", "finger_2_revolute_joint",
      "finger_3_prismatic_joint", "finger_3_revolute_joint",
      "palm_prismatic_joint"
    };
    std::vector<std::string> base = {
      "world_to_base"
    };
    std::vector<std::string> finger;                  // runtime depends
  } names;

  // gripper specific info for joints (determined by the name order above)
  struct {
    std::vector<int> prismatic { 0, 2, 4 };
    std::vector<int> revolute { 1, 3, 5 };
    std::vector<int> palm { 6 };
  } gripper;

  // key dimensions and details
  struct Dim {
    double finger_length = 235e-3;
    double finger_thickness = 0.9e-3;
    double finger_width = 28e-3;
    double E = 200e9;
    double I = (finger_width * std::pow(finger_thickness, 3)) / 12.0;
    double EI = E * I;
    bool fixed_first_segment;                         // runtime depends
    double stiffness_c = 0;                           // runtime depends
    double segment_length = 0;                        // runtime depends
    std::vector<luke::gfloat> joint_stiffness;        // runtime depends

    void update_EI() {
      I = (finger_width * std::pow(finger_thickness, 3)) / 12.0;
      EI = E * I;
    }

    void reset() {
      joint_stiffness.clear();
      update_EI();
    }

  } dim;

  // strain gauge parameters
  struct {
    bool use_armadillo_gauges = true;          // whether to use curve fitting for finger bending
    int order = 3;
    double xpos = 50e-3;
    double cbrt_xpos = std::cbrt(xpos);         // runtime depends
    double xpos_cubed = xpos * xpos * xpos;     // runtime depends
  } gauge;

  // control parameters
  struct {
    bool stepper = true;
    int num_steps = 10;                         // number of stepper motors steps in one chunk
    double pulses_per_s = 2000;                 // stepper motor pulses per second, this sets speed (2000pps = 300rpm)
    Gain kp {100, 40, 1000};                    // proportional gains for gripper xyz motors {x, y, z}
    Gain kd {1, 1, 1};                          // derivative gains for gripper xyz motors {x, y, z}
    double base_kp = 2000;                      // proportional gain for gripper base motions
    double base_kd = 100;                       // derivative gain for gripper base motions

    double time_per_step = 0.0;                 // runtime depends
  } ctrl;

  // hardcoded stiffnesses based on numerical solving
  struct {

    // VALID FOR: real data 1, 235x28x0.9mm fingers, EI=0.34
    struct {

      std::vector<float> N5 { 12.351, 4.728, 5.877, 5.030, 1.927 };
      std::vector<float> N6 { 15.230, 5.923, 6.318, 7.202, 6.383, 2.287 };
      std::vector<float> N7 { 18.861, 7.121, 7.336, 9.602, 8.184, 6.218, 2.000 };
      std::vector<float> N8 { 20.645, 8.041, 8.389, 9.570, 10.123, 8.606, 6.252, 2.079 };
      std::vector<float> N9 { 24.818, 9.799, 9.907, 9.872, 12.627, 11.329, 9.560, 7.562, 2.863 };
      std::vector<float> N10 { 27.302, 11.378, 9.575, 10.845, 14.214, 15.567, 10.791, 9.340, 5.922, 1.816 };
      std::vector<float> N15 { 37.103, 22.431, 15.466, 12.927, 23.115, 15.260, 22.107, 20.695, 21.631, 14.546, 16.221, 12.876, 10.205, 6.744, 2.480 };
      std::vector<float> N20 { 75.206, 23.422, 27.987, 21.172, 16.599, 40.647, 28.657, 19.737, 27.249, 28.612, 34.024, 36.741, 25.092, 21.021, 21.680, 18.867, 14.520, 10.797, 6.767, 1.830 };
      std::vector<float> N25 { 503.022, 16.672, 124.905, 27.550, 24.489, 17.904, 63.461, 34.390, 17.560, 35.540, 69.003, 91.560, 70.231, 63.729, 62.175, 22.117, 6.076, 44.070, 40.950, 65.208, 51.254, 44.330, 25.115, 14.184, 0.500 };
      
      // N30 did not converge to a low error value
      std::vector<float> N30 { 685.194, 18.996, 206.246, 38.443, 27.892, 34.549, 17.070, 57.971, 96.573, 58.083, 11.051, 35.880, 81.274, 140.579, 146.179, 184.276, 219.552, 260.973, 277.488, 303.149, 328.203, 356.418, 351.385, 345.677, 327.709, 298.091, 256.786, 205.697, 144.913, 74.690 };

    } finger_235x28x0p9;

    // VALID FOR: theory data, 235x28x0.8mm fingers, E=200GPa
    struct {

      std::vector<float> N5 { 9.45, 4.33, 4.25, 3.92, 3.05 };
      std::vector<float> N6 { 11.60, 5.41, 5.39, 5.27, 4.83, 4.27 };
      std::vector<float> N7 { 13.00, 6.13, 6.15, 6.14, 5.99, 5.59, 5.35 };
      std::vector<float> N8 { 15.61, 7.44, 7.45, 7.47, 7.42, 7.20, 6.82, 6.74 };
      std::vector<float> N9 { 17.00, 8.16, 8.25, 8.21, 8.17, 8.12, 8.06, 8.04, 8.28 };
      std::vector<float> N10 { 20.30, 9.78, 9.82, 9.81, 9.82, 9.80, 9.70, 9.56, 9.45, 9.55 };
      std::vector<float> N11 { 21.95, 10.60, 10.70, 10.70, 10.71, 10.70, 10.67, 10.59, 10.51, 10.47, 10.63 };
      std::vector<float> N12 { 23.49, 11.37, 11.54, 11.54, 11.53, 11.53, 11.54, 11.52, 11.47, 11.43, 11.45, 11.66 };
      std::vector<float> N13 { 25.00, 12.17, 12.34, 12.40, 12.34, 12.35, 12.37, 12.38, 12.38, 12.37, 12.36, 12.44, 12.68 };
      std::vector<float> N14 { 29.45, 14.50, 14.47, 14.45, 14.47, 14.48, 14.50, 14.52, 14.50, 14.46, 14.40, 14.32, 14.26, 14.22 };
      std::vector<float> N15 { 31.25, 15.38, 15.40, 15.42, 15.44, 15.46, 15.47, 15.48, 15.48, 15.46, 15.44, 15.40, 15.36, 15.32, 15.28 };
      std::vector<float> N16 { 33.25, 16.30, 16.30, 16.31, 16.33, 16.34, 16.36, 16.38, 16.40, 16.40, 16.39, 16.38, 16.36, 16.33, 16.30, 16.28 };
      std::vector<float> N17 { 35.08, 17.13, 17.15, 17.18, 17.20, 17.25, 17.32, 17.38, 17.44, 17.48, 17.49, 17.49, 17.46, 17.41, 17.36, 17.31, 17.28 };
      std::vector<float> N18 { 36.89, 18.01, 17.99, 18.03, 18.10, 18.16, 18.26, 18.35, 18.43, 18.49, 18.53, 18.55, 18.53, 18.50, 18.44, 18.39, 18.33, 18.29 };
      std::vector<float> N19 { 38.81, 18.82, 18.85, 18.91, 19.00, 19.09, 19.20, 19.31, 19.40, 19.49, 19.55, 19.58, 19.60, 19.57, 19.53, 19.47, 19.40, 19.35, 19.31 };
      std::vector<float> N20 { 40.19, 19.61, 19.82, 19.92, 19.99, 20.05, 20.11, 20.14, 20.17, 20.22, 20.28, 20.31, 20.31, 20.29, 20.25, 20.18, 20.11, 20.07, 20.05, 20.13 };
      std::vector<float> N21 { 41.74, 20.50, 20.82, 20.71, 20.82, 20.91, 20.94, 21.06, 21.10, 21.18, 21.24, 21.29, 21.31, 21.32, 21.29, 21.25, 21.19, 21.13, 21.08, 21.08, 21.16 };
      std::vector<float> N22 { 43.40, 21.42, 21.51, 21.72, 21.79, 21.80, 21.91, 21.88, 21.94, 22.02, 22.06, 22.13, 22.19, 22.20, 22.21, 22.19, 22.16, 22.11, 22.07, 22.05, 22.07, 22.16 };
      std::vector<float> N23 { 45.26, 22.09, 22.54, 22.71, 22.59, 22.61, 22.69, 22.78, 22.87, 22.91, 22.97, 23.04, 23.10, 23.15, 23.18, 23.18, 23.17, 23.15, 23.10, 23.07, 23.06, 23.08, 23.18 };
      std::vector<float> N24 { 47.23, 22.87, 23.30, 23.53, 23.63, 23.68, 23.68, 23.68, 23.69, 23.72, 23.76, 23.81, 23.86, 23.92, 23.97, 24.00, 24.01, 24.01, 24.00, 23.98, 23.97, 23.98, 24.05, 24.18 };
      std::vector<float> N25 { 49.02, 23.69, 24.17, 24.43, 24.56, 24.61, 24.63, 24.63, 24.63, 24.59, 24.59, 24.63, 24.69, 24.76, 24.81, 24.85, 24.89, 24.92, 24.94, 24.94, 24.93, 24.95, 24.97, 25.05, 25.19 };
      std::vector<float> N26 { 50.51, 24.51, 25.15, 25.47, 25.38, 25.35, 25.41, 25.49, 25.44, 25.47, 25.56, 25.60, 25.66, 25.74, 25.79, 25.85, 25.91, 25.93, 25.95, 25.97, 25.97, 25.97, 25.98, 26.01, 26.08, 26.22 };
      std::vector<float> N27 { 51.32, 25.99, 25.69, 25.42, 25.97, 25.48, 25.95, 26.07, 25.74, 26.48, 26.17, 26.77, 26.62, 26.86, 26.71, 26.74, 26.57, 26.46, 26.30, 26.08, 25.84, 25.51, 25.19, 24.87, 24.72, 24.90, 25.69 };
      std::vector<float> N28 { 52.82, 27.06, 25.92, 26.96, 25.95, 26.97, 26.19, 26.98, 26.68, 27.00, 27.16, 27.04, 27.44, 27.38, 27.64, 27.69, 27.70, 27.71, 27.68, 27.58, 27.32, 27.08, 26.77, 26.38, 26.06, 25.89, 26.05, 26.80 };
      std::vector<float> N29 { 54.46, 27.70, 27.29, 26.99, 27.58, 27.08, 27.37, 27.81, 27.22, 27.63, 28.31, 28.14, 28.07, 28.43, 28.51, 28.51, 28.41, 28.30, 28.13, 28.01, 27.84, 27.54, 27.23, 26.95, 26.69, 26.48, 26.45, 26.77, 27.68 };
      std::vector<float> N30 { 56.37, 28.77, 28.88, 28.86, 28.63, 29.37, 29.56, 29.66, 29.47, 30.10, 30.31, 30.17, 29.94, 30.39, 31.26, 32.34, 33.19, 33.46, 33.36, 32.93, 32.22, 31.33, 30.34, 29.31, 28.37, 27.61, 27.10, 26.95, 27.32, 28.42 };

      std::vector<int> loops { 232, 261, 257, 235, 29, 150, 111, 99, 22, 22, 3, 6, 23, 20, 19, 99, 95, 96, 91, 95, 94, 20, 500, 500, 500, 500 };
      std::vector<float> errors { 0.019816, 0.019906, 0.019912, 0.019893, 0.01948, 0.019629, 0.018897, 0.017943, 0.017132, 0.018029, 0.014474, 0.019269, 0.018786, 0.019077, 0.019292, 0.0187, 0.018082, 0.01735, 0.017378, 0.016763, 0.016636, 0.015557, 1.426878, 1.903569, 2.170651, 2.408715 };
    
    } theory_235x28x0p8;

    // VALID FOR: theory data, 235x28x0.9mm fingers, E=200GPa
    struct {

      std::vector<float> N5 { 13.99, 6.37, 6.15, 5.62, 4.84 };
      std::vector<float> N6 { 17.00, 7.90, 7.77, 7.48, 6.97, 6.72 };
      std::vector<float> N7 { 19.12, 9.02, 8.96, 8.77, 8.52, 8.28, 8.49 };
      std::vector<float> N8 { 22.74, 10.83, 10.79, 10.64, 10.44, 10.22, 10.06, 10.32 };
      std::vector<float> N9 { 24.82, 11.88, 11.96, 11.82, 11.67, 11.50, 11.37, 11.37, 11.79 };
      std::vector<float> N10 { 29.22, 14.06, 14.06, 13.98, 13.85, 13.73, 13.57, 13.42, 13.37, 13.64 };
      std::vector<float> N11 { 31.55, 15.21, 15.30, 15.24, 15.14, 15.04, 14.92, 14.80, 14.71, 14.76, 15.10 };
      std::vector<float> N12 { 33.78, 16.32, 16.52, 16.46, 16.36, 16.27, 16.17, 16.08, 16.02, 16.01, 16.14, 16.54 };
      std::vector<float> N13 { 39.02, 19.23, 19.07, 19.00, 18.91, 18.87, 18.79, 18.72, 18.64, 18.56, 18.50, 18.50, 18.58 };
      std::vector<float> N14 { 41.75, 20.45, 20.38, 20.33, 20.28, 20.25, 20.22, 20.18, 20.13, 20.08, 20.03, 19.99, 20.00, 20.08 };
      std::vector<float> N15 { 44.35, 21.66, 21.62, 21.61, 21.59, 21.58, 21.58, 21.58, 21.58, 21.59, 21.59, 21.60, 21.61, 21.63, 21.67 };
      std::vector<float> N16 { 47.17, 22.94, 22.88, 22.83, 22.80, 22.79, 22.78, 22.79, 22.80, 22.82, 22.85, 22.89, 22.92, 22.96, 23.02, 23.09 };
      std::vector<float> N17 { 49.85, 24.16, 24.10, 24.08, 24.05, 24.05, 24.08, 24.11, 24.15, 24.18, 24.21, 24.24, 24.28, 24.31, 24.36, 24.42, 24.50 };
      std::vector<float> N18 { 52.45, 25.40, 25.30, 25.29, 25.32, 25.35, 25.41, 25.47, 25.53, 25.59, 25.63, 25.67, 25.70, 25.72, 25.76, 25.79, 25.85, 25.93 };
      std::vector<float> N19 { 55.15, 26.54, 26.49, 26.51, 26.55, 26.63, 26.71, 26.80, 26.88, 26.96, 27.02, 27.07, 27.11, 27.14, 27.17, 27.20, 27.23, 27.29, 27.38 };
      std::vector<float> N20 { 57.83, 27.72, 27.66, 27.69, 27.77, 27.88, 28.00, 28.12, 28.22, 28.32, 28.40, 28.48, 28.53, 28.57, 28.60, 28.61, 28.64, 28.68, 28.73, 28.83 };
      std::vector<float> N21 { 60.32, 28.90, 28.94, 28.88, 28.98, 29.12, 29.24, 29.39, 29.52, 29.65, 29.76, 29.84, 29.92, 29.96, 30.01, 30.04, 30.06, 30.09, 30.12, 30.18, 30.26 };
      std::vector<float> N22 { 61.33, 30.07, 30.31, 30.64, 30.69, 30.66, 30.68, 30.55, 30.54, 30.47, 30.45, 30.46, 30.44, 30.45, 30.46, 30.49, 30.52, 30.57, 30.67, 30.82, 31.04, 31.38 };
      std::vector<float> N23 { 63.99, 31.05, 31.76, 32.04, 31.89, 31.87, 31.88, 31.90, 31.80, 31.74, 31.74, 31.74, 31.76, 31.76, 31.78, 31.81, 31.84, 31.90, 31.98, 32.09, 32.25, 32.48, 32.83 };
      std::vector<float> N24 { 71.57, 36.21, 36.37, 36.37, 36.31, 36.21, 36.09, 35.98, 35.87, 35.77, 35.69, 35.61, 35.53, 35.46, 35.38, 35.30, 35.23, 35.13, 35.05, 34.96, 34.87, 34.80, 34.75, 34.73 };
      std::vector<float> N25 { 74.34, 37.56, 37.72, 37.75, 37.70, 37.63, 37.53, 37.42, 37.33, 37.25, 37.15, 37.07, 37.00, 36.93, 36.87, 36.80, 36.73, 36.65, 36.57, 36.48, 36.40, 36.32, 36.26, 36.21, 36.18 };
      std::vector<float> N26 { 76.91, 38.83, 39.11, 39.24, 39.14, 39.02, 38.92, 38.85, 38.73, 38.63, 38.55, 38.50, 38.42, 38.36, 38.31, 38.25, 38.19, 38.12, 38.06, 37.97, 37.90, 37.83, 37.76, 37.70, 37.65, 37.63 };
      std::vector<float> N27 { 79.55, 40.21, 40.50, 40.47, 40.51, 40.42, 40.33, 40.27, 40.16, 40.09, 40.01, 39.95, 39.90, 39.83, 39.80, 39.73, 39.68, 39.62, 39.56, 39.50, 39.42, 39.35, 39.28, 39.22, 39.15, 39.11, 39.08 };
      std::vector<float> N28 { 82.28, 41.69, 41.66, 41.89, 41.80, 41.83, 41.72, 41.69, 41.60, 41.55, 41.48, 41.43, 41.38, 41.34, 41.29, 41.24, 41.20, 41.14, 41.09, 41.03, 40.96, 40.88, 40.81, 40.75, 40.68, 40.62, 40.57, 40.54 };
      std::vector<float> N29 { 85.09, 42.86, 43.15, 43.16, 43.25, 43.19, 43.13, 43.12, 43.01, 42.96, 42.92, 42.86, 42.81, 42.77, 42.73, 42.68, 42.65, 42.60, 42.56, 42.50, 42.45, 42.38, 42.32, 42.25, 42.18, 42.13, 42.06, 42.02, 41.99 };
      std::vector<float> N30 { 87.97, 44.10, 44.46, 44.61, 44.56, 44.55, 44.54, 44.51, 44.43, 44.38, 44.34, 44.31, 44.25, 44.22, 44.18, 44.15, 44.12, 44.08, 44.03, 43.99, 43.94, 43.88, 43.83, 43.76, 43.70, 43.64, 43.58, 43.52, 43.47, 43.45 };

      std::vector<int> loops { 296, 244, 110, 265, 207, 234, 234, 204, 91, 78, 7, 3, 24, 32, 29, 30, 30, 191, 185, 29, 32, 33, 36, 38, 43, 56 };
      std::vector<float> errors { 0.01991, 0.019921, 0.019758, 0.019631, 0.019437, 0.019613, 0.019526, 0.019239, 0.019065, 0.019502, 0.018542, 0.015368, 0.0193, 0.018971, 0.019288, 0.019647, 0.019703, 0.018989, 0.018819, 0.018175, 0.017745, 0.018386, 0.018174, 0.018487, 0.018556, 0.018725 };
    
    } theory_235x28x0p9;

    // VALID FOR: theory data, 235x28x1.0mm fingers, E=200GPa
    struct {

      std::vector<float> N5 { 18.74, 8.51, 8.21, 7.48, 6.68 };
      std::vector<float> N6 { 23.54, 10.93, 10.70, 10.24, 9.66, 9.66 };
      std::vector<float> N7 { 26.42, 12.44, 12.36, 12.02, 11.63, 11.33, 11.73 };
      std::vector<float> N8 { 31.05, 14.77, 14.73, 14.46, 14.14, 13.82, 13.69, 14.18 };
      std::vector<float> N9 { 36.29, 17.34, 17.27, 17.09, 16.84, 16.57, 16.31, 16.24, 16.63 };
      std::vector<float> N10 { 39.42, 18.93, 19.00, 18.85, 18.62, 18.37, 18.17, 18.02, 18.10, 18.62 };
      std::vector<float> N11 { 45.36, 21.94, 21.76, 21.61, 21.46, 21.32, 21.15, 20.99, 20.88, 20.90, 21.18 };
      std::vector<float> N12 { 48.94, 23.75, 23.68, 23.59, 23.48, 23.35, 23.21, 23.08, 22.95, 22.91, 22.98, 23.26 };
      std::vector<float> N13 { 52.70, 25.40, 25.28, 25.21, 25.16, 25.14, 25.14, 25.17, 25.21, 25.28, 25.36, 25.47, 25.63 };
      std::vector<float> N14 { 56.57, 27.15, 26.96, 26.83, 26.75, 26.72, 26.73, 26.76, 26.83, 26.93, 27.05, 27.19, 27.36, 27.56 };
      std::vector<float> N15 { 60.04, 28.72, 28.55, 28.49, 28.47, 28.49, 28.54, 28.59, 28.66, 28.74, 28.83, 28.95, 29.09, 29.27, 29.50 };
      std::vector<float> N16 { 63.65, 30.25, 30.13, 30.12, 30.16, 30.24, 30.32, 30.42, 30.50, 30.59, 30.66, 30.75, 30.87, 31.00, 31.19, 31.44 };
      std::vector<float> N17 { 69.75, 34.85, 34.85, 34.74, 34.52, 34.33, 34.17, 34.03, 33.89, 33.76, 33.65, 33.53, 33.44, 33.38, 33.36, 33.40, 33.53 };
      std::vector<float> N18 { 73.34, 36.65, 36.58, 36.51, 36.39, 36.25, 36.13, 35.98, 35.88, 35.77, 35.66, 35.58, 35.49, 35.42, 35.38, 35.38, 35.42, 35.54 };
      std::vector<float> N19 { 77.18, 38.29, 38.35, 38.32, 38.25, 38.16, 38.06, 37.94, 37.84, 37.74, 37.66, 37.58, 37.51, 37.45, 37.40, 37.38, 37.39, 37.44, 37.55 };
      std::vector<float> N20 { 80.95, 40.02, 40.06, 40.06, 40.03, 39.98, 39.94, 39.88, 39.82, 39.75, 39.69, 39.63, 39.58, 39.54, 39.50, 39.47, 39.45, 39.46, 39.50, 39.59 };
      std::vector<float> N21 { 84.75, 41.74, 41.76, 41.76, 41.77, 41.77, 41.78, 41.78, 41.78, 41.77, 41.77, 41.76, 41.75, 41.75, 41.74, 41.73, 41.73, 41.72, 41.70, 41.71, 41.70 };
      std::vector<float> N22 { 88.64, 43.62, 43.58, 43.56, 43.54, 43.53, 43.52, 43.51, 43.50, 43.50, 43.50, 43.50, 43.51, 43.51, 43.52, 43.53, 43.54, 43.55, 43.58, 43.60, 43.62, 43.65 };
      std::vector<float> N23 { 92.55, 45.47, 45.42, 45.38, 45.32, 45.28, 45.25, 45.23, 45.22, 45.21, 45.21, 45.23, 45.24, 45.26, 45.28, 45.31, 45.34, 45.37, 45.41, 45.45, 45.50, 45.55, 45.60 };
      std::vector<float> N24 { 96.49, 47.37, 47.27, 47.18, 47.11, 47.05, 47.00, 46.98, 46.96, 46.95, 46.94, 46.95, 46.97, 46.99, 47.02, 47.06, 47.11, 47.16, 47.21, 47.28, 47.34, 47.41, 47.49, 47.57 };
      std::vector<float> N25 { 100.41, 49.25, 49.11, 48.99, 48.90, 48.82, 48.75, 48.70, 48.68, 48.66, 48.65, 48.66, 48.69, 48.71, 48.75, 48.80, 48.85, 48.92, 48.99, 49.05, 49.14, 49.23, 49.32, 49.43, 49.53 };
      std::vector<float> N26 { 104.12, 51.00, 50.88, 50.79, 50.69, 50.60, 50.57, 50.54, 50.54, 50.54, 50.55, 50.59, 50.62, 50.66, 50.71, 50.76, 50.81, 50.87, 50.93, 50.99, 51.07, 51.14, 51.22, 51.31, 51.41, 51.51 };
      std::vector<float> N27 { 107.78, 52.79, 52.68, 52.50, 52.46, 52.38, 52.35, 52.36, 52.35, 52.39, 52.43, 52.46, 52.53, 52.57, 52.63, 52.69, 52.75, 52.81, 52.87, 52.93, 53.00, 53.06, 53.13, 53.21, 53.29, 53.39, 53.49 };
      std::vector<float> N28 { 111.47, 54.70, 54.33, 54.32, 54.14, 54.15, 54.09, 54.13, 54.15, 54.22, 54.25, 54.34, 54.40, 54.48, 54.55, 54.62, 54.69, 54.76, 54.82, 54.88, 54.94, 55.00, 55.06, 55.13, 55.20, 55.28, 55.37, 55.47 };
      std::vector<float> N29 { 115.25, 56.35, 56.21, 55.99, 55.97, 55.89, 55.88, 55.95, 55.96, 56.02, 56.11, 56.18, 56.27, 56.36, 56.44, 56.52, 56.60, 56.68, 56.74, 56.81, 56.88, 56.94, 56.99, 57.05, 57.12, 57.19, 57.27, 57.36, 57.46 };
      std::vector<float> N30 { 119.12, 58.08, 58.00, 57.91, 57.72, 57.68, 57.71, 57.74, 57.77, 57.85, 57.95, 58.05, 58.14, 58.24, 58.34, 58.44, 58.53, 58.61, 58.69, 58.76, 58.82, 58.88, 58.94, 59.00, 59.06, 59.12, 59.19, 59.27, 59.36, 59.45 };

      std::vector<int> loops { 500, 294, 380, 413, 368, 379, 207, 206, 9, 4, 48, 74, 143, 135, 124, 94, 3, 4, 4, 3, 3, 31, 50, 65, 75, 83 };
      std::vector<float> errors { 0.020577, 0.019895, 0.019827, 0.019784, 0.01983, 0.019726, 0.019933, 0.019841, 0.018864, 0.014028, 0.019431, 0.019458, 0.01947, 0.019408, 0.019523, 0.019729, 0.016478, 0.013167, 0.006946, 0.008642, 0.016216, 0.019371, 0.019016, 0.019289, 0.019098, 0.018883 };
    } theory_235x28x1p0;

  } hardcoded_c;

  /* ----- automatically generated settings ----- */

  // is this part of the model in use
  struct InUse {
    bool panda = false;
    bool gripper = false;
    bool finger = false;
    bool base = false;
  } in_use;

  // how many joints for each part
  struct JointNum {
    int panda = 0;
    int gripper = 0;
    int finger = 0;
    int per_finger = 0;
    int base = 0;
  } num;

  VectorStruct<int> idx;
  VectorStruct<int> qposadr;
  VectorStruct<int> qveladr;
  VectorStruct<mjtNum> qpos;
  VectorStruct<mjtNum> qvel;
  VectorStruct<mjtNum*> to_qpos;
  VectorStruct<mjtNum*> to_qvel;

  // joint weld constraint indexes (for freezing/fixing joints)
  struct ConIdx {
    std::vector<int> prismatic;
    std::vector<int> revolute;
    std::vector<int> palm;
  } con_idx;

  // segmented finger geom ids for colour changing fingers
  struct GeomIdx {
    std::vector<int> finger1;
    std::vector<int> finger2;
    std::vector<int> finger3;
    std::vector<int> palm;
  } geom_idx;

  /* ----- Member functions ----- */

  // only resets the automatically generated settings
  void reset() {

    // special case, reset joint stiffness vector
    dim.reset();

    // reset the VectorStructs
    idx.reset();
    qposadr.reset();
    qveladr.reset();
    qpos.reset();
    qvel.reset();
    to_qpos.reset();
    to_qvel.reset();

    // reset other custom structs
    Names names_reset;
    names = names_reset;

    ConIdx con_idx_reset;
    con_idx = con_idx_reset;

    GeomIdx geom_idx_reset;
    geom_idx = geom_idx_reset;

    InUse in_use_reset;
    in_use = in_use_reset;

    JointNum joint_num_reset;
    num = joint_num_reset;
  }

  // printing functions
  void print_idx() {
    print_vec(idx.panda, "panda joint idx");
    print_vec(idx.gripper, "gripper joint idx");
    print_vec(idx.finger, "finger joint idx");
    print_vec(idx.base, "base joint idx");
  }
  void print_in_use() {
    std::cout << "Using: " 
      << "panda = " << (in_use.panda ? "true" : "false")
      << ", gripper = " << (in_use.gripper ? "true" : "false")
      << ", segmented fingers = " << (in_use.finger ? "true" : "false")
      << ", base = " << (in_use.gripper ? "true" : "false")
      << '\n';
  }
  void print_num() {
    std::cout << "Number of joints for: "
      << "panda = " << num.panda
      << ", gripper = " << num.gripper 
      << ", finger = " << num.finger
      << ", per finger = " << num.per_finger 
      << ", base = " << num.base
      << '\n';
  }
  void print_qposadr() {
    print_vec(qposadr.panda, "panda joint qpos addresses");
    print_vec(qposadr.gripper, "gripper joint qpos addresses");
    print_vec(qposadr.finger, "finger joint qpos addresses");
    print_vec(qposadr.base, "base joint qpos addresses");
  }
  void print_qveladr() {
    print_vec(qveladr.panda, "panda joint qvel addresses");
    print_vec(qveladr.gripper, "gripper joint qvel addresses");
    print_vec(qveladr.finger, "finger joint qvel addresses");
    print_vec(qveladr.base, "base joint qvel addresses");
  }
  void print_qpos() {
    std::cout << "Please note, qpos and qvel are no longer used. To see them for "
      "debugging, please run the function update_state() before printing\n";
    print_vec(qpos.panda, "panda joint qpos");
    print_vec(qpos.gripper, "gripper joint qpos");
    print_vec(qpos.finger, "finger joint qpos");
    print_vec(qpos.base, "base joint qpos");
  }
  void print_qvel() {
    std::cout << "Please note, qpos and qvel are no longer used. To see them for "
      "debugging, please run the function update_state() before printing\n";
    print_vec(qvel.panda, "panda joint qvel");
    print_vec(qvel.gripper, "gripper joint qvel");
    print_vec(qvel.finger, "finger joint qvel");
    print_vec(qvel.base, "base joint qvel");
  }
  void print_geom_idx() {
    print_vec(geom_idx.finger1, "finger1 geom idx");
    print_vec(geom_idx.finger2, "finger2 geom idx");
    print_vec(geom_idx.finger3, "finger3 geom idx");
    print_vec(geom_idx.palm, "palm geom idx");
  }
  
};

// global joint settings structure
JointSettings j_;

// create object handler to control graspable objects in simulation
ObjectHandler oh_;

Target target_;     // global state target

// gripper finger state 
// these are not currently used at all!
Gripper finger1_;
Gripper finger2_;
Gripper finger3_;

// time of last stepper step
static double last_step_time_ = 0.0;

constexpr static bool debug = false; // turn on/off debug mode for this file only

/* ----- initialising, setup, and utilities ----- */

void init(mjModel* model, mjData* data)
{
  /* runs once when model is created */

  last_step_time_ = 0.0;

  // extract model information and store it in our global variable j_
  init_J(model, data);

  // set the model to the inital keyframe
  keyframe(model, data, j_.initial_keyframe);

  // initialise the object handler
  oh_.init(model, data);

  // // assign my control function to the mujoco control fcn pointer
  // mjcb_control = control;

}

void init_J(mjModel* model, mjData* data)
{
  /* initialise our global data structure with joint and model information */

  // wipe the global settings structure
  j_.reset();

  // use joint names to get body indexes and qpos/qvel addresses
  get_joint_indexes(model);
  get_joint_addresses(model);
  get_geom_indexes(model);

  if (debug) {
    print_joint_names(model);
  }

  // resize state vectors and find qpos/qvel pointers
  configure_qpos(model, data);

  // calculate constants
  j_.ctrl.time_per_step = j_.ctrl.num_steps / j_.ctrl.pulses_per_s;
  int N = j_.num.per_finger;
  int Ntotal = j_.num.per_finger + j_.dim.fixed_first_segment;
  j_.dim.segment_length = j_.dim.finger_length / float(Ntotal);

  if (j_.dim.fixed_first_segment) {
    j_.dim.stiffness_c = ( j_.dim.EI / (2 * j_.dim.finger_length) ) 
      * ( (float)(N * (N*N + 6*N + 11)) / (float)((N + 1) * (N + 1)) );
  }
  else {
    j_.dim.stiffness_c = ( j_.dim.EI / (2 * j_.dim.finger_length) ) 
       * ( (float)((N + 1)*(N + 2)) / N);
  }

  if (debug) {
    std::cout << "Number of finger joints N is " << j_.num.per_finger << '\n';
    std::cout << "Joint stiffness c is " << j_.dim.stiffness_c << '\n';
  }

  configure_constraints(model, data);
}

void reset(mjModel* model, mjData* data)
{
  /* reset the simulation */

  // reset the targets and disable any constraints
  target_.reset();
  set_all_constraints(model, data, false);

  // wipe object positions and reset
  mj_resetData(model, data);
  keyframe(model, data, j_.reset_keyframe);
  reset_object(model, data);

  // recalculate all object positions/forces
  mj_forward(model, data);
  update_all(model, data);

  // set the joints to the equilibrium position
  calibrate_reset(model, data);

  // now enable all constraints at equilibrium position
  set_all_constraints(model, data, true);
}

void print_joint_names(mjModel* model)
{
  /* print joint names to the terminal */

  for (int i = 0; i < model->njnt; i++) {
    auto x = mj_id2name(model, mjOBJ_JOINT, i);
    std::cout << "i = " << i << " gives jnt name = " << x << '\n';
  }
}

void get_joint_indexes(mjModel* model)
{
  /* Get the indexes of the different joint groups */

  for (std::string name : j_.names.panda) {
    int idx = mj_name2id(model, mjOBJ_JOINT, name.c_str());
    j_.idx.panda.push_back(idx);
  }
  for (std::string name : j_.names.gripper) {
    int idx = mj_name2id(model, mjOBJ_JOINT, name.c_str());
    j_.idx.gripper.push_back(idx);
  }
  for (std::string name : j_.names.base) {
    int idx = mj_name2id(model, mjOBJ_JOINT, name.c_str());
    j_.idx.base.push_back(idx);
  }

  if (j_.idx.panda[0] != -1) j_.in_use.panda = true;
  else j_.in_use.panda = false;

  if (j_.idx.gripper[0] != -1) j_.in_use.gripper = true;
  else j_.in_use.gripper = false;

  if (j_.idx.base[0] != -1) j_.in_use.base = true;
  else j_.in_use.base = false;

  // determine how many joints are being used for each part
  j_.num.panda = j_.names.panda.size() * j_.in_use.panda;
  j_.num.gripper = j_.names.gripper.size() * j_.in_use.gripper;
  j_.num.base = j_.names.base.size() * j_.in_use.base;

  // count how many segment joints we have
  j_.num.finger = 0;
  j_.dim.fixed_first_segment = true;
  for (int i = 0; i < model->njnt; i++) {
    std::string x = mj_id2name(model, mjOBJ_JOINT, i);
    if (x.substr(0,6) == "finger" and x.substr(9, 13) == "segment_joint") {
      j_.num.finger += 1;
      // if we have a segment_joint_0 then there is not a fixed first joint
      if (x.substr(9, 15) == "segment_joint_0") {
        j_.dim.fixed_first_segment = false;
      }
    }
  }

  // // for testing
  // if (j_.num.finger != 27) throw std::runtime_error("j_.num.finger != 27");

  // hence per finger is this divided by 3
  j_.num.per_finger = j_.num.finger / 3;

  if (j_.num.finger > 0) {

    j_.in_use.finger = true;
    int ffs = (int) j_.dim.fixed_first_segment;

    // add the names of every finger joint to the global vector
    for (int i = 1; i <= 3; i++) {
      for (int k = ffs; k < j_.num.per_finger + ffs; k++) {
        // create the joint name string and add it to the vector
        std::string next = "finger_" + std::to_string(i)
          + "_segment_joint_" + std::to_string(k);
        j_.names.finger.push_back(next);
      }
    }

    // now, we want the indexes corresponding to each joint
    for (std::string name : j_.names.finger) {
      int idx = mj_name2id(model, mjOBJ_JOINT, name.c_str());
      j_.idx.finger.push_back(idx);
    }

    // extra safety check
    if (j_.names.finger.size() == 0 or j_.idx.finger[0] == -1) {
      printf("Error: Finger joints not found\n");
      j_.in_use.finger = false;
      j_.num.finger = 0;
    }
  }

  if (debug) {
    j_.print_in_use();
    j_.print_num();
    j_.print_idx();
    std::cout << "Fixed first segment is: " << j_.dim.fixed_first_segment << '\n';
  }
}

void get_geom_indexes(mjModel* model)
{
  /* get the indexes of the geoms for the fingers */

  // each geom has both a 'collision' and 'visual' version, so we collect both
  std::vector<std::string> geom_suffixes { "collision", "visual" };

  int ffs = j_.dim.fixed_first_segment;

  for (std::string geom_tag : geom_suffixes) {

    for (int i = 0; i < j_.num.finger; i++) {

      std::string geom_name = "finger_" + std::to_string(i / j_.num.per_finger + 1)  // finger_X, X=1,2,3
        + "_segment_link_" + std::to_string(i % j_.num.per_finger + 1 + ffs)         // links go 2-10 for 10 segments
        + "_geom_" + geom_tag;

      int x = mj_name2id(model, mjOBJ_GEOM, geom_name.c_str());

      if (i < j_.num.per_finger) {
        j_.geom_idx.finger1.push_back(x);
      }
      else if (i < 2 * j_.num.per_finger) {
        j_.geom_idx.finger2.push_back(x);
      }
      else if (i < 3 * j_.num.per_finger) {
        j_.geom_idx.finger3.push_back(x);
      }
      else {
        throw std::runtime_error("get_geom_indexes() found inconsistent finger segment numbers");
      }
    }

    // now add the hook links
    std::string f1_hook = "finger_1_segment_link_" + std::to_string(j_.num.per_finger + ffs)
      + "_geom_hook_" + geom_tag;
    std::string f2_hook = "finger_2_segment_link_" + std::to_string(j_.num.per_finger + ffs)
      + "_geom_hook_" + geom_tag;
    std::string f3_hook = "finger_3_segment_link_" + std::to_string(j_.num.per_finger + ffs)
      + "_geom_hook_" + geom_tag;

    j_.geom_idx.finger1.push_back(mj_name2id(model, mjOBJ_GEOM, f1_hook.c_str()));
    j_.geom_idx.finger2.push_back(mj_name2id(model, mjOBJ_GEOM, f2_hook.c_str()));
    j_.geom_idx.finger3.push_back(mj_name2id(model, mjOBJ_GEOM, f3_hook.c_str()));

    // now add the palm link
    std::string palm_geom_name = "palm_geom_" + geom_tag;
    j_.geom_idx.palm.push_back(mj_name2id(model, mjOBJ_GEOM, palm_geom_name.c_str()));
  }

  if (debug) {
    j_.print_geom_idx();
  }
}

void get_joint_addresses(mjModel* model)
{
  /* Get the qpos and qvel addresses for each active joint */

  // old (incorrect) code used: model->jnt_qposadr[model->body_jntadr[idx]]

  if (j_.in_use.panda) {
    for (int idx : j_.idx.panda) {
      j_.qposadr.panda.push_back(model->jnt_qposadr[idx]);
      j_.qveladr.panda.push_back(model->jnt_dofadr[idx]);
    }
  }

  if (j_.in_use.gripper) {
    for (int idx : j_.idx.gripper) {
      j_.qposadr.gripper.push_back(model->jnt_qposadr[idx]);
      j_.qveladr.gripper.push_back(model->jnt_dofadr[idx]);
    }
  }

  if (j_.in_use.finger) {
    for (int idx : j_.idx.finger) {
      j_.qposadr.finger.push_back(model->jnt_qposadr[idx]);
      j_.qveladr.finger.push_back(model->jnt_dofadr[idx]);
    }
  }

  if (j_.in_use.base) {
    for (int idx : j_.idx.base) {
      j_.qposadr.base.push_back(model->jnt_qposadr[idx]);
      j_.qveladr.base.push_back(model->jnt_dofadr[idx]);
    }
  }

  if (debug) {
    j_.print_qposadr();
    j_.print_qveladr();
  }
}

bool change_finger_thickness(float thickness)
{
  /* set a new finger width, and correspondingly change EI, requires reset after */

  constexpr bool local_debug = debug;

  if (local_debug) {
    std::cout << "About to change finger thickness from " << j_.dim.finger_thickness
      << " to " << thickness << ", EI is " << j_.dim.EI << '\n';
  }

  // check if thickness is greater than 5mm
  if (thickness > 5e-3) {
    std::cout << "thickness given = " << thickness << '\n';
    throw std::runtime_error("change_finger_thickness() got value above 5mm - make sure you are using SI units!");
  }

  constexpr float tol = 1e-5;

  if (abs(thickness - j_.dim.finger_thickness) < tol) {
    if (local_debug) std::cout << "Finger thickness is the same as current, not changing\n";
    return false;
  }

  j_.dim.finger_thickness = thickness;
  j_.dim.I = (j_.dim.finger_width * std::pow(j_.dim.finger_thickness, 3)) / 12.0;
  j_.dim.EI = j_.dim.E * j_.dim.I;

  if (local_debug) {
    std::cout << "Finger thickness changed, now is " << j_.dim.finger_thickness
      << ", EI is " << j_.dim.EI << '\n';
  }

  return true;
}

void set_finger_stiffness(mjModel* model, std::vector<luke::gfloat> stiffness)
{
  /* set the finger stiffness to a vector sequence of values */

  constexpr bool local_debug = false; // debug
  
  int N = j_.num.per_finger;

  if (stiffness.size() != N) {
    throw std::runtime_error("wrong number of joint stiffnesses passed to set_finger_stiffness(...)");
  }

  // loop over all three fingers
  for (int i = 0; i < 3; i++) {

    // loop from n=1 to N
    for (int n = 1; n < N + 1; n++) {

      int idx = j_.idx.finger[i * N + (n - 1)];

      if (local_debug and i == 0) 
        std::cout << "Finger joint stiffness joint " << i << " is " << stiffness[n - 1] << '\n';

      if (i == 0) {
        j_.dim.joint_stiffness[n - 1] = stiffness[n - 1];
      }

      model->jnt_stiffness[idx] = stiffness[n - 1];

    }
  }
}

void set_finger_stiffness(mjModel* model, mjtNum stiffness)
{
  /* set the stiffness of the flexible finger joints. The input value for stiffness
  determines the behaviour of this function.

  INPUT OPTIONS
  stiffness > 0       -> all joints are set to this stiffness value
  stiffness is -7.5   -> stiffness is calculated with finalised theory derviation
  stiffness is -100   -> use hardcoded stiffness values, convergence on real data (only 0.9mm bending @ 300g)
  stiffness is -101   -> use hardcoded stiffness values, convergence on theory (0.8/0.9/1.0mm @ 300g)

  */

  // macro for hardcoding stiffness values
  #define LUKE_EXPAND_HARDCODED_STIFFNESSES(NAME) \
                switch (N) { \
                  case 5:  set_finger_stiffness(model, j_.hardcoded_c.NAME.N5); break; \
                  case 6:  set_finger_stiffness(model, j_.hardcoded_c.NAME.N6); break; \
                  case 7:  set_finger_stiffness(model, j_.hardcoded_c.NAME.N7); break; \
                  case 8:  set_finger_stiffness(model, j_.hardcoded_c.NAME.N8); break; \
                  case 9:  set_finger_stiffness(model, j_.hardcoded_c.NAME.N9); break; \
                  case 10: set_finger_stiffness(model, j_.hardcoded_c.NAME.N10); break; \
                  case 11: set_finger_stiffness(model, j_.hardcoded_c.NAME.N11); break; \
                  case 12: set_finger_stiffness(model, j_.hardcoded_c.NAME.N12); break; \
                  case 13: set_finger_stiffness(model, j_.hardcoded_c.NAME.N13); break; \
                  case 14: set_finger_stiffness(model, j_.hardcoded_c.NAME.N14); break; \
                  case 15: set_finger_stiffness(model, j_.hardcoded_c.NAME.N15); break; \
                  case 16: set_finger_stiffness(model, j_.hardcoded_c.NAME.N16); break; \
                  case 17: set_finger_stiffness(model, j_.hardcoded_c.NAME.N17); break; \
                  case 18: set_finger_stiffness(model, j_.hardcoded_c.NAME.N18); break; \
                  case 19: set_finger_stiffness(model, j_.hardcoded_c.NAME.N19); break; \
                  case 20: set_finger_stiffness(model, j_.hardcoded_c.NAME.N20); break; \
                  case 21: set_finger_stiffness(model, j_.hardcoded_c.NAME.N21); break; \
                  case 22: set_finger_stiffness(model, j_.hardcoded_c.NAME.N22); break; \
                  case 23: set_finger_stiffness(model, j_.hardcoded_c.NAME.N23); break; \
                  case 24: set_finger_stiffness(model, j_.hardcoded_c.NAME.N24); break; \
                  case 25: set_finger_stiffness(model, j_.hardcoded_c.NAME.N25); break; \
                  case 26: set_finger_stiffness(model, j_.hardcoded_c.NAME.N26); break; \
                  case 27: set_finger_stiffness(model, j_.hardcoded_c.NAME.N27); break; \
                  case 28: set_finger_stiffness(model, j_.hardcoded_c.NAME.N28); break; \
                  case 29: set_finger_stiffness(model, j_.hardcoded_c.NAME.N29); break; \
                  case 30: set_finger_stiffness(model, j_.hardcoded_c.NAME.N30); break; \
                  default: \
                    std::string error_string = "no hardcoded theory stiffness values for this N = "; \
                    error_string += std::to_string(N); \
                    throw std::runtime_error(error_string); \
                }

  // start of function proper
  constexpr bool local_debug = debug;

  int N = j_.num.per_finger;

  // prepare a convenience vector to record stiffness values for each finger
  j_.dim.joint_stiffness.clear();
  j_.dim.joint_stiffness.resize(N);

  if (stiffness > 0) {
    if (local_debug) std::cout << "Finger joint stiffness ALL set to " << stiffness << '\n';
    for (int i : j_.idx.finger) {
      model->jnt_stiffness[i] = stiffness;
    }
    // save stiffness values for one finger, even though they are all the same
    for (int i = 0; i < N; i++) j_.dim.joint_stiffness[i] = stiffness;
  }

  else if (stiffness > -8 and stiffness < -7) {

    if (local_debug) std::cout << "Finger joint stiffness set using finalised theory method (EI*N)/L\n";

    // loop over all three fingers
    for (int i = 0; i < 3; i++) {

      float angle_sum = 0;

      // loop from n=1 to N
      for (int n = 1; n < N + 1; n++) {

        int idx = j_.idx.finger[i * N + (n - 1)];

        // calculate the stiffness for each joint
        float c;
        if (n == 1) {
          c = ((2 * j_.dim.EI) / j_.dim.finger_length) * ((N*N) / (double)(N - (1.0/3.0)));
        }
        else {
          c = (N * j_.dim.EI) / j_.dim.finger_length;
        }

        // save stiffness values for 1st finger
        if (i == 0) {
          j_.dim.joint_stiffness[n - 1] = c;
        }

        if (local_debug and i == 0) {
          std::cout << "finger joint " << n << " has c_n = " << c << '\n';
        }

        model->jnt_stiffness[idx] = c;
      }
    }
  }

  else if (stiffness > -100.5 and stiffness < -99.5) {

    if (local_debug) std::cout << "Finger joint stiffness set using hardcoding for 235x28x0.9mm fingers from real data\n";

    switch (N) {

      case 5:  set_finger_stiffness(model, j_.hardcoded_c.finger_235x28x0p9.N5); break;
      case 6:  set_finger_stiffness(model, j_.hardcoded_c.finger_235x28x0p9.N6); break;
      case 7:  set_finger_stiffness(model, j_.hardcoded_c.finger_235x28x0p9.N7); break;
      case 8:  set_finger_stiffness(model, j_.hardcoded_c.finger_235x28x0p9.N8); break;
      case 9:  set_finger_stiffness(model, j_.hardcoded_c.finger_235x28x0p9.N9); break;
      case 10: set_finger_stiffness(model, j_.hardcoded_c.finger_235x28x0p9.N10); break;
      case 15: set_finger_stiffness(model, j_.hardcoded_c.finger_235x28x0p9.N15); break;
      case 20: set_finger_stiffness(model, j_.hardcoded_c.finger_235x28x0p9.N20); break;
      case 25: set_finger_stiffness(model, j_.hardcoded_c.finger_235x28x0p9.N25); break;
      case 30: set_finger_stiffness(model, j_.hardcoded_c.finger_235x28x0p9.N30); break;

      default:
        std::cout << "N is " << N << '\n';
        throw std::runtime_error("no hardcoded stiffness values for this N");
    }

  }

  else if (stiffness > -101.5 and stiffness < -100.5) {

    float tol = 1e-5;

    if (abs(j_.dim.finger_thickness - 0.8e-3) < tol) {

      if (local_debug) std::cout << "Finger joint stiffness set using hardcoding for 235x28x0.8mm fingers from theory predictions\n";

      LUKE_EXPAND_HARDCODED_STIFFNESSES(theory_235x28x0p8)
    }

    else if (abs(j_.dim.finger_thickness - 0.9e-3) < tol) {

      if (local_debug) std::cout << "Finger joint stiffness set using hardcoding for 235x28x0.9mm fingers from theory predictions\n";

      LUKE_EXPAND_HARDCODED_STIFFNESSES(theory_235x28x0p9)
    }

    else if (abs(j_.dim.finger_thickness - 1.0e-3) < tol) {

      if (local_debug) std::cout << "Finger joint stiffness set using hardcoding for 235x28x1.0mm fingers from theory predictions\n";

      LUKE_EXPAND_HARDCODED_STIFFNESSES(theory_235x28x1p0)
    }

    else {
      std::cout << "finger thickness is " << j_.dim.finger_thickness << '\n';
      throw std::runtime_error("no hardcoded theory stiffness values for this finger thickness");
    }
  }

  else {
    std::cout << "set_finger_stiffness(...) input stiffness was " << stiffness << '\n';
    throw  std::runtime_error("set_finger_stiffness(...) input stiffness not valid");
  }

  if (local_debug)
    print_vec(j_.dim.joint_stiffness, "joint stiffness vector");
}

void configure_qpos(mjModel* model, mjData* data)
{
  /* sort out qpos and qvel information */

  // resize state vectors
  j_.qpos.panda.resize(j_.num.panda);
  j_.qpos.gripper.resize(j_.num.gripper);
  j_.qpos.finger.resize(j_.num.finger);
  j_.qpos.base.resize(j_.num.base);

  j_.qvel.panda.resize(j_.num.panda);
  j_.qvel.gripper.resize(j_.num.gripper);
  j_.qvel.finger.resize(j_.num.finger);
  j_.qvel.base.resize(j_.num.base);

  // resize pointer vectors
  j_.to_qpos.panda.resize(j_.num.panda);
  j_.to_qpos.gripper.resize(j_.num.gripper);
  j_.to_qpos.finger.resize(j_.num.finger);
  j_.to_qpos.base.resize(j_.num.base);

  j_.to_qvel.panda.resize(j_.num.panda);
  j_.to_qvel.gripper.resize(j_.num.gripper);
  j_.to_qvel.finger.resize(j_.num.finger);
  j_.to_qvel.base.resize(j_.num.base);

  // insert the pointers
  if (j_.in_use.panda) {
    for (int i = 0; i < j_.num.panda; i++) {
      j_.to_qpos.panda[i] = &data->qpos[j_.qposadr.panda[i]];
      j_.to_qvel.panda[i] = &data->qvel[j_.qveladr.panda[i]];
    }
  }

  if (j_.in_use.gripper) {
    for (int i = 0; i < j_.num.gripper; i++) {
      j_.to_qpos.gripper[i] = &data->qpos[j_.qposadr.gripper[i]];
      j_.to_qvel.gripper[i] = &data->qvel[j_.qveladr.gripper[i]];
    }
  }
  
  if (j_.in_use.finger) {
    for (int i = 0; i < j_.num.finger; i++) {
      j_.to_qpos.finger[i] = &data->qpos[j_.qposadr.finger[i]];
      j_.to_qvel.finger[i] = &data->qvel[j_.qveladr.finger[i]];
    }
  }

  if (j_.in_use.base) {
    for (int i = 0; i < j_.num.base; i++) {
      j_.to_qpos.base[i] = &data->qpos[j_.qposadr.base[i]];
      j_.to_qvel.base[i] = &data->qvel[j_.qveladr.base[i]];
    }
  }
}

void configure_constraints(mjModel* model, mjData* data)
{
  /* configure equality constraints for gripper motors */

  constexpr char pris_b1[] = "gripper_base_link";
  constexpr char pris_b2[] = "finger_{X}_intermediate";
  constexpr char rev_b1[] = "finger_{X}_intermediate";
  constexpr char rev_b2[] = "finger_{X}";
  constexpr char palm_b1[] = "gripper_base_link";
  constexpr char palm_b2[] = "palm";

  for (int i = 0; i < model->neq; i++) {

    std::string name1 = mj_id2name(model, mjOBJ_BODY, model->eq_obj1id[i]);
    std::string name2 = mj_id2name(model, mjOBJ_BODY, model->eq_obj2id[i]);


    if (debug) {
      std::printf("Constraint %d has ids %d and %d, which are bodies %s and %s\n", 
        i, model->eq_obj1id[i], model->eq_obj2id[i], name1.c_str(), name2.c_str());
    }

    // detect if it is a prismatic joint constraint
    if ((strcmp_w_sub(name1, pris_b1, 3) or strcmp_w_sub(name1, pris_b2, 3)) and
        (strcmp_w_sub(name2, pris_b1, 3) or strcmp_w_sub(name2, pris_b2, 3))) {
      j_.con_idx.prismatic.push_back(i);
    }

    // detect if it is a revolute joint constraint
    if ((strcmp_w_sub(name1, rev_b1, 3) or strcmp_w_sub(name1, rev_b2, 3)) and
        (strcmp_w_sub(name2, rev_b1, 3) or strcmp_w_sub(name2, rev_b2, 3))) {
      j_.con_idx.revolute.push_back(i);
    }

    // detect if it is a palm joint constraint
    if ((strcmp_w_sub(name1, palm_b1, 3) or strcmp_w_sub(name1, palm_b2, 3)) and
        (strcmp_w_sub(name2, palm_b1, 3) or strcmp_w_sub(name2, palm_b2, 3))) {
      j_.con_idx.palm.push_back(i);
    }

    // set constraint to default value of false
    model->eq_active[i] = false;
  }

  if (debug) {
    print_vec(j_.con_idx.prismatic, "prismatic joint constraints");
    print_vec(j_.con_idx.revolute, "revolute joint constraints");
    print_vec(j_.con_idx.palm, "palm joint constraints");
  }
}

void set_all_constraints(mjModel* model, mjData* data, bool set_to)
{
  /* reset all constraints to false */

  for (int i : j_.con_idx.prismatic) {
    set_constraint(model, data, i, set_to);
  }
  for (int i : j_.con_idx.revolute) {
    set_constraint(model, data, i, set_to);
  }
  for (int i : j_.con_idx.palm) {
    set_constraint(model, data, i, set_to);
  }
}

void toggle_constraint(mjModel* model, mjData* data, int id)
{
  set_constraint(model, data, id, not model->eq_active[id]);
}

void set_constraint(mjModel* model, mjData* data, int id, bool set_as)
{
  /* toggle a constraint, if active lock the body in place relative to another */

  if (set_as) {

    // prepare and get indexes of position/rotation data
    mjtNum body1_pos[3];
    mjtNum body2_pos[3];
    mjtNum body1_rot[9];
    mjtNum body2_rot[9];
    int con_id = id * mjNEQDATA; // index where we insert constraint data
    int b1_pos_id = model->eq_obj1id[id] * 3;
    int b2_pos_id = model->eq_obj2id[id] * 3;
    int b1_rot_id = model->eq_obj1id[id] * 9;
    int b2_rot_id = model->eq_obj2id[id] * 9;

    // get the global rotation of the two bodies
    for (int i = 0; i < 9; i++) {
      body1_rot[i] = data->xmat[b1_rot_id + i];
      body2_rot[i] = data->xmat[b2_rot_id + i];
    }

    // get the global position of the two bodies
    for (int i = 0; i < 3; i++) {
      body1_pos[i] = data->xpos[b1_pos_id + i];
      body2_pos[i] = data->xpos[b2_pos_id + i];
    }

    // now find the local rotation, R12 = (R01)^T * R02
    mjtNum R12[9];
    mju_mulMatTMat(R12, body1_rot, body2_rot, 3, 3, 3);

    // subract the vectors from each other, then rotate into frame 1 (from 0)
    mjtNum vdiff[3];
    mjtNum vec12[3];
    vdiff[0] = body2_pos[0] - body1_pos[0];
    vdiff[1] = body2_pos[1] - body1_pos[1];
    vdiff[2] = body2_pos[2] - body1_pos[2];
    mju_mulMatVec(vec12, body1_rot, vdiff, 3, 3);

    // convert local rotation into a quaternion
    mjtNum quat12[4];
    mju_mat2Quat(quat12, R12);

    // insert this info into the constraint
    model->eq_data[con_id + 0] = vec12[0];
    model->eq_data[con_id + 1] = vec12[1];
    model->eq_data[con_id + 2] = vec12[2];
    model->eq_data[con_id + 3] = quat12[0];
    model->eq_data[con_id + 4] = quat12[1];
    model->eq_data[con_id + 5] = quat12[2];
    model->eq_data[con_id + 6] = quat12[3];

    // activate the constraint
    set_as = true;

    /* for testing
    std::cout << "Body 1 position is: ";
    for (int i = 0; i < 3; i++) {
      std::cout << body1_pos[i] << ", ";
    }
    std::cout << '\n';
    std::cout << "Body 2 position is: ";
    for (int i = 0; i < 3; i++) {
      std::cout << body2_pos[i] << ", ";
    }
    std::cout << '\n';
    std::cout << "Body 1 rotation is: ";
    mjtNum b1Quat[4];
    mju_mat2Quat(b1Quat, body1_rot);
    for (int i = 0; i < 4; i++) {
      std::cout << b1Quat[i] << ", ";
    }
    std::cout << '\n';
    std::cout << "Body 2 rotation is: ";
    mjtNum b2Quat[4];
    mju_mat2Quat(b2Quat, body2_rot);
    for (int i = 0; i < 4; i++) {
      std::cout << b2Quat[i] << ", ";
    }
    std::cout << '\n';
    std::cout << "Constraint data is: ";
    for (int i = 0; i < 7; i++) {
      std::cout << model->eq_data[con_id + i] << ", ";
    }
    std::cout << "\n\n";
    */
    
  }

  // set the constraint either active or inactive
  model->eq_active[id] = set_as;
}

void target_constraint(mjModel* model, mjData* data, int id, bool set_as, int type)
{
  /* set a constraint to send a motor to a target position */

  /* THIS FUNCTION IS UNFINISHED AND NOT CURRENTLY IN USE */

  if (set_as) {

    // int con_id = id * mjNEQDATA; // index where we insert constraint data

    // prepare and get indexes of position/rotation data
    mjtNum body1_pos[3];
    mjtNum body2_pos[3];
    mjtNum body1_rot[9];
    mjtNum body2_rot[9];
    int con_id = id * mjNEQDATA; // index where we insert constraint data
    int b1_pos_id = model->eq_obj1id[id] * 3;
    int b2_pos_id = model->eq_obj2id[id] * 3;
    int b1_rot_id = model->eq_obj1id[id] * 9;
    int b2_rot_id = model->eq_obj2id[id] * 9;

    // get the global rotation of the two bodies
    for (int i = 0; i < 9; i++) {
      body1_rot[i] = data->xmat[b1_rot_id + i];
      body2_rot[i] = data->xmat[b2_rot_id + i];
    }

    // get the global position of the two bodies
    for (int i = 0; i < 3; i++) {
      body1_pos[i] = data->xpos[b1_pos_id + i];
      body2_pos[i] = data->xpos[b2_pos_id + i];
    }

    // now find the local rotation, R12 = (R01)^T * R02
    mjtNum R12[9];
    mju_mulMatTMat(R12, body1_rot, body2_rot, 3, 3, 3);

    // subract the vectors from each other, then rotate into frame 1 (from 0)
    mjtNum vdiff[3];
    mjtNum vec12[3];
    vdiff[0] = body2_pos[0] - body1_pos[0];
    vdiff[1] = body2_pos[1] - body1_pos[1];
    vdiff[2] = body2_pos[2] - body1_pos[2];
    mju_mulMatVec(vec12, body1_rot, vdiff, 3, 3);

    // convert local rotation into a quaternion
    mjtNum quat12[4];
    mju_mat2Quat(quat12, R12);

    // insert this info into the constraint
    model->eq_data[con_id + 0] = vec12[0];
    model->eq_data[con_id + 1] = vec12[1];
    model->eq_data[con_id + 2] = vec12[2];
    model->eq_data[con_id + 3] = quat12[0];
    model->eq_data[con_id + 4] = quat12[1];
    model->eq_data[con_id + 5] = quat12[2];
    model->eq_data[con_id + 6] = quat12[3];

    // gripper prismatic
    if (type == 0) {
      model->eq_data[con_id + 0] = 0;
      model->eq_data[con_id + 1] = target_.end.x;
      model->eq_data[con_id + 2] = 0;
      model->eq_data[con_id + 3] = 0;
      model->eq_data[con_id + 4] = 0;
      model->eq_data[con_id + 5] = 0;
      model->eq_data[con_id + 6] = 1;
    }

    // // gripper revolute
    // else if (type == 1) {
    //   float axis[3] = { 0, 0, -1 };
    //   float half_angle = 0.5 * target_.end.get_revolute_joint();
    //   model->eq_data[con_id + 0] = 0;
    //   model->eq_data[con_id + 1] = 0;
    //   model->eq_data[con_id + 2] = 0;
    //   model->eq_data[con_id + 3] = axis[0] * std::sin(half_angle);
    //   model->eq_data[con_id + 4] = axis[1] * std::sin(half_angle);
    //   model->eq_data[con_id + 5] = axis[2] * std::sin(half_angle);
    //   model->eq_data[con_id + 6] = std::cos(half_angle);
    // }

    // // gripper palm
    // else if (type == 2) {
    //   model->eq_data[con_id + 0] = target_.end.z;
    //   model->eq_data[con_id + 1] = 0;
    //   model->eq_data[con_id + 2] = 0;
    //   model->eq_data[con_id + 3] = 0;
    //   model->eq_data[con_id + 4] = 0;
    //   model->eq_data[con_id + 5] = 0;
    //   model->eq_data[con_id + 6] = 1;
    // }

    /* for testing */
    std::cout << "Body 1 position is: ";
    for (int i = 0; i < 3; i++) {
      std::cout << body1_pos[i] << ", ";
    }
    std::cout << '\n';
    std::cout << "Body 2 position is: ";
    for (int i = 0; i < 3; i++) {
      std::cout << body2_pos[i] << ", ";
    }
    std::cout << '\n';
    std::cout << "Body 1 rotation is: ";
    mjtNum b1Quat[4];
    mju_mat2Quat(b1Quat, body1_rot);
    for (int i = 0; i < 4; i++) {
      std::cout << b1Quat[i] << ", ";
    }
    std::cout << '\n';
    std::cout << "Body 2 rotation is: ";
    mjtNum b2Quat[4];
    mju_mat2Quat(b2Quat, body2_rot);
    for (int i = 0; i < 4; i++) {
      std::cout << b2Quat[i] << ", ";
    }
    std::cout << '\n';
    std::cout << "Constraint data is: ";
    for (int i = 0; i < 7; i++) {
      std::cout << model->eq_data[con_id + i] << ", ";
    }
    std::cout << "\n\n";
    
  }

  // set the constraint either active or inactive
  model->eq_active[id] = set_as;
}

void keyframe(mjModel* model, mjData* data, std::string keyframe_name)
{
  /* overload with keyframe name */

  // set model to keyframe with the given name
  int key = mj_name2id(model, mjOBJ_KEY, keyframe_name.c_str());

  keyframe(model, data, key);
}

void keyframe(mjModel* model, mjData* data, int key)
{
  /* to run once to initialise the model to a desired state.
  Code is adapted from testspeed.cc line 117 */

  if (key >= 0) {
    data->time = model->key_time[key];
    mju_copy(data->qpos, model->key_qpos + key * model->nq, model->nq);
    mju_copy(data->qvel, model->key_qvel + key * model->nv, model->nv);
    mju_copy(data->act, model->key_act + key * model->na, model->na);
    // mju_copy(d->mocap_pos, m->key_mpos+i*3*m->nmocap, 3*m->nmocap);
		// mju_copy(d->mocap_quat, m->key_mquat+i*4*m->nmocap, 4*m->nmocap);
  }
  else {
    throw std::runtime_error("keyframe does not exist");
  }

  last_step_time_ = model->key_time[key];
}

void calibrate_reset(mjModel* model, mjData* data)
{
  /* find the equilibrium start position and set the simulation to that */

  static bool first_call = true;
  static std::vector<mjtNum> control_signals;
  static std::vector<mjtNum> qpos_positions;

  if (first_call) {

    constexpr int settle_number = 400; // found using mysimulate and visual inspection

    // loop to settle the simulation ~86ms
    for (int i = 0; i < settle_number; i++) {
      before_step(model, data);
      step(model, data);
      after_step(model, data);
    }

    // see where the joints have settled to equilibrium
    for (int i = 0; i < j_.num.panda; i++) {
      qpos_positions.push_back(*j_.to_qpos.panda[i]);
    }

    for (int i = 0; i < j_.num.base; i++) {
      qpos_positions.push_back(*j_.to_qpos.base[i]);
    }

    for (int i = 0; i < j_.num.gripper; i++) {
      qpos_positions.push_back(*j_.to_qpos.gripper[i]);
    }

    first_call = false;
  }

  int k = 0;

  // apply the equilibrium positions to the joints
  for (int i = 0; i < j_.num.panda; i++) { 
    (*j_.to_qpos.panda[i]) = qpos_positions[k]; 
    k += 1;
  }

  for (int i = 0; i < j_.num.base; i++) {
    (*j_.to_qpos.base[i]) = qpos_positions[k]; 
    k += 1;
  }

  for (int i = 0; i < j_.num.gripper; i++) {
    (*j_.to_qpos.gripper[i]) = qpos_positions[k]; 
    k += 1;
  }

}

void apply_tip_force(mjModel* model, mjData* data, double force, bool reset)
{
  /* apply a horizontal force to the tip of the finger */
  
  static bool first_call = true;
  static std::vector<int> tip_idx;
  static mjtNum tip_mat1[9];
  static mjtNum tip_mat2[9];
  static mjtNum tip_mat3[9];
  static std::vector<mjtNum*> tip_mat{ tip_mat1, tip_mat2, tip_mat3 };

  // the first time this function is called, find the tip body indexes
  if (first_call or reset) {

    if (reset) tip_idx.clear();

    // get the name of the last finger link (hook link is removed by mujoco as fixed joint)
    int tip_num = j_.num.per_finger + j_.dim.fixed_first_segment;
    std::string tip_name = "finger_{X}_segment_link_" + std::to_string(tip_num);

    for (int i = 0; i < model->nbody; i++) {
      std::string name = mj_id2name(model, mjOBJ_BODY, i);
      if (strcmp_w_sub(name, tip_name, 3)) {
        tip_idx.push_back(i);
      }
    }

    if (debug) {
      print_vec(tip_idx, "finger body tip_idx");
    }
    
    if (tip_idx.size() != 3) {
      throw std::runtime_error("tip_idx vector in apply_tip_force(...) has size != 3");
    }

    // find the starting body orientation
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 9; j++) {
        tip_mat[i][j] = data->xmat[tip_idx[i] * 9 + j];
      }
    }

    first_call = false;
  }

  // for lock the fingers in place
  for (int i : j_.con_idx.prismatic) {
    set_constraint(model, data, i, true);
  }
  for (int i : j_.con_idx.revolute) {
    set_constraint(model, data, i, true);
  }

  // loop through and apply force to fingertips
  for (int i = 0; i < 3; i++) {

    // prepare to apply force outwards on fingertips
    mjtNum fvec[3] = { 0, 0, -force };
    mjtNum rotfvec[3];

    // rotate into the tip frame to pull directly horizontal
    mju_mulMatVec(rotfvec, tip_mat[i], fvec, 3, 3);

    // apply force in cartesian space (joint space is qfrc_applied)
    data->xfrc_applied[tip_idx[i] * 6 + 0] = rotfvec[0];
    data->xfrc_applied[tip_idx[i] * 6 + 1] = rotfvec[1];
    data->xfrc_applied[tip_idx[i] * 6 + 2] = rotfvec[2];
  }
}

/* ----- simulation ----- */

void before_step(mjModel* model, mjData* data)
{
  /* before making a simulation step */

  mju_zero(data->ctrl, model->nu);
  mju_zero(data->qfrc_applied, model->nv);
  mju_zero(data->xfrc_applied, 6 * model->nbody);
}

void step(mjModel* model, mjData* data)
{
  /* make a simulation step */

  // mj_step(model, data);

  mj_step1(model, data);

  /* 
  To make the 'leadscrews' non-backdriveable, we want no forces to be
  transferred from the finger to the finger platform/joints. So we will
  try wiping any forces, and trust that the momentum forces are sufficiently
  small.

  The joints to wipe are either:
    finger_1_revolute_joint (1 + panda)
    finger_2_revolute_joint (12 + panda)
    finger_3_revolute_joint (23 + panda)
  or:
    finger_1_segment_joint_1 (2 + panda)
    finger_2_segment_joint_2 (13 + panda)
    finger_3_segment_joint_3 (24 + panda)

  */   

  static std::vector<int> to_wipe {
    j_.idx.gripper[j_.gripper.prismatic[0]],  // 0
    j_.idx.gripper[j_.gripper.prismatic[1]],  // 11
    j_.idx.gripper[j_.gripper.prismatic[2]],  // 22
    j_.idx.gripper[j_.gripper.revolute[0]],   // 1
    j_.idx.gripper[j_.gripper.revolute[1]],   // 12
    j_.idx.gripper[j_.gripper.revolute[2]],   // 23
  };

  control(model, data);   // since ctrl pntr not assigned

  for (int i = 0; i < to_wipe.size(); i++) {
    // all are (nv * 1), and nv = 34 for gripper, which = njnts
    // data->qfrc_passive[to_wipe[i]] = 0;  // passive force
    // data->efc_vel[to_wipe[i]] = 0;       // velocity in constraint space: J*qvel
    // data->efc_aref[to_wipe[i]] = 0;      // reference pseudo-acceleartion
    // data->qfrc_bias[to_wipe[i]] = 0;     // C(qpos, qvel)
    // data->cvel[to_wipe[i]] = 0;          // com-based velcotiy [3D rot; 3D tran]

    // data->qfrc_unc[to_wipe[i]] = 0;
    // data->qacc_unc[to_wipe[i]] = 0;
  }

  // // for testing, applly known force to the end of the finger
  // data->xfrc_applied[11 * 6 + 1] = 10;

  mj_step2(model, data);
  return;

  mj_fwdActuation(model, data);
  mj_fwdAcceleration(model, data);
  mj_fwdConstraint(model, data);


  std::vector<mjtNum> qfrc;
  // std::cout << "qfrc constraint is ";
  for (int i = 0; i < to_wipe.size(); i++) {

    // // wipe forces arising from constraints (contacts)
    // qfrc.push_back(data->qfrc_constraint[to_wipe[i]]);
    // data->qfrc_constraint[to_wipe[i]] = 0;
    // // std::cout << data->qfrc_constraint[to_wipe[i]] << " ";
  } 
  // std::cout << "\n";

  // int j = 0;
  // for (int i : j_.gripper.prismatic) {
  //   data->ctrl[j_.num.panda + i] += -qfrc[j];
  //   j += 1;
  // }
  // for (int i : j_.gripper.revolute) {
  //   data->ctrl[j_.num.panda + i] += -qfrc[j];
  //   j += 1;
  // }
  // mj_fwdActuation(model, data);
  // mj_fwdAcceleration(model, data);
  // mj_fwdConstraint(model, data);

  mj_sensorAcc(model, data);
  mj_checkAcc(model, data);

  // compare forward and inverse solutions if enabled
  // if( mjENABLED(mjENBL_FWDINV) )
  if (model->opt.enableflags and mjENBL_FWDINV)
      mj_compareFwdInv(model, data);

  mj_Euler(model, data);


}

void after_step(mjModel* model, mjData* data)
{
  /* after making a simulation step */

  // compute the contact forces on all bodies
  mj_rnePostConstraint(model, data);

  update_all(model, data);
}

/* ----- control ----- */

void control(const mjModel* model, mjData* data)
{
  /* Control function for mujoco */

  // disable this warning as added objects contribute nv but not nu
  if (false and model->nu != model->nv) {
    printf("Warning from Luke's control function: "
      "model nu (num ctrl inputs) does not equal nv (num DoF). "
      "nu = %i, nv = %i\n", model->nu, model->nv);
    throw std::runtime_error("nu != nv for your model");
  }

  if (j_.in_use.panda) {
    control_panda(model, data);
  }

  if (j_.in_use.gripper) {
    control_gripper(model, data, target_.next);
  }

  if (j_.in_use.base) {
    control_base(model, data);
  }
}

void control_panda(const mjModel* model, mjData* data)
{
  /* control the panda joints */

  // mju_scl(data->ctrl, data->qvel, -0.1 * 100, model->nv);

}

void control_gripper(const mjModel* model, mjData* data, Gripper& target)
{
  /* control the gripper joints */

  double u = 0;

  double force_lim = 100;

  int n = j_.num.panda + j_.num.base;

  // input the control signals
  for (int i : j_.gripper.prismatic) {
    u = ((*j_.to_qpos.gripper[i]) - target.x) * j_.ctrl.kp.x 
      + (*j_.to_qvel.gripper[i]) * j_.ctrl.kd.x;
    // if (abs(u) > force_lim) {
    //   std::cout << "x frc limited from " << u << " to ";
    //   u = force_lim * sign(u);
    //   std::cout << u << '\n';
    // }
    data->ctrl[n + i] = -u;
  }

  for (int i : j_.gripper.revolute) {
    u = ((*j_.to_qpos.gripper[i]) - target.th) * j_.ctrl.kp.y 
      + (*j_.to_qvel.gripper[i]) * j_.ctrl.kd.y;
    // if (abs(u) > force_lim) {
    //   std::cout << "y frc limited from " << u << " to ";
    //   u = force_lim * sign(u);
    //   std::cout << u << '\n';
    // }
    data->ctrl[n + i] = -u;
  }
  
  for (int i : j_.gripper.palm) {
    u = ((*j_.to_qpos.gripper[i]) - target.z) * j_.ctrl.kp.z 
      + (*j_.to_qvel.gripper[i]) * j_.ctrl.kd.z;
    // if (abs(u) > force_lim) {
    //   std::cout << "z frc limited from " << u << " to ";
    //   u = force_lim * sign(u);
    //   std::cout << u << '\n';
    // }
    data->ctrl[n + i] = -u;
  }
}

void control_base(const mjModel* model, mjData* data)
{
  /* control the base joint */

  double u = 0;
  int n = j_.num.panda;

  if (j_.num.base != 1) {
    throw std::runtime_error("base dof does not equal 1");
  }

  for (int i = 0; i < j_.num.base; i++) {
    u = ((*j_.to_qpos.base[i]) - target_.base[i]) * j_.ctrl.base_kp
      + (*j_.to_qvel.base[i]) * j_.ctrl.base_kd;
    data->ctrl[n + i] = -u;
  }
}

void update_state(const mjModel* model, mjData* data)
{
  /* update our record of the model state */

  /* this function has been replaced by accessing qpos and qvel directly via
  their pointers. It can still be used for helpful printing and debugging
  but should never be called in the main loop */

  if (j_.in_use.panda) {
    for (int i = 0; i < j_.num.panda; i++) {
      j_.qpos.panda[i] = data->qpos[j_.qposadr.panda[i]];
      j_.qvel.panda[i] = data->qvel[j_.qveladr.panda[i]];
    }
  }

  if (j_.in_use.gripper) {
    for (int i = 0; i < j_.num.gripper; i++) {
      j_.qpos.gripper[i] = data->qpos[j_.qposadr.gripper[i]];
      j_.qvel.gripper[i] = data->qvel[j_.qveladr.gripper[i]];
    }
  }

  if (j_.in_use.finger) {
    for (int i = 0; i < j_.num.finger; i++) {
      j_.qpos.finger[i] = data->qpos[j_.qposadr.finger[i]];
      j_.qvel.finger[i] = data->qvel[j_.qveladr.finger[i]];
    }
  }

  if (j_.in_use.base) {
    for (int i = 0; i < j_.num.base; i++) {
      j_.qpos.base[i] = data->qpos[j_.qposadr.base[i]];
      j_.qvel.base[i] = data->qvel[j_.qveladr.base[i]];
    }
  }

  // // report state for testing
  // j_.print_qpos();
  // j_.print_qvel();

}

void print_state(const mjModel* model, mjData* data)
{
  /* print the qpos of all the joints */

  update_state(model, data);
  j_.print_qpos();
  j_.print_qvel();
}

void update_all(mjModel* model, mjData* data)
{
  /* update the state of everything in the simulation */

  // update_state(model, data); // NO LONGER NEEDED
  // update_objects(model, data); // NO LONGER NEEDED

  if (j_.ctrl.stepper) {
    update_stepper(model, data);
  }
  else {
    throw std::runtime_error("non-stepper not implemented");
  }

  // // for testing
  // update_state(model, data);
  // j_.print_qpos();
  // target_.end.print();
}

void update_stepper(mjModel* model, mjData* data)
{
  /* update the gripper joint positions and determine equilibirum/target_reached 
  assuming a stepper motor style */

  constexpr static bool log_test_data = true;

  bool stepped = false;

  if (data->time > last_step_time_ + j_.ctrl.time_per_step) {

    // we can optionally log position data to see motor response to steps
    if (log_test_data) {
      Gripper temp;
      temp.set_xyz_m_rad(*j_.to_qpos.gripper[0], *j_.to_qpos.gripper[1], *j_.to_qpos.gripper[6]);
      target_.timedata.add(data->time);
      target_.target_stepperx.add(target_.next.x * 1e6);
      target_.target_steppery.add(target_.next.y * 1e6);
      target_.target_stepperz.add(target_.next.z * 1e6);
      target_.target_basez.add(target_.base[0] * 1e6);
      target_.actual_stepperx.add(temp.x * 1e6);
      target_.actual_steppery.add(temp.y * 1e6);
      target_.actual_stepperz.add(temp.z * 1e6);
      target_.actual_basez.add(*j_.to_qpos.base[0] * 1e6);
    }

    // check if motors are moving, if not, lock them
    update_constraints(model, data);

    // apply a step to any motors still not at the target
    last_step_time_ = data->time;
    target_.next.step_to(target_.end, j_.ctrl.num_steps);
    stepped = true;

    // uncomment these to see ratio of steps to waits
    // std::cout << "step!\n";
  }
  else {
    // std::cout << "wait-";
  }
}

void update_objects(const mjModel* model, mjData* data)
{
  /* update the position of the objects in the simulation */

  for (int i = 0; i < oh_.names.size(); i++) {
    if (oh_.in_use[i]) {
      oh_.qpos[i].update(model, data, oh_.qposadr[i]);
    }
  }

  // // for testing
  // QPos test = get_object_qpos();
  // printf("qpos is xyz (%.3f, %.3f, %.3f)\n", test.x, test.y, test.z);

  // get_object_contact_forces(model, data);
}

void update_constraints(mjModel* model, mjData* data)
{
  /* control toggling of constraints which log motor positions once they finish
  moving */

  static bool old_x = true;
  static bool old_y = true;
  static bool old_z = true;

  bool new_x = target_.x_moving();
  bool new_y = target_.y_moving();
  bool new_z = target_.z_moving();

  // // FOR TESTING - this work was not finished
  // for (int i : j_.con_idx.prismatic) {
  //   target_constraint(model, data, i, not new_x, 0);
  // }
  // for (int i : j_.con_idx.revolute) {
  //   target_constraint(model, data, i, not new_y, 1);
  // }
  // for (int i : j_.con_idx.palm) {
  //   target_constraint(model, data, i, not new_z, 2);
  // }

  // return;

  if (new_x != old_x) {
    if (new_x) {
      // constraint enable is true
      for (int i : j_.con_idx.prismatic) {
        set_constraint(model, data, i, false);
      }
    }
    else {
      // constraint enable is false
      for (int i : j_.con_idx.prismatic) {
        set_constraint(model, data, i, true);
      }
    }
    old_x = new_x;
  }
  
  if (new_y != old_y) {
    if (new_y) {
      // constraint enable is true
      for (int i : j_.con_idx.revolute) {
        set_constraint(model, data, i, false);
      }
    }
    else {
      // constraint enable is false
      for (int i : j_.con_idx.revolute) {
        set_constraint(model, data, i, true);
      }
    }
    old_y = new_y;
  }

  if (new_z != old_z) {
    if (new_z) {
      // constraint enable is true
      for (int i : j_.con_idx.palm) {
        set_constraint(model, data, i, false);
      }
    }
    else {
      // constraint enable is false
      for (int i : j_.con_idx.palm) {
        set_constraint(model, data, i, true);
      }
    }
    old_z = new_z;
  }

  // // for testing
  // std::cout << "Prismatic constraints: ";
  // for (int i : j_.con_idx.prismatic) {
  //   std::cout << (int) model->eq_active[i] << ", ";
  // }
  // std::cout << "\n";
}

/* ----- set gripper target ----- */

bool set_gripper_target_step(int x, int y, int z)
{
  /* set a step target for the gripper motors */

  target_.last_robot = Target::Robot::gripper;

  // return false if target is outside motor limits
  return target_.end.set_xyz_step(x, y, z);
}

bool set_gripper_target_m(double x, double y, double z)
{
  /* set a motor state target for the gripper, returns true when reached */

  target_.last_robot = Target::Robot::gripper;

  // return false if the target is outside motor limits
  return target_.end.set_xyz_m(x, y, z);
}

bool set_gripper_target_m_rad(double x, double th, double z)
{
  /* sets a joint state target for the gripper, returns true when reached */

  target_.last_robot = Target::Robot::gripper;

  // return false if the target is outside motor limits
  return target_.end.set_xyz_m(x, th, z);
}

bool move_gripper_target_step(int x, int y, int z)
{
  /* adjust the gripper target by the indicated number of steps */

  target_.last_robot = Target::Robot::gripper;

  // return false if the target is outside motor limits
  return target_.end.set_xyz_step(target_.end.step.x + x, target_.end.step.y + y, 
    target_.end.step.z + z);
}

bool move_gripper_target_m(double x, double y, double z)
{
  /* adjust gripper target by the indicated distances in metres */

  target_.last_robot = Target::Robot::gripper;

  // return false if the target is outside motor limits
  return target_.end.set_xyz_m(target_.end.x + x, target_.end.y + y,
    target_.end.z + z);
}

bool move_gripper_target_m_rad(double x, double th, double z)
{
  /* adjust the gripper joint values */

  target_.last_robot = Target::Robot::gripper;

  // return false if the target is outside motor limits
  return target_.end.set_xyz_m_rad(target_.end.x + x, target_.end.th + th, 
    target_.end.z + z);
}

bool move_base_target_m(double x, double y, double z)
{
  /* move the base target in x, y, z */

  target_.last_robot = Target::Robot::panda;

  /* only z motion currently implemented */
  target_.base[0] += z;

  // check limits, currently only z movements supported
  double z_min = luke::Target::base_z_min;
  double z_max = luke::Target::base_z_max;

  // check if we have gone outside the limits
  if (target_.base[0] > z_max) {
    target_.base[0] = z_max;
    return false;
  }
  if (target_.base[0] < z_min) {
    target_.base[0] = z_min;
    return false;
  }

  return true;
}

void print_target()
{
  std::cout << "The target gripper state is:";
  target_.end.print();
}

void update_target()
{
  target_.end.update();
}

/* ----- sensing ------ */

gfloat read_armadillo_gauge(const mjData* data, int finger)
{
  /* read the virtual strain gauge for one finger */

  arma::vec joint_values(j_.num.per_finger, arma::fill::zeros);

  // get the joint values for this finger
  for (int i = 0; i < j_.num.per_finger; i++) {
    joint_values(i) = 
      data->qpos[j_.idx.finger[i + finger * j_.num.per_finger]];
  }

  // next convert this into X and Y coordinates
  arma::vec cumulative(j_.num.per_finger, arma::fill::zeros);
  arma::mat finger_xy(j_.num.per_finger + 1, 2, arma::fill::zeros);

  // if first segment is locked
  if (j_.dim.fixed_first_segment)
    finger_xy(0, 0) = j_.dim.segment_length;
  else
    finger_xy(0, 0) = 0;

  for (int i = 0; i < j_.num.per_finger; i++) {

    // keep cumulative total of angular sum
    if (i == 0) {
      cumulative(i) = joint_values(i);
    }
    else {
      cumulative(i) = cumulative(i - 1) + joint_values(i);
    }

    // calculate cartesian coordinates of each joint
    finger_xy(i + 1, 0) = finger_xy(i, 0)
      + j_.dim.segment_length * std::cos(cumulative(i));
    finger_xy(i + 1, 1) = finger_xy(i, 1) 
      + j_.dim.segment_length * std::sin(cumulative(i));
  }

  // polyfit a cubic curve to these joint positions
  arma::vec coeff = arma::polyfit(finger_xy.col(0), finger_xy.col(1), j_.gauge.order);

  // evaluate y at the gauge x position
  gfloat y = 0.0;
  for (int i = 0; i <= j_.gauge.order; i++) {
    y += coeff(i) * std::pow(j_.gauge.xpos, j_.gauge.order - i);
  }

  /* The equation relating the force P to deflection delta is:
        delta = (P * l^3) / (3 * E * I)
     We can approximate the P / 3EI as proportional to our strain, k:
        delta = k * l^3
            k = delta / l^3
     Hence we our approximated strain, k, as our gauge reading
  */

  // // calculate the approximated gauge reading
  // gfloat k = y / j_.gauge.xpos_cubed;

  // // transfer to SI units for force (optional)
  // // THIS IS NOT ACCURATE as L^3 only applies at tip of beam
  // // BETTER TO NOT PROCESS TO SI as it removes this functions dependence on j_.dim.EI
  // gfloat P = k * (3 * j_.dim.EI);

  /* the SI result is not accurate because the finger stiffness is not
  accurate (here we do not have the right E). However, tuning the stiffness
  to be perfect is not helpful as the simulation can become unstable and the
  interaction with the simulated 'motors' is already not realistic */

  /* Finally, we want to scale this data. To get an idea of the size of k, lets
     take some default values:
      xpos = 50mm
      y_max == xpos -> this is from the finger bending to 45deg
     Hence, an absolute maximum value would be:
      k = xpos / cbrt(xpos)
      k = 0.136
     Lets scale this to the range -100, +100

     Lets scale the data to the range -1, +1. First, what is the maximum
     expected force?

     k = P / 3EI
     E = 200e9 Pa
     I = 1/12 * hb^3 where h = 28mm and b = 0.9mm
     I = 1.71e-12 m^4

     hence, with maximum expected force of 20N, we get:

     k = 20 / (3 * 200 * 1.71) * (10e-12 * 10e-9)
     k = 19.6 m^-2
  */

  // k *= (100.0 / 0.136);

  // return y value in millimeters, unprocessed (OLD: return P;)
  return y * 1000;
}

gfloat verify_armadillo_gauge(const mjData* data, int finger,
  std::vector<float>& vec_joint_x, std::vector<float>& vec_joint_y,
  std::vector<float>& vec_coefficients, std::vector<float>& vec_errors)
{
  /* read the virtual strain gauge for one finger */

  arma::vec joint_values(j_.num.per_finger, arma::fill::zeros);

  // get the joint values for this finger
  for (int i = 0; i < j_.num.per_finger; i++) {
    // joint_values(i) = 
    //   data->qpos[j_.idx.finger[i + finger * j_.num.per_finger]];

    joint_values(i) = *j_.to_qpos.finger[i + finger * j_.num.per_finger];
  }

  // next convert this into X and Y coordinates
  arma::vec cumulative(j_.num.per_finger, arma::fill::zeros);
  arma::mat finger_xy(j_.num.per_finger + 1, 2, arma::fill::zeros);

  // if first segment is locked
  if (j_.dim.fixed_first_segment)
    finger_xy(0, 0) = j_.dim.segment_length;
  else
    finger_xy(0, 0) = 0;

  for (int i = 0; i < j_.num.per_finger; i++) {

    // keep cumulative total of angular sum
    if (i == 0) {
      cumulative(i) = joint_values(i);
    }
    else {
      cumulative(i) = cumulative(i - 1) + joint_values(i);
    }

    // calculate cartesian coordinates of each joint
    finger_xy(i + 1, 0) = finger_xy(i, 0)
      + j_.dim.segment_length * std::cos(cumulative(i));
    finger_xy(i + 1, 1) = finger_xy(i, 1) 
      + j_.dim.segment_length * std::sin(cumulative(i));
  }

  // polyfit a cubic curve to these joint positions
  arma::vec coeff = arma::polyfit(finger_xy.col(0), finger_xy.col(1), j_.gauge.order);

  // evaluate y at the gauge x position
  gfloat y = 0.0;
  for (int i = 0; i <= j_.gauge.order; i++) {
    y += coeff(i) * std::pow(j_.gauge.xpos, j_.gauge.order - i);
  }

  /* The equation relating the force P to deflection delta is:
        delta = (P * l^3) / (3 * E * I)
     We can approximate the P / 3EI as proportional to our strain, k:
        delta = k * l^3
            k = delta / l^3
     Hence we our approximated strain, k, as our gauge reading
  */

  // // calculate the approximated gauge reading
  // gfloat k = y / j_.gauge.xpos_cubed;

  // // transfer to SI units for force (optional)
  // gfloat P = k * (3 * j_.dim.EI);

  /* the SI result is not accurate because the finger stiffness is not
  accurate (here we do not have the right E). However, tuning the stiffness
  to be perfect is not helpful as the simulation can become unstable and the
  interaction with the simulated 'motors' is already not realistic */

  /* Finally, we want to scale this data. To get an idea of the size of k, lets
     take some default values:
      xpos = 50mm
      y_max == xpos -> this is from the finger bending to 45deg
     Hence, an absolute maximum value would be:
      k = xpos / cbrt(xpos)
      k = 0.136
     Lets scale this to the range -100, +100

     Lets scale the data to the range -1, +1. First, what is the maximum
     expected force?

     k = P / 3EI
     E = 200e9 Pa
     I = 1/12 * hb^3 where h = 28mm and b = 0.9mm
     I = 1.71e-12 m^4

     hence, with maximum expected force of 20N, we get:

     k = 20 / (3 * 200 * 1.71) * (10e-12 * 10e-9)
     k = 19.6 m^-2
  */

  // k *= (100.0 / 0.136);

  /* ----- only difference between read/verfiy is as follows ----- */

  int num_points = j_.num.per_finger + 1;
  int num_coeff = j_.gauge.order + 1;

  vec_joint_x.resize(num_points);
  vec_joint_y.resize(num_points);
  vec_errors.resize(num_points);
  vec_coefficients.resize(num_coeff);

  // save the joint angles
  for (int i = 0; i < num_points; i++) {
    vec_joint_x[i] = finger_xy(i, 0);
    vec_joint_y[i] = finger_xy(i, 1);
  }

  // save the cubic coefficients
  for (int i = 0; i < num_coeff; i++) {
    vec_coefficients[i] = coeff(i);
  }

  // evaluate the predicted y position and resulting error
  float cum_error = 0;
  for (int i = 0; i < num_points; i++) {
    float y = 0.0;
    for (int j = 0; j < num_coeff; j++) {
      y += vec_coefficients[j] * std::pow(vec_joint_x[i], j_.gauge.order - j);
    }
    float error = vec_joint_y[i] - y;
    vec_errors[i] = error;
    cum_error += abs(error);
  }

  return cum_error / num_points;

  /* ----- end read/verify differences ----- */
}

gfloat verify_small_angle_model(const mjData* data, int finger,
  std::vector<float>& joint_angles, std::vector<float>& joint_pred,
  std::vector<float>& pred_x, std::vector<float>& pred_y, std::vector<float>& theory_y,
  std::vector<float>& theory_x_curve, std::vector<float>& theory_y_curve,
  float force, float finger_stiffness)
{
  /* evaluate the difference in joint angle between the actual and model
  predicted values */

  int ffs =  j_.dim.fixed_first_segment;

  int N = j_.num.per_finger;
  joint_angles.resize(N);
  joint_pred.resize(N);
  pred_x.resize(N + 1);
  pred_y.resize(N + 1);
  theory_y.resize(N + 1 + ffs);
  std::vector<float> theory_x(N + 1 + ffs);

  int theory_N = 50;
  float theory_step = j_.dim.finger_length / (float) theory_N;
  theory_x_curve.resize(theory_N);
  theory_y_curve.resize(theory_N);

  std::vector<float> joint_errors(N);

  float cum_error = 0;
  float cum_pred_angle = 0;

  if (j_.dim.fixed_first_segment) {
    pred_x[0] = j_.dim.segment_length;
    theory_x[1] = j_.dim.segment_length;
    theory_y[1] = (force * std::pow(theory_x[1], 3)) / (3 * j_.dim.EI); // WRONG EQUATION!
  }
  else {
    pred_x[0] = 0; 
  }

  pred_y[0] = 0;
  theory_x[0] = 0;
  theory_y[0] = 0;
  theory_x_curve[0] = 0;
  theory_y_curve[0] = 0;

  // get the joint values for this finger
  for (int i = 0; i < N; i++) {

    // determine the joint stiffness of this joint
    int n = i + 1;
    float c = j_.dim.joint_stiffness[i];

    // actual joint values
    joint_angles[i] = *j_.to_qpos.finger[i + finger * N];

    // predicted joint values
    joint_pred[i] = ((N - n + 1) * force * j_.dim.finger_length) / (N * c);

    // joint angle error
    joint_errors[i] = joint_angles[i] - joint_pred[i];
    cum_error += abs(joint_errors[i]);

    // predicted xy positions
    cum_pred_angle += joint_pred[i];
    pred_x[i + 1] = pred_x[i] + j_.dim.segment_length * std::cos(cum_pred_angle);
    pred_y[i + 1] = pred_y[i] + j_.dim.segment_length * std::sin(cum_pred_angle);

    // // theory y position
    // theory_x[i + 1 + ffs] = theory_x[i + ffs] + j_.dim.segment_length;
    // theory_y[i + 1 + ffs] = (force * std::pow(theory_x[i + 1 + ffs], 3)) / (3 * j_.dim.EI); 

    // basic theory attempt 2
    double theory_factor = (force) / (6.0 * j_.dim.EI);
    double x = (i + 1) * j_.dim.segment_length;
    theory_x[i + 1 + ffs] = x;
    theory_y[i + 1 + ffs] = theory_factor * (-std::pow(x, 3) + 3 * j_.dim.finger_length * std::pow(x, 2));  
    
  }

  fill_theory_curve(theory_x_curve, theory_y_curve, force, theory_N);

  // // approximate free end tangent angle
  // double B = (force * std::pow(j_.dim.finger_length, 2)) / (j_.dim.EI);
  // double phi_0 = 0.5 * B * (1.0 - (1.0/12.0) * std::pow(B, 2));
  // double gamma = M_PI_2;

  // // factors for basic theory
  // double f1 = -force / 6.0;
  // double f2 = (force * std::pow(j_.dim.finger_length, 2)) / 2.0;
  // double f3 = -(force * std::pow(j_.dim.finger_length, 3)) / 3.0;

  // // create theory curve
  // for (int i = 0; i < theory_N - 1; i++) {

  //   // // proportional to L cubed, basic
  //   // theory_x_curve[i + 1] = theory_x_curve[i] + theory_step;
  //   // theory_y_curve[i + 1] = (force * std::pow(theory_x_curve[i + 1], 3)) / (3 * j_.dim.EI); 

  //   // basic theory attempt 2
  //   // double x = j_.dim.finger_length * (1.0 - (i / (float) (theory_N - 2)));
  //   // theory_x_curve[i + 1] = theory_x_curve[i] + theory_step;
  //   // theory_y_curve[i + 1] = (-1.0 / j_.dim.EI) * ((f1 * std::pow(x, 3)) + (f2 * x) + f3); 

  //   double theory_factor = (force / (6.0 * j_.dim.EI));
  //   double x = j_.dim.finger_length * ((i / (float) (theory_N - 2)));
  //   theory_x_curve[i + 1] = x;
  //   theory_y_curve[i + 1] = theory_factor * (-std::pow(x, 3) + 3 * j_.dim.finger_length * std::pow(x, 2));  

  //   // following Bisshopp end angle approximation ... how to get cartesian?

  //   // // // Batista paper, analytical solution
  //   // #ifndef LUKE_PREVENT_BOOST
  //   //   double s = 1.0 - (i / (float) (theory_N - 2));
  //   //   double M0 = 0.0;
  //   //   double alpha = 1.8785;
  //   //   luke_boost::ArcPoint p = luke_boost::get_point(s, force, M0, j_.dim.finger_length,
  //   //     j_.dim.EI, alpha);
  //   //   theory_x_curve[i + 1] = p.x;
  //   //   theory_y_curve[i + 1] = p.y;
  //   // #endif
  // }

  // return average error
  return (gfloat) cum_error / N;
}

void fill_theory_curve(std::vector<float>& theory_X, std::vector<float>& theory_Y, 
  float force, int num)
{
  /* take two vectors (which are wiped) and fill them with the theory curve, this
  is basic bending theory for Euler-Bernoulli beam. Force should be given in NEWTONS */

  theory_X.clear();
  theory_Y.clear();

  theory_X.resize(num);
  theory_Y.resize(num);

  // create theory curve
  for (int i = 0; i < num; i++) {

    double theory_factor = (force / (6.0 * j_.dim.EI));
    double x = j_.dim.finger_length * ((i / (float) (num - 2)));
    theory_X[i] = x;
    theory_Y[i] = theory_factor * (-std::pow(x, 3) + 3 * j_.dim.finger_length * std::pow(x, 2)); 
  }
}

std::vector<gfloat> get_gauge_data(const mjModel* model, mjData* data)
{
  /* Get the position of the finger joints */

  if (not j_.in_use.finger) {
    printf("Error: gauge data has been request without using segments\n");
    return std::vector<gfloat>{0, 0, 0};
  }

  std::vector<gfloat> readings(3);

  // use armadillo to detect finger bending
  if (j_.gauge.use_armadillo_gauges) {
    for (int i = 0; i < 3; i++) {
      readings[i] = read_armadillo_gauge(data, i);
    }
  }
  // use fingertip forces
  else {
    Forces forces = get_object_forces(model, data);
    readings[0] = (gfloat) forces.all.finger1_local[1];
    readings[1] = (gfloat) forces.all.finger2_local[1];
    readings[2] = (gfloat) forces.all.finger3_local[1];
  }

  return readings;  
}

gfloat get_palm_force(const mjModel* model, mjData* data)
{
  /* get the axial force on the palm */

  return (gfloat) oh_.get_palm_force(model, data);
}

std::vector<gfloat> get_panda_state(const mjData* data)
{
  /* Get the state of the panda joints */

  if (not j_.in_use.panda) return std::vector<gfloat>{ 0 };

  std::vector<gfloat> joint_values(j_.num.panda);

  for (int i = 0; i < j_.num.panda; i++) {
    joint_values[i] = data->qpos[j_.idx.panda[i]];
  }

  return joint_values;
}

std::vector<gfloat> get_gripper_state(const mjData* data)
{
  /* Get the state of the gripper joints */

  if (not j_.in_use.gripper) return std::vector<gfloat>{ 0 };

  std::vector<gfloat> joint_values(j_.num.gripper);

  for (int i = 0; i < j_.num.gripper; i++) {
    joint_values[i] = data->qpos[j_.idx.gripper[i]];
  }

  return joint_values;
}

std::vector<gfloat> get_target_state()
{
  /* Get the state of the gripper target */

  return target_.get_target_m();

  // // old code
  // // target_.end.update();
  // gfloat x = target_.end.x;
  // gfloat y = target_.end.y; // or theta?
  // gfloat z = target_.end.z;

  // std::vector<gfloat> target_joint_values = { x, y, z };

  // return target_joint_values;
}

int get_N() 
{
  return j_.num.per_finger + j_.dim.fixed_first_segment;
}

float get_finger_thickness()
{
  return j_.dim.finger_thickness;
}

std::vector<luke::gfloat> get_stiffnesses()
{
  return j_.dim.joint_stiffness;
}

/* ----- environment ----- */

Gripper get_gripper_target()
{
  /* get the target state of the gripper */

  return target_.end;
}

std::vector<std::string> get_objects()
{
  /* get the names of objects in the simulation scene */

  std::vector<std::string> objects;

  for (int i = 0; i < oh_.names.size(); i++) {
    if (oh_.in_use[i]) {
      objects.push_back(oh_.names[i]);
    }
  }
  
  return objects;
}

void reset_object(mjModel* model, mjData* data)
{
  /* reset the live object to its starting position outside the task area */

  oh_.reset_live(model, data);
}

void spawn_object(mjModel* model, mjData* data, std::string name, QPos pose)
{
  /* overload to pass object name not index */

  for (int i = 0; i < oh_.names.size(); i++) {
    if (oh_.names[i] == name and oh_.in_use[i]) {
      spawn_object(model, data, i, pose);
      return;
    }
  }

  throw std::runtime_error("name not found");
}

void spawn_object(mjModel* model, mjData* data, int idx, QPos pose)
{
  /* spawn an object in the simulation with the given pose, and always wipes qvel */

  oh_.spawn_object(model, data, idx, pose);
}

QPos get_object_qpos(mjModel* model, mjData* data)
{
  /* returns the position of the live object in the simulation */

  if (oh_.live_object == -1) {
    QPos empty;
    return empty;
    throw std::runtime_error("no live object");
  }

  if (oh_.live_object >= oh_.names.size())
    throw std::runtime_error("live object exceeds number of named objects");

  // // for testing
  // QPos test = get_object_qpos();
  // printf("qpos is xyz (%.3f, %.3f, %.3f)\n", test.x, test.y, test.z);

  // old, when qpos was updated
  // return oh_.qpos[oh_.live_object];

  return oh_.get_live_qpos(model, data);
}

Forces get_object_forces(const mjModel* model, mjData* data)
{
  /* get the contact forces on the live object */

  // use the faster version of the extract_forces() function
  return oh_.extract_forces(model, data);
}

Forces_faster get_object_forces_faster(const mjModel* model, mjData* data)
{
  /* get the contact forces on the live object */

  // use the faster version of the extract_forces() function
  return oh_.extract_forces_faster(model, data);
}

void set_object_colour(mjModel* model, std::vector<float> rgba)
{
  /* set the colour of the main object */

  oh_.set_colour(model, rgba);
}

void set_ground_colour(mjModel* model, std::vector<float> rgba)
{
  /* randomise the colour of the ground */

  oh_.set_ground_colour(model, rgba);
}

void randomise_all_colours(mjModel* model, std::shared_ptr<std::default_random_engine> generator)
{
  /* randomise the colour of every object but not the ground*/

  oh_.randomise_all_colours(model, generator);
}

void default_colours(mjModel* model)
{
  /* restore colours to default values */

  oh_.default_colours(model);

  std::vector<float> rgba_default { 0.5, 0.5, 0.5, 1.0 };
  set_finger_colour(model, rgba_default, 1);
  set_finger_colour(model, rgba_default, 2);
  set_finger_colour(model, rgba_default, 3);
  set_finger_colour(model, rgba_default, 4); // 4 means palm
}

void set_finger_colour(mjModel* model, std::vector<float> rgba, int finger_num)
{
  /* set the segmented finger all to one colour, finger_num = 1,2,3, or 4 (4 means palm) */

  if (rgba.size() != 3 and rgba.size() != 4) {
    throw std::runtime_error("set_finger_colour() not given a rgba vector of size 3 or 4");
  }

  // make a pointer to a vector so we can flexibly swap between different fingers
  std::vector<int>* fptr;

  // assign this pointer to one of the following options
  if (finger_num == 1) fptr = &j_.geom_idx.finger1;
  else if (finger_num == 2) fptr = &j_.geom_idx.finger2;
  else if (finger_num == 3) fptr = &j_.geom_idx.finger3;
  else if (finger_num == 4) fptr = &j_.geom_idx.palm;
  else {
    throw std::runtime_error("set_finger_colour() expects finger_num equal to either 1,2,3 or 4 (4 is palm)");
  }

  // loop through the vector we assigned and update the colour
  for (int i : *fptr) {
    model->geom_rgba[i * 4 + 0] = rgba[0];
    model->geom_rgba[i * 4 + 1] = rgba[1];
    model->geom_rgba[i * 4 + 2] = rgba[2];

    // if an a value is given, set this too
    if (rgba.size() == 4)
      model->geom_rgba[i * 4 + 3] = rgba[3];
  }
}

/* ----- misc ----- */

int last_action_robot()
{
  /* which robot was the last robot used for a change in target.
        0 = none,
        1 = gripper,
        2 = panda
  */

  return target_.last_robot;
}

bool is_sim_unstable(mjModel* model, mjData* data)
{
  /* detect if the simulation has become unstable */

  if (data->warning[mjWARN_BADQACC].number > 0) {
    return true;
  }

  return false;
}

void print_stiffnesses()
{
  print_vec(get_stiffnesses(), "Joint stiffnesses");
}

} // namespace luke