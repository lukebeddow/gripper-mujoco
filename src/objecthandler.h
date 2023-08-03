#ifndef OBJECTHANDLER_H_
#define OBJECTHANDLER_H_

#include <string>
#include <vector>
#include <array>
#include <chrono>
#include <random>
#include <memory>

#include "mujoco.h"
#include "customtypes.h"

namespace luke
{

typedef std::chrono::high_resolution_clock time_;

// global scene handler for the objects in the scene
struct ObjectHandler {

  /* member variables */

  // define important names
  constexpr static char finger1_body_name[] = "finger_1";
  constexpr static char finger2_body_name[] = "finger_2";
  constexpr static char finger3_body_name[] = "finger_3";
  constexpr static char palm_body_name[] = "palm";
  constexpr static char gnd_geom_name[] = "ground_geom";
  constexpr static char geom_suffix[] = "_geom";    // object body name converts to geom name with this added

  std::vector<std::string> names;
  std::vector<bool> in_use;
  std::vector<int> idx;
  std::vector<QPos> qpos;
  std::vector<QPos> reset_qpos;
  std::vector<int> qposadr;
  std::vector<int> qveladr;
  std::vector<int> geom_id;

  int live_object;
  std::string live_geom;

  // body ids for fingers and palm
  int f1_idx;
  int f2_idx;
  int f3_idx;
  int pm_idx;

  // geom id of the ground
  int gnd_geom_id;

  // extra tolerance when spawning objects in metres
  constexpr static double z_spawn_tolerance =  1e-6;

  // collision groups
  constexpr static unsigned long COL_none = 0;  // 0...000
  constexpr static unsigned long COL_main = 1;  // 0...001
  constexpr static unsigned long COL_dead = 2;  // 0...010
  constexpr static unsigned long COL_all = 3;   // 0...011
  
  constexpr static bool debug = false;

  /* custom types */
  struct Contact {

    // store all information about a contact
    myNum frame;
    myNum local_force_vec;
    myNum global_force_vec;
    std::string name1;
    std::string name2;

    // contact options
    struct {
      bool object;
      bool finger1;
      bool finger2;
      bool finger3;
      bool palm;
      bool ground;
    } with;

    bool with_any() {
      if (with.object or 
          with.finger1 or 
          with.finger2 or 
          with.finger3 or
          with.palm or 
          with.ground) {
        return true;
      }
      return false;
    }
    bool involves(std::string idstr) {
      if (name1.substr(0, idstr.length()) == idstr or
          name2.substr(0, idstr.length()) == idstr) {
        return true;
      }
      return false;
    }
    void check_involves(std::string object_geom) {
      with.object = involves(object_geom);
      with.finger1 = involves("finger_1");
      with.finger2 = involves("finger_2");
      with.finger3 = involves("finger_3");
      with.palm = involves("palm");
      with.ground = involves("ground");
    }
    void print() {
      print_involves();
      std::cout << "contact frame:\n"; frame.print();
      std::cout << "local forces:\n"; local_force_vec.print();
      std::cout << "global forces:\n"; global_force_vec.print();
    }
    void print_involves() {
      std::cout << "contact between " << name1 << " and " << name2
        << ", which involves object = " << with.object << ", fingers " 
        << with.finger1 << " " << with.finger2 << " " << with.finger3
        << ", palm " << with.palm << ", ground " << with.ground << '\n';
    }
  };

  /* member functions */

  // constructor and initilisation
  ObjectHandler();
  void init(mjModel* model, mjData* data);
  void resize();
  void remove_collisions(mjModel* model, mjData* data);
  void settle_objects(mjModel* model, mjData* data);
  void overwrite_keyframe(mjModel* model, mjData* data, int keyid = 0);

  // manipulate object in the scene
  bool check_idx(int idx);
  void set_live(mjModel* model, int idx);
  void move_object(mjData* data, int idx, QPos pose);
  void reset_object(mjModel* model, mjData* data, int idx) ;
  void reset_live(mjModel* model, mjData* data);
  void spawn_object(mjModel* model, mjData* data, int idx, QPos pose);
  QPos get_live_qpos(mjModel* model, mjData* data);

  // get object information
  myNum get_object_net_force(const mjModel* model, mjData* data);
  rawNum get_object_net_force_faster(const mjModel* model, mjData* data);
  std::vector<Contact> get_all_contacts(const mjModel* model, mjData* data);
  myNum rotate_vector(myNum force_vec, double x_rot, double y_rot, double z_rot);
  double get_palm_force(const mjModel* model, mjData* data);
  Forces extract_forces(const mjModel* model, mjData* data);
  Forces_faster extract_forces_faster(const mjModel* model, mjData* data);
  bool check_contact_forces(const mjModel* model, mjData* data);

  // set object properties
  void set_colour(mjModel* model, std::vector<float> rgba);
  void set_all_colours(mjModel* model, std::vector<float> rgba);
  void set_ground_colour(mjModel* model, std::vector<float> rgba);
  void set_friction(mjModel* model, mjtNum sliding_friction);
  void set_friction(mjModel* model, std::vector<mjtNum> friction_triple);
  void randomise_all_colours(mjModel* model, 
    std::shared_ptr<std::default_random_engine> generator);
  void default_colours(mjModel* model);

  // misc
  void print();
};

} // namespace luke

#endif // OBJECTHANDLER_H_