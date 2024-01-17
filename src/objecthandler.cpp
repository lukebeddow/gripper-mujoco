#include "objecthandler.h"

namespace luke
{

// due to excentric c++11, need to declare the consexpr char[] to prevent linker error
constexpr char ObjectHandler::finger1_body_name[];
constexpr char ObjectHandler::finger2_body_name[];
constexpr char ObjectHandler::finger3_body_name[];
constexpr char ObjectHandler::palm_body_name[];
constexpr char ObjectHandler::gnd_geom_name[];
constexpr char ObjectHandler::geom_suffix[];

ObjectHandler::ObjectHandler() 
{
  /* constructor */

  // // set default - no live object
  // live_object = -1;
}

void ObjectHandler::init(mjModel* model, mjData* data)
{
  /* initialise the objects in the scene, must be done after keyframe */

  // wipe to defaults, in case previously init()
  names.clear();
  in_use.clear();
  idx.clear();
  qpos.clear();
  reset_qpos.clear();
  qposadr.clear();
  qveladr.clear();
  geom_id.clear();
  live_objects.clear();
  live_geoms.clear();
  xyz_values.clear();

  // // we cannot start with an object live
  // live_object = -1;
  // live_geom = "";

  // first, run the simluation to let all the objects settle
  settle_objects(model, data);

  // scan every joint in the model to find freejoints (for the objects)
  for (int i = 0; i < model->njnt; i++) {
    auto x = mj_id2name(model, mjOBJ_JOINT, i);
    if (model->jnt_type[i] == mjJNT_FREE) {
      std::string namestr(x);
      names.push_back(namestr);
    }
  }

  // now resize all the class vectors with the number of object names
  resize();

  // go through all the object names
  for (int i = 0; i < names.size(); i++) {

    // get the body index (not joint)
    int j = mj_name2id(model, mjOBJ_BODY, names[i].c_str());

    // if the body is present in the model
    if (j != -1) {
      in_use[i] = true;
      idx[i] = j;
      qposadr[i] = model->jnt_qposadr[model->body_jntadr[j]];
      qpos[i].update(model, data, qposadr[i]);
      reset_qpos[i] = qpos[i];
      qveladr[i] = model->jnt_dofadr[model->body_jntadr[j]];
      std::string geom_name = names[i] + geom_suffix;
      geom_id[i] = mj_name2id(model, mjOBJ_GEOM, geom_name.c_str());
    }
    else {
      in_use[i] = false;
      idx[i] = -1;
      qposadr[i] = -1;
      qpos[i].update(model, data, -1);
      reset_qpos[i] = qpos[i];
      qveladr[i] = -1;
      geom_id[i] = -1;
    }

    // override the reset qpos as we have issues with quaternions with nan values
    reset_qpos[i].qx = 0;
    reset_qpos[i].qy = 0;
    reset_qpos[i].qz = 0;
    reset_qpos[i].qw = 1;

    // get the xyz bounding box of the object
    bool debug_xyz = false;
    std::string numeric_name = "Task object " + std::to_string(i);
    int numeric_i = mj_name2id(model, mjOBJ_NUMERIC, numeric_name.c_str());
    if (numeric_i == -1) {
      if (debug_xyz) std::cout << numeric_name << " not found in objecthandler, ignoring\n";
      xyz_values[i].x = 0;
      xyz_values[i].y = 0;
      xyz_values[i].z = 0;
    }
    else {
      double x = model->numeric_data[model->numeric_adr[numeric_i] + 0];
      double y = model->numeric_data[model->numeric_adr[numeric_i] + 1];
      double z = model->numeric_data[model->numeric_adr[numeric_i] + 2];
      if (debug_xyz) std::cout << numeric_name 
        << " has (x, y, z) >> (" << x << ", " << y << ", " << z << ")\n";
      xyz_values[i].x = x;
      xyz_values[i].y = y;
      xyz_values[i].z = z;
    }
  }

  // get the body ids for fingers and palm (for contact forces)
  f1_idx = mj_name2id(model, mjOBJ_BODY, finger1_body_name);
  f2_idx = mj_name2id(model, mjOBJ_BODY, finger2_body_name);
  f3_idx = mj_name2id(model, mjOBJ_BODY, finger3_body_name);
  pm_idx = mj_name2id(model, mjOBJ_BODY, palm_body_name);
  
  // get the geom id for the ground (for changing colour)
  gnd_geom_id = mj_name2id(model, mjOBJ_GEOM, gnd_geom_name);

  // update the reset keyframe now objects are settled
  overwrite_keyframe(model, data);

  // set up collision types and affinities
  remove_collisions(model, data);

  // apply default object visibility
  set_object_visibility(model, object_visibility);
  
  // for testing
  if (debug) print();
}

void ObjectHandler::remove_collisions(mjModel* model, mjData* data)
{
  /* We want four collision groups:
      ground: collide with everything
      main: collide only with main and ground
      dead: collide only with the ground
      off: collide with nothing

  We can achieve this by using four conaffinities and four contypes:
      ground: type = 11, aff = 11   
      main:   type = 01, aff = 01   (default setting in xml, int = 1)
      dead:   type = 10, aff = 10
      off:    type = 00, aff = 00   (disabled setting in xml, int = 0)

  This is because a collision is only valid if the following is true:
    (contype1 & conaffinity2) || (contype2 & conaffinity1)

  ie: contype of one geom and conaff of other both have a common bit set to one.

  In terms of integers, we have the four groups:
      off = 0     -> COL_none
      main = 1    -> COL_main
      dead = 2    -> COL_dead
      ground = 3  -> COL_all

  In this function, we leave disabled collisions off, then place the ground in
  its new group and the objects in their new group
  */

  for (int i = 0; i < model->ngeom; i++) {

    // some geoms have contacts already disabled (eg visuals), preserve this
    if (model->geom_contype[i] == COL_none or model->geom_conaffinity[i] == COL_none) {
      continue;
    }

    // check the name of this geom
    auto name1 = mj_id2name(model, mjOBJ_GEOM, i);

    // if it doesn't have a name
    if (not name1) {
      // leave in main simulation group
      // std: :cout << "not named " << model->geom_contype[i] << '\n';
      continue;
    }

    // turn the name to a string
    std::string namestr(name1); 

    // if it is the ground geom, set to 11 (already at 01)
    if (namestr == gnd_geom_name) {
      model->geom_contype[i] = COL_all;
      model->geom_conaffinity[i] = COL_all;
      // std::cout << "ground geom " << model->geom_contype[i] << '\n';
      continue;
    }

    // if it is one of our objects
    bool object = false;
    for (int j = 0; j < names.size(); j++) {
      if (i == geom_id[j]) {
        object = true;
        break;
      }
    }
    if (object) {
      // set to dead group 10 (currently at 01)
      model->geom_contype[i] = COL_dead;
      model->geom_conaffinity[i] = COL_dead;
      // std::cout << "object " << model->geom_contype[i] << '\n';
      continue;
    }

    // otherwise, leave in main simulation group
    model->geom_contype[i] = COL_main;
    model->geom_conaffinity[i] = COL_main;
  }
}

void ObjectHandler::settle_objects(mjModel* model, mjData* data)
{
  /* step the simulation to ensure objects have settled */

  // compile time constant
  static constexpr int settle_step_num = 0;

  auto start_time = time_::now();

  for (int i = 0; i < settle_step_num; i++) {
    mj_step(model, data);
  }

  auto end_time = time_::now();

  if (debug)
    std::cout << "Time taken for settling is " 
      << (std::chrono::duration_cast<std::chrono::milliseconds>
        (end_time - start_time).count()) / 1000.0 << " seconds\n";
}

void ObjectHandler::overwrite_keyframe(mjModel* model, mjData* data, int keyid)
{
  /* overwrites the keyframe with the current object xyz positions */

  // loop through all the objects
  for (int i = 0; i < names.size(); i++) {
    // overwrite the keyframe with the current qpos data for x, y, z
    if (in_use[i]) {
      model->key_qpos[qposadr[i] + 0 + keyid * model->nq] = data->qpos[qposadr[i] + 0];
      model->key_qpos[qposadr[i] + 1 + keyid * model->nq] = data->qpos[qposadr[i] + 1];
      model->key_qpos[qposadr[i] + 2 + keyid * model->nq] = data->qpos[qposadr[i] + 2];
    }
  }
}

// helpful functions
void ObjectHandler::resize() 
{
  int size = names.size();
  in_use.resize(size);
  idx.resize(size);
  qpos.resize(size);
  qposadr.resize(size);
  qveladr.resize(size);
  reset_qpos.resize(size);
  geom_id.resize(size);
  xyz_values.resize(size);
}

bool ObjectHandler::check_idx(int idx) 
{
  if (idx < 0 or idx >= names.size()) {
    std::cout << "object idx is out of range - nothing done\n";
    return false;
  }
  if (not in_use[idx]) {
    std::cout << "object idx is not in_use - nothing done\n";
    return false;
  }
  return true;
}

void ObjectHandler::set_live(mjModel* model, int idx)
{
  /* add another live object - no safety checks! */

  // live_object = idx;
  // live_geom = names[idx] + "_geom";

  live_objects.push_back(idx);
  live_geoms.push_back(names[idx] + "_geom");

  // enable collisions, set to 01
  model->geom_contype[geom_id[idx]] = COL_main;
  model->geom_conaffinity[geom_id[idx]] = COL_main;

  // turn on visibility
  model->geom_rgba[geom_id[idx] * 4 + 3] = 1;
}

bool ObjectHandler::is_live(int live_idx)
{
  /* return if an object idx is included in the live objects */

  for (int i = 0; i < live_objects.size(); i++) {
    if (live_idx == live_objects[i]) return true;
  }

  return false;
}

int ObjectHandler::get_live_geom_index(int live_idx)
{
  /* get the index of the live geom for this live_idx
  
  so if we have live_objects [4, 2, 9] and
                live_geoms ["x", "y", "z"]
                
  this function will return:
    0 for live_idx = 4
    1 for live_idx = 2
    2 for live_idx = 9
    error otherwise */

  for (int i = 0; i < live_objects.size(); i++) {
    if (live_idx == live_objects[i]) {
      return i;
    }
  }

  throw std::runtime_error("ObjectHandler::get_live_geom_index() was given a live_idx that is not a live object");
}

void ObjectHandler::move_object(mjData* data, int idx, QPos pose) 
{
  /* move an object to a pose, always wipes qvel */

  // std::cout << "QPos to move the object to:\n";
  // pose.print();

  // move the object
  data->qpos[qposadr[idx]] = pose.x;
  data->qpos[qposadr[idx] + 1] = pose.y;
  data->qpos[qposadr[idx] + 2] = pose.z;
  data->qpos[qposadr[idx] + 3] = pose.qx;
  data->qpos[qposadr[idx] + 4] = pose.qy;
  data->qpos[qposadr[idx] + 5] = pose.qz;
  data->qpos[qposadr[idx] + 6] = pose.qw;
  
  // now wipe the object qvel
  data->qvel[qveladr[idx]] = 0;
  data->qvel[qveladr[idx] + 1] = 0;
  data->qvel[qveladr[idx] + 2] = 0;
  data->qvel[qveladr[idx] + 3] = 0; 
  data->qvel[qveladr[idx] + 4] = 0;
  data->qvel[qveladr[idx] + 5] = 0;
}

void ObjectHandler::reset_object(mjModel* model, mjData* data, int idx) 
{
  /* reset an object to its starting pose - does not check if its live! */

  if (not check_idx(idx)) return;

  move_object(data, idx, reset_qpos[idx]);

  // // if we have reset the live object
  // if (idx == live_object) {
  //   live_object = -1;
  //   live_geom = "";

  //   // disable collisions,set to 10
  //   model->geom_contype[geom_id[idx]] = COL_dead;
  //   model->geom_conaffinity[geom_id[idx]] = COL_dead;

  //   // check if we should make the object invisible
  //   if (not object_visibility) {
  //     model->geom_rgba[geom_id[idx] * 4 + 3] = 0;
  //   }
  // }
}

void ObjectHandler::reset_live(mjModel* model, mjData* data) 
{
  // if (live_object == -1) return;
  // reset_object(model, data, live_object);

  if (live_objects.size() == 0) return;

  for (int live_idx : live_objects) {

    reset_object(model, data, live_idx);

    // disable collisions,set to 10
    model->geom_contype[geom_id[live_idx]] = COL_dead;
    model->geom_conaffinity[geom_id[live_idx]] = COL_dead;

    // check if we should make the object invisible
    if (not object_visibility) {
      model->geom_rgba[geom_id[live_idx] * 4 + 3] = 0;
    }
  }

  // remove all the live objects
  live_objects.clear();
  live_geoms.clear();
}

QPos ObjectHandler::spawn_object(mjModel* model, mjData* data, int new_idx, QPos pose) 
{
  /* spawn a new live object */

  if (not check_idx(new_idx)) {
    QPos empty;
    return empty;
  }

  // // first reset the live object
  // reset_live(model, data);

  // override the z height of the given pose to use reset value
  pose.z = reset_qpos[new_idx].z + z_spawn_tolerance;

  // move this object into the given pose
  move_object(data, new_idx, pose);

  // make the object live if it isn't already
  if (not is_live(new_idx)) set_live(model, new_idx);

  QPos new_qpos;
  new_qpos.update(model, data, qposadr[new_idx]);

  return new_qpos;
}

std::vector<QPos> ObjectHandler::get_live_qpos(mjModel* model, mjData* data)
{
  /* get the qpos of the live objects */

  if (live_objects.size() == 0)
    throw std::runtime_error("ObjectHandler::get_live_qpos() failed as there is no live object\n");

  std::vector<QPos> out(live_objects.size());

  for (int i = 0; i < live_objects.size(); i++) {
    out[i].update(model, data, qposadr[live_objects[i]]);
  }
  
  return out;
}

Vec3 ObjectHandler::get_object_xyz(int obj_idx)
{
  /* return the xyz bounding box of a given object index */

  if (not check_idx(obj_idx)) {
    throw std::runtime_error("ObjectHandler::get_object_xyz() recieved out of bounds 'obj_idx'");
  }

  return xyz_values[obj_idx];
}

// print functions
void ObjectHandler::print()
{
  std::cout << "object scene s_: ";
  if (names.size() == 0) {
    std::cout << "no named objects\n";
    return;
  }
  std::cout << "live objects = ";
  for (int i = 0; i < live_objects.size(); i++) {
     std::cout << i + 1 << ". has idx " << live_objects[i] << "; ";
  }
  for (int i = 0; i < names.size(); i++) {
    printf("\n\t%d. name: %s, in_use: %d, idx: %d, qposadr: %d, qveladr: %d, geom_id: %d, ", 
      i, names[i].c_str(), (int)in_use[i], idx[i], qposadr[i], qveladr[i], geom_id[i]);
    printf("xyz = (%.3f, %.3f, %.3f) quat = (%.3f, %.3f, %.3f, %.3f)",
      qpos[i].x, qpos[i].y, qpos[i].z, qpos[i].qx, qpos[i].qy, qpos[i].qz,
      qpos[i].qw);
  }
  std::cout << "\n";
}

// inspect contact forces
myNum ObjectHandler::get_object_net_force(const mjModel* model, mjData* data, int live_idx)
{
  /* get the net force in world frame on the live object */

  // if (live_object == -1) {
  //   myNum empty;
  //   return empty;
  // }

  // get the net forces (pointer encompasses whole array)
  mjtNum* F = data->cfrc_ext + idx[live_idx] * 6;

  // cfrc_ext stores forces as [torque, force], so swap order
  mjtNum F_swap[6] = { F[3], F[4], F[5], F[0], F[1], F[2] };

  // create my mjtNum wrapper, nr=6, nc=1
  myNum out(F_swap, 6, 1);

  // for testing
  // out.print();

  return out;
}

rawNum ObjectHandler::get_object_net_force_faster(const mjModel* model, mjData* data, int live_idx)
{
  /* get the net force in world frame on the live object */

  // if (live_object == -1) {
  //   rawNum empty;
  //   return empty;
  // }

  // get the net forces (pointer encompasses whole array)
  mjtNum* F = data->cfrc_ext + idx[live_idx] * 6;

  // cfrc_ext stores forces as [torque, force], so swap order
  mjtNum F_swap[6] = { F[3], F[4], F[5], F[0], F[1], F[2] };

  // create my mjtNum wrapper, nr=6, nc=1
  rawNum out(F_swap, 6, 1);

  // for testing
  // out.print();

  return out;
}

std::vector<ObjectHandler::Contact> 
ObjectHandler::get_all_contacts(const mjModel* model, mjData* data)
{
  /* get all the contacts on the live objects */

  // create arrays (matrices) to store mujoco data
  mjtNum frame[9];            // contact frame rotation matrix wrt base (3x3)
  mjtNum global_force_vec[6]; // contact forces/torques in global frame (6x1)
  mjtNum local_force_vec[6];  // contact forces/torques in local frame (6x1)
  mjtNum force_vec[3];        // local forces only (3x1)
  mjtNum torque_vec[3];       // local torques only (3x1)
  mjtNum force_global[3];     // global forces only (3x1)
  mjtNum torque_global[3];    // global torques only (3x1)

  // containers to output the data from this function
  Contact contact;
  std::vector<Contact> contact_vec;

  constexpr bool debug_fcn = true;

  if (debug_fcn) std::cout << "the number of contacts is " << data->ncon << '\n';

  // loop through all of the contacts at this time step
  for (int i = 0; i < data->ncon; i++) {

    // check that the contact is not excluded
    if (data->contact[i].exclude != 0) {
      // 0: include, 1: in gap, 2: fused, 3: equality, 4: no dofs
      continue;
    }

    // check that the contact involves the live object
    auto name1 = mj_id2name(model, mjOBJ_GEOM, data->contact[i].geom1);
    auto name2 = mj_id2name(model, mjOBJ_GEOM, data->contact[i].geom2);

    if (debug_fcn) {
      std::printf("name1 is %s, name2 is %s, live_object_geoms are: ", 
        name1, name2);
      for (std::string x : live_geoms) std::cout << x << ", ";
      std::cout << "\n";
    }

    // if we have a null pointer (non-named geom)
    if (not name1 or not name2) {
      continue;
    }

    // convert to std::string
    std::string strname1(name1);
    std::string strname2(name2);

    // save object names
    contact.name1 = strname1;
    contact.name2 = strname2;

    // check which objects/geoms this contact involves
    contact.check_involves(live_geoms);

    // get the contact frame wrt to world frame (needs to be transposed)
    mju_transpose(frame, data->contact[i].frame, 3, 3);

    // get the local forces in this contact (requires mj_rnePostConstraint(...))
    mj_contactForce(model, data, i, local_force_vec);

    // get the force/torque vectors - note they are reverse order [t; f]
    force_vec[0] = local_force_vec[0];  // x force
    force_vec[1] = local_force_vec[1];  // y force
    force_vec[2] = local_force_vec[2];  // z force
    torque_vec[0] = local_force_vec[3]; // x torque
    torque_vec[1] = local_force_vec[4]; // y torque
    torque_vec[2] = local_force_vec[5]; // z torque

    // rotate force/torque vectors into the global frame
    mju_mulMatVec(force_global, frame, force_vec, 3, 3);
    mju_mulMatVec(torque_global, frame, torque_vec, 3, 3);

    // save the global force vector with forces, then torques
    global_force_vec[0] = force_global[0];
    global_force_vec[1] = force_global[1];
    global_force_vec[2] = force_global[2];
    global_force_vec[3] = torque_global[0];
    global_force_vec[4] = torque_global[1];
    global_force_vec[5] = torque_global[2];
    
    // save information in our return container
    contact.frame.init(frame, 3, 3);
    contact.local_force_vec.init(local_force_vec, 6, 1);
    contact.global_force_vec.init(global_force_vec, 6, 1);
    contact.name1 = name1;
    contact.name2 = name2;

    contact_vec.push_back(contact);

    if (debug_fcn) {
      contact.print();
      contact.print_involves();
    }
  }

  return contact_vec;
}

myNum ObjectHandler::rotate_vector(myNum force_vec, double x_rot, double y_rot,
  double z_rot)
{
  /* return the magnitude of a force give a 6x1 force vector. Note that this
  function ignores the three torque components */

  // three force components
  mjtNum f_local[3] { force_vec[0], force_vec[1], force_vec[2] };

  // global forces we will get
  mjtNum f_rotated[3];
  mju_zero3(f_rotated);

  // for rotation results
  mjtNum rotzy[9];
  mjtNum rotzyx[9];

  // x rotation matrix
  mjtNum rotx[9] { 1,          0,           0,
                  0, cos(x_rot), -sin(x_rot),
                  0, sin(x_rot),  cos(x_rot) };

  // y rotation matrix
  mjtNum roty[9] {  cos(y_rot), 0, sin(y_rot),
                            0, 1,          0,
                  -sin(y_rot), 0, cos(y_rot) };

  // z rotation matrix
  mjtNum rotz[9] { cos(z_rot), -sin(z_rot), 0,
                  sin(z_rot),  cos(z_rot), 0,
                  0,           0,          1 };

  // combine all rotations
  mju_mulMatMat(rotzy, rotz, roty, 3, 3, 3);
  mju_mulMatMat(rotzyx, rotzy, rotx, 3, 3, 3);

  // apply rotation to local forces
  mju_mulMatVec(f_rotated, rotzyx, f_local, 3, 3);

  // save in wrapper class
  myNum out(f_rotated, 3, 1);

  return out;
}

double ObjectHandler::get_palm_force(const mjModel* model, mjData* data)
{
  /* get only the axial force on the palm */

  std::vector<Contact> contacts = get_all_contacts(model, data);
  myNum palm_global(6, 1);
  myNum palm_local(3, 1);

  for (int i = 0; i < contacts.size(); i++) {

    if (contacts[i].with.palm) {
      palm_global += contacts[i].global_force_vec;
    }
  }

  myNum r4(&data->xmat[pm_idx * 9], 3, 3);

  palm_local = palm_global.rotate3_by(r4.transpose());

  // take only the axial load, [0]
  return palm_local[0];
}

Forces_faster ObjectHandler::extract_forces_faster(const mjModel* model, mjData* data)
{
  /* a new version of extract_forces() which intends to be faster in the following:
        - use raw c arrays instead of copying them into myNum
          -> they are only copied at the end of the function for output purposes
        - compute only the bare minimum calculations on contacts
          -> ordinarily, every contact is gathered and then parsed
          -> in this function, only relevant contacts are gathered
  this function will be more error prone but aims to increase speed */

  // if (live_object == -1) {
  //   Forces_faster empty;
  //   return empty;
  // }

  if (live_objects.size() == 0) {
    Forces_faster empty;
    return empty;
  }

  // create arrays (matrices) to store mujoco contact data
  mjtNum frame[9];            // contact frame rotation matrix wrt base (3x3)
  mjtNum global_force_vec[6]; // contact forces/torques in global frame (6x1)
  mjtNum local_force_vec[6];  // contact forces/torques in local frame (6x1)
  mjtNum force_vec[3];        // local forces only (3x1)
  mjtNum torque_vec[3];       // local torques only (3x1)
  mjtNum force_global[3];     // global forces only (3x1)
  mjtNum torque_global[3];    // global torques only (3x1)

  /* create raw arrays which will then be converted to myNum and outputed,
  these correspond to the fields of the Forces struct */

  // // contact vectors with the object
  // mjtNum obj_glob_sum[6] = {};
  // mjtNum obj_glob_f1[6] = {};
  // mjtNum obj_glob_f2[6] = {};
  // mjtNum obj_glob_f3[6] = {};
  // mjtNum obj_glob_palm[6] = {};
  // mjtNum obj_glob_gnd[6] = {};
  // mjtNum obj_loc_f1[3] = {};
  // mjtNum obj_loc_f2[3] = {};
  // mjtNum obj_loc_f3[3] = {};
  // mjtNum obj_loc_palm[3] = {};

  // vector of the above for multiple live objects
  std::vector<ObjSumMatrices> objfrc(live_objects.size());

  // contact vectors taking into account everything
  mjtNum all_glob_f1[6] = {};
  mjtNum all_glob_f2[6] = {};
  mjtNum all_glob_f3[6] = {};
  mjtNum all_glob_palm[6] = {};
  mjtNum all_loc_f1[3] = {};
  mjtNum all_loc_f2[3] = {};
  mjtNum all_loc_f3[3] = {};
  mjtNum all_loc_palm[3] = {};

  // contact vectors with the ground
  mjtNum gnd_glob_f1[6] = {};
  mjtNum gnd_glob_f2[6] = {};
  mjtNum gnd_glob_f3[6] = {};
  mjtNum gnd_loc_f1[3] = {};
  mjtNum gnd_loc_f2[3] = {};
  mjtNum gnd_loc_f3[3] = {};

  // all the above need to be zeroed

  // contact container, used for the 'check_involves' member function
  Contact contact;

  // loop through all of the contacts at this time step
  for (int i = 0; i < data->ncon; i++) {

    // check that the contact is not excluded
    if (data->contact[i].exclude != 0) {
      // 0: include, 1: in gap, 2: fused, 3: equality, 4: no dofs
      continue;
    }

    // check that the contact involves the live object
    auto name1 = mj_id2name(model, mjOBJ_GEOM, data->contact[i].geom1);
    auto name2 = mj_id2name(model, mjOBJ_GEOM, data->contact[i].geom2);

    // for testing: check the geom names
    // std::printf("name1 is %s, name2 is %s, live_geom is %s\n", 
    //   name1, name2, live_geom.c_str());

    // if we have a null pointer (non-named geom)
    if (not name1 or not name2) {
      continue;
    }

    // save object names (as std::string)
    contact.name1 = name1;
    contact.name2 = name2;

    // check which objects/geoms this contact involves
    contact.check_involves(live_geoms);

    // is this contact with entities we care about?
    if (not contact.with_any()) continue;

    // get the contact frame wrt to world frame (needs to be transposed)
    mju_transpose(frame, data->contact[i].frame, 3, 3);

    // get the local forces in this contact (requires mj_rnePostConstraint(...))
    mj_contactForce(model, data, i, local_force_vec);

    // get the force/torque vectors - note they are reverse order [t; f]
    force_vec[0] = local_force_vec[0];  // x force
    force_vec[1] = local_force_vec[1];  // y force
    force_vec[2] = local_force_vec[2];  // z force
    torque_vec[0] = local_force_vec[3]; // x torque
    torque_vec[1] = local_force_vec[4]; // y torque
    torque_vec[2] = local_force_vec[5]; // z torque

    // rotate force/torque vectors into the global frame
    mju_mulMatVec(force_global, frame, force_vec, 3, 3);
    mju_mulMatVec(torque_global, frame, torque_vec, 3, 3);

    // save the global force vector with forces, then torques
    global_force_vec[0] = force_global[0];
    global_force_vec[1] = force_global[1];
    global_force_vec[2] = force_global[2];
    global_force_vec[3] = torque_global[0];
    global_force_vec[4] = torque_global[1];
    global_force_vec[5] = torque_global[2];

    // if the contact involves the object
    for (int i = 0; i < live_objects.size(); i++) {
      if (contact.with.live_object[i]) {
        mju_addTo(objfrc[i].obj_glob_sum, global_force_vec, 6);
        if (contact.with.finger1) {
          mju_addTo(objfrc[i].obj_glob_f1, global_force_vec, 6);
        }
        if (contact.with.finger2) {
          mju_addTo(objfrc[i].obj_glob_f2, global_force_vec, 6);
        }
        if (contact.with.finger3) {
          mju_addTo(objfrc[i].obj_glob_f3, global_force_vec, 6);
        }
        if (contact.with.palm) {
          mju_addTo(objfrc[i].obj_glob_palm, global_force_vec, 6);
        }
        if (contact.with.ground) {
          mju_addTo(objfrc[i].obj_glob_gnd, global_force_vec, 6);
        }
      }
    }

    // if the contact involves the gripper
    if (contact.with.finger1) {
      mju_addTo(all_glob_f1, global_force_vec, 6);
      if (contact.with.ground) {
        mju_addTo(gnd_glob_f1, global_force_vec, 6);
      }
    }
    if (contact.with.finger2) {
      mju_addTo(all_glob_f2, global_force_vec, 6);
      if (contact.with.ground) {
        mju_addTo(gnd_glob_f2, global_force_vec, 6);
      }
    }
    if (contact.with.finger3) {
      mju_addTo(all_glob_f3, global_force_vec, 6);
      if (contact.with.ground) {
        mju_addTo(gnd_glob_f3, global_force_vec, 6);
      }
    }
    if (contact.with.palm) {
      mju_addTo(all_glob_palm, global_force_vec, 6);
    }

    // // for testing
    // contact.print();
    // contact.print_involves();
  }

  // get the rotation matrices for each finger body
  mjtNum r1[9];
  mjtNum r2[9];
  mjtNum r3[9];
  mjtNum r4[9];
  mju_transpose(r1, &data->xmat[f1_idx * 9], 3, 3);
  mju_transpose(r2, &data->xmat[f2_idx * 9], 3, 3);
  mju_transpose(r3, &data->xmat[f3_idx * 9], 3, 3);
  mju_transpose(r4, &data->xmat[pm_idx * 9], 3, 3);

  // rotate the finger vectors from the world frame to the finger body frame
  // x = axial force, y = tangential force (ie strain gauge force)
  // these forces are observed to all be +ve under object contact (outwards bend)
  // mju_mulMatMat(result, lhs, rhs, lhs.nr, lhs.nc, rhs.nc);
  for (int i = 0; i < live_objects.size(); i++) {
    mju_mulMatMat(objfrc[i].obj_loc_f1, r1, objfrc[i].obj_glob_f1, 3, 3, 1);
    mju_mulMatMat(objfrc[i].obj_loc_f2, r2, objfrc[i].obj_glob_f2, 3, 3, 1);
    mju_mulMatMat(objfrc[i].obj_loc_f3, r3, objfrc[i].obj_glob_f3, 3, 3, 1);
    mju_mulMatMat(objfrc[i].obj_loc_palm, r4, objfrc[i].obj_glob_palm, 3, 3, 1); // moved from below
  }
  
  mju_mulMatMat(all_loc_f1, r1, all_glob_f1, 3, 3, 1);
  mju_mulMatMat(all_loc_f2, r2, all_glob_f2, 3, 3, 1);
  mju_mulMatMat(all_loc_f3, r3, all_glob_f3, 3, 3, 1);

  mju_mulMatMat(gnd_loc_f1, r1, gnd_glob_f1, 3, 3, 1);
  mju_mulMatMat(gnd_loc_f2, r2, gnd_glob_f2, 3, 3, 1);
  mju_mulMatMat(gnd_loc_f3, r3, gnd_glob_f3, 3, 3, 1);

  // rotate the palm global force vector into the local frame
  // x = axial force, y/z unimportant
  // x force observed to be -ve under object contact (compression)
  // mju_mulMatMat(obj_loc_palm, r4, obj_glob_palm, 3, 3, 1);
  mju_mulMatMat(all_loc_palm, r4, all_glob_palm, 3, 3, 1);

  Forces_faster out;     // we will output this struct of force information

  // how many objects do we have
  out.obj.resize(live_objects.size());

  // get net force on object
  for (int i = 0; i < live_objects.size(); i++) {

    out.obj[i].net = get_object_net_force_faster(model, data, live_objects[i]);

    // copy across data for output
    out.obj[i].sum.init(objfrc[i].obj_glob_sum, 6, 1);
    out.obj[i].finger1.init(objfrc[i].obj_glob_f1, 6, 1);
    out.obj[i].finger2.init(objfrc[i].obj_glob_f2, 6, 1);
    out.obj[i].finger3.init(objfrc[i].obj_glob_f3, 6, 1);
    out.obj[i].palm.init(objfrc[i].obj_glob_palm, 6, 1);
    out.obj[i].ground.init(objfrc[i].obj_glob_gnd, 6, 1);
    out.obj[i].finger1_local.init(objfrc[i].obj_loc_f1, 3, 1);
    out.obj[i].finger2_local.init(objfrc[i].obj_loc_f2, 3, 1);
    out.obj[i].finger3_local.init(objfrc[i].obj_loc_f3, 3, 1);
    out.obj[i].palm_local.init(objfrc[i].obj_loc_palm, 3, 1);
  }

  out.all.finger1.init(all_glob_f1, 6, 1);
  out.all.finger2.init(all_glob_f2, 6, 1);
  out.all.finger3.init(all_glob_f3, 6, 1);
  out.all.palm.init(all_glob_palm, 6, 1);
  out.all.finger1_local.init(all_loc_f1, 3, 1);
  out.all.finger2_local.init(all_loc_f2, 3, 1);
  out.all.finger3_local.init(all_loc_f3, 3, 1);
  out.all.palm_local.init(all_loc_palm, 3, 1);

  out.gnd.finger1.init(gnd_glob_f1, 6, 1);
  out.gnd.finger2.init(gnd_glob_f2, 6, 1);
  out.gnd.finger3.init(gnd_glob_f3, 6, 1);
  out.gnd.finger1_local.init(gnd_loc_f1, 3, 1);
  out.gnd.finger2_local.init(gnd_loc_f2, 3, 1);
  out.gnd.finger3_local.init(gnd_loc_f3, 3, 1);
  
  out.empty = false;

  return out;
}

bool ObjectHandler::check_contact_forces(const mjModel* model, mjData* data)
{
  /* check that the sum of contact forces = net object force */

  // if (live_object == -1) return true;

  bool good = true;

  std::vector<Contact> forces = get_all_contacts(model, data);

  for (int live_idx : live_objects) {

    myNum net_force = get_object_net_force(model, data, live_idx);

    // sum the contact forces in the global frame
    myNum sum_force(6, 1);
    for (int i = 0; i < forces.size(); i++) {
      sum_force += forces[i].global_force_vec;
    }

    // for testing
    std::cout << "net force is:\n";
    net_force.print();
    std::cout << "sum of forces is:\n";
    sum_force.print();

    // now determine the difference
    sum_force -= net_force;
    mjtNum mag = sum_force.magnitude();
    constexpr double tol = 1e-5;

    // for testing
    std::cout << "the difference magnitude is " << mag << '\n';

    if (mag > tol) good = false;
  }

  return good;
}

bool ObjectHandler::apply_antiroll(mjData* data)
{
  /* smooth out any very low velocities to prevent rolling without any external
  contact. Checking for contacts has been removed as the threshold to stop rolling
  is so low */

  constexpr mjtNum threshold = 1e-6; // 1um per second

  for (int i = 0; i < live_objects.size(); i++) {

      int idx = live_objects[i];

      rawNum vel(&data->qvel[qveladr[idx]], 6, 1);

      if (vel.magnitude3() < threshold) {
        data->qvel[qveladr[idx]] = 0;
        data->qvel[qveladr[idx] + 1] = 0;
        data->qvel[qveladr[idx] + 2] = 0;
        data->qvel[qveladr[idx] + 3] = 0; 
        data->qvel[qveladr[idx] + 4] = 0;
        data->qvel[qveladr[idx] + 5] = 0;
      }
  }

  return true;
}

// change object parameters

void ObjectHandler::set_object_visibility(mjModel* model, bool visible)
{
  /* set the visbility of all objects, except the live object which must
  always be visible. Setting the background objects invisible speeds up
  rendering times and is recommended when using a camera */

  // set all objects to the given visibility (alpha=0 in rgba)
  for (int i : geom_id) {
    model->geom_rgba[i * 4 + 3] = visible;
  }

  // // finally, set the live object as visible (cannot be invisible)
  // if (live_object != -1) {
  //   int rgba_idx = geom_id[live_object] * 4;
  //   model->geom_rgba[rgba_idx + 3] = 1;
  // }

  // finally, set the live objects as visible (cannot be invisible)
  for (int live_idx : live_objects) {
    int rgba_idx = geom_id[live_idx] * 4;
    model->geom_rgba[rgba_idx + 3] = 1;
  }

  // save the change
  object_visibility = visible;
}

void ObjectHandler::set_colour(mjModel* model, std::vector<float> rgba)
{
  /* set the colour of the live object */

  // if (live_object == -1) {
  //   return;
  // }

  if (live_objects.size() == 0) {
    return;
  }

  if (rgba.size() != 3 and rgba.size() != 4) {
    throw std::runtime_error("set_colour() must receive a 3 element rgb vector or a 4 element rgba vector");
  }

  for (int live_idx : live_objects) {

    // index for rgba entries in mjModel
    int rgba_idx = geom_id[live_idx] * 4;

    // set the colour parameters
    model->geom_rgba[rgba_idx + 0] = rgba[0];
    model->geom_rgba[rgba_idx + 1] = rgba[1];
    model->geom_rgba[rgba_idx + 2] = rgba[2];

    if (rgba.size() == 4)
      model->geom_rgba[rgba_idx + 3] = rgba[3];

  }

}

void ObjectHandler::set_all_colours(mjModel* model, std::vector<float> rgba)
{
  /* set all the objects to be one colour */

  for (int i : geom_id) {

    // set the colour parameters
    model->geom_rgba[i * 4 + 0] = rgba[0];
    model->geom_rgba[i * 4 + 1] = rgba[1];
    model->geom_rgba[i * 4 + 2] = rgba[2];
    
    // if an a value is given, set this too
    if (rgba.size() == 4)
      model->geom_rgba[i * 4 + 3] = rgba[3];
  }
}

void ObjectHandler::set_ground_colour(mjModel* model, std::vector<float> rgba)
{
  if (rgba.size() != 3 and rgba.size() != 4) {
    throw std::runtime_error("set_ground_colour() must receive a 3 element rgb vector or a 4 element rgba vector");
  }

  // index for rgba entries in mjModel
  int rgba_idx = gnd_geom_id * 4;

  // set the colour parameters
  model->geom_rgba[rgba_idx + 0] = rgba[0];
  model->geom_rgba[rgba_idx + 1] = rgba[1];
  model->geom_rgba[rgba_idx + 2] = rgba[2];

  if (rgba.size() == 4)
    model->geom_rgba[rgba_idx + 3] = rgba[3];
}

void ObjectHandler::set_friction(mjModel* model, mjtNum sliding_friction)
{
  /* set the live object sliding friction only */

  std::vector<mjtNum> f { sliding_friction };
  set_friction(model, f);
}

void ObjectHandler::set_friction(mjModel* model, std::vector<mjtNum> friction_triple)
{
  /* set the friction values for the live object. In mujoco these are:

        { sliding_friction, torsional_friction, rolling_friction }
        with defaults { 1, 0.005, 0.0001 }

  Sliding friction acts along both axes of the tangent plane. Torsional friction
  acts around the contact normal. Rolling friction acst around both axes of the
  tangent plane.

  These 3 parameters are combined into 5dof frictional contacts, and the two 3dof
  vectors are combined in a contact based on the solmix and priority attributes.
  In general, whatever geom has higher values will be used - unless you specify a
  higher priority for the less friction geom.
  */

  // if (live_object == -1) {
  //   return;
  // }

  if (live_objects.size() == 0) {
    return;
  }

  if (friction_triple.size() == 0 or friction_triple.size() > 3) {
    throw std::runtime_error("set_friction() must receive a vector from 1-3 elements");
  }

  for (int live_idx : live_objects) {

    // index for friction entries in mjModel
    int fidx = geom_id[live_idx] * 3;

    model->geom_friction[fidx + 0] = friction_triple[0]; // sliding friction

    if (friction_triple.size() > 1)
      model->geom_friction[fidx + 1] = friction_triple[1]; // torsional friction

    if (friction_triple.size() > 2)
      model->geom_friction[fidx + 2] = friction_triple[2]; // rolling friction
  }
}

void ObjectHandler::randomise_all_colours(mjModel* model, 
  std::shared_ptr<std::default_random_engine> generator)
{
  /* randomise the colours of all objects */

  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  // we should ignore a as object_visibility handles this
  std::vector<float> rgba(4);

  for (int i : geom_id) {

    // get random numbers from [0.0, 1.0]
    rgba[0] = distribution(*generator);
    rgba[1] = distribution(*generator);
    rgba[2] = distribution(*generator);
    // rgba[3] = distribution(*generator);

    // set the colour parameters
    model->geom_rgba[i * 4 + 0] = rgba[0];
    model->geom_rgba[i * 4 + 1] = rgba[1];
    model->geom_rgba[i * 4 + 2] = rgba[2];
    // model->geom_rgba[i * 4 + 3] = 1.0; // a is currently not randomised
  }
}

void ObjectHandler::default_colours(mjModel* model)
{
  /* restore all objects to default colour */

  for (int i : geom_id) {

    // set the colour parameters
    model->geom_rgba[i * 4 + 0] = 0.5;
    model->geom_rgba[i * 4 + 1] = 0.5;
    model->geom_rgba[i * 4 + 2] = 0.5;
    model->geom_rgba[i * 4 + 3] = 1.0;

  }

  model->geom_rgba[gnd_geom_id * 4 + 0] = 0.5;
  model->geom_rgba[gnd_geom_id * 4 + 1] = 0.5;
  model->geom_rgba[gnd_geom_id * 4 + 2] = 0.5;
  model->geom_rgba[gnd_geom_id * 4 + 3] = 0.5;
}

int ObjectHandler::is_object_geom(int id)
{
  /* determine if a geom id is an object */

  for (int i = 0; i < live_objects.size(); i++) {
    if (id == geom_id[live_objects[i]]) return i + 1;
  }

  return 0;
}

bool ObjectHandler::is_ground_geom(int id)
{
  /* determine if a geom id is the ground */

  if (id == gnd_geom_id) {
    return true;
  }

  return false;
}

} // namespace luke