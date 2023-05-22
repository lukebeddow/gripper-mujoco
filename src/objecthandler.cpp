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

  // set default - no live object
  live_object = -1;
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

  // we cannot start with an object live
  live_object = -1;
  live_geom = "";

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
  in_use.resize(names.size());
  idx.resize(names.size());
  qpos.resize(names.size());
  qposadr.resize(names.size());
  qveladr.resize(names.size());
  reset_qpos.resize(names.size());
  geom_id.resize(names.size());
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
  /* set the live object - no safety checks! */

  live_object = idx;
  live_geom = names[idx] + "_geom";

  // enable collisions, set to 01
  model->geom_contype[geom_id[idx]] = COL_main;
  model->geom_conaffinity[geom_id[idx]] = COL_main;

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
  /* reset an object to its starting pose */

  if (not check_idx(idx)) return;

  move_object(data, idx, reset_qpos[idx]);

  // if we have reset the live object
  if (idx == live_object) {
    live_object = -1;
    live_geom = "";

    // disable collisions,set to 10
    model->geom_contype[geom_id[idx]] = COL_dead;
    model->geom_conaffinity[geom_id[idx]] = COL_dead;
  }
}

void ObjectHandler::reset_live(mjModel* model, mjData* data) 
{
  if (live_object == -1) return;
  reset_object(model, data, live_object);
}

void ObjectHandler::spawn_object(mjModel* model, mjData* data, int idx, QPos pose) 
{
  /* spawn a new live object */

  if (not check_idx(idx)) return;

  // first reset the live object
  reset_live(model, data);

  // override the z height of the given pose to use reset value
  pose.z = reset_qpos[idx].z + z_spawn_tolerance;

  // move this object into the given pose
  move_object(data, idx, pose);

  // set our object as the live object
  set_live(model, idx);
}

QPos ObjectHandler::get_live_qpos(mjModel* model, mjData* data)
{
  /* get the qpos of the live object */

  if (live_object == -1)
    throw std::runtime_error("ObjectHandler::get_live_qpos() failed as there is no live object\n");

  QPos out;

  out.update(model, data, qposadr[live_object]);

  return out;
}

// print functions
void ObjectHandler::print()
{
  std::cout << "object scene s_: ";
  if (names.size() == 0) {
    std::cout << "no named objects\n";
    return;
  }
  std::cout << "live object is " << live_object;
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
myNum ObjectHandler::get_object_net_force(const mjModel* model, mjData* data)
{
  /* get the net force in world frame on the live object */

  if (live_object == -1) {
    myNum empty;
    return empty;
  }

  // get the net forces (pointer encompasses whole array)
  mjtNum* F = data->cfrc_ext + idx[live_object] * 6;

  // cfrc_ext stores forces as [torque, force], so swap order
  mjtNum F_swap[6] = { F[3], F[4], F[5], F[0], F[1], F[2] };

  // create my mjtNum wrapper, nr=6, nc=1
  myNum out(F_swap, 6, 1);

  // for testing
  // out.print();

  return out;
}

rawNum ObjectHandler::get_object_net_force_faster(const mjModel* model, mjData* data)
{
  /* get the net force in world frame on the live object */

  if (live_object == -1) {
    rawNum empty;
    return empty;
  }

  // get the net forces (pointer encompasses whole array)
  mjtNum* F = data->cfrc_ext + idx[live_object] * 6;

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
  /* get all the contacts on the live object */

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

  // for testing
  // std::cout << "the number of contacts is " << data->ncon << '\n';

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

    // convert to std::string
    std::string strname1(name1);
    std::string strname2(name2);

    // save object names
    contact.name1 = strname1;
    contact.name2 = strname2;

    // check which objects/geoms this contact involves
    contact.check_involves(live_geom);

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

    // // for testing
    // contact.print();
    // contact.print_involves();
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

Forces ObjectHandler::extract_forces(const mjModel* model, mjData* data)
{
  /* extract the forces on the live object */

  if (live_object == -1) {
    Forces empty;
    return empty;
  }

  Forces out;
  out.obj.net = get_object_net_force(model, data);
  std::vector<Contact> contacts = get_all_contacts(model, data);

  // sum the contact forces in global frame
  for (int i = 0; i < contacts.size(); i++) {

    // if the contact involves the object
    if (contacts[i].with.object) {
      out.obj.sum += contacts[i].global_force_vec;
      if (contacts[i].with.finger1) 
        out.obj.finger1 += contacts[i].global_force_vec;
      if (contacts[i].with.finger2) 
        out.obj.finger2 += contacts[i].global_force_vec;
      if (contacts[i].with.finger3) 
        out.obj.finger3 += contacts[i].global_force_vec;
      if (contacts[i].with.palm) 
        out.obj.palm += contacts[i].global_force_vec;
      if (contacts[i].with.ground) 
        out.obj.ground += contacts[i].global_force_vec;
    }

    // if the contact involves the gripper
    if (contacts[i].with.finger1) {
      out.all.finger1 += contacts[i].global_force_vec;
      // testing! seperate out contacts with the ground
      if (contacts[i].with.ground) {
        out.gnd.finger1 += contacts[i].global_force_vec;
      }
    }
    if (contacts[i].with.finger2) {
      out.all.finger2 += contacts[i].global_force_vec;
      if (contacts[i].with.ground) {
        out.gnd.finger2 += contacts[i].global_force_vec;
      }
    }
    if (contacts[i].with.finger3) {
      out.all.finger3 += contacts[i].global_force_vec;
      if (contacts[i].with.ground) {
        out.gnd.finger3 += contacts[i].global_force_vec;
      }
    }
    if (contacts[i].with.palm) {
      out.all.palm += contacts[i].global_force_vec;
    }

  }

  // get the rotation matrices for each finger body
  myNum r1(&data->xmat[f1_idx * 9], 3, 3);
  myNum r2(&data->xmat[f2_idx * 9], 3, 3);
  myNum r3(&data->xmat[f3_idx * 9], 3, 3);
  myNum r4(&data->xmat[pm_idx * 9], 3, 3);

  // rotate the finger vectors from the world frame to the finger body frame
  // x = axial force, y = tangential force (ie strain gauge force)
  // these forces are observed to all be +ve under object contact (outwards bend)
  out.obj.finger1_local = out.obj.finger1.rotate3_by(r1.transpose());
  out.obj.finger2_local = out.obj.finger2.rotate3_by(r2.transpose());
  out.obj.finger3_local = out.obj.finger3.rotate3_by(r3.transpose());
  out.all.finger1_local = out.all.finger1.rotate3_by(r1.transpose());
  out.all.finger2_local = out.all.finger2.rotate3_by(r2.transpose());
  out.all.finger3_local = out.all.finger3.rotate3_by(r3.transpose());
  out.gnd.finger1_local = out.gnd.finger1.rotate3_by(r1.transpose());
  out.gnd.finger2_local = out.gnd.finger2.rotate3_by(r2.transpose());
  out.gnd.finger3_local = out.gnd.finger3.rotate3_by(r3.transpose());


  // rotate the palm global force vector into the local frame
  // x = axial force, y/z unimportant
  // x force observed to be -ve under object contact (compression)
  out.obj.palm_local = out.obj.palm.rotate3_by(r4.transpose());
  out.all.palm_local = out.all.palm.rotate3_by(r4.transpose());

  return out;
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

  if (live_object == -1) {
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

  // contact vectors with the object
  mjtNum obj_glob_sum[6] = {};
  mjtNum obj_glob_f1[6] = {};
  mjtNum obj_glob_f2[6] = {};
  mjtNum obj_glob_f3[6] = {};
  mjtNum obj_glob_palm[6] = {};
  mjtNum obj_glob_gnd[6] = {};
  mjtNum obj_loc_f1[3] = {};
  mjtNum obj_loc_f2[3] = {};
  mjtNum obj_loc_f3[3] = {};
  mjtNum obj_loc_palm[3] = {};

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
    contact.check_involves(live_geom);

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
    if (contact.with.object) {
      // out.obj.sum += contact.global_force_vec;
      mju_addTo(obj_glob_sum, global_force_vec, 6);
      if (contact.with.finger1) 
        // out.obj.finger1 += contact.global_force_vec;
        mju_addTo(obj_glob_f1, global_force_vec, 6);
      if (contact.with.finger2) 
        // out.obj.finger2 += contact.global_force_vec;
        mju_addTo(obj_glob_f2, global_force_vec, 6);
      if (contact.with.finger3) 
        // out.obj.finger3 += contact.global_force_vec;
        mju_addTo(obj_glob_f3, global_force_vec, 6);
      if (contact.with.palm) 
        // out.obj.palm += contact.global_force_vec;
        mju_addTo(obj_glob_palm, global_force_vec, 6);
      if (contact.with.ground) 
        // out.obj.ground += contact.global_force_vec;
        mju_addTo(obj_glob_gnd, global_force_vec, 6);
    }

    // if the contact involves the gripper
    if (contact.with.finger1) {
      // out.all.finger1 += contact.global_force_vec;
      mju_addTo(all_glob_f1, global_force_vec, 6);
      if (contact.with.ground) {
        // out.gnd.finger1 += contact.global_force_vec;
        mju_addTo(gnd_glob_f1, global_force_vec, 6);
      }
    }
    if (contact.with.finger2) {
      // out.all.finger2 += contact.global_force_vec;
      mju_addTo(all_glob_f2, global_force_vec, 6);
      if (contact.with.ground) {
        // out.gnd.finger2 += contact.global_force_vec;
        mju_addTo(gnd_glob_f2, global_force_vec, 6);
      }
    }
    if (contact.with.finger3) {
      // out.all.finger3 += contact.global_force_vec;
      mju_addTo(all_glob_f3, global_force_vec, 6);
      if (contact.with.ground) {
        // out.gnd.finger3 += contact.global_force_vec;
        mju_addTo(gnd_glob_f3, global_force_vec, 6);
      }
    }
    if (contact.with.palm) {
      // out.all.palm += contact.global_force_vec;
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
  mju_mulMatMat(obj_loc_f1, r1, obj_glob_f1, 3, 3, 1);
  mju_mulMatMat(obj_loc_f2, r2, obj_glob_f2, 3, 3, 1);
  mju_mulMatMat(obj_loc_f3, r3, obj_glob_f3, 3, 3, 1);

  mju_mulMatMat(all_loc_f1, r1, all_glob_f1, 3, 3, 1);
  mju_mulMatMat(all_loc_f2, r2, all_glob_f2, 3, 3, 1);
  mju_mulMatMat(all_loc_f3, r3, all_glob_f3, 3, 3, 1);

  mju_mulMatMat(gnd_loc_f1, r1, gnd_glob_f1, 3, 3, 1);
  mju_mulMatMat(gnd_loc_f2, r2, gnd_glob_f2, 3, 3, 1);
  mju_mulMatMat(gnd_loc_f3, r3, gnd_glob_f3, 3, 3, 1);

  // rotate the palm global force vector into the local frame
  // x = axial force, y/z unimportant
  // x force observed to be -ve under object contact (compression)
  mju_mulMatMat(obj_loc_palm, r4, obj_glob_palm, 3, 3, 1);
  mju_mulMatMat(all_loc_palm, r4, all_glob_palm, 3, 3, 1);

  Forces_faster out;     // we will output this struct of force information

  // get net force on object
  out.obj.net = get_object_net_force_faster(model, data);

  // copy across data for output
  out.obj.sum.init(obj_glob_sum, 6, 1);
  out.obj.finger1.init(obj_glob_f1, 6, 1);
  out.obj.finger2.init(obj_glob_f2, 6, 1);
  out.obj.finger3.init(obj_glob_f3, 6, 1);
  out.obj.palm.init(obj_glob_palm, 6, 1);
  out.obj.ground.init(obj_glob_gnd, 6, 1);
  out.obj.finger1_local.init(obj_loc_f1, 3, 1);
  out.obj.finger2_local.init(obj_loc_f2, 3, 1);
  out.obj.finger3_local.init(obj_loc_f3, 3, 1);
  out.obj.palm_local.init(obj_loc_palm, 3, 1);

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

  if (live_object == -1) return true;

  myNum net_force = get_object_net_force(model, data);
  std::vector<Contact> forces = get_all_contacts(model, data);

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

  if (mag > tol) return false;

  return true;
}

// change object parameters

void ObjectHandler::set_colour(mjModel* model, std::vector<float> rgba)
{
  /* set the colour of the live object */

  if (live_object == -1) {
    return;
  }

  if (rgba.size() != 3 and rgba.size() != 4) {
    throw std::runtime_error("set_colour() must receive a 3 element rgb vector or a 4 element rgba vector");
  }

  // index for rgba entries in mjModel
  int rgba_idx = geom_id[live_object] * 4;

  // set the colour parameters
  model->geom_rgba[rgba_idx + 0] = rgba[0];
  model->geom_rgba[rgba_idx + 1] = rgba[1];
  model->geom_rgba[rgba_idx + 2] = rgba[2];

  if (rgba.size() == 4)
    model->geom_rgba[rgba_idx + 3] = rgba[3];

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

  if (live_object == -1) {
    return;
  }

  if (friction_triple.size() == 0 or friction_triple.size() > 3) {
    throw std::runtime_error("set_friction() must receive a vector from 1-3 elements");
  }

  // index for friction entries in mjModel
  int fidx = geom_id[live_object] * 3;

  model->geom_friction[fidx + 0] = friction_triple[0]; // sliding friction

  if (friction_triple.size() > 1)
    model->geom_friction[fidx + 1] = friction_triple[1]; // torsional friction

  if (friction_triple.size() > 2)
    model->geom_friction[fidx + 2] = friction_triple[2]; // rolling friction

}

void ObjectHandler::randomise_all_colours(mjModel* model, 
  std::shared_ptr<std::default_random_engine> generator)
{
  /* randomise the colours of all objects */

  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  std::vector<float> rgba(4);

  for (int i : geom_id) {

    // get random numbers from [0.0, 1.0]
    rgba[0] = distribution(*generator);
    rgba[1] = distribution(*generator);
    rgba[2] = distribution(*generator);
    rgba[3] = distribution(*generator);

    // set the colour parameters
    model->geom_rgba[i * 4 + 0] = rgba[0];
    model->geom_rgba[i * 4 + 1] = rgba[1];
    model->geom_rgba[i * 4 + 2] = rgba[2];
    model->geom_rgba[i * 4 + 3] = 1.0; // a is currently not randomised
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

} // namespace luke