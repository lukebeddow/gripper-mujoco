#!/usr/bin/env python3

import yaml
import os
from lxml import etree
import numpy as np

# objects yaml file
objects_yaml_file = "define_objects.yaml"

# get relevant path information
filepath = os.path.dirname(os.path.abspath(__file__))
description_path = os.path.dirname(filepath)

# import the dictionary of object information
with open(filepath + "/" + objects_yaml_file) as file:
  object_details = yaml.safe_load(file)

# define key xml snippets in the form of a function, returning formatted snippet
def get_object_xml(name, quat, mass, diaginertia):
  # this snippet creates the main object in mujoco
  object_xml = """
  <body name="{0}" pos="0 0 0">
    <inertial pos="0 0 0" quat="{1}" mass="{2}" diaginertia="{3}"/>
    <freejoint name="{0}"/>
    <geom name="{0}_geom" type="mesh" mesh="{0}"/>
  </body>\n
  """.format(name, quat, mass, diaginertia)
  return object_xml

def get_asset_xml(name, filepath, xscale, yscale, zscale, refquat):
  # this snippet defines the object mesh
  mesh_xml = """
  <mesh name="{0}" file="{1}"
        scale="{2} {3} {4}"
        refquat="{5}"
  />\n
  """.format(name, filepath, xscale, yscale, zscale, refquat)
  return mesh_xml

def get_details_xml(name, z_rest):
  # this snippet is for me to save any extrsa relevant information
  details_xml = """
  <object_details name="{0}" z_rest="{1}"/>\n
  """.format(name, z_rest)
  return details_xml

# function to add xml to a tree
def add_chunk(tree, parent_tag, xml_string_to_add):
  """
  This function adds a chunk of xml text under the bracket of the given
  parent tag into the given tree
  """

  # print(xml_string_to_add)

  # extract the xml text from
  new_tree = etree.fromstring(xml_string_to_add)

  # get the root of the parent tree
  root = tree.getroot()

  # special case where we are adding at the root
  if parent_tag == "@root":
    root.append(new_tree)
    return

  # search recursively for all instances of the parent tag
  tags = root.findall(".//" + parent_tag)

  if len(tags) > 1:
    raise RuntimeError("more than one parent tag found")
  elif len(tags) == 0:
    raise RuntimeError("parent tag not found")

  for t in tags:
    t.append(new_tree)
  
  return

if __name__ == "__main__":

  object_root = etree.Element("mujoco")
  assets_root = etree.Element("mujoco")
  detail_root = etree.Element("mujoco")

  object_tree = etree.ElementTree(object_root)
  assets_tree = etree.ElementTree(assets_root)
  detail_tree = etree.ElementTree(detail_root)

  # loop through top level objects in the yaml file
  for object in object_details:

    if object_details[object]["include"] is False:
      continue

    # density = 1000 

    for density in [100]:

      # extract key information
      name_root = object_details[object]["name_root"]
      name_suffix = object_details[object]["suffix"]
      name_path = object_details[object]["path"]

      spawn_axis = object_details[object]["spawn"]["axis"]
      spawn_height = object_details[object]["spawn"]["rest"]

      scale_num = object_details[object]["scale"]["num"]
      x_max = object_details[object]["scale"]["max"]["x"]
      x_min = object_details[object]["scale"]["min"]["x"]
      y_max = object_details[object]["scale"]["max"]["y"]
      y_min = object_details[object]["scale"]["min"]["y"]
      z_max = object_details[object]["scale"]["max"]["z"]
      z_min = object_details[object]["scale"]["min"]["z"]

      inertial_type = object_details[object]["inertial"]["type"]
      frame_align = object_details[object]["inertial"]["align"]

      fillet_used = object_details[object]["fillet"]["used"]
      if fillet_used:
        fillet_step = object_details[object]["fillet"]["step"]
        fillet_max = object_details[object]["fillet"]["max"]
        fillet_min = object_details[object]["fillet"]["min"]
        # work out how many fillet steps
        fillet_num = int((fillet_max - fillet_min) / fillet_step) + 1

      qx = object_details[object]["quat"]["x"]
      qy = object_details[object]["quat"]["y"]
      qz = object_details[object]["quat"]["z"]
      qw = object_details[object]["quat"]["w"]

      # work out scale increments
      if scale_num > 1:
        x_increment = (x_max - x_min) / (scale_num - 1)
        y_increment = (y_max - y_min) / (scale_num - 1)
        z_increment = (z_max - z_min) / (scale_num - 1)

      # loop through scaling number
      for i in range(scale_num):

        # work out this scale factor
        if scale_num == 1:
          xscale = 1.0
          yscale = 1.0
          zscale = 1.0
        else:
          xscale = x_min + i * x_increment
          yscale = y_min + i * y_increment
          zscale = z_min + i * z_increment

        # swap scaling for inertia based on frame alignment
        scales = [xscale, yscale, zscale]
        align = [
          0 if frame_align[0] == 'x' else (1 if frame_align[0] == 'y' else 2),
          0 if frame_align[1] == 'x' else (1 if frame_align[1] == 'y' else 2),
          0 if frame_align[2] == 'x' else (1 if frame_align[2] == 'y' else 2)
        ]
        xscale_in = scales[align[0]]
        yscale_in = scales[align[1]]
        zscale_in = scales[align[2]]

        if inertial_type == "cuboid":

          # extract dimensions
          x = xscale_in * object_details[object]["inertial"]["x"]
          y = yscale_in * object_details[object]["inertial"]["y"]
          z = zscale_in * object_details[object]["inertial"]["z"]

          # calculate the mass
          mass = x * y * z * density

          # calculate the diaginertia
          ixx = (1.0/12.0) * mass * (y**2 + z**2)
          iyy = (1.0/12.0) * mass * (x**2 + z**2)
          izz = (1.0/12.0) * mass * (x**2 + y**2)

        elif inertial_type == "sphere":
          
          # extract dimensions
          rx = xscale_in * object_details[object]["inertial"]["r"]
          ry = yscale_in * object_details[object]["inertial"]["r"]
          rz = zscale_in * object_details[object]["inertial"]["r"]

          # calculate the mass
          mass = (4.0/3.0) * np.pi * rx * ry * rz * density

          # calculate the diaginertia
          ixx = (1.0/5.0) * mass * (ry**2 + rz**2)
          iyy = (1.0/5.0) * mass * (rx**2 + rz**2)
          izz = (1.0/5.0) * mass * (rx**2 + ry**2)

        elif inertial_type == "cylinder":

          # extract dimensions
          rx = xscale_in * object_details[object]["inertial"]["r"]
          ry = yscale_in * object_details[object]["inertial"]["r"]
          h = zscale_in * object_details[object]["inertial"]["h"]
          
          # calculate the mass
          mass = np.pi * rx * ry * h * density

          # calculate the diaginertia
          ixx = (1.0/12.0) * mass * (3 * rx * ry + h**2)
          iyy = (1.0/12.0) * mass * (3 * rx * ry + h**2)
          izz = (1.0/2.0) * mass * rx * ry

        else:
          raise RuntimeError("inertial type not one of 'cuboid', 'sphere', 'cylinder'")

        # now generate the final xml name
        if fillet_used:
          obj_filenames = [
            "{0}_{1}".format(name_root, fillet_min + j * fillet_step)
            for j in range(fillet_num)
          ]
          names = [
            "{0}_{1}_{2}_{3}".format(
              name_root,
              fillet_min + j * fillet_step,
              name_suffix,
              i
            ) for j in range(fillet_num)
          ]
        else:
          obj_filenames = ["{0}".format(name_root, name_suffix)]
          names = ["{0}_{1}_{2}".format(name_root, name_suffix, i)]
        
        # format key data ready to insert into xml snippets
        quat = f"{qw} {qx} {qy} {qz}"
        quat_conj = f"{qw} {-qx} {-qy} {-qz}" # note quaternion conjugate used, see mujoco docs
        diaginertia = "{0:.6f} {1:.6f} {2:.6f}".format(ixx, iyy, izz)
        
        # check which axis the z rest is
        if spawn_axis == "x": z_rest = spawn_height * xscale
        elif spawn_axis == "y": z_rest = spawn_height * yscale
        elif spawn_axis == "z": z_rest = spawn_height * zscale
        else: raise RuntimeError("spawn axis not one of 'x', 'y', 'z'")

        # loop through every fillet option and create the xml snippets
        for k, name in enumerate(names):

          path = "models/{0}/{1}.STL".format(name_path, obj_filenames[k])

          object_xml = get_object_xml(name, quat, mass, diaginertia)
          asset_xml = get_asset_xml(name, path, xscale, yscale, zscale, quat_conj)
          detail_xml = get_details_xml(name, z_rest)

          add_chunk(object_tree, "@root", object_xml)
          add_chunk(assets_tree, "@root", asset_xml)
          add_chunk(detail_tree, "@root", detail_xml)

  # add the ground as the final element in the object tree
  ground_xml = """
  <body name="ground" pos="0 0 0">
    <geom name="ground_geom" type="plane" size="1 1 1"/>
  </body>
  """
  add_chunk(object_tree, "@root", ground_xml)

  # finally, save the trees
  object_tree.write(filepath + "/objects.xml")
  assets_tree.write(filepath + "/assets.xml")
  detail_tree.write(filepath + "/details.xml")



    


