#!/usr/bin/env python3

from datetime import datetime
# import pickle
import dill as pickle
import os

class ModelSaver:

  def __init__(self, path, root=None):
    """
    Saves learned models at relative path
    """

    if not root:
      self.root = "/home/luke/mymujoco/rl/"
    else:
      if root[-1] != '/': root += '/'
      self.root = root

    self.default_num = 1
    self.file_ext = ".pickle"
    self.file_num = "{:03d}"
    self.date_str = "%d-%m-%Y_%H:%M"
    self.folder_names = "train_{}/"
    self.last_loadpath = ""

    self.in_folder = False
    self.folder = ""

    if path[-1] != '/': path += '/'
    self.rel_path = path

    use_root = False
    if use_root: self.path = self.root + self.rel_path
    else:
      self.path = path

    # create directories if they don't already exist
    if not os.path.exists(self.path):
      os.makedirs(self.path)

  def get_file_num(self, file):
    """
    Return the numbering of the file, eg XYZ_004 returns 4
    """
    # check if we should remove the file extension
    if file[-len(self.file_ext):] == self.file_ext:
      file = file[:-len(self.file_ext)]

    # extract the file numbering
    num_len = len(self.file_num.format(0))
    return int(file[-num_len:])

  def get_recent_file(self, path, id=None):
    """
    Get the path to the most recent save in the path, or return None if empty
    """

    # get all files with pickle extension in the target directory
    pkl_files = [x for x in os.listdir(path) if x.endswith(self.file_ext)]

    # if there are no candidate files
    if len(pkl_files) == 0: return None

    # remove the file extension
    files = [x[:-len(self.file_ext)] for x in pkl_files]

    imax = -1
    num_max = -1

    # loop through the files, clip of the number at the end eg '003'
    for i in range(len(files)):
      num = self.get_file_num(files[i])
      if num > num_max:
        # get the index of the biggest file ending
        num_max = num
        imax = i
      # if we are searching for a specific id
      if id != None:
        if id == num:
          imax = i
          break

    # return the path to this file
    if path[-1] != '/': path += '/'
    return path + pkl_files[imax]

  def get_most_recent(self, label=None):
    """
    Get the most recent model trained in the path. First we check the root of
    the path, if that fails we check all the folders for the most recent
    timestamp, and then get the most recent in there
    """

    root_recent = self.get_recent_file(self.path)

    if root_recent != None:
      print("Found the most recent file:", root_recent)
      return root_recent

    else:
      print("No recent files found")

    # # else we need to check the folder names
    # subdir = [x for x in os.listdir(self.path) if os.path.isdir(x)]

    # # find which name is the most recent

  def new_folder(self, label=None, suffix=None):
    """
    Create a new training folder
    """

    if self.in_folder:
      self.exit_folder()

    # get the current time and date
    now = datetime.now()
    time_stamp = now.strftime(self.date_str)

    # if we will add an extra label
    if label != None:
      time_stamp = label + '_' + time_stamp
    if suffix != None:
      time_stamp += '_' + suffix

    # create the folder name
    folder_name = self.folder_names.format(time_stamp)

    # create the new folder
    if not os.path.exists(self.path + folder_name):
      os.makedirs(self.path + folder_name)
    else:
      # if the folder name already exists, add a distinguishing number
      i = 1
      while os.path.exists(self.path + folder_name[:-1] + f"_{i}"):
        i += 1
      os.makedirs(self.path + folder_name[:-1] + f"_{i}")
      folder_name = folder_name[:-1] + f"_{i}" + '/'

    # enter the folder
    self.enter_folder(folder_name)

  def enter_folder(self, foldername, folderpath=None):
    """
    Enter a folder for future saving and loading
    """

    if folderpath != None:
      self.path = folderpath

    if os.path.exists(self.path + foldername):

      if self.in_folder: self.exit_folder()

      if foldername[-1] != '/': foldername += '/'

      self.in_folder = True
      self.folder = foldername

    else:
      raise RuntimeError("folder name does not exist")

  def exit_folder(self):
    """
    Stop saving in a special training folder
    """

    self.in_folder = False
    self.folder = ""

  def save(self, name, pyobj=None, txtstr=None, txtonly=None, txtlabel=None):
    """
    Save the given object using pickle
    """

    savepath = self.path

    if self.in_folder: savepath += self.folder

    # if only saving a text file
    if txtonly != None:
      savename = name + '.txt'
      print(f"Saving text only {savepath + savename}")
      with open(savepath + savename, 'w') as openfile:
        openfile.write(txtstr)
      return savepath + savename
  
    # find out what the most recent file number in the savepath was
    most_recent = self.get_recent_file(savepath)

    if most_recent == None: save_id = self.default_num
    else:
      save_id = 1 + self.get_file_num(most_recent)

    # create the file name
    savename = name + '_' + self.file_num.format(save_id) + self.file_ext

    # save
    print(f"Saving file {savepath + savename} with pickle ... ", end="")
    with open(savepath + savename, 'wb') as openfile:
      pickle.dump(pyobj, openfile)
      print("finished")

    # if we are asked to save a .txt file too
    if txtstr != None:
      ext = '.txt'
      if txtlabel != None: ext = '_' + txtlabel + ext
      txtname = name + '_' + self.file_num.format(save_id) + ext
      with open(savepath + txtname, 'w') as openfile:
        openfile.write(txtstr)
        print(f"Saved also: {txtname}")

    return savepath + savename

  def load(self, folderpath=None, foldername=None, id=None):
    """
    Load a model, by default loads the most recent in the current folder
    """

    loadpath = self.path

    # if a path to a file is specified
    if folderpath != None:
      loadpath = folderpath

    # if the path to the folder is given
    if foldername != None:
      loadpath += foldername
      if loadpath[-1] != '/': loadpath += '/'
      loadpath = self.get_recent_file(loadpath, id)

    else:
      # default: find in current folder folder
      loadpath = self.path

      if self.in_folder: 
        loadpath += self.folder

      new_loadpath = self.get_recent_file(loadpath, id)

      if new_loadpath == None:
        print(f"No model found at path {new_loadpath}")
      else:
        loadpath = new_loadpath

    print(f"Loading file {loadpath} with pickle ... ", end="")
    with open(loadpath, 'rb') as f:
      loaded_obj = pickle.load(f)
      print("finished")

    self.last_loadpath = loadpath

    return loaded_obj


if __name__ == "__main__":

  obj = ModelSaver('models/demosave')

  pyobj = [1, 2, 3, 4, 5]

  obj.new_folder(label="cluster")
  obj.new_folder(label="cluster")
  obj.new_folder(label="cluster")

  obj.save("hyperparams", txtstr="testing 1 2 3", txtonly=True)
  obj.save("network1", pyobj=pyobj)
  obj.save("network1", pyobj=pyobj)
  obj.save("network1", pyobj=pyobj)
  obj.save("network1", pyobj=pyobj)

  obj.load()

  obj.exit_folder()

  obj.save("network2", pyobj=pyobj)
  obj.save("network2", pyobj=pyobj)
  obj.save("network2", pyobj=pyobj)
  obj.save("network2", pyobj=pyobj)

  obj.load()

    