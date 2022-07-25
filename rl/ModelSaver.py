#!/usr/bin/env python3

from datetime import datetime
import dill as pickle
import bz2
import os
import shutil

# global variable, do we compress saved files with bz2 library
use_compression = True

def compressed_pickle(title, data):
 with bz2.BZ2File(title, 'w') as f: 
  pickle.dump(data, f)

def decompress_pickle(file):
 data = bz2.BZ2File(file, 'rb')
 data = pickle.load(data)
 return data

class ModelSaver:

  def __init__(self, path, root=None):
    """
    Saves learned models at relative path
    """

    # user set parameters 
    self.default_num = 1             # starting number for saving files with numbers
    self.file_ext = ".pbz2" if use_compression else ".pickle" # saved file extension
    self.file_num = "{:03d}"         # digit format for saving files with numbers
    self.date_str = "%d-%m-%Y-%H:%M" # date string, must be seperated by '-'
    self.folder_names = "train_{}/"  # default name of created folders, then formatted with date

    # if we are given a root, we can use abs paths not relative
    if not root:
      # assume the root is the path to the current running files
      self.root = pathhere = os.path.dirname(os.path.abspath(__file__)) + "/"
      use_root = False
    else:
      if self.root[-1] != '/': self.root += '/'
      self.root = root
      use_root = True

    self.in_folder = False
    self.folder = ""
    self.last_loadpath = ""

    # ensure path ends in trailing slash
    if path[-1] != '/': path += '/'
    self.rel_path = path

    # if a root is given, add this to the path
    if use_root: self.path = self.root + self.rel_path
    else:
      # save the path
      self.path = path

    # create directories if they don't already exist
    if not os.path.exists(self.path):
        try:
          os.makedirs(self.path)
        except FileExistsError:
          # file must have just been created
          pass

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
    Get the path to the highest index save in the path, or return None if empty,
    for example, if we have file_001, file_002, file_004, we return file_004
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

  def get_recent_folder(self, path, target_index=None, x_most_recent=None):
    """
    Get the most recent folder based on the date string stamp
    """

    # get all files with pickle extension in the target directory
    folders = [x for x in os.listdir(path) if os.path.isdir(path + x)]

    # create an intial entry, we compare our folders with this
    folder_elem = -1
    day = 0; month = 0; year = 0; hour = 0; min = 0; index = 0
    most_recent = [folder_elem, [year, month, day, hour, min, index]]

    all_entries = []

    # loop through all folder names to find which is the most recent
    for i, name in enumerate(folders):

      # TIME STRING MUST BE SEPERATED BY '-'
      # PARTS SEPERATED BY DASHES '-' CANNOT BE NUMERIC EXCEPT DATE eg not train-4-cluster
      dash_seperated = name.split('-')
      target = ["day", "month", "year", "hour", "min"]
      ind = 0
      for str in dash_seperated:
        if ind == 0 and len(str) >= 2 and str[-2:].isnumeric():
          target[0] = str[-2:]
          ind += 1
        elif ind > 0 and ind < 3:
          target[ind] = str
          ind += 1
        elif ind == 3:
          if str[2] != ':':
            raise RuntimeError("colon (:) not found in date string in ModelSaver"
              ", check to ensure only date information is seperated by dashes (-)"
              " unless the item between the dashes does not trigger isnumeric() == True")
          h, m = str.split(':')
          target[3] = h
          target[4] = m[:2]

      # do our folder names have trailing indexes, eg train_cluster..._array_11
      trailing_index = name.split('_')
      trailing_index = trailing_index[-1]
      if trailing_index.isnumeric():
        trailing_index = int(trailing_index)
      else:
        trailing_index = 0

      # convert date to a set of integers
      day = int(target[0])
      month = int(target[1])
      year = int(target[2])
      hour = int(target[3])
      min = int(target[4])

      # create an entry for this folder
      entry = [i, [year, month, day, hour, min, trailing_index]]
      all_entries.append(entry)

      # compare with current most recent to see if this is more recent
      for d in range(len(entry[1])):

        # if we are at the end and we have a target for the trailing index
        if (d == len(entry[1]) - 1 and target_index != None
            and entry[d] == target_index):
          most_recent[0] = entry[0]
          most_recent[1] = entry[1][:]
          break

        # if this entry is more recent, overwrite most recent
        if entry[d] > most_recent[d]:
          most_recent[0] = entry[0]
          most_recent[1] = entry[1][:]
          break
        elif entry[d] == most_recent[d]: continue
        else: break
          
    if most_recent[0] == -1:
      print("No recent folders found at path:", path)
      return None

    # if we don't want the most recent, but the x-th (eg 2nd, 4th,...)
    if x_most_recent != None:
      # sort all of the entries
      sorted_entries = []
      for i in range(len(all_entries)):
        inserted = False
        for j in range(len(sorted_entries)):
          for k in range(len(all_entries[0][1])):
            if (sorted_entries[j][1][k] < all_entries[i][1][k]):
              sorted_entries.insert(j, all_entries[i])
              inserted = True
              break
            elif sorted_entries[j][1][k] == all_entries[i][1][k]: continue
            else: break
          if inserted == True: break
        if inserted == False: sorted_entries.append(all_entries[i])
      if x_most_recent < 0 or x_most_recent >= len(sorted_entries):
        print("x_most recent (", x_most_recent, "), out of bounds - max is", 
              len(sorted_entries))
        return None
      return folders[sorted_entries[x_most_recent][0]]

    # return the name of the most recent folder
    return folders[most_recent[0]]

  def get_most_recent(self, label=None):
    """
    Get the most recent model trained in the path. First we check the root of
    the path, if that fails we check all the folders for the most recent
    timestamp, and then get the most recent in there
    """

    # see if there is a model file in the root
    root_recent = self.get_recent_file(self.path)

    if root_recent != None:
      print("Found the most recent file:", root_recent)
      return root_recent

    # see if there is a recent folder
    recent_folder = self.get_recent_folder(self.path)

    if recent_folder == None:
      print(f"No recent file or recent folder found in {self.path}")
      return None

    # go into this recent folder to find a recent file
    file_in_folder = self.get_recent_file(self.path + recent_folder)

    if file_in_folder == None:
      i = 1
      while True:
        print(f"No recent file found in {self.path + recent_folder}")
        recent_folder = self.get_recent_folder(self.path, x_most_recent=i)
        if recent_folder == None: break
        file_in_folder = self.get_recent_file(self.path + recent_folder)
        if file_in_folder == None: i += 1
        else:
          print("Found the most recent file:", recent_folder + "/" + file_in_folder)
          return recent_folder + "/" + file_in_folder
      print("No recent files found at all, giving up")
      return None

    print("Found the most recent file:", recent_folder + "/" + file_in_folder)
    return recent_folder + "/" + file_in_folder

  def new_folder(self, name=None, label=None, suffix=None, notimestamp=None):
    """
    Create a new training folder
    """

    if self.in_folder:
      self.exit_folder()

    save_label = ""

    folder_name = ""

    # if we have been given a full name for the folder
    if name != None:
      folder_name = name

    # else auto generate a name for the folder using given label information
    else:

      # add in a user specified label -> train_{label}
      if label != None:
        save_label += label

      # add in a timestamp unless told not to -> train_{DD-MM-YYYY-hr:min}
      if notimestamp != True:
        now = datetime.now()
        time_stamp = now.strftime(self.date_str)
        if not save_label.endswith('_') and len(save_label) > 0: save_label += '_'
        save_label += time_stamp

      # add in a user specified suffix -> train_{DD-MM-YYYY-hr:min}_{suffix}
      if suffix != None:
        if not save_label.endswith('_') and len(save_label) > 0: save_label += '_'
        save_label += suffix

      # create the folder name
      folder_name = self.folder_names.format(save_label)

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
    if use_compression:
      compressed_pickle(savepath + savename, pyobj)
    else:
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

  def copy_files(self, copyfilepath, copyfilename, copyto=None):
    """
    Copy a file from one location to the default save location.
    """

    copypath = self.root + self.path

    if self.in_folder: copypath += self.folder

    if copyto != None:
      copypath = copyto

    if copypath[-1] != '/': copypath += '/'

    try:
      shutil.copyfile(copyfilepath + copyfilename, copypath + copyfilename)
    except FileNotFoundError as e:
      print(f"Copyfile has failed, check you are running in mymujoco/rl, error message is {e}")
      return None

    return self.path

  def load(self, folderpath=None, foldername=None, id=None):
    """
    Load a model, by default loads the most recent in the current folder
    """

    # default is loading from the main path
    loadpath = self.path

    # if a different path to a file is specified
    if folderpath != None:
      loadpath = folderpath

    # if the file name is specified
    if foldername != None:
      loadpath += foldername
      if loadpath[-1] != '/': loadpath += '/'
      loadpath = self.get_recent_file(loadpath, id)

    else:
      # default: find file in current folder folder
      loadpath = self.path

      if self.in_folder: 
        loadpath += self.folder

      new_loadpath = self.get_recent_file(loadpath, id)

      if new_loadpath == None:
        print(f"No model found at path {new_loadpath}")
      else:
        loadpath = new_loadpath

    print(f"Loading file {loadpath} with pickle ... ", end="")
    if use_compression:
      loaded_obj = decompress_pickle(loadpath)
    else:
      with open(loadpath, 'rb') as f:
        loaded_obj = pickle.load(f)
    print("finished")

    self.last_loadpath = loadpath

    return loaded_obj


if __name__ == "__main__":

  obj = ModelSaver('models/demosave')

  # pyobj = [1, 2, 3, 4, 5]

  # obj.new_folder()
  # obj.new_folder(label="cluster")
  # obj.new_folder(label="cluster")

  # obj.save("hyperparams", txtstr="testing 1 2 3", txtonly=True)
  # obj.save("network1", pyobj=pyobj)
  # obj.save("network1", pyobj=pyobj)
  # obj.save("network1", pyobj=pyobj)
  # obj.save("network1", pyobj=pyobj)

  # obj.load()

  # obj.exit_folder()

  # obj.save("network2", pyobj=pyobj)
  # obj.save("network2", pyobj=pyobj)
  # obj.save("network2", pyobj=pyobj)
  # obj.save("network2", pyobj=pyobj)

  # obj.load()

  print(obj.get_most_recent())

    