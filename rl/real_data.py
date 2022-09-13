#!/usr/bin/env python3

import csv
from matplotlib import pyplot as plt
import numpy as np

path = "/home/luke/Pictures/finger_bending_pics"

filename = "csv_f3_{0}g.csv"

masses = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]

real_data = []

fig, axs = plt.subplots(5, 2)

for j, mass in enumerate(masses):

  fullpath = path + "/" + filename.format(mass)

  data_entry = []

  with open(fullpath, newline='') as csvfile:
    
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')

    for i, row in enumerate(reader):

      if i == 0: continue # this row is simply commas (eg ,,,,,,)

      columns = ', '.join(row).split(',')

      datapoint = [0, 0]
      datapoint[0] = float(columns[4])
      datapoint[1] = float(columns[5])
      
      data_entry.append(datapoint)

  data_entry = np.array(data_entry)

  # zero the data
  offset = data_entry[0]
  data_entry[:, 0] -= offset[0]
  data_entry[:, 1] -= offset[1]

  real_data.append(data_entry)

length = real_data[0][-1][0]
maxdef = real_data[-1][-1][1]
factor = 235 / length

# scale all
for i, d in enumerate(real_data):

  d *= factor

  print(d)

  if i < 5:
    row = i
    col = 0
  else:
    row = i - 5
    col = 1

  axs[row][col].plot(d[:,0], d[:,1], label="{0}g".format(masses[i]))
  axs[row][col].axis("equal")
  axs[row][col].set_xlim([0, length * factor])
  axs[row][col].set_ylim([0, maxdef * factor])
  axs[row][col].legend(loc="upper left")

plt.show()