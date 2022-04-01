#!/usr/bin/env python3

# script to plot finger data

# raw data
finger_1_x = [ 
  0,
0.023175,
0.0459481,
0.0681783,
0.0898142,
0.110878,
0.131451,
0.151656,
0.171645,
0.191525
]

finger_1_y = [ 
  0,
0.00271359,
0.00779595,
0.0148855,
0.0236224,
0.03366,
0.0446694,
0.0563401,
0.0683767,
0.0805918
]

finger_1_coeff = [ 
  -4.56156,
2.83258,
0.045613,
8.22E-05
]

finger_2_x = [
  0,
0.0228516,
0.0445205,
0.0646846,
0.0833377,
0.100708,
0.117178,
0.133222,
0.149215,
0.165163
]

finger_2_y = [ 
  0,
0.00471705,
0.0133715,
0.0251127,
0.0391308,
0.0547103,
0.0712381,
0.0881809,
0.105171,
0.122203
]

finger_2_coeff = [
  -12.6411,
6.41935,
0.0244705,
0.000363735
]

finger_1_errors = [ 
  -8.22E-05,
0.000109781,
8.02E-05,
-2.75E-05,
-0.000100909,
-8.54E-05,
7.22E-06,
0.000103164,
7.95E-05,
-8.39E-05,
]

finger_2_errors = [ 
  -0.000363735,
0.000592825,
0.000310147,
-0.000271788,
-0.000539161,
-0.000311799,
0.000203215,
0.000515401,
0.000225849,
-0.000360958
]

from matplotlib import pyplot as plt
import numpy as np

f1_x = np.array(finger_1_x)
f1_y = np.array(finger_1_y)
f1_c = np.array(finger_1_coeff)

f2_x = np.array(finger_2_x)
f2_y = np.array(finger_2_y)
f2_c = np.array(finger_2_coeff)

f1_xpred = np.arange(f1_x[0], f1_x[-1], 0.001)
f2_xpred = np.arange(f2_x[0], f2_x[-1], 0.001)

f1_ypred = np.zeros(len(f1_xpred))
f2_ypred = np.zeros(len(f2_xpred))

for i in range(len(f1_xpred)):
  f1_ypred[i] = (f1_c[0] * f1_xpred[i] ** 3
                + f1_c[1] * f1_xpred[i] ** 2
                + f1_c[2] * f1_xpred[i] ** 1
                + f1_c[3])

for i in range(len(f2_xpred)):
  f2_ypred[i] = (f2_c[0] * f2_xpred[i] ** 3
                + f2_c[1] * f2_xpred[i] ** 2
                + f2_c[2] * f2_xpred[i] ** 1
                + f2_c[3])

fig, ax = plt.subplots(1, 1)

ax.plot(f1_xpred, f1_ypred, label="Finger 1 cubic approximation")
ax.plot(f1_x, f1_y, '*', label="Finger 1 actual points")
ax.set_aspect('equal', 'box')

ax.plot(f2_xpred, f2_ypred, label="Finger 2 cubic approximation")
ax.plot(f2_x, f2_y, '*', label="Finger 2 actual points")

ax.set_aspect('equal', 'box')
# ax.set_title("Curve fitting gripper fingers")
ax.set_xlabel("X / m")
ax.set_ylabel("Y / m")
ax.legend()

f1error = np.mean(np.abs(np.array(finger_1_errors)))
f2error = np.mean(np.abs(np.array(finger_2_errors)))

print("Finger 1 average error:", f1error)
print("Finger 2 average error:", f2error)

ax.text(0.1, 0.010, "Finger 1 average error: %.0f $\mu$m" % (f1error * 1e6))
ax.text(0.1, 0.000, "Finger 2 average error: %.0f $\mu$m" % (f2error * 1e6))

plt.show()

