#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code for python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import seaborn as sns
from scipy.stats import norm, multivariate_normal


# Import data
data = pd.read_csv("../../LNPR_BOOK_CODES/sensor_data/sensor_data_700.txt",
                   delimiter=" ", header=None, names=("date", "time", "ir", "lidar"))

# extract data during 12 to 16
d = data[(data["time"] < 160000) & (data["time"] >= 120000)]
d = d.loc[:, ["ir", "lidar"]]
# print(d)

# Draw graph of 2-D gaussian distribution
# sns.jointplot(d["ir"], d["lidar"], d, kind="kde")
# plt.show()

# get variances
print("variance of IR sensor: ", d.ir.var())
print("variance of LiDAR: ", d.lidar.var())

diff_ir = d.ir - d.ir.mean()
diff_lidar = d.lidar - d.lidar.mean()

a = diff_ir * diff_lidar
print("covariance: ", sum(a)/(len(d)-1))
print(d.mean())

# easier way to get covariance
print()
print(d.cov())

# draw gaussian distribution
# c = d.cov().values + np.array([[0, -20], [-20, 0]])  # modify covariance
# irlidar = multivariate_normal(mean=d.mean().values.T, cov=d.cov().values)
# irlidar = multivariate_normal(mean=d.mean().values.T, cov=c)
# x, y = np.mgrid[0:40, 710:750]  # make x-y coordinate on 2-D plane
# x is 2-D list with 40*40, add 3rd dimention and make 40*40*2
# pos = np.empty(x.shape+(2,))
# substitute x and y of 3rd dimention
# pos[:, :, 0] = x
# pos[:, :, 1] = y
# get x and y coordinate value and corresponding density
# cont = plt.contour(x, y, irlidar.pdf(pos))
# cont.clabel(fmt='%1.1e')  # set a format
# plt.show()

data_2 = pd.read_csv("../../LNPR_BOOK_CODES/sensor_data/sensor_data_200.txt",
                     delimiter=" ", header=None, names=("date", "time", "ir", "lidar"))
d2 = data_2.loc[:, ["ir", "lidar"]]  # extract data on IR and LiDAR
# sns.jointplot(d2["ir"], d2["lidar"], d2, kind="kde")
# print(d2.cov())
# print(d2.mean())
# print(d2)
# plt.show()

x2, y2 = np.mgrid[280:340, 190:230]
pos2 = np.empty(x2.shape+(2,))
pos2[:, :, 0] = x2
pos2[:, :, 1] = y2
# make 2-D gaussian distribution
irlidar2 = multivariate_normal(mean=d2.mean().values.T, cov=d2.cov().values)
# cont2 = plt.contour(x2, y2, irlidar2.pdf(pos2))  # make contour of the density
# cont2.clabel(fmt='%1.1e')
# plt.show()

x3, y3 = np.mgrid[0:200, 0:100]
pos3 = np.empty(x3.shape+(2,))
pos3[:, :, 0] = x3
pos3[:, :, 1] = y3
a = multivariate_normal(mean=[50, 50], cov=[[50, 0], [0, 100]])
b = multivariate_normal(mean=[100, 50], cov=[[125, 0], [0, 25]])
c = multivariate_normal(mean=[150, 50], cov=[
                        [100, -25*math.sqrt(3)], [-25*math.sqrt(3), 50]])
# for e in [a, b, c]:
#     plt.contour(x3, y3, e.pdf(pos3))
# plt.gca().set_aspect('equal')
# plt.gca().set_xlabel('x')
# plt.gca().set_ylabel('y')
# plt.show()

# get eigen values and vectors of gaussian distribution:c
eig_vals, eig_vec = np.linalg.eig(c.cov)
print("eig_vals: ", eig_vals)
print("eig_vec: ", eig_vec)
print("eigen vector 1: ", eig_vec[:, 0])
print("eigen vector 2: ", eig_vec[:, 1])

plt.contour(x3, y3, c.pdf(pos3))
v = 2*math.sqrt(eig_vals[0])*eig_vec[:, 0]
plt.quiver(c.mean[0], c.mean[1], v[0], v[1], color="red",
           angles='xy', scale_units='xy', scale=1)
v = 2*math.sqrt(eig_vals[1])*eig_vec[:, 1]
plt.quiver(c.mean[0], c.mean[1], v[0], v[1], color="blue",
           angles='xy', scale_units='xy', scale=1)
plt.gca().set_aspect('equal')
# plt.show()

V = eig_vec
L = np.diag(eig_vals)
print("calculated covariance: \n", V.dot(L.dot(np.linalg.inv(V))))
print("original covariance: \n", np.array(
    [[100, -25*math.sqrt(3)], [-25*math.sqrt(3), 50]]))
