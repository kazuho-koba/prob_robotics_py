#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code for python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import seaborn as sns
from scipy.stats import norm


# Import data
data = pd.read_csv("../LNPR_BOOK_CODES/sensor_data/sensor_data_600.txt",
                   delimiter=" ", header=None, names=("date", "time", "lr", "lidar"))

# Draw a histogram of the data
# data["lidar"].hist(bins=max(data["lidar"])-min(data["lidar"]), align='left')
# plt.show()

# Draw a graph along time
# data.lidar.plot()
# plt.show()

# Draw a graph of the data, categorize by the time
data["hour"] = [e//10000 for e in data.time]
d = data.groupby("hour")
# d.lidar.mean().plot()
# plt.show()

# Draw histograms of the data by each hours
# d.lidar.get_group(6).hist()
# d.lidar.get_group(14).hist()
# plt.show()

# calculate probability to get each values on a certain time
each_hour = {i: d.lidar.get_group(
    i).value_counts().sort_index() for i in range(24)}
freqs = pd.concat(each_hour, axis=1)  # conbine data
freqs = freqs.fillna(0)  # fill the data indicates NA by 0
probs = freqs/len(data)
# print(probs)

# Draw heatmaps of each probabilities
# sns.heatmap(probs)
# sns.jointplot(data["hour"], data["lidar"], data, kind="kde")
# plt.show()

# calculate P(z) and P(t)
p_t = pd.DataFrame(probs.sum())  # get a sumation of each columns
p_t.transpose()
# print(p_t)
# print()
# p_t.plot()
# plt.show()
# print(p_t.sum())

p_z = pd.DataFrame(probs.transpose().sum())
# p_z.plot()
p_z.transpose()
# print(p_z)
# plt.show()
# print()
# print(p_z.sum())

# calculate P(z|t) and plot them at t=6 and t=14
cond_z_t = probs/p_t[0]
# print(cond_z_t)
# (cond_z_t[6]).plot.bar(color="blue", alpha=0.5)
# (cond_z_t[14]).plot.bar(color="orange", alpha=0.5)
# plt.show()

# get P(z=630|t=13) by Bayes' theorem
cond_t_z = probs.transpose()/probs.transpose().sum()  # get P(t|z)
# probability of the state which the sensor value indicates 630, whenever the time is
print("P(z=630) = ", p_z[0][630])
# probability of the state which the time is 13 o-clock
print("P(t=13) = ", p_t[0][13])
print("P(t=13|z=630) = ", cond_t_z[630][13])
print("Bayes P(z=630|t=13) = ", cond_t_z[630][13]*p_z[0][630]/p_t[0][13])
# the probability which the sensor indicates 630 at 13 pm
print("answer P(z=630|t=13) = ", cond_z_t[13][630])


# infer current time from the sensor data
def bayes_estimation(sensor_value, current_estimation):
    new_estimation = []
    for i in range(24):
        new_estimation.append(cond_z_t[i][sensor_value]*current_estimation[i])
    return new_estimation/sum(new_estimation) # normalization

# infer the time when the sensor value is 630
estimation = bayes_estimation(630, p_t[0])
# print(p_t[0])
# plt.plot(estimation)
# plt.show()

# infer the tiem when the sensor values are 630, 632, 636, in series
# actually, this is a sensor data gained at 5 am
value_5 = [630, 632, 636]
estimation = p_t[0]
for v in value_5:
    estimation = bayes_estimation(v, estimation)
# plt.plot(estimation)
# plt.show()

# infer the tiem when the sensor values are 617, 624, 619, in series
# actually, this is a sensor data gained at 11 am
value_11 = [617, 624, 619]
estimation = p_t[0]
for v in value_11:
    estimation = bayes_estimation(v, estimation)
plt.plot(estimation)
plt.show()