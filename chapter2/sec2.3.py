#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code for python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from scipy.stats import norm


# Import data
data = pd.read_csv("../../LNPR_BOOK_CODES/sensor_data/sensor_data_200.txt",
                   delimiter=" ", header=None, names=("date", "time", "lr", "lidar"))
# Check the key value of the data
mean1 = sum(data["lidar"].values)/len(data["lidar"].values)
mean2 = data["lidar"].mean()
# Calculate variance by its definition
zs = data["lidar"].values
mean = sum(zs)/len(zs)
diff_square = [(z-mean)**2 for z in zs]
sampling_var = sum(diff_square)/(len(zs))  # sample variance
unbiased_var = sum(diff_square)/(len(zs)-1)  # unbiased variance
# Calculate standard deviation by its definition
stddev1 = math.sqrt(sampling_var)
stddev2 = math.sqrt(unbiased_var)


# Draw a Gaussian distribution
def p(z, mu=209.7, dev=23.4):
    return math.exp(-(z-mu)**2/(2*dev))/math.sqrt(2*math.pi*dev)


zs = range(190, 230)
ys = [p(z) for z in zs]
# print(ys)
# plt.plot(zs, ys)
# plt.show()


# Draw a probability distribution as if sensor value is limited to integer number
def prob(z, width=0.5):
    return width*(p(z-width)+p(z+width))


zs = range(190, 230)
ys = [prob(z) for z in zs]
freqs = pd.DataFrame(data["lidar"].value_counts())
freqs["probs"] = freqs["lidar"]/len(data["lidar"])

# plt.bar(zs, ys, color="red", alpha=0.3)  # alpha makes graph transparent
f = freqs["probs"].sort_index()
# plt.bar(f.index, f.values, color="blue", alpha=0.3)
# plt.show()

# draw an probability density function(pdf) by built-in func of scipy
zs = range(190, 230)
ys = [norm.pdf(z, mean1, stddev1) for z in zs]
# plt.plot(zs, ys)
# plt.show()

# draw an cumulative distribution function(cdf) by built-in func of scipy
ys = [norm.cdf(z, mean1, stddev1) for z in zs]
# plt.plot(zs, ys, color="red")
# plt.show()

# draw an pdf by other method
ys = [norm.cdf(z+0.5, mean1, stddev1)-norm.cdf(z-0.5, mean1, stddev1)
      for z in zs]
# plt.bar(zs, ys)
# plt.show()

# get expectation value of dices
samples = [random.choice([1, 2, 3, 4, 5, 6]) for i in range(10000)]
e = sum(samples)/len(samples)
print(e)
