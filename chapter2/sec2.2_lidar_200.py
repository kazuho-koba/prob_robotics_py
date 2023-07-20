#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code for python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# Import data
data = pd.read_csv("../../LNPR_BOOK_CODES/sensor_data/sensor_data_200.txt",
                   delimiter=" ", header=None, names=("date", "time", "lr", "lidar"))

# Check the contents of the data
# print(data)
# print(data["lidar"][0:5])

# Draw a histogram
# data["lidar"].hist(bins=max(data["lidar"])-min(data["lidar"]), align='left')
# plt.show()

# Check the key value of the data
mean1 = sum(data["lidar"].values)/len(data["lidar"].values)
mean2 = data["lidar"].mean()
print(mean1, mean2)

# Draw a histogram with the mean indicator
# data["lidar"].hist(bins=max(data["lidar"])-min(data["lidar"]),
#                    color="orange", align='left')
# plt.vlines(mean1, ymin=0, ymax=5000, color="red")
# plt.show()

# Calculate variance by its definition
zs = data["lidar"].values
mean = sum(zs)/len(zs)
diff_square = [(z-mean)**2 for z in zs]
sampling_var = sum(diff_square)/(len(zs))  # sample variance
unbiased_var = sum(diff_square)/(len(zs)-1)  # unbiased variance
print(sampling_var)
print(unbiased_var)

# Calculate variance by the built-in func of pandas
pandas_sampling_var = data["lidar"].var(ddof=False)  # sample variance
pandas_default_var = data["lidar"].var()  # unbiased variance (by default)
print(pandas_sampling_var)
print(pandas_default_var)

# Calculate variance by the built-in func of numpy
numpy_default_var = np.var(data["lidar"])  # sample variance (by default)
numpy_unbiased_var = np.var(data["lidar"], ddof=1)  # unbiased variance
print(numpy_default_var)
print(numpy_unbiased_var)

# Calculate standard deviation by its definition
stddev1 = math.sqrt(sampling_var)
stddev2 = math.sqrt(unbiased_var)

# Calculate standard deviation by the built-in func of pandas
pandas_stddev = data["lidar"].std()

print(stddev1)
print(stddev2)
print(pandas_stddev)


# look up the freqency of the each sensor data's value
freqs = pd.DataFrame(data["lidar"].value_counts())
freqs["probs"] = freqs["lidar"]/len(data["lidar"])
print(freqs.transpose())
print(sum(freqs["probs"]))

# Plot the probability of each sensor data output
# freqs["probs"].sort_index().plot.bar()
# plt.show()


# Simulate the sensor value that you can get at next sampling
def drawing():
    return freqs.sample(n=1, weights="probs").index[0]


print(drawing())

# Simulate the sensor value for N times and draw a simulated histogram
print(len(data)) # 58988
# samples = [drawing() for i in range(len(data))] # it takes long time
samples = [drawing() for i in range(1000)]
simulated = pd.DataFrame(samples, columns=["lidar"])
p = simulated["lidar"]
p.hist(bins=max(p)-min(p), color="orange", align='left')
plt.show()
