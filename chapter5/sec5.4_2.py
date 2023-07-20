#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code for python3
import sys  # noqa
sys.path.append('../scripts/')  # noqa
from robot import *  # noqa
import math
import pandas as pd
from scipy.stats import multivariate_normal

m = Map()
m.append_landmark(Landmark(1, 0))

distance = []
direction = []

for i in range(1000):
    c = Camera(m)  # generate a new camera per every iteration
    d = c.data(np.array([0.0, 0.0, 0.0]).T)
    if len(d) > 0:
        distance.append(d[0][0][0])
        direction.append(d[0][0][1])
    # print(i)

df = pd.DataFrame()
df["distance"] = distance
df["direction"] = direction
print(df)
print(df.std())
print(df.mean())