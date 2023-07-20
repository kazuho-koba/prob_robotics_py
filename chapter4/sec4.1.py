#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code for python3

from scipy.stats import norm, multivariate_normal
import seaborn as sns
import random
import math
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import pandas as pd
import matplotlib
import sys
from ideal_robot import *


class Robot(IdealRobot):
    pass


world = World(30, 0.1)

for i in range(100):
    circling = Agent(0.2, 10.0/180*math.pi)
    r = Robot(np.array([0, 0, 0]).T, sensor=None, agent=circling)
    world.append(r)

world.draw()
