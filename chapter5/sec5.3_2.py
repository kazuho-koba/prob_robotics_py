#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code for python3
import sys  # noqa
sys.path.append('../scripts/')  # noqa
from robot import *  # noqa
import math
import copy
from scipy.stats import multivariate_normal
import pandas as pd


world = World(40.0, 0.1)

initial_pose = np.array([0, 0, 0]).T
robots = []


for i in range(50):
    r = Robot(initial_pose, sensor=None, agent=Agent(0.0, 0.1))
    # register to the world so that it can move in your animation
    world.append(r)
    robots.append(r)  # register to reference list of the object

world.draw()

poses = pd.DataFrame([[math.sqrt(r.pose[0]**2+r.pose[1]**2), r.pose[2]]
                     for r in robots], columns=['r', 'theta'])
print(poses.transpose())
print()
print(poses["theta"].var())
print(poses["theta"].mean())
print(poses["r"].var())
print(poses["r"].mean())
# print("on = " + str(math.sqrt(poses["theta"].var()/poses["r"].mean())))
# print("nn = " + str(math.sqrt(poses["r"].var()/poses["r"].mean())))
print("oo = " + str(math.sqrt(poses["theta"].var()/poses["theta"].mean())))
