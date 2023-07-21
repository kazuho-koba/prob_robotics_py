#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code for python3
import sys  
from robot import *
from mcl import *
import math
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse

def sigma_ellipse(p, cov, n):
    eig_vals, eig_vec = np.linalg.eig(cov)
    ang = math.atan2(eig_vec[:,0][1], eig_vec[:,0][0])/math.pi*180
    return Ellipse(p, width=2*n*math.sqrt(eig_vals[0]),height=2*n*math.sqrt(eig_vals[1]),
                   angle=ang, fill=False, color='blue', alpha=0.5)

class KalmanFilter:
    def __init__(self, envmap, init_pose,
                 motion_noise_stds={"nn":0.23, "no":0.001, "on":0.11, "oo":0.05}):
        self.belief = multivariate_normal(mean=np.array([0.0, 0.0, math.pi/4]),
                                          cov=np.diag([0.1, 0.2, 0.01]))
        self.pose = self.belief.mean

    def motion_update(self, nu, omega, time):
        pass

    def observation_update(self, observation):
        pass

    def draw(self, ax, elems):
        # xy平面上の誤差3σ範囲
        e = sigma_ellipse(self.belief.mean[0:2], self.belief.cov[0:2, 0:2], 3)
        elems.append(ax.add_patch(e))

        # θ方向の誤差3σ範囲
        x, y, c = self.belief.mean
        sigma3 = math.sqrt(self.belief.cov[2, 2])*3
        xs = [x + math.cos(c - sigma3), x, x + math.cos(c + sigma3)]
        ys = [y + math.sin(c - sigma3), y, y + math.sin(c + sigma3)]
        elems += ax.plot(xs, ys, color = 'blue', alpha=0.5)

def trial_gaussian():
    time_interval = 0.1
    world = World(30, time_interval, debug=False)

    # 地図生成、ランドマーク追加
    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]: m.append_landmark(Landmark(*ln))
    world.append(m)

    # ロボット生成
    initial_pose = np.array([0, 0, 0]).T
    kf = KalmanFilter(m, initial_pose)
    circling = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, kf)
    r = Robot(initial_pose, sensor=Camera(m), agent=circling, color='red')
    world.append(r)

    world.draw()
trial_gaussian()