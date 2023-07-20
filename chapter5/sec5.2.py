#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code for python3
import sys  # noqa
sys.path.append('../scripts/')  # noqa
from robot import *  # noqa
import math


class EstimationAgent(Agent):
    def __init__(self, nu, omega, estimator):
        super().__init__(nu, omega)
        self.estimator = estimator

    def draw(self, ax, elems):
        self.estimator.draw(ax, elems)
        elems.append(ax.text(0, 0, "hoge", fontsize=10))


class Particle:
    def __init__(self, init_pose):
        self.pose = init_pose


class Mcl:
    def __init__(self, init_pose, num):
        self.particles = [Particle(init_pose) for i in range(num)]

    def draw(self, ax, elems):
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]
        vxs = [math.cos(p.pose[2]) for p in self.particles]
        vys = [math.sin(p.pose[2]) for p in self.particles]
        elems.append(ax.quiver(xs, ys, vxs, vys, color="blue", alpha=0.5))


world = World(30, 0.1)

# generate a map and add 3 landmarks
m = Map()
for ln in [(-4, 2), (2, -3), (3, 3)]:
    m.append_landmark(Landmark(*ln))
world.append(m)

# generate & deploy robots
initial_pose = np.array([2, 2, math.pi/6]).T
estimator = Mcl(initial_pose, 100)  # generate a particle filter
circling = EstimationAgent(0.2, 10.0/180*math.pi, estimator)
r = Robot(initial_pose, sensor=Camera(m), agent=circling)
world.append(r)

# execute animation
world.draw()
