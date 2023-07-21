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


# 雑音や誤差を含むロボットモデルを多数回走らせて、どの程度の姿勢誤差が生じるかを見積もるコード。
# この結果を元にsec5.3.py（またはそれ以降）でパーティクルをどの程度分散させるか決める

world = World(40, 0.1, False)

initial_pose = np.array([0, 0, 0]).T
robots = []
r = Robot(initial_pose, sensor=None, agent=Agent(0.1, 0.0))

for i in range(100):
    copy_r = copy.copy(r)
    copy_r.distance_until_noise = copy_r.noise_pdf.rvs()    # 最初に雑音が発生するタイミングを変える
    world.append(copy_r)    # ワールドに登録する（これでアニメーションが動く）
    robots.append(copy_r)   # オブジェクトのリストを登録

world.draw()

# 実行してみると、ロボットは円弧状に広がり、主に向きについて誤差が発生する（40秒で4m並進すること自体に対してはあまり誤差が生じていない）
# ことがわかる（Robotクラスで小石を踏んだときの誤差はθにしか生じないようにしているので、並進方向のばらつきが少ない）
poses = pd.DataFrame([[math.sqrt(r.pose[0]**2 + r.pose[1]**2), r.pose[2]] for r in robots], columns = ['r', 'theta'])
print(poses.transpose())
print(poses['theta'].var())
print(poses['theta'].mean())
print(math.sqrt(poses['theta'].var()/poses["r"].mean()))    # これが標準偏差σ_{ωv}である


# poses = pd.DataFrame([[math.sqrt(r.pose[0]**2+r.pose[1]**2), r.pose[2]]
#                      for r in robots], columns=['r', 'theta'])
# print(poses.transpose())
# print()
# print(poses["theta"].var())
# print(poses["theta"].mean())
# print(poses["r"].var())
# print(poses["r"].mean())
# print("on = " + str(math.sqrt(poses["theta"].var()/poses["r"].mean())))
# print("nn = " + str(math.sqrt(poses["r"].var()/poses["r"].mean())))
# print("oo = " + str(math.sqrt(poses["theta"].var()/poses["theta"].mean())))
