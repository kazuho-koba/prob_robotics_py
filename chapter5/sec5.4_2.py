#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code for python3
from robot import *  # noqa
import math
import pandas as pd
from scipy.stats import multivariate_normal

########################################
# カメラによる観測誤差をシミュレーションにより評価するためのコード
########################################
m = Map()
m.append_landmark(Landmark(1, 0))

distance = []
direction = []

for i in range(1000):
    c = Camera(m)  # バイアス（何の？観測の？）の影響も考慮するためカメラは毎回新規作成する
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
# 実行すると、距離の標準偏差が0.55、角度の標準偏差が0.13くらいになる
# 本に比べてだいぶ大きいが理由がわからない（robot.pyにあるCameraクラスのノイズ設定は本と同じように思えるが・・・？