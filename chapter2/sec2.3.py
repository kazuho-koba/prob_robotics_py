#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code for python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm

# Import data
data = pd.read_csv("../LNPR_BOOK_CODES/sensor_data/sensor_data_200.txt",
                   delimiter=" ", header=None, names=("date", "time", "lr", "lidar"))

# センサ値が平均209.7、分散23.7の正規分布（ガウス分布）として得られると仮定して、値zが得られる確率を返す関数を実装する
def p(z, mu=209.7, dev=23.4):
    return math.exp(-(z-mu)**2/(2*dev))/math.sqrt(2*math.pi*dev)

# 実装した関数pを使ってガウス分布を表示する
zs = range(190, 230)
ys = [p(z) for z in zs]
plt.plot(zs, ys)
plt.show()

# センサ値が整数に限定される場合の離散化したガウス分布を表示する（各整数xについてx-0.5~x+0.5の区間で積分して表示）
def prob(z, width = 0.5):
    # ガウス分布の関数の積分は難しいので台形公式で近似
    return width * (p(z-width) + p(z+width))
zs = range(190, 230)
ys = [prob(z) for z in zs]
# 離散化したガウス分布の描画
plt.bar(zs, ys, color='red', alpha=0.3)     # alphaはグラフ描画の透明度を決める引数

# センサ値（実績）も再度プロット
freqs = pd.DataFrame(data["lidar"].value_counts()).reset_index()    # lidarのセンサ値とそれが得られた回数のデータフレームを得る（lidarというラベルが消えるので、後でつけ直すつもりで一度リセット）
freqs.columns = ['lidar', 'count']                                  # lidar, countというラベルをつけ直す
freqs["probs"] = freqs["count"]/sum(freqs["count"])                 # 確率の列を追加
plt.bar(freqs['lidar'], freqs['probs'], color='blue', alpha=0.3)    # 確率分布を表示

# 図を表示
plt.show()