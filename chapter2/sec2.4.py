#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code for python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm
import seaborn as sns

# Import data
data = pd.read_csv("../LNPR_BOOK_CODES/sensor_data/sensor_data_600.txt",
                   delimiter=" ", header=None, names=("date", "time", "lr", "lidar"))

# lidarデータをヒストグラムにする（引数binsは各区間（ビン）の数で、今回はビンの幅を1にするためにデータの数だけ用意した）
plt.figure()
data["lidar"].hist(bins = max(data["lidar"]) - min(data["lidar"]), align = 'left')


# 表示してみると、マルチモーダル（モード：最頻値が複数ある）ことがわかる。
# とりあえず一度時系列にプロットして眺めてみる。
plt.figure()
data.lidar.plot()

# 山が3つあることがわかった。時分秒の6桁で1秒ごとに取得しているデータを時間でわけてみる
data["hour"] = [e//10000 for e in data.time]
# print(data)
d = data.groupby("hour")    # データを時間ごとにグループ分けする
plt.figure()
d.lidar.mean().plot()       # 各グループの平均値をプロットする

# 6時台の値が大きく、14時台の値が小さくなるらしい。ここに絞って再度ヒストグラムを表示してみる
plt.figure()
d.lidar.get_group(6).hist()
d.lidar.get_group(14).hist()

# 時刻と値ごとにそれが得られる確率を求める
common_index = np.arange(607, 645)     # 予め共通のインデックス（lidarセンサ値に対応）を用意しておく、時間ごとに出てこないセンサ値があったりして結合がうまく行かないので
each_hour = {i:d.lidar.get_group(i).value_counts().sort_index() for i in range(24)}
each_hour_df = {i: pd.DataFrame(data, index=common_index) for i, data in each_hour.items()}
for key in each_hour_df.keys():
    each_hour_df[key].rename({each_hour_df[key].columns[0]: key}, axis='columns', inplace=True)
freqs = pd.concat(each_hour_df.values(), axis=1)    # concatで連結
freqs.columns = freqs.columns.get_level_values(0)
# print(freqs.head())
freqs = freqs.fillna(0)                 # 欠損値を0で埋める
probs = freqs/len(data)                 # 頻度から確率に
# print(probs)

# 確率をヒートマップ化する
plt.figure()
sns.heatmap(probs)

# 確率をジョイントプロットで表示する
# plt.figure()
# sns.jointplot(x=data["hour"], y=data["lidar"], data=data, kind="kde")

plt.figure()
p_t = pd.DataFrame(probs.sum())             # 各列を合計
p_t.plot()
# print(p_t.transpose())
print(p_t.sum()) # 確率の合計なので1になるはず

plt.figure()
p_z = pd.DataFrame(probs.transpose().sum()) # 各列を合計（転置前でいうと各行ということに
p_z.plot()


plt.show() # 図を表示
