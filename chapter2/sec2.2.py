#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code for python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# Import data
data = pd.read_csv("../LNPR_BOOK_CODES/sensor_data/sensor_data_200.txt",
                   delimiter=" ", header=None, names=("date", "time", "lr", "lidar"))


#########
# 2.2節
#########
# lidarデータをヒストグラムにする（引数binsは各区間（ビン）の数で、今回はビンの幅を1にするためにデータの数だけ用意した）
data["lidar"].hist(bins = max(data["lidar"]) - min(data["lidar"]), align = 'left')
plt.show() # 図を表示

# lidarデータの平均値を出力する
mean1 = sum(data["lidar"].values)/len(data["lidar"].values)     # 定義に則った計算
mean2 = data["lidar"].mean()                                    # pandasの機能による計算
print(mean1, mean2)

# lidarデータを再度ヒストグラムに、今度は平均値を表示する
data["lidar"].hist(bins = max(data["lidar"]) - min(data["lidar"]), color = 'orange', align = 'left')
plt.vlines(mean1, ymin=0, ymax=5000, color='red')
plt.show() # 図を表示

# 分散と標準偏差を計算する
# 定義から計算する
zs = data["lidar"].values
mean = sum(zs)/len(zs)
diff_square = [(z-mean)**2 for z in zs]

sampling_var = sum(diff_square)/len(zs)           # 標本分散
unbiased_var = sum(diff_square)/(len(zs) - 1)     # 不偏分散
print('comutation based on their definitions: ')
print(sampling_var)
print(unbiased_var)

stddev1 = math.sqrt(sampling_var)
stddev2 = math.sqrt(unbiased_var)
print(stddev1, stddev2)
print()

# pandasの機能で計算する
print('computation based on pandas functions: ')
pandas_sampling_var = data["lidar"].var(ddof = False)   # 標本分散
pandas_default_var = data["lidar"].var()                # 不偏分散（こちらがデフォルト）

print(pandas_sampling_var)
print(pandas_default_var)

pandas_stddev = data["lidar"].std()
print(pandas_stddev)
print()

# numpyの機能で計算する
print('computation based on numpy functions: ')
numpy_sampling_var = np.var(data["lidar"])              # 標本分散（こちらがデフォルト
numpy_unbiased_var = np.var(data["lidar"], ddof = 1)    # 不偏分散

print(numpy_sampling_var)
print(numpy_unbiased_var)
print()
print()


# 確率分布を考える。まずは過去N個のセンサデータから素朴に確率分布を求める：
# すなわち、ある値xが得られる確率 = xが得られた回数／N　という計算をする
freqs = pd.DataFrame(data["lidar"].value_counts()).reset_index()    # lidarのセンサ値とそれが得られた回数のデータフレームを得る（lidarというラベルが消えるので、後でつけ直すつもりで一度リセット）
freqs.columns = ['lidar', 'count']                                  # lidar, countというラベルをつけ直す

print(freqs.transpose())                                # 転置して表示
freqs["probs"] = freqs["count"]/sum(freqs["count"])     # 確率の列を追加
print(freqs.transpose())                                # 再度表示
print()
print(sum(freqs["probs"]))      # 確率の合計値を表示（1になるはず）
print()
# 確率分布を表示する
plt.bar(freqs['lidar'], freqs['probs'])
plt.show()

# 得られた確率分布をベースに新たなセンサ値をシミュレーションデータとして得る
def drawing():
    a = freqs.sample(n=1, weights="probs")["lidar"]
    return a
drawed_data = drawing()
print(drawed_data.values[0])
