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
# 2.1節
#########
# データの中身を確認する
print(data)         # データ全体
print()
print(type(data))   # データの型

# データのうちlidarのセンサ値だけ出力する
print(data["lidar"])

