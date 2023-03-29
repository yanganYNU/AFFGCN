"""
coding:utf-8
@Time    : 2022/7/16 5:32
@Author  : Alex-杨安
@FileName: weather_08_02.py
@Software: PyCharm
"""

import pandas as pd
import numpy as np
import time
from utils import Count_num_miss_value, one_hot

time_start = time.time()
print('reading weather data...')
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
wx_data = pd.read_csv('data/test.csv')
print(wx_data.head(5))
Count_num_miss_value(wx_data)
for i in range(16):
    j = str(i+1)
    wx_data[j] = np.nan
print(wx_data.head(5))
Count_num_miss_value(wx_data)
for index, row in wx_data.iterrows():
    onehot = []

    skyc1 = row[11]
    skyc1 = skyc1.split(',')
    skyc1 = list(map(int, skyc1))
    onehot.extend(skyc1)

    skyc2 = row[12]
    skyc2 = skyc2.split(',')
    skyc2 = list(map(int, skyc2))
    onehot.extend(skyc2)

    wxcodes = row[13]
    wxcodes = wxcodes.split(',')
    wxcodes = list(map(int, wxcodes))
    onehot.extend(wxcodes)

    one_hot(index, wx_data, onehot)
    if index % 2000 == 0:
        print(wx_data[index:index+1],'\n')
wx_data.drop(['wxcodes'], axis=1, inplace=True)  # 在本来的data 上删除
wx_data.drop(['skyc1'], axis=1, inplace=True)  # 在本来的data 上删除
wx_data.drop(['skyc2'], axis=1, inplace=True)  # 在本来的data 上删除
print(wx_data.head(5))
np.savez('A:/traffic_prediction/AFFGCN/AFFGCN/data/PEMS08/weather.npz', wx_data)
