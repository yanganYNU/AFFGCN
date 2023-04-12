"""
coding:utf-8
@Time    : 2022/7/11 2:53
@Author  : Alex-杨安
@FileName: train.py
@Software: PyCharm
"""
import pandas as pd
import numpy as np
import time
from utils import split_valid, one_hot_skyc, one_hot
from utils import Count_num_miss_value, one_hot_wxcodes

time_start = time.time()
print('reading weather data...')
pd.set_option('display.width', None)
# pd.set_option('display.max_rows', None)
wx_data = pd.read_csv('data/PeMS04.csv')
Count_num_miss_value(wx_data)
wx_data = wx_data.interpolate(method='linear', limit_direction='backward')
wx_data = wx_data.interpolate(method='linear', limit_direction='forward').round(2)
wx_data = wx_data.interpolate(method='pad', limit_direction='forward')
Count_num_miss_value(wx_data)
print(wx_data)

for index, row in wx_data.iterrows():
    skyc1 = row[14]
    skyc1 = one_hot_skyc(skyc1)
    wx_data.loc[index, 'skyc1'] = skyc1

    skyc2 = row[15]
    skyc2 = one_hot_skyc(skyc2)
    wx_data.loc[index, 'skyc2'] = skyc2

    skyc3 = row[16]
    skyc3 = one_hot_skyc(skyc3)
    wx_data.loc[index, 'skyc3'] = skyc3

    wxcodes = row[17]
    wxcodes = one_hot_wxcodes(wxcodes)
    wx_data.loc[index, 'wxcodes'] = wxcodes
    valid = row[0]
    str_valid, minute = split_valid(valid)
    if minute != 0 and minute != 5 and minute != 10 and minute != 15 and minute != 20 and minute != 25 and minute != 30 and minute != 35 and minute != 40 and minute != 45 and minute != 50 and minute != 55:
        wx_data.drop(index, axis=0, inplace=True)
    if index % 2000 == 0:
        print(wx_data[index:index+1])
Count_num_miss_value(wx_data)
wx_data.drop(['valid'], axis=1, inplace=True)
for i in range(25):
    j = str(i + 1)
    wx_data[j] = np.nan
for index, row in wx_data.iterrows():
    onehot = []

    skyc1 = row[13]
    skyc1 = skyc1.split(',')
    skyc1 = list(map(int, skyc1))
    onehot.extend(skyc1)

    skyc2 = row[14]
    skyc2 = skyc2.split(',')
    skyc2 = list(map(int, skyc2))
    onehot.extend(skyc2)

    skyc3 = row[15]
    skyc3 = skyc3.split(',')
    skyc3 = list(map(int, skyc3))
    onehot.extend(skyc3)

    wxcodes = row[16]
    wxcodes = wxcodes.split(',')
    wxcodes = list(map(int, wxcodes))
    onehot.extend(wxcodes)

    one_hot(index, wx_data, onehot)

wx_data.drop(['wxcodes'], axis=1, inplace=True)  # 在本来的data 上删除
wx_data.drop(['skyc1'], axis=1, inplace=True)  # 在本来的data 上删除
wx_data.drop(['skyc2'], axis=1, inplace=True)  # 在本来的data 上删除
wx_data.drop(['skyc3'], axis=1, inplace=True)  # 在本来的data 上删除
print(wx_data)
np.savez('A:/traffic_prediction/AFFGCN/AFFGCN/data/PEMS04/weather.npz', wx_data)
time_end = time.time()
time_sum = time_end - time_start
print('time of final use：', round((time_end - time_start), 2), 's')
