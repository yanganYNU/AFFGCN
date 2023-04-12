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
from utils import split_valid, change_minte_data, one_hot_skyc, get_tidy_data
from utils import Count_num_miss_value, insert_data, one_hot_wxcodes

time_start = time.time()
print('reading weather data...')
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
wx_data = pd.read_csv('data/PeMS08.csv', nrows=2934)
Count_num_miss_value(wx_data)
cols = list(wx_data)
print(cols)
cols.insert(14, cols.pop(cols.index('wxcodes')))
cols.insert(13, cols.pop(cols.index('skyc2')))
cols.insert(12, cols.pop(cols.index('skyc1')))
print(cols)
wx_data = wx_data.loc[:, cols]

for index, row in wx_data.iterrows():
    valid = row[0]
    str_valid, minute = split_valid(valid)
    wx_data.loc[index, 'valid'] = change_minte_data(str_valid, minute)

    skyc1 = row[12]
    skyc1 = one_hot_skyc(skyc1)
    wx_data.loc[index, 'skyc1'] = skyc1

    skyc2 = row[13]
    skyc2 = one_hot_skyc(skyc2)
    wx_data.loc[index, 'skyc2'] = skyc2

    wxcodes = row[14]
    wxcodes = one_hot_wxcodes(wxcodes)
    wx_data.loc[index, 'wxcodes'] = wxcodes


wx_data = wx_data.interpolate(method='linear', limit_direction='backward')
wx_data = wx_data.interpolate(method='linear', limit_direction='forward').round(2)
wx_data = wx_data.interpolate(method='pad', limit_direction='forward')
print('processing missing values and timestamps...')
print(wx_data.head(5))
Count_num_miss_value(wx_data)

wx_seq = pd.DataFrame(columns=wx_data.columns)
wx_seq.loc[0] = ['2016/7/1 0:00', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                 np.nan, '1,0,0,0,0,0', '0,0,0,0,1,0', '1,0,0,0']
year = '2016'
month = 7
day = 1
hour = 0
minute = 0
current_time = [0]
print('processing missing timestamps...')
for i in range(17855):
    tmstp, minute, hour, day, month = get_tidy_data(minute, hour, day, month, year)
    wx_seq = insert_data(i, tmstp, wx_data, wx_seq, time_start, current_time)

wx_seq = wx_seq.interpolate(method='linear', limit_direction='backward')
wx_seq = wx_seq.interpolate(method='linear', limit_direction='forward')
wx_seq = wx_seq.interpolate(method='pad', limit_direction='forward')
Count_num_miss_value(wx_seq)
print(wx_seq.head(5))
wx_seq.drop(['valid'], axis=1, inplace=True)
print(wx_seq.head(5))

wx_seq.to_csv(path_or_buf='A:/traffic_prediction/AFFGCN/weather/PEMS08/data/test.csv', index=False)

time_end = time.time()
time_sum = time_end - time_start
print('time of final use：', round((time_end - time_start), 2), 's')
