"""
coding:utf-8
@Time    : 2022/7/11 2:53
@Author  : Alex-杨安
@FileName: utils.py
@Software: PyCharm
"""

import pandas as pd
import numpy as np
import time


def Count_num_miss_value(wx_data):
    t = 0
    for index, row in wx_data.iterrows():
        for column_name, Series_values in row.iteritems():
            if Series_values != Series_values:
                t = t + 1
    print('the number of missing values are', t)


def one_hot(index, wx_data, onehot):
    for i in range(25):
        j = str(i + 1)
        wx_data.loc[index, j] = onehot[i]
    if index % 2000 == 0:
        print(wx_data[index:index+1])


def one_hot_wxcodes(wxcodes):
    # # 'HZ' 'RA' 'BR' 'RA BR' 'FG' 'BR BCFG' 'BCFG'
    if wxcodes == 'HZ':
        wxcodes = '1,0,0,0,0,0,0'
    elif wxcodes == 'RA':
        wxcodes = '0,1,0,0,0,0,0'
    elif wxcodes == 'BR':
        wxcodes = '0,0,1,0,0,0,0'
    elif wxcodes == 'RA BR':
        wxcodes = '0,0,0,1,0,0,0'
    elif wxcodes == 'FG':
        wxcodes = '0,0,0,0,1,0,0'
    elif wxcodes == 'BR BCFG':
        wxcodes = '0,0,0,0,0,1,0'
    elif wxcodes == 'BCFG':
        wxcodes = '0,0,0,0,0,0,1'
    return wxcodes


def one_hot_skyc(skyc):
    if skyc == 'CLR':
        # skyc = [[1,0,0,0,0,0]]
        skyc = '1,0,0,0,0,0'
        # skyc = [(1, 0, 0, 0, 0, 0)]
    elif skyc == 'FEW':
        # skyc = [[0,1,0,0,0,0]]
        skyc = '0,1,0,0,0,0'
        # skyc = [(0, 1, 0, 0, 0, 0)]
    elif skyc == 'VV ':
        # skyc = [[0,0,1,0,0,0]]
        skyc = '0,0,1,0,0,0'
        # skyc = [(0, 0, 1, 0, 0, 0)]
    elif skyc == 'SCT':
        # skyc = [[0, 0, 0, 1, 0, 0]]
        skyc = '0,0,0,1,0,0'
        # skyc = [(0, 0, 0, 1, 0, 0)]
    elif skyc == 'BKN':
        # skyc = [[0, 0, 0, 0, 1, 0]]
        skyc = '0,0,0,0,1,0'
        # skyc = [(0, 0, 0, 0, 1, 0)]
    elif skyc == 'OVC':
        # skyc = [[0, 0, 0, 0, 0, 1]]
        skyc = '0,0,0,0,0,1'
        # skyc = [(0, 0, 0, 0, 0, 1)]
    return skyc


def split_valid(valid):
    valid = pd.to_datetime(valid)
    year = str(valid.year)
    month = str(valid.month)
    day = str(valid.day)
    hour = str(valid.hour)
    minute = valid.minute
    str_valid = year + '/' + month + '/' + day + ' ' + hour
    return str_valid, minute
