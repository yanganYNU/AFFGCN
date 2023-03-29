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
    print('the number of missing values are',t)

def change_minte_data(str_valid, minute):
    if minute < 5:
        str_valid = str_valid + ':00'
    elif minute >= 5 and minute < 10:
        str_valid = str_valid + ':05'
    elif minute >= 10 and minute < 15:
        str_valid = str_valid + ':10'
    elif minute >= 15 and minute < 20:
        str_valid = str_valid + ':15'
    elif minute >= 20 and minute < 25:
        str_valid = str_valid + ':20'
    elif minute >= 25 and minute < 30:
        str_valid = str_valid + ':25'
    elif minute >= 30 and minute < 35:
        str_valid = str_valid + ':30'
    elif minute >= 35 and minute < 40:
        str_valid = str_valid + ':35'
    elif minute >= 40 and minute < 45:
        str_valid = str_valid + ':40'
    elif minute >= 45 and minute < 50:
        str_valid = str_valid + ':45'
    elif minute >= 50 and minute < 55:
        str_valid = str_valid + ':50'
    else:
        str_valid = str_valid + ':55'
    return str_valid

def one_hot(index,wx_data,onehot):
    for i in range(16):
        j = str(i+1)
        wx_data.loc[index, j] = onehot[i]
def one_hot_wxcodes(wxcodes):
    # ['HZ', 'BR', 'BLSA HZ', 'BLSA']
    if wxcodes == 'HZ':
        wxcodes = '1,0,0,0'
    elif wxcodes == 'BR':
        wxcodes = '0,1,0,0'
    elif wxcodes == 'BLSA HZ':
        wxcodes = '0,0,1,0'
    elif wxcodes == 'BLSA':
        wxcodes = '0,0,0,1'
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


def get_tidy_data(minute, hour, day, month, year):
    minute = minute + 5
    if minute == 60:
        minute = 0
        hour = hour + 1
    if hour == 24:
        hour = 0
        day = day + 1
    if day == 32:
        month = month + 1
        day = 1
    str_minute = str(minute)
    str_hour = str(hour)
    str_day = str(day)
    str_month = str(month)
    if minute < 10:
        str_minute = '0' + str_minute
    tmstp = year + '/' + str_month + '/' + str_day + ' ' + str_hour + ':' + str_minute
    return tmstp, minute, hour, day, month


def insert_data(i, tmstp, df_data, data_seq, time_start, current_time):
    df_empty = [tmstp, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan, np.nan]
    raw_data = df_data[df_data.valid == tmstp]

    if raw_data.empty:
        data_seq.loc[i + 1] = df_empty
        data_seq = data_seq.interpolate(method='linear', limit_direction='forward')

    else:
        if raw_data.index[0] % 250 == 0:
            print(raw_data)
            time_end = time.time()
            time_sum = (time_end - time_start)
            time_current = time_sum - current_time[-1]
            current_time.append(time_sum)
            print('current time used：', round(time_current, 1), 's')  # round((time_sum-time_current),2)
            print('total time used：', round(time_sum, 1), 's\n')
        # if raw_data.index[0] == 2933:
        #     print(raw_data, '\n')
        data_seq = pd.concat([data_seq, raw_data], ignore_index=True)
    return data_seq
