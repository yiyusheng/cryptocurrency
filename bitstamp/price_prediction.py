#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  21 09:55:12 2018

@author: yiyusheng
"""

import pandas as pd
import numpy as np
import talib
import sklearn
import scipy
from pandas import Series

#%% Load data
file_path = '/home/yiyusheng/Data/COIN/bitstampUSD.csv'
with open(file_path,'rb') as csvfile:
    data = pd.read_csv(csvfile, sep=',',header=None, names = ['time','price','volumn'])
data['time'] = pd.to_datetime(data['time'],unit='s')
data = data.set_index('time')

#%% Convert tick data to OHLC data
ohlc = data['price'].resample('1D').ohlc()
vol = data['volumn'].resample('1D').sum()
data_ohlcv = pd.concat([ohlc,vol],axis=1)
data_ohlcv = data_ohlcv.dropna(subset = ['open'])

#%% Extract price
cl = data_ohlcv['close'].as_matrix()

#%% Extract Technical Indicators
sma6 = talib.SMA(cl,timeperiod = 6)
sma12 = talib.SMA(cl,timeperiod = 12)
macd, macdsignal, macdhist = talib.MACD(cl, fastperiod=12, slowperiod=26, signalperiod=9)
upperband, middleband, lowerband = talib.BBANDS(cl, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)

#%% Build target, trainset, testset, cross validation
valid_ind = max(np.isnan(sma6).sum(),np.isnan(sma12).sum(),np.isnan(macd).sum())
return_rate = cl[1:]/cl[:-1] - 1
target = np.concatenate((np.array([0]),np.round(return_rate*100,-1)))

dataset = pd.concat([Series(cl),Series(sma6),Series(sma12),Series(macd),Series(middleband),Series(target)],axis=1)
dataset.columns = ['price','sma6','sma12','macd','middleband','target']
dataset.index = data_ohlcv.index
dataset = dataset.iloc[valid_ind+1:]

train = dataset.iloc[:1500]
test = dataset.iloc[1500:]
cv1 = train[:300]
cv2 = train[300:600]
cv3 = train[600:900]
cv4 = train[900:1200]
cv5 = train[1200:]

#dataset['target'].value_counts()

