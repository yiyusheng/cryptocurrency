#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 17:20:29 2018

@author: yiyusheng
"""

import numpy as np 
import pandas as pd 
import datetime
import os

# %%define a conversion function for the native timestamps in the csv file
def dateparse (time_in_secs):    
    return datetime.datetime.fromtimestamp(float(time_in_secs))

dir = '/home/yiyusheng/Data/COIN/bitcoin-historical-data/'
print('Data listing...')
print(os.listdir(dir))

# read in the data and apply our conversion function, this spits out a DataFrame with the DateTimeIndex already in place
print('Using bitstampUSD_1-min_data...')
data = pd.read_csv(dir+'bitstampUSD_1-min_data_2012-01-01_to_2018-03-27.csv', parse_dates=True, date_parser=dateparse, index_col=[0])

print('Total null open prices: %s' % data['Open'].isnull().sum())

# %%First thing is to fix the data for bars/candles where there are no trades. 
# Volume/trades are a single event so fill na's with zeroes for relevant fields...
data['Volume_(BTC)'].fillna(value=0, inplace=True)
data['Volume_(Currency)'].fillna(value=0, inplace=True)
data['Weighted_Price'].fillna(value=0, inplace=True)

# next we need to fix the OHLC (open high low close) data which is a continuous timeseries so
# lets fill forwards those values...
data['Open'].fillna(method='ffill', inplace=True)
data['High'].fillna(method='ffill', inplace=True)
data['Low'].fillna(method='ffill', inplace=True)
data['Close'].fillna(method='ffill', inplace=True)

# check how we are looking now, should be nice and clean...
print(data)

# %%The first thing we need are our trading signals. The Turtle strategy was based on daily data and
# they used to enter breakouts (new higher highs or new lower lows) in the 22-60 day range roughly.
# We are dealing with minute bars here so a 22 minute new high isn't much to get excited about. Lets
# pick an equivalent to 60 days then. They also only considered Close price so lets do the same...

signal_lookback = 60 * 24 * 60 # days * hours * minutes

# here's our signal columns
data['Buy'] = np.zeros(len(data))
data['Sell'] = np.zeros(len(data))

# this is our 'working out', you could collapse these into the .loc call later on and save memory 
# but I've left them in for debug purposes, makes it easier to see what is going on
data['RollingMax'] = data['Close'].shift(1).rolling(signal_lookback, min_periods=signal_lookback).max()
data['RollingMin'] = data['Close'].shift(1).rolling(signal_lookback, min_periods=signal_lookback).min()
data.loc[data['RollingMax'] < data['Close'], 'Buy'] = 1
data.loc[data['RollingMin'] > data['Close'], 'Sell'] = -1

# lets now take a look and see if its doing something sensible
import matplotlib
import matplotlib.pyplot as plt

fig,ax1 = plt.subplots(1,1)
ax1.plot(data['Close'])
y = ax1.get_ylim()
ax1.set_ylim(y[0] - (y[1]-y[0])*0.4, y[1])

ax2 = ax1.twinx()
ax2.set_position(matplotlib.transforms.Bbox([[0.125,0.1],[0.9,0.32]]))
ax2.plot(data['Buy'], color='#77dd77')
ax2.plot(data['Sell'], color='#dd4444')

# %% Get data in a interval, recognized by the start timepoint, interval(mininutes) and number of point
def get_target_data(start_ts,interval,p_count,DT):
    start_ts = convert_date(start_ts)
    point_set = [start_ts+datetime.timedelta(minutes=interval)*x for x in range(0,p_count)]
    return DT[DT.index.isin(point_set)]

def convert_date(date):
    if(type(date)==datetime.datetime):
        return date
    elif(type(date)==str):
        return datetime.datetime.strptime(date,'%Y-%m-%d %H:%M')
    else:
        return 0
    
# %% Get base data to compare
def get_base_data(start_ts,interval,end_ts,DT):
    start_ts = convert_date(start_ts)
    end_ts = convert_date(end_ts)
    length_period = end_ts-start_ts
    length_period = length_period.total_seconds()
    length_period = int(length_period/60)
    point_set = [start_ts+datetime.timedelta(minutes=interval)*x for x in range(0,length_period/interval)]
    return DT[DT.index.isin(point_set)]
    
# %% Moving windows to evaluate their similarity
def moving_windows(target_data,base_data,f_str):
    len_td = len(target_data)
    len_bd = len(base_data)
    loop_count = len_bd-len_td
    f = globals()[f_str]
    results = np.zeros(loop_count)
    for i in range(0,loop_count):
        results[i] = f(target_data['Close'],base_data[i:i+len_td]['Close'])
    return [results,base_data.index[0:loop_count]]
            
def get_coefficient(arr1,arr2):
    return np.corrcoef(arr1,arr2)[0][1]

# %% plot figures tto compare the target_data and the base_data 
def comp_figure(base_data,target_data,base_start,interval=60,add_count=24*2):
    start_ts_add = max(target_data.index)+datetime.timedelta(minutes=interval)
    start_ts_add = start_ts_add.to_datetime()
    target_data_add = get_target_data(start_ts_add,interval,add_count,data)
    target_data = target_data.append(target_data_add)
    
    len_td = len(target_data)
    base_data_extract = base_data[base_data.index >= base_start[0]]
    base_data_extract = base_data_extract[0:len_td]
    plt.subplot(2,1,1)
    plot_figure(target_data)
    plt.subplot(2,1,2)
    plot_figure(base_data_extract)

def plot_figure(DT):
    plt.plot(DT.index,DT['Close'])    
# %% Main
target_data = get_target_data('2018-03-15 0:0',60,24*3,data)    
base_data = get_base_data('2017-01-01 0:0',60,'2018-01-01 0:0',data)
result,date = moving_windows(target_data,base_data,'get_coefficient')    
   
base_start = date[result == max(result)]   
comp_figure(base_data,target_data,base_start,60,0)    
    
    
    
    
    
    
    
    
    
    