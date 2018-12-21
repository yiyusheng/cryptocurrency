#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 09:55:12 2017

@author: yiyusheng
"""

import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import sklearn
import scipy

from pandas import Series
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

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

#%% Extract Technical Indicators
cl = data_ohlcv['close'].as_matrix()
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

#%% LR: Build estimator and fit model
lr = sklearn.linear_model.LogisticRegression()
lr_scores = cross_val_score(lr,train.ix[:,train.columns!='target'],train['target'],cv=5,n_jobs=-1)
lr.fit(train.ix[:,train.columns!='target'],train['target'])
lr_train_mse = np.mean((lr.predict(train.ix[:,train.columns!='target'])-train['target'])**2)
lr_test_mse = np.mean((lr.predict(test.ix[:,test.columns!='target'])-test['target'])**2)

#%% GBDT: Build estimator and fit model
gbdt = sklearn.ensemble.GradientBoostingClassifier()
gbdt.fit(train.ix[:,train.columns!='target'],train['target'])
gbdt_train_mse = np.mean((gbdt.predict(train.ix[:,train.columns!='target'])-train['target'])**2)
gbdt_test_mse = np.mean((gbdt.predict(test.ix[:,test.columns!='target'])-test['target'])**2)

#%% RF: Build estimator and fit model
rf = sklearn.ensemble.RandomForestClassifier()
rf.fit(train.ix[:,train.columns!='target'],train['target'])
rf_train_mse = np.mean((rf.predict(train.ix[:,train.columns!='target'])-train['target'])**2)
rf_test_mse = np.mean((rf.predict(test.ix[:,test.columns!='target'])-test['target'])**2)

#%% GBDT: Build estimator and fit model
svm = sklearn.svm.SVC()
param_svm = {'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),'kernel': ['rbf'], 'class_weight':['balanced', None]}
rsearch_svm = RandomizedSearchCV(estimator = svm, param_distributions = param_svm, n_iter=1000, n_jobs=-1)
rsearch_svm.fit(train.ix[:,train.columns!='target'],train['target'])
print(rsearch_svm.best_score_)
print(rsearch_svm.best_estimator_)
print(rsearch_svm.best_params_)

#%% NN: Build estimator and fit model
from sklearn.neural_network import MLPClassifier
nn = MLPClassifier()
nn.fit(train.ix[:,train.columns!='target'],train['target'])
nn_train_mse = np.mean((nn.predict(train.ix[:,train.columns!='target'])-train['target'])**2)
nn_test_mse = np.mean((nn.predict(test.ix[:,test.columns!='target'])-test['target'])**2)

#%% MSE: Build estimator and fit model
mse = pd.DataFrame([[lr_train_mse,gbdt_train_mse,rf_train_mse,svm_train_mse,nn_train_mse],
                   [lr_test_mse,gbdt_test_mse,rf_test_mse,svm_test_mse,nn_test_mse]],
                   columns=['LR','GBDT','RF','SVM','NN'],  index= ['train','test'])