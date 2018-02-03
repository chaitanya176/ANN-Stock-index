#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 19:18:07 2017

@author: Chaitanya
"""

import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense

import pandas as pd
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from neupy import algorithms, estimators


def create_datset(datset, look_back=1):
    datX, datY = [], []
    for i in range(len(datset)-look_back-1):
        x = datset[i:(i+look_back), 0]
        datX.append(x)
        datY.append(datset[i + look_back, 0])
    return numpy.array(datX), numpy.array(datY)


numpy.random.seed(4)

dataframe = read_csv('/Users/Chaitanya/Documents/IE 530/DataYahoo5YStd.csv',usecols=[4], engine='python', skipfooter=3)
datset = dataframe.values
datset = datset.astype('float32')

tr_size = int(len(datset) * 0.60)
te_size = len(datset) - tr_size
tr, te = datset[0:tr_size,:], datset[tr_size:len(datset),:]

look_back = 3
trX, trY = create_datset(tr, look_back)
teX, teY = create_datset(te, look_back)

model = Sequential()
model.add(Dense(4, input_dim=look_back, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='linear'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trX, trY, epochs=1000, batch_size=1, verbose=1)

trainScore = model.evaluate(trX, trY, verbose=0)
print('train-Score:(%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('test-Score:(%.2f RMSE)' % (testScore, math.sqrt(testScore)))

trPredict = model.predict(trX)
tePredict = model.predict(teX)

trPredictPlot = numpy.empty_like(datset)
trPredictPlot[:, :] = numpy.nan
trPredictPlot[look_back:len(trPredict)+look_back, :] = trPredict

tePredictPlot = numpy.empty_like(datset)
tePredictPlot[:, :] = numpy.nan
tePredictPlot[len(trPredict)+(look_back*2)+1:len(datset)-1, :] = tePredict

plt.plot(datset)
plt.plot(tePredictPlot)
plt.show()

df = pd.read_csv('/Users/Chaitanya/Documents/IE 530/DataYahoo5YStd.csv',usecols=[4])
x_train, x_test, y_train, y_test = train_test_split(preprocessing.minmax_scale(df),preprocessing.minmax_scale(df), train_size=0.65,)

nw = algorithms.GRNN(std=2, verbose=False)
nw.train(x_train, y_train)

y_predicted = nw.predict(x_test)
estimators.mse(y_predicted, y_test)