#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 19:20:06 2017

@author: Chaitanya
"""
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def create_datset(datset, look_back=1):
    datX, datY = [], []
    for i in range(len(datset)-look_back-1):
        x= datset[i:(i+look_back), 0]
        datX.append(x)
        datY.append(datset[i + look_back, 0])
    return numpy.array(datX), numpy.array(datY)

numpy.random.seed(4)

dataframe = read_csv('/Users/Chaitanya/Documents/IE 530/DataYahoo5YStd.csv', usecols=[4], engine='python', skipfooter=3)
datset = dataframe.values
datset = datset.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
datset = scaler.fit_transform(datset)

tr_size = int(len(datset) * 0.60)
te_size = len(datset) - tr_size
tr, te = datset[0:tr_size,:], datset[tr_size:len(datset),:]

look_back = 1
trX, trY = create_datset(tr, look_back)
teX, teY = create_datset(te, look_back)

trX = numpy.reshape(trX, (trX.shape[0], 1, trX.shape[1]))
teX = numpy.reshape(teX, (teX.shape[0], 1, teX.shape[1]))

model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trX, trY, epochs=50, batch_size=25, verbose=2)

trPredict = model.predict(trX)
tePredict = model.predict(teX)

trPredict = scaler.inverse_transform(trPredict)
trY = scaler.inverse_transform([trY])
tePredict = scaler.inverse_transform(tePredict)
teY = scaler.inverse_transform([teY])

trScore = math.sqrt(mean_squared_error(trY[0], trPredict[:,0]))
print('train-Score: %.2f RMSE' % (trScore))
teScore = math.sqrt(mean_squared_error(teY[0], tePredict[:,0]))
print('test-Score: %.2f RMSE' % (teScore))

trPredictPlot = numpy.empty_like(datset)
trPredictPlot[:, :] = numpy.nan
trPredictPlot[look_back:len(trPredict)+look_back, :] = trPredict

tePredictPlot = numpy.empty_like(datset)
tePredictPlot[:, :] = numpy.nan
tePredictPlot[len(trPredict)+(look_back*2)+1:len(datset)-1, :] = tePredict

plt.plot(scaler.inverse_transform(datset))
plt.plot(tePredictPlot)
plt.show()
