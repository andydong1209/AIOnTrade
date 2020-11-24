#!/usr/bin/python
# -*- coding: utf-8 -*-

# forecast.py

import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
import pandas_datareader.data as web

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Obtain stock information from IEX
start_date = datetime.datetime(2016, 1, 1)
start = '2016-01-01'
end_date = datetime.datetime(2018, 12, 31)
end = '2018-12-31'
beg_date = start_date-timedelta(days=365) # (2015, 1, 10)
beg = '2015-01-02'

src = "iex"
lags = 5
# ts = web.DataReader("SPY", src, beg_date, end_date) # 1006

# or obtain stock information from CSV(from IEX Data)
ts = pd.read_csv('IEX_SPY.csv', index_col = 0) # 1006

# Create the new lagged DataFrame
tslag = pd.DataFrame(index = ts.index)
tslag["Today"] = ts["close"]
tslag["Volume"] = ts["volume"]

# Create the shifted lag series of prior trading period close values
for i in range(0, lags):
	tslag["Lag%s" % str(i+1)] = ts["close"].shift(i+1)
	
# Create the returns DataFrame
tsret = pd.DataFrame(index=tslag.index)
tsret["Volume"] = tslag["Volume"]
tsret["Today"] = tslag["Today"].pct_change()*100.0

# If any of the values of percentage returns equal zero, set them to
# a small number (stops issues with QDA model in Scikit-Learn)
for i, x in enumerate(tsret["Today"]):
	if abs(x) < 0.0001:
		tsret.iloc[i, 1] = 0.0001

# Create the lagged percentage returns columns
for i in range(0, lags):
	tsret["Lag%s" % str(i+1)] = tslag["Lag%s" % str(i+1)].pct_change()*100.0

# print(tsret.tail())
# print(tsret[0:10])
tsret = tsret[tsret.index >= '2015-01-12'] # 1000
# tsret = tsret[lag+1:]

# Create the "Direction" column (+1 or -1) indicating an up/down day
tsret["Direction"] = np.sign(tsret["Today"]) # 1000
tsret = tsret[tsret.index >= start] # 754

# Use the prior two days of returns as predictor values, with direction as the response
X = tsret[["Lag1", "Lag2"]] # 754
y = tsret["Direction"] # 754
# print(X)
# print(y)


# ***********************************************************************************

# The test data is split into two parts: Before and after 1st Jan 2018.
start_test = '2018-01-01' # datetime.datetime(2018, 1, 1)

# Create training and test sets
X_train = X[X.index < start_test] # 503
y_train = y[y.index < start_test] # 503

#print(X_train)
#print(y_train)

X_test = X[X.index >= start_test] # 251
y_test = y[y.index >= start_test] # 251
#print(X_test)
#print(y_test)

# Create the (parametrised) models
print("Hit Rates/Confusion Matrices:\n")


models = [("LR", LogisticRegression())]
# Iterate through the models
# Train each of the models on the training set
# model[1].fit(X_train, y_train)

for m in models:
	m[1].fit(X_train, y_train)
	pred = m[1].predict(X_test)
	print(m[0], m[1].score(X_test, y_test))
	print(confusion_matrix(pred, y_test))

#Create a single prediction
X_test.loc(1)
ind = pd.Index([datetime.datetime(2019, 1, 1)])
S1 = pd.Series(0.2, index=ind)
S2 = pd.Series(0.5, index=ind)
X1_test = pd.DataFrame({'Lag1':S1, 'Lag2':S2})
print(X1_test)
pred1 = m[1].predict(X1_test)
print(pred1)
