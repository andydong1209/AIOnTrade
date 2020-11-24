#!/usr/bin/python
# -*- coding: utf-8 -*-

# forecast.py

from __future__ import print_function

import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from datetime import timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Obtain stock information from Yahoo Finance
start_date = datetime.datetime(2001, 1, 10)
end_date = datetime.datetime(2005, 12, 31)
lags = 5

# start_train = datetime.datetime(2005, 1, 1) + timedelta(days=lags+2)
# start_test = datetime.datetime(2005, 9, 1)

ts = web.DataReader("^GSPC", "yahoo", start_date-timedelta(days=365), end_date)
# print(ts)
	
# Create the new lagged DataFrame
tslag = pd.DataFrame(index = ts.index)
tslag["Today"] = ts["Adj Close"]
tslag["Volume"] = ts["Volume"]

# Create the shifted lag series of prior trading period close values
for i in range(0, lags):
	tslag["Lag%s" % str(i+1)] = ts["Adj Close"].shift(i+1)
	
# Create the returns DataFrame
tsret = pd.DataFrame(index=tslag.index)
tsret["Volume"] = tslag["Volume"]
tsret["Today"] = tslag["Today"].pct_change()*100.0

# If any of the values of percentage returns equal zero, set them to
# a small number (stops issues with QDA model in Scikit-Learn)
for i, x in enumerate(tsret["Today"]):
	if abs(x) < 0.0001:
		tsret.iloc[i, 1] = 0.0001
		#tsret["Today"][i] = 0.0001

# Create the lagged percentage returns columns
for i in range(0, lags):
	tsret["Lag%s" % str(i+1)] = tslag["Lag%s" % str(i+1)].pct_change()*100.0

print(tsret.tail())
print(tsret[0:10])
tsret = tsret[tsret.index >= datetime.datetime(2000, 1, 19)]

# Create the "Direction" column (+1 or -1) indicating an up/down day
tsret["Direction"] = np.sign(tsret["Today"])
# print(tsret) # 1503

tsret = tsret[tsret.index >= start_date]
# print(tsret) # 1250

# Use the prior two days of returns as predictor
# values, with direction as the response
X = tsret[["Lag1", "Lag2"]] # 1250
y = tsret["Direction"] # 1250
# print(X)
# print(y)

# The test data is split into two parts: Before and after 1st Jan 2005.
start_test = datetime.datetime(2005, 1, 1)

# Create training and test sets
X_train = X[X.index < start_test]
y_train = y[y.index < start_test]

#print(X_train)
#print(y_train)

X_test = X[X.index >= start_test]
y_test = y[y.index >= start_test]
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
ind = pd.Index([datetime.datetime(2006, 1, 1)])
S1 = pd.Series(0.2, index=ind)
S2 = pd.Series(0.5, index=ind)
X1_test = pd.DataFrame({'Lag1':S1, 'Lag2':S2})
print(X1_test)
pred1 = m[1].predict(X1_test)
print(pred1)
