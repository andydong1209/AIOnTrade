import math
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import statsmodels.api as sm
import matplotlib.pyplot as plt

'''
gld = web.DataReader('GLD', data_source='yahoo', start='5/23/2006', end='11/30/2007')
gdx = web.DataReader('GDX', data_source='yahoo', start='5/23/2006', end='11/30/2007')
'''

gld = pd.read_csv('GLD.csv', index_col = 0, parse_dates = True)
gdx = pd.read_csv('GDX.csv', index_col = 0, parse_dates = True)

'''
h5 = pd.HDFStore('exp3_6_2.h5', complevel=9, complib='blosc')
h5['gld_price'] = gld
h5['gdx_price'] = gdx
h5.close()

h5 = pd.HDFStore('exp3_6_2.h5')
gld = h5['gld_price']
gdx = h5['gdx_price']
h5.close()
'''

print(gld.info())
print(gld[0:10])
print(gdx.info())
print(gld[0:10])


OBS = 252
y = np.array(gld['Adj Close'][0:OBS])
x = np.array(gdx['Adj Close'][0:OBS])
X = sm.add_constant(x)


plt.figure(figsize=(14, 10))
plt.subplot(211)
plt.plot(y, lw=1.5, label='gld')
plt.plot(y, 'ro')
plt.grid(True)
plt.legend(loc=0)
plt.ylabel('$Price$')
plt.title('Time Serice of Prices')

plt.subplot(212)
plt.plot(x, 'g', lw=1.5, label='gdx')
plt.plot(x, 'bx')
plt.grid(True)
plt.legend(loc=0)
plt.ylabel('$Price$')
plt.xlabel('$Date$')
plt.show()


model = sm.OLS(y, X)
results = model.fit()
hedgeRatio = results.params[1]
print(hedgeRatio)

gld_adjClose = gld['Adj Close']
gdx_adjClose = gdx['Adj Close']
spread = gld_adjClose - hedgeRatio * gdx_adjClose
spreadTrain = spread[0:OBS]


plt.figure(figsize=(14, 7))
plt.plot(spread, lw=1.5, label='spread')
plt.plot(spreadTrain, 'ro', label='Train')
plt.grid(True)
plt.legend(loc=0)
plt.xlabel('$Date$')
plt.ylabel('$Spread Price$')
plt.title('Pair Trade Performance')
plt.show()


spreadMean = spreadTrain.mean()
spreadStd = spreadTrain.std()
print(spreadMean)
print(spreadStd)

zscore = (spread - spreadMean) / spreadStd
print(zscore[:10])
print(zscore[-10:])

longs = zscore <= -2
shorts = zscore >= 2
exits = np.abs(zscore) <= 1

positions = np.array(len(spread)*[None, None])
positions.shape = (len(spread), 2)

for i, b in enumerate(shorts):
	if b:
		positions[i] = [-1, 1]
		
for i, b in enumerate(longs):
	if b:
		positions[i] = [1, -1]

for i, b in enumerate(exits):
	if b:
		positions[i] = [0, 0]

for i, b in enumerate(positions):
	if b[0] == None :
		positions[i] = positions[i-1]

print(positions[:10])
print(positions[-10:])

OBS = 385
cl1 = np.array(gld['Adj Close'][0:OBS])
cl2 = np.array(gdx['Adj Close'][0:OBS])

ret_cl1 = np.diff(cl1) / cl1[:-1]
ret_cl2 = np.diff(cl2) / cl2[:-1]

dailyret = np.concatenate((ret_cl1, ret_cl2), axis=0)
dailyret = np.reshape(dailyret, (OBS-1, 2), order = 'F')

PL = positions[:-1] * dailyret
pnl = np.sum(PL, axis = 1)

plt.figure(figsize=(14, 7))
plt.plot(pnl, lw=1.5, label='pnl')
plt.grid(True)
plt.legend(loc=0)
plt.xlabel('$Date$')
plt.ylabel('$Profit_Loss$')
plt.title('Strategy PL')
plt.show()


total = np.sum(pnl)
print(total)

sharpTrainset = math.sqrt(252)*np.mean(pnl[0:251])/np.std(pnl[0:251])
sharpTestset = math.sqrt(252)*np.mean(pnl[252:OBS-1])/np.std(pnl[252:OBS-1])

print('sharpTrainset: ', sharpTrainset)
print('sharpTestset: ', sharpTestset)
