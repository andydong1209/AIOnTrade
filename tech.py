import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt

#sp500 = web.DataReader('^GSPC', data_source='yahoo', start='1/1/2000', end='4/14/2014')
sp500 = pd.read_csv('GSPC_2000_2014.csv', index_col = 0, parse_dates = True)
print(sp500.info())
print(sp500[0:20])



#sp500['42d'] = np.round(pd.rolling_mean(sp500['Close'], window=42), 2)
#sp500['252d'] = np.round(pd.rolling_mean(sp500['Close'], window=252), 2)

sp500['42d'] = np.round(sp500['Close'].rolling(window=42, center=False).mean(), 2)
sp500['252d'] = np.round(sp500['Close'].rolling(window=252, center=False).mean(), 2)

'''
plt.figure(figsize=(10, 6))
plt.plot(sp500['Close'])
plt.grid(True)
plt.show()

plt.figure(figsize=(18, 10))
plt.plot(sp500[['Close', '42d', '252d']])
plt.grid(True)
plt.show()

sp500[['Close', '42d', '252d']].plot(grid=True, figsize=(18, 10))
plt.show()
'''

sp500['42-252'] = sp500['42d'] - sp500['252d']
print(sp500['42-252'].tail())

SD = 50
sp500['Regime'] = np.where(sp500['42-252'] > SD, 1, 0)
sp500['Regime'] = np.where(sp500['42-252'] < -SD, -1, sp500['Regime'])
print(sp500['Regime'].value_counts())

'''
plt.figure(figsize=(18, 10))
plt.plot(sp500[['Regime']])
plt.ylim([-1.1, 1.1])
plt.grid(True)
plt.show()
'''

sp500['Market'] = np.log(sp500['Close']/sp500['Close'].shift(1))
sp500['Strategy'] = sp500['Regime'].shift(1) * sp500['Market']

'''
sp500[['Market', 'Strategy']].cumsum().apply(np.exp).plot(grid=True, figsize=(18, 10))
plt.show()
'''