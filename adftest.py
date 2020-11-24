# Import the Time Series library
import statsmodels.tsa.stattools as ts

# Import Datetime and the Pandas DataReader
from datetime import datetime
import pandas_datareader.data as web
import pandas as pd

# Download the Amazon OHLCV data from 1/1/2000 to 1/1/2015
amzn = web.DataReader("AMZN", "yahoo", datetime(2000,1,1), datetime(2015,1,1))
# amzn = pd.read_csv("amazon.csv", index_col=0, parse_dates=True, infer_datetime_format=True)

# Output the results of the Augmented Dickey-Fuller test for Amazon
# with a lag order value of 1
result = ts.adfuller(amzn['Adj Close'], 1)

print(result)

print(result[0])
print(result[4]['5%'])


