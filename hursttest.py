from numpy import cumsum, log, polyfit, sqrt, std, subtract
import numpy as pd
from numpy.random import randn
from datetime import datetime
import pandas as pd
import pandas_datareader.data as web


def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)
    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0

# Create a Random Walk, Mean-Reverting and Trending Series
rw = log(cumsum(randn(100000))+1000)
mr = log(randn(100000)+1000)
tr = log(cumsum(randn(100000)+1)+1000)

# Output the Hurst Exponent for each of the above series
# and the price of Amazon (the Adjusted Close price) for
# the ADF test given above in the article
print("Hurst(RW): %s" % hurst(rw))
print("Hurst(MR): %s" % hurst(mr))
print("Hurst(TR): %s" % hurst(tr))

# amzn = web.DataReader("AMZN", "yahoo", datetime(2000,1,1), datetime(2015,1,1))
amzn = pd.read_csv('amazon.csv', index_col=0, parse_dates=True, infer_datetime_format=True)
amznts = amzn['Adj Close'].values
print("Hurst(AMZN): %s" % hurst(amznts))

usdcad = pd.read_csv('USDCAD_FED.csv', index_col=0, parse_dates=True, infer_datetime_format=True)
usdcadts = usdcad.values
print("Hurst(USDCAD): %s" % hurst(usdcadts))
cad1ts = usdcadts[-250:]
cad2ts = usdcadts[-500:]
cad3ts = usdcadts[-1000:]
print("Hurst(CAD1): %s" % hurst(cad1ts))
print("Hurst(CAD2): %s" % hurst(cad2ts))
print("Hurst(CAD3): %s" % hurst(cad3ts))
