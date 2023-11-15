import os, dotenv;
import numpy as np;
import pandas as pd;
from datetime import datetime, timedelta;

import yfinance as yf;

#
# Fetch data sources
#

start_date 	= '2016-01-01';
end_date 	= '2023-01-01';

# 1762 Observations in DD, DJIA, DJTA, SPX, VIX but not W5000
dd = yf.Ticker('^DJI').history(start=start_date, end=end_date);
dd.index = pd.to_datetime(dd.index.date);

djia = yf.Ticker('^DJT').history(start=start_date, end=end_date);
djia.index = pd.to_datetime(djia.index.date);

djta = yf.Ticker('^DJT').history(start=start_date, end=end_date);
djta.index = pd.to_datetime(djta.index.date);

spx = yf.Ticker('^SPX').history(start=start_date, end=end_date);
spx.index = pd.to_datetime(spx.index.date);

wilshire = yf.Ticker('^W5000').history(start=start_date, end=end_date);
wilshire.index = pd.to_datetime(wilshire.index.date);

vix = yf.Ticker('^VIX').history(start=start_date, end=end_date);
vix.index = pd.to_datetime(vix.index.date);


# Wilshire5000 inexplicably contains 1756 observations.
# Filter other dataframes to match wilshire size.

wilshire_dates = list(wilshire.index.astype(str));

dd 		= dd[dd.index.astype(str).isin(wilshire_dates)];
djia 	= djia[djia.index.astype(str).isin(wilshire_dates)];
djta 	= djta[djta.index.astype(str).isin(wilshire_dates)];
spx 	= spx[spx.index.astype(str).isin(wilshire_dates)];
vix 	= vix[vix.index.astype(str).isin(wilshire_dates)];


#
# Now that every constituent dataframe contains 1756 rows, organize closing prices into a single dataframe
#

market_data = pd.DataFrame({
	'DD': dd['Close'],
	'DJIA': djia['Close'],
	'DJTA': djta['Close'],
	'SPX': spx['Close'],
	'VIX': vix['Close'],
});

# >>> market_data
#                       DD          DJIA          DJTA          SPX        VIX
# 2016-01-04  17148.939453   7352.589844   7352.589844  2012.660034  20.700001
# 2016-01-05  17158.660156   7363.950195   7363.950195  2016.709961  19.340000
# 2016-01-06  16906.509766   7217.049805   7217.049805  1990.260010  20.590000
# 2016-01-07  16514.099609   6995.390137   6995.390137  1943.089966  24.990000
# 2016-01-08  16346.450195   6946.359863   6946.359863  1922.030029  27.010000
# ...                  ...           ...           ...          ...        ...
# 2022-12-23  33203.929688  13564.759766  13564.759766  3844.820068  20.870001
# 2022-12-27  33241.558594  13530.309570  13530.309570  3829.250000  21.650000
# 2022-12-28  32875.710938  13298.360352  13298.360352  3783.219971  22.139999
# 2022-12-29  33220.800781  13496.230469  13496.230469  3849.280029  21.440001
# 2022-12-30  33147.250000  13391.910156  13391.910156  3839.500000  21.670000
#
# [1756 rows x 5 columns]


































