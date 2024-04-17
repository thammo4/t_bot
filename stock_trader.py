import os;
import dotenv;
import random;
import warnings;
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import datetime;
import re;

from scipy.optimize import minimize;
import scipy.stats as stats;

import yfinance as yf;
from uvatradier import Account, Quotes, OptionsData, EquityOrder, OptionsOrder;
from fredapi import Fred;

from dow30 import DOW30;

warnings.filterwarnings("ignore");

dotenv.load_dotenv();

random.seed(10);

tradier_acct = os.getenv('tradier_acct');
tradier_token = os.getenv('tradier_token');
fred_api_key = os.getenv('fred_api_key');

acct = Account(tradier_acct, tradier_token);
quotes = Quotes(tradier_acct, tradier_token);
equity_order = EquityOrder(tradier_acct, tradier_token);
options_data = OptionsData(tradier_acct, tradier_token);
options_order = OptionsOrder(tradier_acct, tradier_token);

fred = Fred(api_key=fred_api_key);


# random sample basket of symbols to begin, constructed via random.sample(DOW30, 5)
dow_sample = ['MRK', 'AMZN', 'CSCO', 'KO', 'V'];


#
# Create dataframe with OHLCV bar data for each of the five stocks
#

# EXAMPLE DATA FRAME
# >>> ko
#                                 Open       High        Low      Close   Volume  Dividends  Stock Splits
# Datetime                                                                                               
# 2024-04-16 09:30:00-04:00  58.250000  58.340000  58.200001  58.279999   348159        0.0           0.0
# 2024-04-16 09:35:00-04:00  58.270000  58.330002  58.200600  58.330002   168652        0.0           0.0
# 2024-04-16 09:40:00-04:00  58.320000  58.320000  58.220001  58.290298    95514        0.0           0.0
# 2024-04-16 09:45:00-04:00  58.299999  58.380001  58.290001  58.290001   158552        0.0           0.0
# 2024-04-16 09:50:00-04:00  58.299999  58.365002  58.250000  58.264999   165672        0.0           0.0
# ...                              ...        ...        ...        ...      ...        ...           ...
# 2024-04-16 15:35:00-04:00  58.019402  58.077000  57.990002  58.014999   142380        0.0           0.0
# 2024-04-16 15:40:00-04:00  58.014999  58.095001  58.005001  58.070000   181327        0.0           0.0
# 2024-04-16 15:45:00-04:00  58.070000  58.150002  58.064999  58.145000   145891        0.0           0.0
# 2024-04-16 15:50:00-04:00  58.139999  58.180000  58.105000  58.180000   411167        0.0           0.0
# 2024-04-16 15:55:00-04:00  58.174999  58.189999  58.009998  58.049999  1178399        0.0           0.0
#
# [78 rows x 7 columns]

mrk 	= yf.Ticker('MRK').history(start=str(datetime.date.today()), interval='5m');
amzn 	= yf.Ticker('AMZN').history(start=str(datetime.date.today()), interval='5m');
csco 	= yf.Ticker('CSCO').history(start=str(datetime.date.today()), interval='5m');
ko 		= yf.Ticker('KO').history(start=str(datetime.date.today()), interval='5m');
v 		= yf.Ticker('V').history(start=str(datetime.date.today()), interval='5m');


#
# Retrieve latest three-month T-BILL rate of return from the FED (Risk-Free Rate)
#

# >>> risk_free_rate
# 5.25
risk_free_rate = fred.get_series_latest_release('DTB3').iloc[-1];




#
# Create feature dataframe - Price Changes, Volatility, Price Increase/Decrease
#

# EXAMPLE OUTPUT
#
# >>> df_features
#            Open        High         Low       Close  Volume  Dividends  Stock Splits  PriceChange  PercentChange  Volatility  RollingVolatility  IsPriceIncrease Symbol
# 0    183.440002  184.259995  183.389999  183.884995  867343        0.0           0.0     0.439987       0.002398    0.373084           0.377521                1   AMZN
# 1    183.880005  184.170197  183.789993  183.949997  465566        0.0           0.0     0.065002       0.000353    0.373084           0.430396                1   AMZN
# 2    183.949997  184.169998  183.529999  184.130005  453791        0.0           0.0     0.180008       0.000979    0.373084           0.495784                1   AMZN
# 3    184.141296  184.279999  183.880005  184.235001  535816        0.0           0.0     0.104996       0.000570    0.373084           0.446945                1   AMZN
# 4    184.229996  184.660004  184.229996  184.604996  716329        0.0           0.0     0.369995       0.002008    0.373084           0.387679                0   AMZN
# ..          ...         ...         ...         ...     ...        ...           ...          ...            ...         ...                ...              ...    ...
# 355  272.700012  272.700012  271.739990  271.809998   34015        0.0           0.0    -0.929993      -0.003410    0.467535           0.418317                1      V
# 356  271.704987  272.079987  271.589996  271.970001   30573        0.0           0.0     0.160004       0.000589    0.467535           0.462803                0      V
# 357  272.013000  272.013000  271.579987  271.910004   47791        0.0           0.0    -0.059998      -0.000221    0.467535           0.462624                1      V
# 358  271.950012  272.329987  271.920105  272.290009   42213        0.0           0.0     0.380005       0.001398    0.467535           0.424184                1      V
# 359  272.350006  272.630005  272.170013  272.309998   75365        0.0           0.0     0.019989       0.000073    0.467535           0.345275                0      V
#
# [360 rows x 13 columns]

data = {ticker: yf.Ticker(ticker).history(period='1d', interval='5m') for ticker in dow_sample};
features = pd.DataFrame();

for symbol, df in data.items():
	df['PriceChange'] = df['Close'].diff();
	df['PercentChange'] = df['Close'].pct_change();
	df['Volatility'] = df['Close'].std();
	df['RollingVolatility'] = df['Close'].rolling(window=6).std();
	# df['IsPriceIncrease'] = (df['Close'] > df['Open']).astype(int);
	df['IsPriceIncrease'] = (df['Close'].shift(-1) > df['Close']).astype(int);
	df = df.dropna();

	df['Symbol'] = symbol;
	features = pd.concat([features, df], axis=0);

df_features = features.groupby('Symbol').apply(lambda x: x.iloc[:-1]).reset_index(drop=True);


#
# Conduct One Sample T-Test using Merck's `PercentChange` as the random variable
# • H0: μ = 0 (no percentage change occurred)
# • Ha: μ ≠ 0 (percentage change has diverged significantly from zero)
#
# • Significance level: α = .05
#

# SAMPLE OUTPUT
#
# >>> t_stat, p_val = stats.ttest_1samp(df_features.query('Symbol == "MRK"')['PercentChange'],0)
# >>> t_stat
# -0.7936897739561891
# >>> p_val
# 0.43002069777872187

t_test, p_val = stats.ttest_1samp(df_features.query('Symbol == "MRK"')['PercentChange'], 0);