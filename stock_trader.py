# For yfinance interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
import os;
import dotenv;
import random;
import requests;
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

tradier_acct_live = os.getenv('tradier_acct_live');
tradier_token_live = os.getenv('tradier_token_live');

fred_api_key = os.getenv('fred_api_key');

acct = Account(tradier_acct, tradier_token);
quotes = Quotes(tradier_acct, tradier_token);
equity_order = EquityOrder(tradier_acct, tradier_token);
options_data = OptionsData(tradier_acct, tradier_token);
options_order = OptionsOrder(tradier_acct, tradier_token);

fred = Fred(api_key=fred_api_key);


#
# Feature-engineering function to define random variables for analysis
#

def create_features (df):
	# Price-related features
	df['LogReturns'] = np.log(df['Close']/df['Close'].shift(1));
	df['PriceChange'] = df['Close'].diff();
	df['PercentChange'] = df['Close'].pct_change();
	df['IsPriceIncrease'] = (df['Close'].shift(-1) > df['Close']).astype(int);

	# Volatility-related features
	df['Volatility'] = df['Close'].std();
	df['VolatilityNormalized'] = df['Volatility']/df['Close'].mean();
	df['VolatilityRolling'] = df['Close'].rolling(window=6).std();

	df = df.dropna();

	return df;

#
# Make OHLCV bar data bitemporal with ValidFrom/ValidTo
#

def make_bitemporal (df):
	df['ValidFrom'] = pd.to_datetime(df.index);
	df['ValidTo'] = df['ValidFrom'].shift(-1);
	df.iloc[-1, df.columns.get_loc('ValidTo')] = pd.Timestamp(str(datetime.datetime.now()));

	return df;



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

# >>> df_features
#            Open        High         Low       Close  Volume  ...  VolatilityNormalized  VolatilityRolling                 ValidFrom                    ValidTo  Symbol
# 0    183.830002  183.940002  183.490204  183.675003  423050  ...              0.006042           0.286561 2024-04-17 09:55:00-04:00  2024-04-17 10:00:00-04:00    AMZN
# 1    183.690002  183.789993  183.332397  183.449997  415565  ...              0.006042           0.343334 2024-04-17 10:00:00-04:00  2024-04-17 10:05:00-04:00    AMZN
# 2    183.455002  183.854996  183.455002  183.695007  327193  ...              0.006042           0.307909 2024-04-17 10:05:00-04:00  2024-04-17 10:10:00-04:00    AMZN
# 3    183.695007  183.759995  183.339996  183.539993  331583  ...              0.006042           0.187451 2024-04-17 10:10:00-04:00  2024-04-17 10:15:00-04:00    AMZN
# 4    183.529999  183.589996  183.360001  183.399994  254919  ...              0.006042           0.166346 2024-04-17 10:15:00-04:00  2024-04-17 10:20:00-04:00    AMZN
# ..          ...         ...         ...         ...     ...  ...                   ...                ...                       ...                        ...     ...
# 355  272.589996  272.850006  272.528198  272.779999   31460  ...              0.001829           0.232457 2024-04-17 15:30:00-04:00  2024-04-17 15:35:00-04:00       V
# 356  272.730011  272.760010  272.209991  272.290009   38812  ...              0.001829           0.248567 2024-04-17 15:35:00-04:00  2024-04-17 15:40:00-04:00       V
# 357  272.309998  272.894989  272.260010  272.779999   49877  ...              0.001829           0.184507 2024-04-17 15:40:00-04:00  2024-04-17 15:45:00-04:00       V
# 358  272.725006  273.100006  272.679993  273.054993   68019  ...              0.001829           0.253919 2024-04-17 15:45:00-04:00  2024-04-17 15:50:00-04:00       V
# 359  273.059998  273.230011  272.760010  273.179993  143960  ...              0.001829           0.316608 2024-04-17 15:50:00-04:00  2024-04-17 15:55:00-04:00       V
#
# [360 rows x 17 columns]
#
#
# >>> df_features.T
#                                             0                          1    ...                        358                        359
# Open                                 183.830002                 183.690002  ...                 272.725006                 273.059998
# High                                 183.940002                 183.789993  ...                 273.100006                 273.230011
# Low                                  183.490204                 183.332397  ...                 272.679993                  272.76001
# Close                                183.675003                 183.449997  ...                 273.054993                 273.179993
# Volume                                   423050                     415565  ...                      68019                     143960
# Dividends                                   0.0                        0.0  ...                        0.0                        0.0
# Stock Splits                                0.0                        0.0  ...                        0.0                        0.0
# LogReturns                            -0.000898                  -0.001226  ...                   0.001008                   0.000458
# PriceChange                           -0.164993                  -0.225006  ...                   0.274994                      0.125
# PercentChange                         -0.000897                  -0.001225  ...                   0.001008                   0.000458
# IsPriceIncrease                               0                          1  ...                          1                          0
# Volatility                             1.099571                   1.099571  ...                   0.498631                   0.498631
# VolatilityNormalized                   0.006042                   0.006042  ...                   0.001829                   0.001829
# VolatilityRolling                      0.286561                   0.343334  ...                   0.253919                   0.316608
# ValidFrom             2024-04-17 09:55:00-04:00  2024-04-17 10:00:00-04:00  ...  2024-04-17 15:45:00-04:00  2024-04-17 15:50:00-04:00
# ValidTo               2024-04-17 10:00:00-04:00  2024-04-17 10:05:00-04:00  ...  2024-04-17 15:50:00-04:00  2024-04-17 15:55:00-04:00
# Symbol                                     AMZN                       AMZN  ...                          V                          V
#
# [17 rows x 360 columns]



data = {ticker: yf.Ticker(ticker).history(period='1d', interval='5m') for ticker in dow_sample};
features = pd.DataFrame();

for symbol, df in data.items():
	df = create_features(df);
	df = make_bitemporal(df);

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