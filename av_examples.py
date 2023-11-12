import os, dotenv;

import numpy as np;
import pandas as pd;

from alpha_vantage.timeseries import TimeSeries;
from alpha_vantage.foreignexchange import ForeignExchange;
from alpha_vantage.techindicators import TechIndicators;
from alpha_vantage.sectorperformance import SectorPerformances;


#
# Load Alpha Vantage API Keys from .env file
#

dotenv.load_dotenv();

alpha_vantage_api_key = os.getenv('alpha_vantage_api_key');



#
# Instantiate base objects for:
# 	• time series
# 	• foreign exchange
# 	• technical indicators
# 	• sector performance
#

ts = TimeSeries(key=alpha_vantage_api_key, output_format='pandas');
fx = ForeignExchange(key=alpha_vantage_api_key, output_format='pandas');
ti = TechIndicators(key=alpha_vantage_api_key, output_format='pandas');


# fx = ForeignExchange(key=alpha_vantage_api_key, output_format='pandas');
# ti = TechIndicators(key=alpha_vantage_api_key, ouput_format='pandas');
# sp = SectorPerformances(key=alpha_vantage_api_key, ouput_format='pandas');


#
# Daily closing stock prices for Capital One
#

cof, meta_cof = ts.get_daily(symbol='COF', outputsize='full');


#
# Intraday stock prices for Proctor & Gamble
#

pg, meta_pg = ts.get_intraday(symbol='PG', interval='1min', outputsize='full');


#
# Foreign exchange rate for Euros -> US Dollars
#
# Alpha Vantage API calls return a 2 element list.
# The second element is normally meta data, but this forex call returns an empty string.
#

fx_euro_usd, tmp_none = fx.get_currency_exchange_rate(from_currency='EUR', to_currency='USD');


#
# Technical Analysis Indicator - Simple Fifty-Day Moving Average for Linde plc
#

sma_lin, sma_lin_meta = ti.get_sma(symbol='MCD', interval='monthly', time_period=50, series_type='close');