import os;
import dotenv;
import random;
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import datetime;

import yfinance as yf;
from uvatradier import Account, Quotes;

dotenv.load_dotenv();

tradier_acct = os.getenv('tradier_acct');
tradier_token = os.getenv('tradier_token');

print(f'acct: {tradier_acct}');
print(f'token: {tradier_token}');

colors = ['#FF5733', '#33FFCE', '#335BFF', '#FF33F6', '#75FF33'];


def plot_price (symbol, start_date='2024-01-01', end_date=str(datetime.date.today()), interval='1d', show_plot=False):
	symbol = symbol.upper();
	data = yf.Ticker(symbol).history(start=start_date, end=end_date, interval=interval);
	plt.figure(figsize=(10, 5));
	plt.plot(data.index, data['Close'], label=symbol);
	plt.title(f'{symbol} Closing Price ({start_date}, {end_date})');
	plt.xlabel('Date'); plt.ylabel('Close Price');
	plt.grid(True);
	plt.legend();
	if show_plot:
		plt.show();



# history(start="2020-06-02", end="2020-06-07", interval="1m")

random.seed(10);


ag_symbols = ['AGFS', 'IBA', 'BIOX', 'VFF', 'ALCO', 'LMNR', 'TRC', 'AVD', 'CDZI', 'LOCL', 'AGRO', 'ALG', 'VITL', 'BHIL', 'ANDE', 'AGCO', 'CALM', 'FMC', 'SMG', 'LND'];
ag_basket = random.sample(ag_symbols, 4); print(f'Basket: {ag_basket}');

# account = Account(tradier_acct, tradier_token);
# quotes = Quotes(tradier_acct, tradier_token);


