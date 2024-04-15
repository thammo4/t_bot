import os;
import dotenv;
import random;
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import datetime;
import re;

from scipy.optimize import minimize;
import scipy.stats as stats;

import yfinance as yf;
from uvatradier import Account, Quotes, OptionsData, EquityOrder, OptionsOrder;

dotenv.load_dotenv();

tradier_acct = os.getenv('tradier_acct');
tradier_token = os.getenv('tradier_token');

acct = Account(tradier_acct, tradier_token);
quotes = Quotes(tradier_acct, tradier_token);
equity_order = EquityOrder(tradier_acct, tradier_token);
options_data = OptionsData(tradier_acct, tradier_token);
options_order = OptionsOrder(tradier_acct, tradier_token);

print(f'acct: {tradier_acct}');
print(f'token: {tradier_token}');

colors = ['#FF5733', '#33FFCE', '#335BFF', '#FF33F6', '#75FF33'];


#
# Simp function to plot closing prices using Yahoo! Finance API
#

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


#
# Log likelihood function for set of normally distributed rand vars
#

# def ll_normal (mu, sigma, data):
def ll_normal (params, data):
	mu = params[0]; sigma = params[1];
	n = len(data);
	log_L = -.50*(n*np.log(2*np.pi*np.power(sigma,2)) + np.sum(np.power(data-mu,2))/np.power(sigma,2));
	return -log_L;



# history(start="2020-06-02", end="2020-06-07", interval="1m")

random.seed(10);


ag_symbols = ['AGFS', 'IBA', 'BIOX', 'VFF', 'ALCO', 'LMNR', 'TRC', 'AVD', 'CDZI', 'LOCL', 'AGRO', 'ALG', 'VITL', 'BHIL', 'ANDE', 'AGCO', 'CALM', 'FMC', 'SMG', 'LND'];
ag_basket = random.sample(ag_symbols, 4); print(f'Basket: {ag_basket}');

# account = Account(tradier_acct, tradier_token);
# quotes = Quotes(tradier_acct, tradier_token);



#
# AGCO Daily Returns
#

agco = yf.Ticker('AGCO').history(period='max');
agco['Return'] = agco['Close'].diff();
agco = agco.iloc[1:];



#
# Approxmiate mu, sigma^2 via Max Likelihood Estimation
#


# MLE RESULT:
#
# >>> theta_MLE
#   message: CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH
#   success: True
#    status: 0
#       fun: 12020.581163848386
#         x: [ 1.443e-02  1.076e+00]
#       nit: 6
#       jac: [-5.457e-04  1.819e-03]
#      nfev: 24
#      njev: 8
#  hess_inv: <2x2 LbfgsInvHessProduct with dtype=float64>

# CHECK:
#
# >>> agco['Return'].mean();
# 0.014434131756101167
# >>> agco['Return'].std();
# 1.0762002361954381

theta_0 = [0,1];
theta_MLE = minimize(fun=ll_normal, x0=theta_0, args=(agco['Return'],), method='L-BFGS-B', bounds=[(None,None), (1e-6,None)]);