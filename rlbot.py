import os, dotenv;
import numpy as np;
import pandas as pd;

# Packages for RIDGE Regression
from sklearn.linear_model import Ridge, RidgeCV;
from sklearn.model_selection import RepeatedKFold;

from datetime import datetime, timedelta;

# Packages for data acquisition
import yfinance as yf;
from fredapi import Fred;

from uvatradier import Tradier, Account, EquityOrder; # tradier will be used to execute trades


#
# Define params/hyperparams for RL Agent
#

EPISODES = 100;
DAYS_PER_EPISODE = 10;

ALPHA = .10;
GAMMA = .995;
EPSILON = .05;

ACCT_BAL_0 = 1000;


SHARES_PER_TRADE = 10;


#
# Load API Keys
#

dotenv.load_dotenv();

fred_api_key = os.getenv('fred_api_key');

tradier_acct = os.getenv('tradier_acct');
tradier_token = os.getenv('tradier_token');


#
# Instantiate FRED and Tradier Objects
#

fred = Fred(api_key = fred_api_key);



#
# Fetch historic stock data for universe of discourse
#

blue_chip_stocks = [
	'AA', 'ABBV', 'AXP', 'BA', 'BAC', 'C',
	'CAT', 'CI', 'CVX', 'DD', 'DIS', 'GE',
	'GM', 'HD', 'HPQ', 'IBM', 'JNJ', 'JPM',
	'KO', 'MCD', 'MMM', 'MRK', 'PFE', 'PG',
	'T', 'VZ', 'WMT', 'XOM'
];


stock_basket = ['DD', 'IBM', 'JPM', 'KO', 'VZ', 'XOM'];

start_date 	= datetime.today() - timedelta(5*365);
end_date 	= datetime.today();

# >>> stock_data
#
#                    DD         IBM         JPM         KO         VZ         XOM
# Date                                                                           
# 2018-11-14  82.521179  114.913956  107.330002  49.759998  58.939999   77.389999
# 2018-11-15  83.588631  116.099426  110.070000  49.740002  59.080002   78.190002
# 2018-11-16  84.243340  116.223709  109.989998  50.169998  60.209999   78.959999
# 2018-11-19  82.264992  115.019119  110.830002  50.509998  60.619999   79.220001
# 2018-11-20  80.229713  112.045891  108.449997  49.380001  59.459999   76.970001
# ...               ...         ...         ...        ...        ...         ...
# 2023-11-06  69.540001  148.970001  144.080002  56.970001  35.639999  105.870003
# 2023-11-07  68.430000  148.830002  144.009995  57.180000  35.939999  104.209999
# 2023-11-08  68.410004  148.029999  144.720001  57.090000  35.770000  102.930000
# 2023-11-09  67.830002  146.619995  144.289993  56.660000  35.619999  102.959999
# 2023-11-10  68.739998  149.020004  146.429993  56.720001  35.709999  103.750000
#
# [1256 rows x 6 columns]


# Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
dd = yf.download('DD', start=start_date, end=end_date, interval='1d');
ibm = yf.download('IBM', start=start_date, end=end_date, interval='1d');
jpm = yf.download('JPM', start=start_date, end=end_date, interval='1d');
ko = yf.download('KO', start=start_date, end=end_date, interval='1d');
vz = yf.download('VZ', start=start_date, end=end_date, interval='1d');
xom = yf.download('XOM', start=start_date, end=end_date, interval='1d');

stock_data = pd.DataFrame({
	'DD' : dd['Close'],
	'IBM': ibm['Close'],
	'JPM': jpm['Close'],
	'KO' : ko['Close'],
	'VZ' : vz['Close'],
	'XOM': xom['Close']
});


#
# Supervised Learning Feature Mining for Augmented State Space Representation
# 	I. SVM-Volatility
# 	II. Cluster-Based Market Regime Detection
# 	(III. Something with a random forest would be cool/probably useful because noisy/non-linear financial data)
# 	(IV. Logistic Regression anything)
#


#
# A. Build SVM-based Volatility Model
#

# 1. fetch volatility data from FRED

# >>> for x in volatility_basket:
# ...     print(f'{x}: {fred.get_series_info(x)["title"]}');
#
# VIXCLS: CBOE Volatility Index: VIX
# OVXCLS: CBOE Crude Oil ETF Volatility Index
# GVZCLS: CBOE Gold ETF Volatility Index
# RVXCLS: CBOE Russell 2000 Volatility Index
# VXNCLS: CBOE NASDAQ 100 Volatility Index
# EVZCLS: CBOE EuroCurrency ETF Volatility Index

volatility_basket = ['VIXCLS', 'OVXCLS', 'GVZCLS', 'RVXCLS', 'VXNCLS', 'EVZCLS']

vix = fred.get_series('VIXCLS');
ovx = fred.get_series('OVXCLS');
gvz = fred.get_series('GVZCLS');
rvx = fred.get_series('RVXCLS');
vxn = fred.get_series('VXNCLS');
evz = fred.get_series('EVZCLS');


volatility_data = pd.DataFrame({
	'VIX': vix,
	'OVX': ovx,
	'GVZ': gvz,
	'RVX': rvx,
	'VXN': vxn,
	'EVZ': evz
});

# The Gold ETF Volatility Index started most recently (2008-06-03).
# Filter out rows whose date precedes June 03, 2008.

volatility_data = volatility_data[volatility_data.index >= pd.to_datetime('2008-06-03')];






# 2. For each column in stock_data, construct a SVM-vix model:
# 	• Create a new dataframe with one column for a single stock's daily closing price and the other columns filled with volatility data
# 	• Convert closing price -> daily price change
# 	• Response variable = daily_price_change
# 	• Feature variables = covariates vix data
# 		• Partition vix data into training/testing
# 		• Fix SVM to training data
# 		• Put vix testing data into fitted svm to produce predicted increase/decrease labels for testing data
# 		• Assess SVM performance by comparing predicted labels to actual increase/decrease for each test observation

# (Assume that the reinforcement learning Q-learning algorithm is implemented/prepared)

# 3. Incorporate the SVM into the RL Agent state space.
# 	• SVM will produce a label (e.g. price increases or price decreases) for training day that RL agent will encounter during the learning phase.
# 	• When RL agent loops episodes/steps, lookup each stock's SVM-VIX label for the current iteration's day, t.
# 		• If label_t = increase -> Encourage agent to buy by inflating R_[t+1]
# 		• If label_t = decrease -> Discourage agent from buying by retarding R_[t+1]




#
# B. Build Cluster-Based Market Regime Detection Model
#

# 1. Get a bunch of data sets that heuristically reflect market conditions
# 	• Examples:
# 		• Market: 	DJIA, SP500 (SP500), NASDAQ Composite, Russell 2000, Wilshire 5000, CBOE ^VIX
# 		• Economy: 	GDP, Consumer Confidence Index (UMCSENT), Consumer Price Index (CPIAUCSL), Producer Price Index (PPIACO), Purchasing Manager's Index, Unemployment Rate (UNRATE), Retail Sales (RSXFS)
# 		• Govt: 	Yield Curve
# 		• Housing: 	New Homes Built (HOUST), New Building Permits Issued (PERMIT), New Home Sales (HSN1F)
# 		• Fed: 		Federal Funds Rate (FEDFUNDS), Discount/Primary Credit Rate (DISCOUNT), Reserve Balances with Federal Reserve Banks (WRESBAL), Federal Reserve Total Assets (WALCL), Federal Funds Rate Monthly (FEDFUNDSM)

# 2. Build cluster-based model to classify sets of covariates into three categories:
# 	• I. 	Bull Market
# 	• II. 	Bear Market
# 	• III. 	Neutral Market

# 3. Incorporate cluster model output into RL Agent State Space
# 	• Cluster model will output a label (bull/bear/neutral) for each day the RL agent encounters during its learning/training phase.
# 	• As agent iterates over each day, it will lookup the market_regime label from the cluster model.




#
# C. Construct RIDGE Regression Model To Predict Future Stock Prices (`predict_stock_return`)
#

# 1. Get a bunch of data series that might be useful for predicting a blue-chip stock's daily closing price.
# 	• Examples:
# 		• Historic: Past daily closing prices of the stock under consideration (e.g. past prices of DD,IBM,...,XOM)
# 		• Market: 	DJIA, SP500 (SP500), NASDAQ Composite, Russell 2000, Wilshire 5000, CBOE ^VIX
# 		• Economy: 	GDP, Consumer Confidence Index (UMCSENT), Consumer Price Index (CPIAUCSL), Producer Price Index (PPIACO), Purchasing Manager's Index, Unemployment Rate (UNRATE), Retail Sales (RSXFS)
# 		• Govt: 	Yield Curve
# 		• Housing: 	New Homes Built (HOUST), New Building Permits Issued (PERMIT), New Home Sales (HSN1F)
# 		• Fed: 		Federal Funds Rate (FEDFUNDS), Discount/Primary Credit Rate (DISCOUNT), Reserve Balances with Federal Reserve Banks (WRESBAL), Federal Reserve Total Assets (WALCL), Federal Funds Rate Monthly (FEDFUNDSM)


# 2. Organize all of the data series into a sensible data frame -> partition the buncha-datasets into training and testing samples
# 	• Maybe something here about cross-validation

# 3. Fit a RIDGE regression model to the training data


# 4. Stick the testing dataset's covariates into the RIDGE model to assess its performance.
# 	• If (model is trash) -> get a new set of predictor variables and try again until it is nice

# 5. Rewrite the `predict_stock_return` function to incorporate the RIDGE model
# 	• Input: (Ticker Symbol of a Single Stock, Trading Day Date)
# 	• Output: RIDGE-estimated closing price for the given Ticker Symbol on the given Trading Day Date.







#
# Define RL Action Space
#

MAX_SHARES_PER_STOCK 	= 10;
MIN_BAL_TO_TRADE 		= stock_data.min().min(); # we buy until we can't!

actions = ('hold', 'buy', 'sell');

# def get_action (acct_bal, stock_prices, portfolio, wager_frac):
# 	actions = [];

# 	action = ('hold', 0);

# 	for stock in stock_basket:
# 		price = stock_prices[stock];
# 		if should_buy(acct_bal, stock, portfolio):
# 			num_shares = int((acct_bal*wager_frac) / price);
# 			action = ('buy', num_shares);
# 		elif should_sell(stock, portfolio):
# 			num_shares = calculate_shares_to_sell(portfolio, stock);
# 			action = ('sell', num_shares);

# 		actions.append(action);

# 	return actions;












def get_action (acct_bal, stock_prices, portfolio, estimated_edge, win_prob):
	actions = [];

	action = ('hold', 0);
	for stock in stock_basket:
		price = stock_prices[stock];
		trade_fraction = kelly_wager(estimated_edge, win_prob);
		if should_buy (acct_bal, stock, portfolio):
			num_shares = int((acct_bal*trade_fraction) / price);
			action = ('buy', num_shares);
		elif should_sell(stock, portfolio):
			num_shares = calculate_shares_to_sell(portfolio, stock);
			action = ('sell', num_shares);

		actions.append(action);

	return actions;


def kelly_wager (edge, prob):
	loss_prob = 1 - prob;
	return (estimated_edge*prob - loss_prob) / estimated_edge;



def should_buy (acct_bal, stock, portfolio, stock_prices, threshold):
	is_good_buy = False;

	expected_return = predict_stock_return(stock, stock_prices);

	if expected_return > threshold and acct_bal > MIN_BAL_TO_TRADE:
		is_good_buy = True;

	return is_good_buy;

def predict_stock_return (stock, stock_prices):
	# PUT A PREDICTIVE MODEL OF STOCK PRICES HERE
	return .05;


def should_sell (stock, portfolio):
	'''This function should implement stock selling logic based upon trends, time held, price, etc...'''
	print('hello, should_sell!');

def calculate_shares_to_sell (portfolio, stock):
	'''This function should determine the number of owned shares with which to part'''
	print('hello, calculate_shares_to_sell!');






#
# RIDGE Regression Example (for temporary reference during development)
# 	• NB: for python, `alpha` is the ordinary lambda tuning parameter
#

df = pd.read_csv("https://raw.githubusercontent.com/Statology/Python-Guides/main/mtcars.csv")[['mpg', 'wt', 'drat', 'qsec', 'hp']];

# Define covariates (X) and response (Y)
# For stock trading:
# 	• df_X = Factors that contribute to / affect the closing price of a stock
# 	• df_Y = Closing price of a stock
df_X = df[['mpg', 'wt', 'drat', 'qsec']];
df_Y = df['hp'];

# Perform 10-fold cross validation three times
# For stock trading:
# 	• This can probably stay the same
xval = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1);

# Define model and fit to training data
# For stock trading:
# 	• This can probably stay the same
model = RidgeCV(alphas=np.arange(.01, 1, .01), cv=xval, scoring='neg_mean_absolute_error');
model.fit(df_X, df_Y);

print('Tuning parameter (lambda) corresponding to lowest test MSE: {}'.format(model.alpha_));

# Prediction with RIDGE Model
# For stock trading:
# 	• Update the df_new to contain the future values of the covariates from df_X
df_new = pd.DataFrame({'mpg':[24], 'wt':[2.5], 'drat':[3.5], 'qsec':[18.5]});
df_predicted = model.predict(df_new);

print(f'Predicted HP: {df_predicted[0]}');