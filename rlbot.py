import os, dotenv;
import numpy as np;
import pandas as pd;
from datetime import datetime, timedelta;

# Packages for RIDGE Regression
from sklearn.linear_model import Ridge, RidgeCV;
from sklearn.model_selection import RepeatedKFold;

# Packages for data acquisition
import yfinance as yf;
from fredapi import Fred;

# Execute paper trade orders on Tradier brokerage platform
# https://pypi.org/project/uvatradier/
from uvatradier import Tradier, Account, EquityOrder;

# Packages to implement neural network that approximates action-value function
from tensorflow.keras.models import Sequential;
from tensorflow.keras.layers import Dense;
from tensorflow.keras.optimizers import Adam;


#
# Define params/hyperparams for RL Agent
#

EPISODE_COUNT 		= 150;
DAYS_PER_EPISODE 	= 10;

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
# Instantiate FRED
#

fred = Fred(api_key = fred_api_key);


#
# Instantiate Tradier Account and EquityOrder.
#

account 		= Account(tradier_acct, tradier_token);
equity_order 	= EquityOrder(tradier_acct, tradier_token);



#
# Fetch historic stock data for universe of discourse
#


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


#
#  These commands download data for Yahoo Finance.
# To avoid making excessive calls to the yfinance API, we save them locally as CSVs and load them.
#

# dd = yf.download('DD', start=start_date, end=end_date, interval='1d');
# ibm = yf.download('IBM', start=start_date, end=end_date, interval='1d');
# jpm = yf.download('JPM', start=start_date, end=end_date, interval='1d');
# ko = yf.download('KO', start=start_date, end=end_date, interval='1d');
# vz = yf.download('VZ', start=start_date, end=end_date, interval='1d');
# xom = yf.download('XOM', start=start_date, end=end_date, interval='1d');

dd 	= pd.read_csv('dd.csv');
ibm = pd.read_csv('ibm.csv');
jpm = pd.read_csv('jpm.csv');
ko 	= pd.read_csv('ko.csv');
vz 	= pd.read_csv('vz.csv');
xom = pd.read_csv('xom.csv');

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



#######################################
# A. Build SVM-based Volatility Model #
#######################################



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

# vix = fred.get_series('VIXCLS');
# ovx = fred.get_series('OVXCLS');
# gvz = fred.get_series('GVZCLS');
# rvx = fred.get_series('RVXCLS');
# vxn = fred.get_series('VXNCLS');
# evz = fred.get_series('EVZCLS');


# volatility_data = pd.DataFrame({
# 	'VIX': vix,
# 	'OVX': ovx,
# 	'GVZ': gvz,
# 	'RVX': rvx,
# 	'VXN': vxn,
# 	'EVZ': evz
# });

# The Gold ETF Volatility Index started most recently (2008-06-03).
# Filter out rows whose date precedes June 03, 2008.

# volatility_data = volatility_data[volatility_data.index >= pd.to_datetime('2008-06-03')];

# Read the volatility_data dataframe from a local CSV to avoid excessive API calls to FRED
volatility_basket 	= ['VIXCLS', 'OVXCLS', 'GVZCLS', 'RVXCLS', 'VXNCLS', 'EVZCLS']
volatility_data 	= pd.read_csv("fred_volatility_data.csv", index_col=0);

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





########################################################
# B. Build Cluster-Based Market Regime Detection Model #
########################################################


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






###############################################################################################
# C. Construct RIDGE Regression Model To Predict Future Stock Prices (`predict_stock_return`) #
###############################################################################################


# 1. Get a bunch of data series that might be useful for predicting a blue-chip stock's daily closing price.
# 	• Examples:
# 		• Historic: Past daily closing prices of the stock under consideration (e.g. past prices of DD,IBM,...,XOM)
# 		• Market: 	DJIA, SP500 (SP500), NASDAQ Composite, Russell 2000, Wilshire 5000, CBOE ^VIX
# 		• Economy: 	GDP, Consumer Confidence Index (UMCSENT), Consumer Price Index (CPIAUCSL), Producer Price Index (PPIACO), Purchasing Manager's Index, Unemployment Rate (UNRATE), Retail Sales (RSXFS)
# 		• Govt: 	Yield Curve
# 		• Housing: 	New Homes Built (HOUST), New Building Permits Issued (PERMIT), New Home Sales (HSN1F)
# 		• Fed: 		Federal Funds Rate (FEDFUNDS), Discount/Primary Credit Rate (DISCOUNT), Reserve Balances with Federal Reserve Banks (WRESBAL), Federal Reserve Total Assets (WALCL), Federal Funds Rate Monthly (FEDFUNDSM)


# TO DO: ENSURE THAT EACH OF THE BELOW FRED DATAFRAMES CONTAIN DAILY DATA (e.g. not weekly/monthly/quarterly)
# 	• Fill in the inter-observation missing daily days values with preceding observation's value (simple but whatever)

overall_market_ids 	= ['DJIA', 'DJTA', 'WILL5000IND']; 								# DJIA, DJTA start 2013-11-14
macro_economic_ids 	= ['GDP', 'UMCSENT', 'CPIAUCSL', 'PPIACO', 'UNRATE', 'RSXFS']; 	# RSXFS starts 1992; need to adjust because some monthly basis, some quarterly basis.
federal_reserve_ids = ['FEDFUNDS', 'WALCL', 'WRESBAL']; 							# WALCL starts 2002-12-18; freqs = [monthly, weekly, weekly]




# 2. Organize all of the data series into a sensible data frame -> partition the buncha-datasets into training and testing samples
# 	• Maybe something here about cross-validation

# 3. Fit a RIDGE regression model to the training data

# 4. Stick the testing dataset's covariates into the RIDGE model to assess its performance.
# 	• If (model is trash) -> get a new set of predictor variables and try again until it is nice

# 5. Rewrite the `predict_stock_return` function to incorporate the RIDGE model
# 	• Input: (Ticker Symbol of a Single Stock, Trading Day Date)
# 	• Output: RIDGE-estimated closing price for the given Ticker Symbol on the given Trading Day Date.




#
# Define RL State Space Representation
#

class State:
	def __init__ (self, portfolio, acct_bal, market_data, economic_indicators):
		self.portfolio = portfolio;
		self.acct_bal = acct_bal;
		self.market_data = market_data;
		self.economic_indicators = economic_indicators



#
# Define RL Action Space
#

MAX_SHARES_PER_STOCK 	= 10;
MIN_BAL_TO_TRADE 		= stock_data.min().min(); # we buy until we can't!

actions = ('hold', 'buy', 'sell');

def execute_action (state, action, stock, kelly_frac):
	if action == 'buy':
		# Retrieve estimated probability of "win" (e.g. that the stock's price will increase)
		# Retrieve expected return from RIDGE regression model.
		# 	• RIDGE model will produce a predicted future price of the stock.
		# 	• Using the price at which we bought the stock and the RIDGE-predicted future price, we can compute the expected return.
		expected_return, win_prob = predict_stock_return(stock, state.market_data);
		num_shares = calculate_shares_to_buy(state.acct_bal, stock, expected_return, win_prob, kelly_frac);
		buy_stock(state, stock, num_shares);
	elif action == 'sell':
		num_shares = calculate_shares_to_sell(state.portfolio, stock, kelly_frac);
		sell_stock(state, stock, num_shares);


def kelly_wager (prob_win, prob_loss, loss_frac, win_frac):
	return (prob_win / loss_frac) - (prob_loss / win_frac);


def predict_stock_return (stock, stock_prices):
	# PUT A RIDGE REGRESSION PREDICTIVE MODEL OF STOCK PRICES HERE
	return .05;


def calculate_shares_to_buy (acct_bal, stock_price, expected_return, win_prob):
	kelly_frac = win_prob * expected_return - (1-win_prob);

	if kelly_frac < 0 or kelly_frac > 1:
		return 0;

	num_shares = int((kelly_frac*acct_bal) / stock_price);

	return min(num_shares, MAX_SHARES_PER_STOCK);


def calculate_shares_to_sell (portfolio, stock):
	'''This function will determine the appropriate number of shares to sell of `stock`'''
	num_shares_held = portfolio[stock];
	return num_shares_held;



#
# Define Feed-Forward Artificial Neural Network containing a single hidden layer with nonlinear activation functions.
#


#
# Define neural network I/O dimensions
#

# ann_input_size = state_count_A + state_count_B;
# ann_output_size = len(ACTIONS);

# q_network = q_ann(ann_input_size, ann_output_size);



#
# Train RL Agent
#

for ep in range(EPISODE_COUNT):
	# state_A = ep * DAYS_PER_EPISODE % state_count_A;
	# state_B = ep * DAYS_PER_EPISODE % state_count_B;

	acct_bal = ACCT_BAL_0;

	# shares_held_A = 0;
	# shares_held_B = 0;

	for day in range(DAYS_PER_EPISODE):

		#
		# Represent state with one-hot encoded value
		#

		# state = one_hot_states(state_A, state_B);

		#
		# Choose action per epsilon-greedy policy
		#

		# action_index = choose_action(state_A, state_B, q_network);

		# action_outcome = execute_action(state_A, state_B, action_index, shares_held_A, shares_held_B, acct_bal);

		# print('priceA = ', closing_price_A(state_A));
		# print('priceB = ', closing_price_B(state_B));

		# next_state_A = action_outcome['next_state_A']; 	print('nextA = ', next_state_A);
		# next_state_B = action_outcome['next_state_B']; 	print('nextB = ', next_state_B);
		# shares_held_A = action_outcome['shares_A']; 	print('sharesA = ', shares_held_A);
		# shares_held_B = action_outcome['shares_B']; 	print('sharesB = ', shares_held_B);
		# reward = action_outcome['reward']; 				print('reward = ', reward);
		# acct_bal = action_outcome['bal']; 				print('account balance = ', acct_bal);


		#
		# Compute target weights
		#

		# next_state 	= one_hot_states(next_state_A, next_state_B);
		# q_vals_next = q_network.predict(next_state.reshape(1,-1));
		# target 		= reward + GAMMA * np.amax(q_vals_next[0]);


		#
		# Update target weight parameters for neural network
		#

		# q_vals = q_network.predict(state.reshape(1, -1));
		# q_vals[0][action_index] = target;
		# q_network.fit(state.reshape(1,-1), q_vals, epochs=1, verbose=0);


		#
		# Define terminal condition (e.g. ran out of money)
		#

		if acct_bal <= 0:
			print('!!!!!! BANKRUPT !!!!!!!!');
			break;


		#
		# Update Q-function
		#

		# state_A, state_B = next_state_A, next_state_B;
		# print('Q VALUES\n', q_vals);

		print('-------------');
		print('\n');
