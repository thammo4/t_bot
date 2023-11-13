import os, dotenv;
import numpy as np;
import pandas as pd;

from datetime import datetime, timedelta;

import yfinance as yf; # Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]

from fredapi import Fred;
from uvatradier import Tradier, Quotes;


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

quotes = Quotes(tradier_acct, tradier_token);



#
# Fetch historic stock data for universe of discourse
#

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
# Build SVM-based Volatility Model
#

# 1. fetch volatility data from FRED

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
# Build Cluster-Based Market Regime Detection Model
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



















