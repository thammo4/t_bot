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
# Retrieve initial account balance from Tradier.
# This will just be the current total_cash in the account.
#

ACCT_BAL_0 = account.get_account_balance()['total_cash'][0];























































































