import os, dotenv;
import numpy as np;
import pandas as pd;
from fredapi import Fred;

#
# Load API keys from .env file
#

dotenv.load_dotenv();
fred_api_key = os.getenv('fred_api_key');


#
# Instantiate FRED object
#

fred = Fred(api_key=fred_api_key);


#
# Retrieve DJIA Data (Response)
#

djia = fred.get_series('DJIA');



#
# Fetch Savings Account Data (Covariates)
#

sndr = fred.get_series('SNDR');
mmty = fred.get_series('MMTY');
ndr12mcd = fred.get_series('NDR12MCD');
ty12mcd = fred.get_series('TY12MCD');
ty3mcd = fred.get_series('TY3MCD');

