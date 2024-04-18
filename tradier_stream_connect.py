import os;
import dotenv;
import random;
import requests;
import threading;
import warnings;
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import datetime;
import time;

from uvatradier import Account, Quotes, OptionsData, EquityOrder, OptionsOrder;

dotenv.load_dotenv();
tradier_acct = os.getenv('tradier_acct');
tradier_token = os.getenv('tradier_token');

tradier_acct_live = os.getenv('tradier_acct_live');
tradier_token_live = os.getenv('tradier_token_live');


def tradier_http_stream():
	while True:
		r = requests.post(url='https://api.tradier.com/v1/markets/events/session', data={}, headers={'Authorization':f'Bearer {tradier_token_live}', 'Accept':'application/json'});

		if r.status_code == 200:
			session_info = r.json()['stream'];
			stream_url = session_info['url'];
			stream_session_id = session_info['sessionid'];

			print(f'Connecting to stream {stream_url} with session ID {stream_session_id}');
			time.sleep(240);
		else:
			print("Get f#%$^d");

#
# Initialize thread to handle streaming session
#

thread = threading.Thread(target=tradier_http_stream());
thread.start();
