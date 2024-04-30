# FILE NAME: `tradier_mrkt_stream.py`
import time;
import asyncio, websockets;
import json

from stock_trader import *

some_stocks = ["BA", "CAT", "CSCO", "CVX", "GS", "HD", "HON"]; 		# 4/19/24
wss_test_stocks = ['OMC', 'RL', 'BG', 'ICE', 'MKTX', 'GS', 'CE']; 	# 4/24/24
wss_test_stocks = ['LMT', 'COP', 'C', 'AAPL', 'R', 'V']; 			# 4/29/24


print('\n');
async def ws_connect():
	async with websockets.connect('wss://ws.tradier.com/v1/markets/events', ssl=True, compression=None) as websock:
		# payload = json.dumps({'symbols':DOW30, 'sessionid':'5033215d-2e0c-4e71-80f7-a1b78e4abe78', 'linebreak':False});
		# payload = json.dumps({'symbols':['SYF', 'MRK'], 'sessionid':'05cae2b5-414b-4f0a-86c7-8f3488710dc2', 'linebreak':False});
		# payload = json.dumps({'symbols':wss_test_stocks, 'sessionid':'5f4fc990-2f96-4893-bc63-783efb4e534b', 'linebreak':False});
		payload = json.dumps({'symbols':['KMI'], 'sessionid':'c5810199-e441-429b-bdad-78f202b1c146', 'linebreak':False});
		await websock.send(payload);
		print(f'{payload}');

		async for message in websock:
			print(f'{message}');


asyncio.run(ws_connect());