# FILE: `ws_market_data.py`
import os, dotenv;
import requests;
import time;

import asyncio;
import websockets;
import json;

from dow30 import DOW30;

dotenv.load_dotenv();

tradier_token_live = os.getenv("tradier_token_live");

def tradier_http_stream():
	while True:
		r = requests.post(url='https://api.tradier.com/v1/markets/events/session', headers={'Authorization':f'Bearer {tradier_token_live}', 'Accept':'application/json'});
		if r.status_code == 200:
			session_info = r.json()['stream'];
			return session_info['sessionid'];
		else:
			print('f#%@^!@k');
			time.sleep(10);

async def ws_connect(session_id, symbol_list):
	async with websockets.connect('wss://ws.tradier.com/v1/markets/events', ssl=True, compression=None) as websocket:
		payload = json.dumps({'symbols':symbol_list, 'sessionid':session_id, 'linebreak':False});
		await websocket.send(payload);
		# print(f'Payload: {payload}');

		#
		# Print market events to standard output
		#

		async for message in websocket:
			print(f'{message}');



#
# Start listening to the web socket
#

session_id = tradier_http_stream();
asyncio.run(ws_connect(session_id, DOW30));

