import time;
import asyncio, websockets;
import json

from stock_trader import *

some_stocks = ["BA", "CAT", "CSCO", "CVX", "GS", "HD", "HON"];

async def ws_connect():
	async with websockets.connect('wss://ws.tradier.com/v1/markets/events', ssl=True, compression=None) as websock:
		# payload = f'{"symbols":"{random.sample(DOW30,6)}", "sessionid":"552a006f-b9dc-474c-80b7-9430234c9b0c", "linebreak":false}';
		# payload = '{"symbols":["AMZN", "AXP", "AMGN", "AAPL", "BA", "CAT", "CSCO", "CVX", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "KO", "JPM", "MCD", "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "CRM", "VZ", "V", "WMT", "DIS", "DOW"], "sessionid":"09f44474-be25-45b8-ba4a-5fd4f9bc6a35", "linebreak":false}';
		# payload = '{"symbols":some_stocks, "sessionid":"b82e2f38-1b75-489d-9970-394489e91374", "linebreak":False}';
		payload = json.dumps({'symbols':some_stocks, 'sessionid':'b82e2f38-1b75-489d-9970-394489e91374', 'linebreak':False});
		await websock.send(payload);
		print(f'---> {payload}');

		async for message in websock:
			print(f'---> {message}');


asyncio.run(ws_connect());
