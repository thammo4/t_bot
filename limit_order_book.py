# mrk_syf_stream.json
# {"type":"quote","symbol":"MRK","bid":125.62,"bidsz":2,"bidexch":"M","biddate":"1713545563000","ask":125.64,"asksz":1,"askexch":"Q","askdate":"1713545563000"},
from stock_trader import *


print(pd.json_normalize(tradier_exchange_list).T);


#
# Convert json file -> Central Limit Order Book
#

def create_clob (file, symbol='', exchange=''):
	stream_data = pd.read_json(file); # print(f'STREAM DATA\n{stream_data.head(10)}\n');

	stream_data = stream_data.query("type == 'quote'")[['bidexch', 'biddate', 'bidsz', 'bid', 'ask', 'asksz', 'askdate', 'askexch', 'symbol']];



	# if symbol is not None:
	if len(symbol) > 0:
		stream_data = stream_data.query("symbol == @symbol");

	# if exchange is not None:
	if len(exchange) > 0:
		stream_data = stream_data.query("bidexch == @exchange");
		stream_data = stream_data.query("askexch == @exchange");

	#
	# Market stream data returns unix timestamps to the millisecond -> Convert to datetime
	#

	stream_data['biddate'] = pd.to_datetime(stream_data['biddate'], unit='ms');
	stream_data['askdate'] = pd.to_datetime(stream_data['askdate'], unit='ms');

	return stream_data;



#
# Plot change in bid/ask through trading day
#

def plot_bid_ask (clob):
	plt.figure(figsize=(14,7));
	plt.plot(clob['biddate'], clob['bid'], label='Bid Price');
	plt.plot(clob['askdate'], clob['ask'], label='Ask Price');
	plt.fill_between(clob['biddate'], clob['bid'], clob['ask'], color='gray', alpha=3/10, label='Spread');
	plt.title(f"{list(clob['symbol'])[0]} Bid/Ask");
	plt.xlabel('Time'); plt.ylabel('Price');
	plt.legend();
	plt.show();


df_clob = create_clob(file='market_streams/market_stream_april22.json', symbol='ICE', exchange='N'); print(f'DF CLOB\n{df_clob}');