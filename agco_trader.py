import os, dotenv;
import time, schedule;
from uvatradier import Account, EquityOrder;
import logging;

dotenv.load_dotenv();

tradier_acct = os.getenv('tradier_acct');
tradier_token = os.getenv('tradier_token');

acct = Account(tradier_acct, tradier_token);
equity_order = EquityOrder(tradier_acct, tradier_token);

def run_scheduler():
	while True:
		schedule.run_pending();
		time.sleep(1.0);

def buy_shares():
    print("Buying shares...")
    response = equity_order.order(symbol='AGCO', side='buy', quantity=5, order_type='market', duration='day');
    print(acct.get_positions());
    print(response)

def sell_shares():
    print("Selling shares...")
    response = equity_order.order(symbol='AGCO', side='sell', quantity=5, order_type='market', duration='day')
    print(acct.get_positions());
    print(response)



tradier_acct = os.getenv('tradier_acct');
tradier_token = os.getenv('tradier_token');


acct = Account(tradier_acct, tradier_token);
equity_order = EquityOrder(tradier_acct, tradier_token);

schedule.every().day.at("09:35").do(buy_shares);
schedule.every().day.at("15:55").do(sell_shares);

if __name__ == '__main__':
	print('Running....');
	run_scheduler();



