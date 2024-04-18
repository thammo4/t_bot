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
    print("Buying shares...");
    response = equity_order.order(symbol='AGCO', side='buy', quantity=5, order_type='market', duration='day'); print(response);
    print('Positions:'); print(acct.get_positions()); print('\n');
    print('P&L'); print(acct.get_gainloss()); print('\n');
    print('Done.');
    print('------------------------------');
    print('\n');

def sell_shares():
    print("Selling shares...")
    response = equity_order.order(symbol='AGCO', side='sell', quantity=5, order_type='market', duration='day'); print(response);
    print('Positions:'); print(acct.get_positions()); print('\n');
    print('P&L'); print(acct.get_gainloss()); print('\n');
    print('Done.');
    print('------------------------------');
    print('\n');

tradier_acct = os.getenv('tradier_acct');
tradier_token = os.getenv('tradier_token');


acct = Account(tradier_acct, tradier_token);
equity_order = EquityOrder(tradier_acct, tradier_token);

schedule.every(5).minutes.do(sell_shares);
schedule.every(5).minutes.do(buy_shares);

if __name__ == '__main__':
	print('Running....');
	run_scheduler();



