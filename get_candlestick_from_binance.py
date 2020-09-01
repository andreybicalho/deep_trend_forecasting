"""
KLINE_INTERVAL_12HOUR = '12h'
KLINE_INTERVAL_15MINUTE = '15m'
KLINE_INTERVAL_1DAY = '1d'
KLINE_INTERVAL_1HOUR = '1h'
KLINE_INTERVAL_1MINUTE = '1m'
KLINE_INTERVAL_1MONTH = '1M'
KLINE_INTERVAL_1WEEK = '1w'
KLINE_INTERVAL_2HOUR = '2h'
KLINE_INTERVAL_30MINUTE = '30m'
KLINE_INTERVAL_3DAY = '3d'
KLINE_INTERVAL_3MINUTE = '3m'
KLINE_INTERVAL_4HOUR = '4h'
KLINE_INTERVAL_5MINUTE = '5m'
KLINE_INTERVAL_6HOUR = '6h'
KLINE_INTERVAL_8HOUR = '8h'
"""

import argparse

from binance_api import BinanceAPI
from log import init_logger

from datetime import datetime

logger = init_logger(__name__, show_debug=True, log_to_file=False)

parser = argparse.ArgumentParser()
parser.add_argument("--k", dest="binance_api_key", nargs='?', default='./binance_api_key.json')
parser.add_argument("--s", dest="symbol", nargs='?', default='BTCUSDT')
parser.add_argument("--i", dest="interval", nargs='?', default='1d')
parser.add_argument("--rs", dest="range_start", nargs='?', default='1 Dec, 2017')
parser.add_argument("--re", dest="range_end", nargs='?', default=str(datetime.now()))
args = parser.parse_args()

print(args)

kwargs = {"logger": logger}
binance = BinanceAPI(api_key_file_path=args.binance_api_key, **kwargs)

candlestick = binance.get_historical_candlestick(symbol=args.symbol,
            interval=args.interval, range_start=args.range_start, range_end=args.range_end)

binance.export_candlestick_to_csv(output_file=f'{args.symbol}_{args.interval}.csv', candlestick=candlestick)
