from binance.client import Client
import json
import pandas as pd
import argparse
from log import init_logger
from collections import namedtuple
from datetime import timedelta, datetime, date

class BinanceAPI:

    def __init__(self, api_key_file_path, **kwargs):
        self.logger = kwargs.get('logger', init_logger(__name__, kwargs.get("show_debug", False), kwargs.get('log_to_file', False)))
        self.logger.info(msg='Binance API initialized!')
        self.candlestick_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'unk1', 'unk2', 'unk3', 'unk4', 'unk5', 'unk6', 'unk7']
        
        self._api_key = self._load_api_key(api_key_file_path)
        self._client = Client(self._api_key.get('api_key'), self._api_key.get('secret_key'), {"verify": False, "timeout": 20})

    def _load_api_key(self, path):
        with open(path) as json_file:
            return json.load(json_file)

    def export_candlestick_to_csv(self, output_file="output.csv", candlestick=None):
        df = pd.DataFrame(candlestick, columns = self.candlestick_columns)
        df.to_csv(output_file, index=False)

    def get_historical_candlestick(self, symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1HOUR, date=None, range_start="1 Dec, 2017", range_end=None):
        if date is not None:
            self.logger.info(msg=f'getting {symbol} pair for {interval} interval on date {date}')
            return self._client.get_historical_klines(symbol, interval, date)
        
        self.logger.info(msg=f'getting {symbol} pair for {interval} interval from {range_start} to {range_end}')
        return self._client.get_historical_klines(symbol, interval, range_start, range_end)

    

if __name__ == '__main__':
    logger = init_logger(__name__, show_debug=True, log_to_file=False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--k", dest="binance_api_key", nargs='?', default='./binance_api_key.json')
    args = parser.parse_args()

    logger.info(msg=args.binance_api_key)

    kwargs = {"logger": logger}
    
    binance = BinanceAPI(api_key_file_path=args.binance_api_key, **kwargs)

    candlestick = binance.get_historical_candlestick(symbol='BTCUSDT',
            interval=Client.KLINE_INTERVAL_1HOUR, range_start=str(date.today()), range_end=str(datetime.now()))

    binance.export_candlestick_to_csv(output_file='binance_1h.csv', candlestick=candlestick)
