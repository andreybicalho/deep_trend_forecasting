# Deep Trend Forecasting

# Get Binance Candlestick Dataset
````
python get_candlestick_from_binance.py --k=binance_api_key.json --s=BTCUSDT --i=1d --rs="1 Dec, 2017" --o=BTCUSDT_1d.csv
````

# Training
````
python train.py --d=BTCUSDT_1d.csv --ws=30 --e=50
````

# Predicting
````
python predict.py --d=BTCUSDT_1d.csv --ws=30 --ts=7
````

# Forecasting on Binance Exchange

## Most Recent Candlestick: 
````
python trend_forecasting.py --k=binance_api_key.json --s=BTCUSDT --i=1d --rs="1 Dec, 2017" --ws=60 --e=30 --st=0.8 --fs=14 --v
````

## Downloaded candlestick:
````
python trend_forecasting.py --ws=60 --e=30 --st=0.8 --fs=30 --inds=BTCUSDT_4h.csv
````