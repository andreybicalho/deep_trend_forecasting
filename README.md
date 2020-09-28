# Deep Trend Forecasting

# Get Binance Candlestick Dataset
````
python get_candlestick_from_binance.py --k=binance_api_key.json --s=BTCUSDT --i=1d --rs="1 Dec, 2017"
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

* Train and forecast with the most recent candlestick: 
````
python trend_forecasting.py --k=binance_api_key.json --s=BTCUSDT --i=1h --rs="1 Dec, 2017" --ws=200 --e=30 --st=0.9 --fs=30 --v
````

* Train and forecast with downloaded candlestick:
````
python trend_forecasting.py --ws=60 --e=30 --st=0.8 --fs=30 --d=BTCUSDT_4h.csv
````

* Forecast with the available model and dataset:
````
python trend_forecasting.py --d=BTCUSDT_1h.csv --ws=200 --fs=3 --v --m=model
````