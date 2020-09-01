import numpy as np
import tensorflow as tf
import os
import argparse
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

from binance_api import BinanceAPI
from log import init_logger

parser = argparse.ArgumentParser()
parser.add_argument("--k", dest="binance_api_key", nargs='?', default='./binance_api_key.json')
parser.add_argument("--s", dest="symbol", nargs='?', default='BTCUSDT')
parser.add_argument("--i", dest="interval", nargs='?', default='1d')
parser.add_argument("--rs", dest="range_start", nargs='?', default='1 Dec, 2017')
parser.add_argument("--re", dest="range_end", nargs='?', default=str(datetime.now()))
parser.add_argument("--inds", dest="input_dataset", nargs='?')
parser.add_argument("--ws", dest="window_size", nargs='?', type=int, default=30)
parser.add_argument("--e", dest="epochs", nargs='?', type=int, default=50)
parser.add_argument("--st", dest="split_time", nargs='?', type=float, default=0.8) # 80% for training
parser.add_argument("--fs", dest="forecast_size", nargs='?', type=int, default=7)
parser.add_argument("--v", dest="verbose", action='store_true')
args = parser.parse_args()

logger = init_logger(__name__, show_debug=args.verbose, log_to_file=False)

WINDOW_SIZE = args.window_size
EPOCHS = args.epochs

def load_dataset(dataset_path='dataset.csv'):
  closing_prices = []
  with open(dataset_path) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
      closing_prices.append(float(row[4]))

  return closing_prices

def get_dataset(**kwargs):
    binance = BinanceAPI(api_key_file_path=args.binance_api_key, **kwargs)

    candlestick = binance.get_historical_candlestick(symbol=args.symbol,
                interval=args.interval, range_start=args.range_start, range_end=args.range_end)

    binance.export_candlestick_to_csv(output_file=f'{args.symbol}_{args.interval}.csv', candlestick=candlestick)

    closing_prices = []
    for row in candlestick:
      closing_prices.append(float(row[4]))

    return closing_prices

def windowed_dataset(series, window_size, batch_size, shuffle_buffer_size):
  series = tf.expand_dims(series, axis=-1)
  ds = tf.data.Dataset.from_tensor_slices(series)
  ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
  ds = ds.flat_map(lambda w: w.batch(window_size + 1))
  ds = ds.shuffle(shuffle_buffer_size)
  ds = ds.map(lambda w: (w[:-1], w[1:]))
  return ds.batch(batch_size).prefetch(1)

def model_forecast(model, series, window_size):
  ds = tf.data.Dataset.from_tensor_slices(series)
  ds = ds.window(window_size, shift=1, drop_remainder=True)
  ds = ds.flat_map(lambda w: w.batch(window_size))
  ds = ds.batch(32).prefetch(1)
  forecast = model.predict(ds)
  return forecast

def plot_series(time, series, format="-", start=0, end=None, color='red'):
  plt.plot(time[start:end], series[start:end], format, color=color)
  plt.xlabel("Time")
  plt.ylabel("Price")
  plt.grid(True)

kwargs = {"logger": logger}
DATASET = get_dataset(kwargs=kwargs) if args.input_dataset is None else load_dataset(dataset_path=args.input_dataset)

series = np.array(DATASET)
time = np.array(range(len(DATASET)))

scaler = MinMaxScaler(feature_range=(0, 1))
series = scaler.fit_transform(series.reshape(-1, 1))
series = series.reshape(-1, ) # reshape back to (x, )

SPLIT_TIME = int(len(DATASET) * args.split_time)
time_train = time[:SPLIT_TIME]
x_train = series[:SPLIT_TIME]
time_valid = time[SPLIT_TIME:]
x_valid = series[SPLIT_TIME:]

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 1000

train_set = windowed_dataset(x_train, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE, shuffle_buffer_size=SHUFFLE_BUFFER_SIZE)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=64, 
                      kernel_size=5,
                      strides=1,
                      padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.LSTM(128, return_sequences=True),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1)
])

model.compile(#loss=tf.keras.losses.Huber(),
              loss='mae',
              optimizer='adam',
              metrics=["mae"])

history = model.fit(train_set, epochs=EPOCHS)

# checking results on valid set
if args.verbose:
  rnn_forecast = model_forecast(model, series[..., np.newaxis], WINDOW_SIZE)
  rnn_forecast = rnn_forecast[SPLIT_TIME - WINDOW_SIZE:, -1, 0]
  print(f'\nMean absolute error: {tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast[:-1]).numpy()}') # -1 element, the new one, that was predicted
  plt.figure(figsize=(10, 6))
  plt.title('Forecasting on valid set')
  plot_series(time_valid, x_valid, color='orange')
  time_forecast = np.array(range(min(time_valid), max(time_valid)+2)) # create one more element for the x axis (for the new one that was predicted). +2 because arrays starts at zero and we are dealing with max and min of the elements instead of its actual size
  plot_series(time_forecast, rnn_forecast, color='purple')
  plt.show()
  #-----------------------------------------------------------
  # Retrieve a list of list results on training and test data
  # sets for each training epoch
  #-----------------------------------------------------------
  loss = history.history['loss']
  epochs=range(len(loss)) # Get number of epochs
  #------------------------------------------------
  # Plot training and validation loss per epoch
  #------------------------------------------------
  plt.plot(epochs, loss, 'r')
  plt.title('Training loss')
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend(["Loss"])
  plt.show()

def add_last_pred(serie_values, time_steps, forecasts):
  last_pred = forecasts[len(forecasts) -1]  
  serie_values = np.append(serie_values, [last_pred], 0)
  time_steps = np.append(time_steps, [len(serie_values)], 0)
  return serie_values, time_steps

NUM_PREDS_IN_THE_FUTURE = args.forecast_size
new_series = series
new_time = time
for i in range(NUM_PREDS_IN_THE_FUTURE):
  rnn_forecast = model_forecast(model, new_series[..., np.newaxis], WINDOW_SIZE)
  rnn_forecast = rnn_forecast[:, -1, 0]
  new_series, new_time = add_last_pred(new_series, new_time, rnn_forecast)

plt.figure(figsize=(10, 6))
plt.title('Trend Forecasting')
plot_series(new_time, new_series, color='gray')
plot_series(time, series, color='orange')
plt.show()