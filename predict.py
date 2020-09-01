import numpy as np
import tensorflow as tf
import os
import argparse
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from log import init_logger

print(tf.__version__)

tf.random.set_seed(51)
np.random.seed(51)

parser = argparse.ArgumentParser()
parser.add_argument("--d", dest="dataset", nargs='?')
parser.add_argument("--in", dest="input_model", nargs='?', default='model')
parser.add_argument("--ws", dest="window_size", nargs='?', type=int, default=30)
parser.add_argument("--ts", dest="trend_size", nargs='?', type=int, default=7)
parser.add_argument("--v", dest="verbose", action='store_true')
args = parser.parse_args()

logger = init_logger(__name__, show_debug=args.verbose, log_to_file=False)

DATASET = args.dataset if args.dataset is not None else 'btc_price.csv'
WINDOW_SIZE = args.window_size
TREND_SIZE = args.trend_size

logger.debug(msg=f'Window Size: {WINDOW_SIZE}')


def load_dataset(dataset=DATASET):
  closing_prices = []
  with open(dataset) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
      closing_prices.append(float(row[4]))

  return closing_prices

def plot_series(time, series, format="-", start=0, end=None, color='red'):
  plt.plot(time[start:end], series[start:end], format, color=color)
  plt.xlabel("Time")
  plt.ylabel("Price")
  plt.grid(True)

closing_prices = load_dataset(DATASET)

logger.debug(msg=f'num of samples: {len(closing_prices)}')

series = np.array(closing_prices)
time = np.array(range(len(closing_prices)))

scaler = MinMaxScaler(feature_range=(0, 1))
series = scaler.fit_transform(series.reshape(-1, 1))
series = series.reshape(-1, ) # reshape back to (x, )

logger.debug(msg=f'series shape and type: {series.shape}, {type(series)}')
logger.debug(msg=f'time shape: {time.shape}')

if args.verbose:
  plt.figure(figsize=(10, 6))
  plt.title('Dataset')
  plot_series(time, series, color='gray')
  plt.show()

def size(dataset):
  for num, _ in enumerate(dataset):
    pass
  return num

def model_forecast(model, series, window_size):
  ds = tf.data.Dataset.from_tensor_slices(series)
  ds = ds.window(window_size, shift=1, drop_remainder=True)
  ds = ds.flat_map(lambda w: w.batch(window_size))
  ds = ds.batch(32).prefetch(1)
  logger.debug(msg=f'ds batch size: {size(ds)}')
  forecast = model.predict(ds)
  return forecast

logger.info(msg=f'Loading model from {args.input_model} directory...')
model = tf.keras.models.load_model(args.input_model)

rnn_forecast = model_forecast(model, series[..., np.newaxis], WINDOW_SIZE)
logger.debug(msg=f'rnn_forecast after aplying model_forecast: {rnn_forecast.shape}')
rnn_forecast = rnn_forecast[:, -1, 0]

logger.debug(msg=f'series shape: {series.shape}')
logger.debug(msg=f'time shape: {time.shape}')
logger.debug(msg=f'rnn_forecast shape: {rnn_forecast.shape}')

def plot_forecast_against_groundtruth(series, time, forecast):
  logger.debug(msg=f'\n\nPloting forecast against groundtruth!')
  logger.debug(msg=f'series shape: {series.shape}')
  logger.debug(msg=f'time shape: {time.shape}')
  logger.debug(msg=f'forecast shape: {forecast.shape}')

  plt.figure(figsize=(10, 6))
  plt.title('Trend Prediction')
  plot_series(time, series, color='orange')
  diff = (len(time) - len(forecast))
  logger.debug(msg=f'\nforecast time len: {len(time[diff:])} --- forecast length: {len(forecast)}')
  plot_series(time[diff:], forecast, color='purple')
  plt.show()

if args.verbose:
  plot_forecast_against_groundtruth(series, time, rnn_forecast)

def add_last_pred(series, time, forecasts):
  last_pred = forecasts[len(forecasts) -1]  
  series = np.append(series, [last_pred], 0)
  new_time_step = len(series)
  time = np.append(time, [new_time_step], 0)
  return series, time

# predict trend
NUM_PREDS_IN_THE_FUTURE = TREND_SIZE
for i in range(NUM_PREDS_IN_THE_FUTURE):
  rnn_forecast = model_forecast(model, series[..., np.newaxis], WINDOW_SIZE)
  rnn_forecast = rnn_forecast[:, -1, 0]
  series, time = add_last_pred(series, time, rnn_forecast)
  #plot_forecast_against_groundtruth(series, time, rnn_forecast)

plot_forecast_against_groundtruth(series, time, rnn_forecast)