import numpy as np
import tensorflow as tf
import os
import argparse
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

print(tf.__version__)

tf.random.set_seed(51)
np.random.seed(51)

parser = argparse.ArgumentParser()
parser.add_argument("--d", dest="dataset", nargs='?')
parser.add_argument("--ws", dest="window_size", nargs='?', type=int, default=30)
parser.add_argument("--ts", dest="trend_size", nargs='?', type=int, default=7)
args = parser.parse_args()

DATASET = args.dataset if args.dataset is not None else 'btc_price.csv'
WINDOW_SIZE = args.window_size
TREND_SIZE = args.trend_size

print(f'Window Size: {WINDOW_SIZE}')


def load_dataset(dataset=DATASET):
  closing_prices = []
  with open(dataset) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
      closing_prices.append(float(row[4]))

    #closing_prices = np.flip(closing_prices) # flipping if dataset is in desc ordering
  return closing_prices

def plot_series(time, series, format="-", start=0, end=None, color='red'):
  plt.plot(time[start:end], series[start:end], format, color=color)
  plt.xlabel("Time")
  plt.ylabel("Price")
  plt.grid(True)

closing_prices = load_dataset(DATASET)

print(f'num of samples: {len(closing_prices)}')

series = np.array(closing_prices)
time = np.array(range(len(closing_prices)))

scaler = MinMaxScaler(feature_range=(0, 1))
series = scaler.fit_transform(series.reshape(-1, 1))
series = series.reshape(-1, ) # reshape back to (x, )

print(f'series shape and type: {series.shape}, {type(series)}')
print(f'time shape: {time.shape}')

#plt.figure(figsize=(10, 6))
#plot_series(time, series, color='gray')
#plt.show()

def size(dataset):
  for num, _ in enumerate(dataset):
    pass
  return num

def model_forecast(model, series, window_size):
  ds = tf.data.Dataset.from_tensor_slices(series)
  print(f'ds size: {size(ds)}')
  ds = ds.window(window_size, shift=1, drop_remainder=True)
  print(f'ds window size: {size(ds)}')
  ds = ds.flat_map(lambda w: w.batch(window_size))
  print(f'ds flat map size: {size(ds)}')
  ds = ds.batch(32).prefetch(1)
  print(f'ds batch size: {size(ds)}')
  forecast = model.predict(ds)
  return forecast


model = tf.keras.models.load_model('model')

rnn_forecast = model_forecast(model, series[..., np.newaxis], WINDOW_SIZE)
print(f'rnn_forecast after aplying model_forecast: {rnn_forecast.shape}')
rnn_forecast = rnn_forecast[:, -1, 0]

print(f'series shape: {series.shape}')
print(f'time shape: {time.shape}')
print(f'rnn_forecast shape: {rnn_forecast.shape}')

#print(f'Mean absolute error: {tf.keras.metrics.mean_absolute_error(series, rnn_forecast[:-1]).numpy()}\n\n')

def plot_forecast_against_groundtruth(series, time, forecast):
  print(f'\n\nPloting forecast against groundtruth!')
  print(f'series shape: {series.shape}')
  print(f'time shape: {time.shape}')
  print(f'forecast shape: {forecast.shape}')

  plt.figure(figsize=(10, 6))
  plt.title('Trend Prediction')
  plot_series(time, series, color='orange')

  diff = (len(time) - len(forecast))

  print(f'\nforecast time len: {len(time[diff:])} --- forecast length: {len(forecast)}')

  #time = np.array(range(len(forecast)))
  plot_series(time[diff:], forecast, color='purple')
  plt.show()

plot_forecast_against_groundtruth(series, time, rnn_forecast)

#
def add_last_pred(series, time, forecasts):
  last_pred = forecasts[len(forecasts) -1]  
  series = np.append(series, [last_pred], 0)
  new_time_step = len(series)
  #print(f'last time: {time[new_time_step -1]}')
  #print(f'length time and the last time step to add: {new_time_step}')
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