import numpy as np
import tensorflow as tf
import os
import argparse
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import MinMaxScaler

from log import init_logger

print(tf.__version__)

tf.random.set_seed(51)
np.random.seed(51)

parser = argparse.ArgumentParser()
parser.add_argument("--d", dest="dataset", nargs='?')
parser.add_argument("--ws", dest="window_size", nargs='?', type=int, default=30)
parser.add_argument("--e", dest="epochs", nargs='?', type=int, default=50)
parser.add_argument("--st", dest="split_time", nargs='?', type=float, default=0.8)  # default 80% for training
parser.add_argument("--out", dest="model_output", nargs='?', default='model')
parser.add_argument("--v", dest="verbose", action='store_true')
args = parser.parse_args()

logger = init_logger(__name__, show_debug=args.verbose, log_to_file=False)

DATASET = args.dataset if args.dataset is not None else 'btc_price_dataset.csv'
WINDOW_SIZE = args.window_size
EPOCHS = args.epochs

logger.debug(msg=f'Window Size: {WINDOW_SIZE}')
logger.debug(msg=f'Epochs: {EPOCHS}')


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

logger.debug(msg=f'series shape and type: {series.shape}, {type(series)}')

scaler = MinMaxScaler(feature_range=(0, 1))
series = scaler.fit_transform(series.reshape(-1, 1))
series = series.reshape(-1, )  # reshape back to (x, )

logger.debug(msg=f'series shape and type after scaling: {series.shape}, {type(series)}')
logger.debug(msg=f'time axis shape: {time.shape}')

if args.verbose:
    plt.figure(figsize=(10, 6))
    plt.title('Dataset')
    plot_series(time, series)
    plt.show()


def windowed_dataset(series, window_size, batch_size, shuffle_buffer_size):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    return ds.batch(batch_size).prefetch(1)


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


# Training
SPLIT_TIME = int(len(closing_prices) * args.split_time)
time_train = time[:SPLIT_TIME]
x_train = series[:SPLIT_TIME]
time_valid = time[SPLIT_TIME:]
x_valid = series[SPLIT_TIME:]

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 1000

train_set = windowed_dataset(x_train, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE,
                             shuffle_buffer_size=SHUFFLE_BUFFER_SIZE)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=128,
                           kernel_size=5,
                           strides=1,
                           padding="causal",
                           activation="relu",
                           input_shape=[None, 1]),
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(  # loss=tf.keras.losses.Huber(),
    loss='mae',
    optimizer='adam',
    metrics=["mae"])

history = model.fit(train_set, epochs=EPOCHS)

model.save(args.model_output)

# checking results on valid set
rnn_forecast = model_forecast(model, series[..., np.newaxis], WINDOW_SIZE)
rnn_forecast = rnn_forecast[SPLIT_TIME - WINDOW_SIZE:, -1, 0]

logger.debug(
    msg=f'\nMean absolute error: {tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast[:-1]).numpy()}')  # -1 element, the new one, that was predicted

logger.debug(msg=f'\nx_valid shape: {x_valid.shape}')
logger.debug(msg=f'time_valid shape: {time_valid.shape}')

if args.verbose:
    plt.figure(figsize=(10, 6))
    plt.title('Forecasting on valid set')
    plot_series(time_valid, x_valid, color='orange')

    time_forecast = np.array(range(min(time_valid), max(
        time_valid) + 2))  # create one more element for the x axis (for the new one that was predicted). +2 because arrays starts at zero and we are dealing with max and min of the elements instead of its actual size
    logger.debug(msg=f'\nrnn_forecast shape: {rnn_forecast.shape}')
    logger.debug(msg=f'time_forecast shape: {time_forecast.shape}')
    plot_series(time_forecast, rnn_forecast, color='purple')
    plt.show()

# -----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
# -----------------------------------------------------------
loss = history.history['loss']
epochs = range(len(loss))  # Get number of epochs
# ------------------------------------------------
# Plot training and validation loss per epoch
# ------------------------------------------------
plt.plot(epochs, loss, 'r')
plt.title('Training loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss"])
plt.show()

logger.info(msg=f'Model is saved at {args.model_output} directory')
