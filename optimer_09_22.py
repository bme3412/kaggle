from keras.models import Sequential
from keras.layers import GRU, Dense

import pandas as pd
import numpy as np

data = pd.read_csv('/Users/brendan/Desktop - Brendan’s MacBook Air/kaggle/optiver-trading-at-the-close/train.csv')

def train_test_split(data, train_ratio=0.8):
    train_size = int(len(data)* train_ratio)
    train, test = data[0:train_size],data[train_size:]
    return train, test

def create_multivariate_sequences(data, seq_length, feature_columns, target_column):
    xs = []
    ys = []

    for i in range(len(data) - seq_length):
        x = data[i:(i+seq_length)][feature_columns].values
        y = data.iloc[i+seq_length][target_column]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


# Example usage:
feature_columns = ['imbalance_size', 'reference_price', 'bid_price', 'ask_price', 'wap']
target_column = 'target'
seq_length = 3

train_size = int(len(data) * 0.8)
train = data.iloc[:train_size]
test = data.iloc[train_size:]



X_train, y_train = create_multivariate_sequences(train, seq_length, feature_columns, target_column)
X_test, y_test = create_multivariate_sequences(test, seq_length, feature_columns, target_column)


input_shape = (3,5)

model = Sequential()
model.add(GRU(50, return_sequences=True, input_shape=(input_shape)))
model.add(GRU(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))
