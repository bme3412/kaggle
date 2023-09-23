from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
from sklearn.preprocessing import StandardScaler
from keras.regularizers import l1, l2, l1_l2
from keras.layers import Dropout

import pandas as pd
import numpy as np

# Constants and data preparation
lr = 0.001
optimizer = Adam(learning_rate=lr)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
data = pd.read_csv('train_clean.csv')
feature_columns = ['imbalance_size', 'imbalance_buy_sell_flag', 'reference_price', 'bid_price', 'ask_price', 'wap']
target_column = 'target'

# Scaling
scaler = StandardScaler()
data[feature_columns] = scaler.fit_transform(data[feature_columns])
if data[target_column].sum() == 0:
    raise ValueError("All targets are zeros. Please check the dataset!")

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

# Splitting and sequence creation
seq_length = 3
train, test = train_test_split(data)
X_train, y_train = create_multivariate_sequences(train, seq_length, feature_columns, target_column)
X_test, y_test = create_multivariate_sequences(test, seq_length, feature_columns, target_column)

# Model definition and training
input_shape = (seq_length, len(feature_columns))
reg = l1_l2(l1=1e-5, l2=1e-4)

model = Sequential()
model.add(GRU(100, return_sequences=True, activation='tanh', input_shape=input_shape, kernel_initializer='glorot_uniform', kernel_regularizer=reg))
model.add(Dropout(0.2))  # Example of adding a dropout layer
model.add(GRU(100, activation='tanh', return_sequences=True, kernel_initializer='glorot_uniform', kernel_regularizer=reg))
model.add(Dropout(0.2))  # Example of adding another dropout layer
model.add(GRU(50, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=reg))
model.add(Dense(1, kernel_initializer='glorot_uniform'))
model.compile(optimizer=optimizer, loss='mean_absolute_error')

model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), callbacks=[reduce_lr, early_stopping])

# Evaluation
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
