from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

lr = 0.001
optimizer = Adam(learning_rate=lr)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.0001)
data = pd.read_csv('train_clean.csv')

feature_columns = ['imbalance_size', 'imbalance_buy_sell_flag','reference_price', 'bid_price', 'ask_price', 'wap']

target_column = 'target'

# Scaling
scaler = StandardScaler()
data[feature_columns] = scaler.fit_transform(data[feature_columns])

# Check for Zero Targets
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


# Example usage:
feature_columns = ['imbalance_size', 'imbalance_buy_sell_flag','reference_price', 'bid_price', 'ask_price', 'wap']
target_column = 'target'
seq_length = 3

train_size = int(len(data) * 0.8)
train = data.iloc[:train_size]
test = data.iloc[train_size:]



X_train, y_train = create_multivariate_sequences(train, seq_length, feature_columns, target_column)
X_test, y_test = create_multivariate_sequences(test, seq_length, feature_columns, target_column)


input_shape = (seq_length, len(feature_columns))

model = Sequential()
model.add(GRU(100, return_sequences=True, activation='tanh', input_shape=input_shape,  kernel_initializer='glorot_uniform'))
model.add(GRU(100, activation='tanh', return_sequences=True, kernel_initializer='glorot_uniform'))
model.add(GRU(50, activation='tanh', kernel_initializer='glorot_uniform'))
model.add(Dense(1, kernel_initializer='glorot_uniform'))
model.compile(optimizer=optimizer, loss='mean_squared_error')

model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), callbacks=[reduce_lr])


# Evaluation
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# predicitons
predictions = model.predict(X_test)

# visualzie
import matplotlib.pyplot as plt

plt.plot(y_test, label="True Values")
plt.plot(predictions, label="Predictions")
plt.legend()
plt.title("Model Predictions vs True Values")
plt.show()

# model saving
model.save('optimer_model1.h5')