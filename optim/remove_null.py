import pandas as pd
import numpy as np

data = pd.read_csv('/Users/brendan/Desktop - Brendan’s MacBook Air/kaggle/optiver-trading-at-the-close/train.csv')

print(len(data))

data_remove_nan = pd.read_csv('/Users/brendan/Desktop - Brendan’s MacBook Air/kaggle/optiver-trading-at-the-close/train.csv').dropna()
data_remove_nan.to_csv('train_clean.csv')