
# coding: utf-8

# In[227]:

import numpy as np
import nltk
import pickle
import pandas as pd
import sys
import keras

import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev0\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from keras import backend as K
from keras.models import model_from_json

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[228]:

train_df = pd.read_csv("btc_train.csv")


# In[229]:

y_raw = train_df["last_price"]
x_raw = train_df.values[:,1:]

x_raw = scaler.fit_transform(x_raw)

log_count = len(y_raw)
shift = 150
shift_y = 150
x_train = np.zeros((log_count-shift-shift_y,shift,15))
y_train = np.zeros(log_count-shift-shift_y)


# In[230]:

for i in range(log_count-shift_y-shift):
    x_train[i] = x_raw[i:i+shift]
    y_train[i] = y_raw[i+shift+shift_y]


# In[223]:

model = Sequential()
model.add(LSTM(32,input_shape=(shift,15), activation='relu', return_sequences=False))
model.add(Dense(1, activation='linear'))
model.compile(loss='mae', optimizer='adam', metrics=['mean_absolute_percentage_error'])
print(model.summary())


# In[231]:

x_train_xgb = x_train.reshape(-1, x_train.shape[1]*x_train.shape[2])


# In[232]:

split = int(x_train_xgb.shape[0]*0.8)
x_train_xgb.shape


# In[234]:

dtrain = xgb.DMatrix(x_train_xgb[:split], label=y_train[:split])
dtest = xgb.DMatrix(x_train_xgb[split:], label=y_train[split:])

params = {"objective": "reg:linear", "booster":"gbtree", 'max_depth':'4', 'eta':'0.02', 'subsample':'0.7', 'eval_metric':'mae' , 'verbose':1}
params['nthread'] = 8   
evallist  = [(dtest,'eval')]
num_round = 1000
gbm_1 = xgb.train(params, dtrain, num_round, evallist)  


# In[127]:

gbm_1.predict(dtest)


# In[135]:

for a,b in zip (gbm_1.predict(dtest), y_train[1200:]):
    print(a)
    print(b)
    print("====")

