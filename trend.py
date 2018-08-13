import requests
import pandas as pd
import numpy as np
import pickle
import _thread
import xgboost as xgb
import math
import json
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

class CryptoTrend():

    def __init__(self, EXCHANGE, SHIFT_X, SHIFT_Y, MODEL_NAME, THREAD, COIN):

        self.EXCHANGE = EXCHANGE
        self.SHIFT_X = SHIFT_X
        self.SHIFT_Y = SHIFT_Y
        self.MODEL_NAME = MODEL_NAME
        self.THREAD = THREAD
        self.COIN = COIN
        if COIN == 'BTC':
            self.url = "http://jumpin.cc/BTC/WhaleSignal/"
        elif COIN == 'ETH':
            self.url = "http://jumpin.cc/BTC/WhaleSignalETH/"

    def request_data(self, url, path):

        res = requests.get(url)
        f = open(path,"w")
        f.write(res.text)
        f.close()
        return(True)

    def get_train(self):
        url = self.url + "training_data_{}_log.txt".format(self.EXCHANGE)
        path = "training_data_raw/{}/{}.csv".format(self.COIN, self.EXCHANGE)
        try:
            self.request_data(url, path)

        except:
            print("error requesting {} data".format(self.EXCHANGE))

        print("requesting {} data complete".format(self.EXCHANGE))

    def get_test(self):
        url = self.url + "training_data_{}_log_500.txt".format(self.EXCHANGE)
        path = "latest_data_raw/{}/{}.csv".format(self.COIN, self.EXCHANGE)
        try:
            self.request_data(url, path)

        except:
            print("error requesting {} data".format(self.EXCHANGE))

        print("requesting {} data complete".format(self.EXCHANGE))


    def parse_train_data(self, exchange, coin, model_name):
  
        train_df = pd.read_csv("training_data_raw/{}/{}.csv".format(coin, exchange))

        train_df = train_df.dropna()

        y_raw = train_df["ohlcv_close"].values
        x_raw = train_df.drop(['time', 'last_price', 'ohlcv_close'], axis=1).values

        log_count = len(y_raw)
        feats = len(x_raw[0])
        shift = self.SHIFT_X
        shift_y = self.SHIFT_Y

        x_train = np.zeros((log_count-shift-shift_y,shift,feats))
        y_train_trend = np.zeros(log_count-shift-shift_y)

        for i in range(log_count-shift_y-shift):

            x_train[i] = x_raw[i:i+shift]
            diff = y_raw[i+shift+shift_y]-y_raw[i+shift]

            if diff/y_raw[i+shift] > 0.02 :

                y_train_trend[i] = 0

            elif diff/y_raw[i+shift] > 0.01:

                y_train_trend[i] = 1

            elif diff/y_raw[i+shift] < -0.02:

                y_train_trend[i] = 3

            elif diff/y_raw[i+shift] < -0.01:

                y_train_trend[i] = 2

            else:

                y_train_trend[i] = 4
   
        self.save_pickle(x_train, 'training_data/{}/x_train_{}_{}.pickle'.format(coin, exchange, model_name))
        self.save_pickle(y_train_trend, 'training_data/{}/y_train_{}_{}.pickle'.format(coin, exchange, model_name))       

        return(True)
    
    
    def parse_test_data(self, exchange, coin, model_name):
  
        train_df = pd.read_csv("latest_data_raw/{}/{}.csv".format(coin, exchange))
        train_df = train_df.dropna()

        x_raw = train_df.drop(['time', 'last_price', 'ohlcv_close'], axis=1).values

        shift = self.SHIFT_X

        latest = x_raw[-shift:].reshape((1, x_raw[-shift:].shape[0], x_raw[-shift:].shape[1]))
        latest = latest.reshape(-1, latest.shape[1]*latest.shape[2])
    
        self.save_pickle(latest, 'latest_data/{}/latest_{}_{}.pickle'.format(coin, exchange, model_name))
        return(True)

    def parse_train(self):

        try:
            self.parse_train_data(self.EXCHANGE, self.COIN, self.MODEL_NAME)

        except:
            print("error parsing {} training data".format(self.EXCHANGE))

        print("parsing {} training data complete".format(self.EXCHANGE))

    
    def parse_test(self):

        try:
            self.parse_test_data(self.EXCHANGE, self.COIN, self.MODEL_NAME)

        except:
            print("error parsing latest {} data".format(self.EXCHANGE))

        print("parsing latest {} data complete".format(self.EXCHANGE))

                
    def save_pickle(self, object_, path):

        with open(path, 'wb') as handle:
            pickle.dump(object_, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    def train_model(self, exchange, coin, model_name):

        with open('training_data/{}/x_train_{}_{}.pickle'.format(coin, exchange, model_name), 'rb') as handle:
            x_train = pickle.load(handle)

        with open('training_data/{}/y_train_{}_{}.pickle'.format(coin, exchange, model_name), 'rb') as handle:
            y_train_trend = pickle.load(handle)

        for i in range(5):
            print("class {} instances = {}".format(i,sum(y_train_trend==i)))

        idxs = []
        for cla in range(5):
            idxs.append([idx for idx,i in enumerate(y_train_trend) if i == cla])

        idxs[-1] = resample(idxs[-1], n_samples=len(idxs[0])+len(idxs[1])+len(idxs[2])+len(idxs[3]), random_state=0)

        x_train_combine = x_train[idxs[0]]
        y_train_trend_combine = y_train_trend[idxs[0]]
        
        for cla in range(4):
            x_train_combine = np.vstack((x_train_combine, x_train[idxs[cla+1]]))
            y_train_trend_combine = np.hstack((y_train_trend_combine,y_train_trend[idxs[cla+1]]))


        x_train_xgb = x_train_combine.reshape(-1, x_train_combine.shape[1]*x_train_combine.shape[2])
        x_train, x_test, y_train_trend, y_test_trend = train_test_split(x_train_xgb, y_train_trend_combine, test_size=0.3, random_state=42)

        rf = RandomForestClassifier(n_estimators=50, max_depth=15, max_features=int(math.sqrt(x_train.shape[1])/4.2), random_state=0, n_jobs=self.THREAD)
        rf.fit(x_train, y_train_trend)

        print("\nConfusuin Matrix :")
        print(confusion_matrix(rf.predict(x_test), y_test_trend))
        print("\nF1 Score : {}".format(f1_score(rf.predict(x_test), y_test_trend, average='macro')))
        print("\n=============================================\n")
        self.save_pickle(rf, 'model/{}/rf_{}_{}.pickle'.format(coin, exchange, model_name))

        
    def train(self):

        try:
            self.train_model(self.EXCHANGE , self.COIN, self.MODEL_NAME)

        except Exception as e:
            print(e)
            print("error while Training {} {} data".format(self.EXCHANGE, self.COIN))

    
    def predict_trend(self, exchange, coin, model_name):

        with open('model/{}/rf_{}_{}.pickle'.format(coin, exchange, model_name), 'rb') as handle:
            rf = pickle.load(handle)

        with open('latest_data/{}/latest_{}_{}.pickle'.format(coin, exchange, model_name), 'rb') as handle:
            latest = pickle.load(handle)

        predict = rf.predict_proba(latest)[0]

        print("\n##############################################################################\n")
        print("Prediction of [{}] [{}/USDT] Market Based On [{}] Model".format(exchange, coin, model_name))
        print("Bull : %.2f, Weak Bull : %.2f, Weak Bear : %.2f, Bear : %.2f, Neutral : %.2f" % (predict[0], predict[1], predict[2], predict[3], predict[4]))
        print("\n##############################################################################\n")

        return (predict)
        

    def save_predict(self, prediction):

        dic = {}
        dic['name'] = self.EXCHANGE
        for prob, cla in zip(prediction,['bull','wbull','bear','wbear','n']):
            dic[cla]="%.2f" % prob
        with open('predict/{}/predict_{}_{}.json'.format(self.COIN, self.EXCHANGE,self.MODEL_NAME), 'w') as outfile:
            json.dump(dic, outfile)

    def predict(self):

        predict = self.predict_trend(self.EXCHANGE, self.COIN, self.MODEL_NAME)
        self.save_predict(predict)