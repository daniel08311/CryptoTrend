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

    def __init__(self, EXCHANGES, SHIFT_X, SHIFT_Y, MODEL_NAME):

        self.EXCHANGES = EXCHANGES

        self.SHIFT_X = SHIFT_X

        self.SHIFT_Y = SHIFT_Y

        self.MODEL_NAME = MODEL_NAME

    

    def request_data(self, exchange):

        res = requests.get("http://jumpin.cc/BTC/WhaleSignalETH/training_data_{}_log.txt".format(exchange))

        f = open("training_data_raw/eth_{}.csv".format(exchange),"w")

        f.write(res.text)

        f.close()

        return(True)

    

    def get(self):

        for exchange in self.EXCHANGES:

            try:

                _thread.start_new_thread(self.request_data, (exchange,))

            except:

                print("error requesting {} data".format(exchange))

        print("requesting data complete")

            

    def parse_data(self, exchange, model_name):

  

        train_df = pd.read_csv("training_data_raw/eth_{}.csv".format(exchange))

        train_df = train_df.dropna()



        y_raw = train_df["ohlcv_close"].values

        x_raw = train_df.drop(['time', 'last_price', 'ohlcv_close'], axis=1).values



        log_count = len(y_raw)

        feats = len(x_raw[0])

        shift = self.SHIFT_X

        shift_y = self.SHIFT_Y



        x_train = np.zeros((log_count-shift-shift_y,shift,feats))

        y_train_trend = np.zeros(log_count-shift-shift_y)

        latest = x_raw[-shift:].reshape((1, x_raw[-shift:].shape[0], x_raw[-shift:].shape[1]))

        latest = latest.reshape(-1, latest.shape[1]*latest.shape[2])



        for i in range(log_count-shift_y-shift):

            x_train[i] = x_raw[i:i+shift]

            diff = y_raw[i+shift+shift_y]-y_raw[i+shift]

            if diff/y_raw[i+shift] > 0.015 :

                y_train_trend[i] = 0

            elif diff/y_raw[i+shift] > 0.0075:

                y_train_trend[i] = 1

            elif diff/y_raw[i+shift] < -0.015:

                y_train_trend[i] = 3

            elif diff/y_raw[i+shift] < -0.0075:

                y_train_trend[i] = 2

            else:

                y_train_trend[i] = 4

        

        self.save_pickle(x_train, 'training_data/x_train_{}_{}.pickle'.format(exchange, model_name))

        self.save_pickle(y_train_trend, 'training_data/y_train_{}_{}.pickle'.format(exchange, model_name))

        self.save_pickle(latest, 'latest_data/latest_{}_{}.pickle'.format(exchange, model_name))

        return(True)

    

    def parse(self):

        for exchange in self.EXCHANGES:

            try:

                _thread.start_new_thread(self.parse_data, (exchange, self.MODEL_NAME,))

            except:

                print("error parsing {} data".format(exchange))

        print("parsing data complete")



                

    def save_pickle(self, object_, path):

        with open(path, 'wb') as handle:

            pickle.dump(object_, handle, protocol=pickle.HIGHEST_PROTOCOL)

    

    def train_model(self, exchange, model_name):

        

        with open('training_data/x_train_{}_{}.pickle'.format(exchange, model_name), 'rb') as handle:

            x_train = pickle.load(handle)



        with open('training_data/y_train_{}_{}.pickle'.format(exchange, model_name), 'rb') as handle:

            y_train_trend = pickle.load(handle)



        print("Currently Training on [{}] Data, Model [{}]".format(exchange, model_name))

        

        for i in range(5):

            print("class {} instances = {}".format(i,sum(y_train_trend==i)))



        idxs = []

        for cla in range(5):

            idxs.append([idx for idx,i in enumerate(y_train_trend) if i == cla])



        idxs[-1] = resample(idxs[-1], n_samples=3000, random_state=0)


        x_train_combine = x_train[idxs[0]]
        y_train_trend_combine = y_train_trend[idxs[0]]
        
        for cla in range(4):
            x_train_combine = np.vstack((x_train_combine, x_train[idxs[cla+1]]))
            y_train_trend_combine = np.hstack((y_train_trend_combine,y_train_trend[idxs[cla+1]]))


        x_train_xgb = x_train_combine.reshape(-1, x_train_combine.shape[1]*x_train_combine.shape[2])

        x_train, x_test, y_train_trend, y_test_trend = train_test_split(x_train_xgb, y_train_trend_combine, test_size=0.3, random_state=42)

        rf = RandomForestClassifier(n_estimators=50, max_depth=15, max_features=int(math.sqrt(x_train.shape[1])/4.2), random_state=0, n_jobs=14)

        rf.fit(x_train, y_train_trend)

        print("\nConfusuin Matrix :")

        print(confusion_matrix(rf.predict(x_test), y_test_trend))

        print("\nF1 Score : {}".format(f1_score(rf.predict(x_test), y_test_trend, average='macro')))

        
        self.save_pickle(rf, 'model/rf_{}_{}.pickle'.format(exchange, model_name))

        

    def train(self):

         for exchange in self.EXCHANGES:

            try:

                self.train_model(exchange, self.MODEL_NAME)

            except Exception as e:

                print(e)

                print("error while Training {} data".format(exchange))

    

    def predict_trend(self, exchange, model_name):

        with open('model/rf_{}_{}.pickle'.format(exchange, model_name), 'rb') as handle:

            rf = pickle.load(handle)

        with open('latest_data/latest_{}_{}.pickle'.format(exchange, model_name), 'rb') as handle:

            latest = pickle.load(handle)

        predict = rf.predict_proba(latest)[0]

        print("Prediction of [{}] Market Based On [{}] Model".format(exchange, model_name))

        print("Bull : {}, Weak Bull : {}, Weak Bear : {}, Bear : {}, Neutral : {}".format(predict[0], predict[1], predict[2], predict[3], predict[4]))

        print("\n===================================================\n")

        return (predict)
        

    def save_predict(self, predictions):

        data = []
        for exchange, predicts in zip(self.EXCHANGES, predictions):
            dic = {}
            dic['name'] = exchange
            for prob, cla in zip(predicts,['bull','wbull','bear','wbear','n']):
                dic[cla]="%.2f" % prob
            data.append(dic)    
        json_data = json.dumps(data)
        with open('predict_{}.json'.format(self.MODEL_NAME), 'w') as outfile:
            json.dump(data, outfile)

    def predict(self):

        predictions = []

        for exchange in self.EXCHANGES:

            predictions.append(self.predict_trend(exchange, self.MODEL_NAME))

        self.save_predict(predictions)
