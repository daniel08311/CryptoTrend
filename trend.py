import requests

import pandas as pd

import numpy as np

import pickle

import _thread



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

            if diff/y_raw[i+shift] > 0.01 :

                y_train_trend[i] = 0



            elif diff/y_raw[i+shift] < -0.01:

                y_train_trend[i] = 1



            else:

                y_train_trend[i] = 2

        

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

        

        for i in range(3):

            print("class {} instances = {}".format(i,sum(y_train_trend==i)))



        idxs = []

        for cla in range(3):

            idxs.append([idx for idx,i in enumerate(y_train_trend) if i == cla])



        idxs[2] = resample(idxs[2], n_samples=len(idxs[0])+len(idxs[1]), random_state=0)



        x_train = np.vstack((x_train[idxs[0]],x_train[idxs[1]],x_train[idxs[2]]))

        y_train_trend = np.hstack((y_train_trend[idxs[0]],y_train_trend[idxs[1]],y_train_trend[idxs[2]]))



        x_train_xgb = x_train.reshape(-1, x_train.shape[1]*x_train.shape[2])



        x_train, x_test, y_train_trend, y_test_trend = train_test_split(x_train_xgb, y_train_trend, test_size=0.3, random_state=42)



        rf = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=0, n_jobs=14)

        rf.fit(x_train, y_train_trend)



        print("\nConfusuin Matrix :")

        print(confusion_matrix(rf.predict(x_test), y_test_trend))

        print("\nF1 Score : {}".format(f1_score(rf.predict(x_test), y_test_trend, average='macro')))

        print("\n===========================================\n") 

        

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

        print("BULL : {}, BEAR : {}, NEUTRAL : {}".format(predict[0], predict[1], predict[2]))

        print("\n===================================================\n")

        

    def predict(self):

        for exchange in self.EXCHANGES:

            self.predict_trend(exchange, self.MODEL_NAME)





