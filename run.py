import trend
import time
import argparse
import json
import os
import telegram

parser = argparse.ArgumentParser()
parser.add_argument("-exchange", help="Choose a exchange to train with", type=str, nargs='?', default="binance")
parser.add_argument("-coin", help="Choose BTC or ETH", type=str, nargs='?', default="ETH")
parser.add_argument("-name", help="Name your trained model", type=str, nargs='?', default="1-hour")
parser.add_argument("-thread", help="How many threads do you want use while training?", type=int, nargs='?', default=4)
parser.add_argument("-shiftx", help="Give the numbers of consecutive trading logs to train", type=int, nargs='?', default=200)
parser.add_argument("-shifty", help="How far do you want to predict? For example, 20 means predicting trend after 20 trades from now", type=int, nargs='?', default=100)
parser.add_argument("-train", help="Train model", const=1, nargs='?')
parser.add_argument("-test", help="Predict latest trend(error if no model present)", const=1, nargs='?')
parser.add_argument("-ls", help="list all the saved model names", const=1, nargs='?')
# parser.add_argument("ls_model", help="List all models", type=bool, nargs='?', default=False)

args = parser.parse_args()

if not os.path.exists("training_data"):
    os.mkdir("training_data")

if not os.path.exists("training_data/{}".format(args.coin)):
    os.mkdir("training_data/{}".format(args.coin))

if not os.path.exists("training_data_raw"):
    os.mkdir("training_data_raw")

if not os.path.exists("training_data_raw/{}".format(args.coin)):
    os.mkdir("training_data_raw/{}".format(args.coin))

if not os.path.exists("model"):
    os.mkdir("model")

if not os.path.exists("model/{}".format(args.coin)):
    os.mkdir("model/{}".format(args.coin))

if not os.path.exists("latest_data"):
    os.mkdir("latest_data")

if not os.path.exists("latest_data/{}".format(args.coin)):
    os.mkdir("latest_data/{}".format(args.coin))

if not os.path.exists("latest_data_raw"):
    os.mkdir("latest_data_raw")

if not os.path.exists("latest_data_raw/{}".format(args.coin)):
    os.mkdir("latest_data_raw/{}".format(args.coin))

if not os.path.exists("predict"):
    os.mkdir("predict")

if not os.path.exists("predict/{}".format(args.coin)):
    os.mkdir("predict/{}".format(args.coin))

if args.ls:
    try:
        with open('model_list.json','r') as data_file:    
            lists = json.load(data_file)
            for k,v in lists.items():
                print(k)
    except:
        print("No Models Yet !!")

else:
    try:
        with open('model_list.json','r') as data_file:    
            lists = json.load(data_file)
            if args.exchange + "_" + args.name not in lists:
                lists[args.exchange + "_" + args.name] = True

        with open('model_list.json','w') as data_file:    
            json.dump(lists, data_file)

    except:
        lists = {}
        with open('model_list.json','w') as data_file:    
            lists[args.exchange + "_" + args.name] = True
            json.dump(lists, data_file)
    
    trend = trend.CryptoTrend(args.exchange, args.shiftx, args.shifty, args.name, args.thread, args.coin)
    
    if(args.train):
        print("\n#####################################")
        print("#  Training on [{}] data ".format(args.exchange))
        print("#  Coin       :  {}   ".format(args.coin))
        print("#  Shiftx     :  {}   ".format(args.shiftx))
        print("#  Shifty     :  {}   ".format(args.shifty))
        print("#  Model Name :  {}   ".format(args.name))
        print("#  Threads    :  {}   ".format(args.thread))
        print("#####################################\n")
        trend.get_train()
        trend.parse_train()
        trend.train()
    
    if(args.test):
        
        trend.get_test()
        trend.parse_test()
        trend.predict()

