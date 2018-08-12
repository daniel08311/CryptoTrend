import trend
import time
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("-exchange", help="Choose a exchange to train with", type=str, nargs='?', default="binance")
parser.add_argument("-name", help="Name your trained model", type=str, nargs='?', default="1-hour")
parser.add_argument("-thread", help="How many threads do you want use while training?", type=int, nargs='?', default=4)
parser.add_argument("-shiftx", help="Give the numbers of consecutive trading logs to train", type=int, nargs='?', default=200)
parser.add_argument("-shifty", help="How far do you want to predict? For example, 20 means predciting trend after 20 trades from now", type=int, nargs='?', default=100)
parser.add_argument("-ls", help="list all the saved model names", const=1, nargs='?')
# parser.add_argument("ls_model", help="List all models", type=bool, nargs='?', default=False)

args = parser.parse_args()

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
            lists[args.name + "_" + args.name] = True
            json.dump(lists, data_file)

    test = trend.CryptoTrend(args.exchange, args.shiftx, args.shifty, args.name, args.thread)
    test.get()
    test.parse()
    test.train()
    test.predict()

