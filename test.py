import trend
import time

# import schedule
EXCHANGES = ['binance', 'bitfinex', 'bittrex', 'huobipro']
a = trend.CryptoTrend(EXCHANGES,400,300,'3-Hour')
# a.get()
# time.sleep(30)
# a.parse()
# time.sleep(20)
# a.train()
a.predict()

