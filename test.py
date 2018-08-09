import trend


import time

# import schedule

test = trend.CryptoTrend(['binance', 'bitfinex', 'bittrex', 'huobipro'],50,50,'30-minute')

test.get()

time.sleep(30)

test.parse()

time.sleep(30)

test.train()

test.predict()

