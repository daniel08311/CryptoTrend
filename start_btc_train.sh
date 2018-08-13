#!/bin/sh  

while true  

    do  

    python3 run.py -exchange binance -coin BTC -name 1h -shiftx 200 -shifty 100 -thread 4 -train
    python3 run.py -exchange binance -coin BTC -name 30m -shiftx 100 -shifty 50 -thread 4 -train
    python3 run.py -exchange bitfinex -coin BTC -name 1h -shiftx 200 -shifty 100 -thread 4 -train
    python3 run.py -exchange bitfinex -coin BTC -name 30m -shiftx 100 -shifty 50 -thread 4 -train
    python3 run.py -exchange bittrex -coin BTC -name 1h -shiftx 200 -shifty 100 -thread 4 -train
    python3 run.py -exchange bittrex -coin BTC -name 30m -shiftx 100 -shifty 50 -thread 4 -train
    python3 run.py -exchange huobipro -coin BTC -name 1h -shiftx 200 -shifty 100 -thread 4 -train
    python3 run.py -exchange huobipro -coin BTC -name 30m -shiftx 100 -shifty 50 -thread 4 -train

    secs=$((10 * 60))
    while [ $secs -gt 0 ]; do
        echo -ne "Cooldown for next round of BTC training in $secs\033[0K\r"
        sleep 1
        : $((secs--))
    done

done
