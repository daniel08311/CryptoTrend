#!/bin/sh  

while true  

    do  

    python3 run.py -exchange binance -coin ETH -name 1h -shiftx 200 -shifty 100 -thread 4 -test
    python3 run.py -exchange binance -coin ETH -name 30m -shiftx 100 -shifty 50 -thread 4 -test
    python3 run.py -exchange bitfinex -coin ETH -name 1h -shiftx 200 -shifty 100 -thread 4 -test
    python3 run.py -exchange bitfinex -coin ETH -name 30m -shiftx 100 -shifty 50 -thread 4 -test
    python3 run.py -exchange bittrex -coin ETH -name 1h -shiftx 200 -shifty 100 -thread 4 -test
    python3 run.py -exchange bittrex -coin ETH -name 30m -shiftx 100 -shifty 50 -thread 4 -test
    python3 run.py -exchange huobipro -coin ETH -name 1h -shiftx 200 -shifty 100 -thread 4 -test
    python3 run.py -exchange huobipro -coin ETH -name 30m -shiftx 100 -shifty 50 -thread 4 -test

    secs=$((1 * 10))
    while [ $secs -gt 0 ]; do
        echo -ne "Cooldown for next round of ETH training in $secs\033[0K\r"
        sleep 1
        : $((secs--))
    done

done
