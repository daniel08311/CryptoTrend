#!/bin/sh  

while true  

do  

python3 run.py -exchange binance -name 30m -shiftx 100 -shifty 50 -thread 2

sleep 30  

done
