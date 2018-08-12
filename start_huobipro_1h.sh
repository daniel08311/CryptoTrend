#!/bin/sh  

while true  

do  

python3 run.py -exchange huobipro -name 1h -shiftx 200 -shifty 100 -thread 4

sleep 30  

done