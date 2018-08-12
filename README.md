# CryptoTrend
### Argument Discription ###
  * -h, --help      
    + Show help message and exit
    
  * -exchange       
    + Choose a exchange to train with
  
  * -name
    + Name your trained model
  
  * -thread
    + How many threads do you want use while training?
  
  * -shiftx 
    + Give the numbers of consecutive trading logs to train
    + (Using 50 trades from the past to train rougly equals training with the past 30 minitue data)
  
  * -shifty
    + How far do you want to predict? For example, 20 means predicting trend after 20 trades from now
    + (Predicting 50 trades into future roughly equals predicting trend after 30 minutes)
  
  * -ls 
    + List all the saved model names

### Sample Usage ###
  * Train binance exchange data with 3 threads and name the model "1h":
      ```
        python3 run.py -exchange binance -name 1h -shiftx 200 -shifty 100 -thread 3
      ```
      
  * Train bitfinex exchange data with 2 threads and name the model "30m":
      ```
        python3 run.py -exchange bitfinex -name 1h -shiftx 100 -shifty 50 -thread 2
      ```
  * List the models you have trained:
    ```
      python3 run.py -ls
    ```
### Some Notes ###
 * After each training, the prediction result with the latest trade will be put in predict/
 * There are some default .sh files which runs the training ( run all the .sh file to repeatedly train the four exchange ) 
