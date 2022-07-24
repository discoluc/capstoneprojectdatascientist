# Short Term Stock Price Prediction with Machine Learning
## Udacity Capstone Project

### Table of Contents
1. [Summary](#summary)
2. [Installation](#installation)
3. [Instructions](#instruction)
4. [Files](#files)
5. [Licensing, Authors, and Acknowledgements](#licensing)


## Summary <a name="summary"></a>
This is a summary of the following [paper](/Paper/article3.pdf). There you find my complete text.
Accurately predicting stock prices is a highly sought-after mastery. In recent years Long Short Term Memory (LSTM), a type of recurrent neural network, has proven to be highly promising in achieving better predictions. Using a simple LSTM with only 25 neurons already can predict loosely the future stock prices based on the stock prices of the previous 30 days. The refined model with two LSTM layers with 150 neurons and two dense layers is very successful in predicting the future close price.

### Project Definition
The accurate forecast of stock prices, traded on an exchange, is inarguably one of the most challenging topics in the field of asset pricing. The stock market is described as an unpredictable, dynamic and non-linear construct. Predicting the prices of its stocks is an ambitious undertaking and depends on many variables including but not limited to the global economy, the company's metrics and local and global political situation. 

Historically there are two main prediction methods. The fundamental analysis which is a qualitative analysis of the company is interested in finding the true value of a stock and comparing it to the actual traded value. The evaluator utilizes several performance criteria e.g. the P/E ratio to truly assess the underlying stock. Secondly, the technical analysis, which is solely based on the past price of the stock e.g. in form of closing or opening prices as time-series. It is rather a short-term prediction using factors like the Moving Average (MA) or the Exponential Moving Average (EMA). Its basic assumption is that every significant information about the stock is already considered in the stock price.

The fast computational development has led to the point that Machine learning techniques have a significant application in financial problems. The use of artificial neural networks has found more and its way into the field of stock price prediction. Here a recurrent neural network (RNN) has been found very proven, more precisely the Long Short Term Memory (LSTM). Its advantage is being
able to process entire sequences of data rather than only one single data point. It has proven to be very practical with time series data such as our historical stock prices. 

In this project, I create an application that is going to predict the closing price of any given stock on which it is trained. For the sake of convenience, this report only considers the stock price prediction of the Apple Inc. $AAPL stock.

A train-test cycle is used to test the accuracy of the prediction. The root mean squared error is used to measure the accuracy of the model. The RMSE is a good measure to compare prediction with actual values and makes it easier to compare different models

In the next section, I will analyze the data and give some insights about the data. Afterwords I will explain the methodology and discuss the results. In the end, I give an outlook for further developments and improvements.

### Data Analysis

The dataset was collected with the help of yfinance
python plugin. In my case, it is replacing the not working
Yahoo Finance API. The collected data frame contains  
  • 1108837 rows  
  • 10 columns  
  • price information for 457 unique stocks  
  • Data of the last 10 years (including 17/07/2022)  
and was collected on 18/07/2022. I use a list of 457 randomly
selected stock tickers to generate the dataset.


The structure and the first 5 lines of the table filtered
for Apple Inc. are displayed in table 1. For each day and
each stock (denoted by its ticker (Ticker) and company name
(Name) the dataset contains the price of the stock in dollar
when opening the market (Open), closing the market (Close).
Furthermore there the days lowest (Low) and highest intra-day
price (High is available, as well as the daily traded volume
(Volume).
Looking at the Apple Inc. stock there are 2514 columns
regarding the apple stock with a minimum closing price of
12.17 and a maximum closing price of 182.01. There are no
missing values (see table 2) in the dataset, and the date and
value columns are correctly set as a datetime respectively float
columns. 

The highly volatile closing price of the stock
is shown for its last 10 years. In recent years the stock prices
have increased roughly sevenfold. In the next section, I am
going to explain the methodology, followed by a chapter discussing the results

### Conclusion 
In this paper, I looked at the problem of forecasting accurately stock prices. In the beginning, I give an overview of why this topic matters so much for the financial world. Additionally, I present some real-life data on the stock prices of Apple stock and some data insights before explaining the methodology. Stock prices can be treated as time-series data, thus LSTM neural networks are predestinated as a suitable algorithm for predicting stock prices. The simple model is falling quite short in predicting the stock prices, however, my most complicated model is roughly 11 times better than the simple one. 

Even though the model is quite good at predicting stock prices, in a real-life scenario it probably will perform poorly due to unforeseen and disruptive events (e.g. COVID19 or Ukraine War).


Generating test and training sets were more complex than usual since I am dealing with time series data. Also, I needed to put together a timeframe of past stock prices as my feature variables with its corresponding stock price prediction. This was quite tricky to generate.


For future research, I will try and compare different machine learning algorithms and compare their performance with the LSMT. Furthermore one could try to play with the parameters, and layers in my current network to make even better predictions of the future stock price. Also, I will use different stocks and see if the performance of the models is going to change.



## Installation <a name="installation"></a>
The code only uses Pandas, Numpy, SciKit , Keras, Matplotly, Tensorflow  and was tested on a Python version 3.9.2.
There should be no necessary  other libraries to run the code here beyond these.
| **Package** | **Version** |
|---------------|-------------|
| keras         | 2.9.0       |
| matplotlib    | 3.5.1       |
| numpy         | 1.22.3      |
| pandas        | 1.43        |
| scxikit-learn | 1.1.1       |
| tensorflow    | 2.91        |
| yfinance      | 0.1.74      |


## Instructions <a name="instruction"></a>:
1. Financial stock data is already downloaded. This skip can be skipped, unless you want newer data. In the Data folder the Apple Stock prices are prepared until the 18th of July. If you want newer stock prices use the get_data_analyze_date.py and run the `get_hist_data(ticker, timeframe="max")` with a ticker of your choice (e.g. for Apple the ticker is "AAPL").


2. To train a model yourself, run the following command in the app's directory to run the model
    `python main.py`



## Files <a name=files></a>

* [Data/](/Data) # All the prepared data
  * [apple.csv](/Data/apple.csv) #Downloaded stock price date for apple
  * [get_data_analyze_date.py](/Data/get_data_analyze_date.py)
  * [stock_prices_10yrs.csv](/Data/stock_prices_10yrs.csv) # Stock prices for the last ten years for all tickers in the ticker_full.csv
  * [ticker_full.csv](/Data/ticker_full.csv)
* [Paper/](/Paper)
  * [Figures/](/Paper/Figures)
  * [article3.aux](/Paper/article3.aux)
  * [article3.bbl](/Paper/article3.bbl)
  * [article3.blg](/Paper/article3.blg)
  * [article3.dvi](/Paper/article3.dvi)
  * [article3.log](/Paper/article3.log)
  * [article3.out](/Paper/article3.out)
  * [article3.pdf](/Paper/article3.pdf)
  * [article3.synctex.gz](/Paper/article3.synctex.gz)
  * [article3.tex](/Paper/article3.tex)
  * [article3.toc](/Paper/article3.toc)
  * [sample.bib](/Paper/sample.bib)
  * [SelfArx.cls](/Paper/SelfArx.cls)
  * [texput.log](/Paper/texput.log)
* [.gitignore](/.gitignore)
* [main.py](/main.py)
* [README.md](/README.md)
* [results.ipynb](/results.ipynb)

All downloaded data is in the Data folder. It includes a python file with helper functions to download stock price information of any given ticker using yfinance. The stock_prices_10yrs.csv consists of all stock tickers in the ticker_full.csv. The apple.csv is the dataset that is used to train the model and predict prices. In the Paper folder are all the tex files needed to compile the paper yourself using latex.

Running the main.py runs a script that trains a model, tests it and saves graphs for each model. In the results.ipynb the results of my model are saved.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
This scriptdeveloped during an exercise regarding the [Udacity Data Science Nanodegree](https://www.udacity.com/school-of-data-science), feel free to use the code as you like.



