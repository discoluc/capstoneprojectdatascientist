# Short Term Stock Price Prediction with Machine Learning
## Udacity Capstone Project

### Table of Contents
1. [Summary](#summary)
2. [Installation](#installation)
3. [Instructions](#instruction)
4. [Files](#files)
5. [Licensing, Authors, and Acknowledgements](#licensing)


## Summary <a name="summary"></a>
Accurately predicting stock prices is a highly sought after mastery. In recent years Long Short Term Memory (LSTM), a type of recurrent neural network, have proven to be highly promising in achieving better predictions. Using a simple LSTM with only 25 neuros already is able to predict loosely the future stock prices based on the stock prices of the previous 30 day. The refined model with two LSTM layers with 150 neurons and two dense layer is very sucessful in predicting the future close price.
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

* [Data/](.\capstoneprojectdatascientist-1\Data) # All the prepared data
  * [apple.csv](.\capstoneprojectdatascientist-1\Data\apple.csv) #Downloaded stock price date for apple
  * [get_data_analyze_date.py](.\capstoneprojectdatascientist-1\Data\get_data_analyze_date.py)
  * [stock_prices_10yrs.csv](.\capstoneprojectdatascientist-1\Data\stock_prices_10yrs.csv) # Stock prices for the last ten years for all tickers in the ticker_full.csv
  * [ticker_full.csv](.\capstoneprojectdatascientist-1\Data\ticker_full.csv)
* [Paper/](.\capstoneprojectdatascientist-1\Paper)
  * [Figures/](.\capstoneprojectdatascientist-1\Paper\Figures)
  * [article3.aux](.\capstoneprojectdatascientist-1\Paper\article3.aux)
  * [article3.bbl](.\capstoneprojectdatascientist-1\Paper\article3.bbl)
  * [article3.blg](.\capstoneprojectdatascientist-1\Paper\article3.blg)
  * [article3.dvi](.\capstoneprojectdatascientist-1\Paper\article3.dvi)
  * [article3.log](.\capstoneprojectdatascientist-1\Paper\article3.log)
  * [article3.out](.\capstoneprojectdatascientist-1\Paper\article3.out)
  * [article3.pdf](.\capstoneprojectdatascientist-1\Paper\article3.pdf)
  * [article3.synctex.gz](.\capstoneprojectdatascientist-1\Paper\article3.synctex.gz)
  * [article3.tex](.\capstoneprojectdatascientist-1\Paper\article3.tex)
  * [article3.toc](.\capstoneprojectdatascientist-1\Paper\article3.toc)
  * [sample.bib](.\capstoneprojectdatascientist-1\Paper\sample.bib)
  * [SelfArx.cls](.\capstoneprojectdatascientist-1\Paper\SelfArx.cls)
  * [texput.log](.\capstoneprojectdatascientist-1\Paper\texput.log)
* [.gitignore](.\capstoneprojectdatascientist-1\.gitignore)
* [main.py](.\capstoneprojectdatascientist-1\main.py)
* [README.md](.\capstoneprojectdatascientist-1\README.md)
* [results.ipynb](.\capstoneprojectdatascientist-1\results.ipynb)

All downloaded data is in the Data folder. It includes a python file with helper functions to download stock price information of any given ticker using yfinance. The stock_prices_10yrs.csv consists of all stock tickers in the ticker_full.csv. The apple.csv is the dataset which is used to train the model and to predict prices on. In the Paper folder are all the tex files needed to compile the paper yourself using latex.

Running the main.py runs a script which trains a model, tests it und saves graphs for each model. In the results.ipynb the results of my model is saved.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
This scriptdeveloped during an exercise regarding the [Udacity Data Science Nanodegree](https://www.udacity.com/school-of-data-science), feel free to use the code as you like.



