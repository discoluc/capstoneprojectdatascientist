from black import out
import numpy as np
import pandas as pd
import math
import yfinance as yf
import matplotlib.pyplot as plt

stocklist = pd.read_csv("ticker_full.csv")



def get_stock_info(ticker):
    """
    Get the info about a ticker e.g. name of company.
    ticker...Input the Ticker of a certain stock

    output....json with stock info
    """

    stock_info = yf.Ticker(ticker).info
    return stock_info


def get_hist_data(ticker, timeframe="max"):
    """
    Get the historical daily market data of a certain ticker for a certain period.
    Market Date includes the open, close, high, low preices of the day and the trading volumen.

    ticker... Input the ticker of the stock of interest
    timeframe... the period of interest (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

    """

    hist = yf.Ticker(ticker).history(period=timeframe)
    return hist


def generate_consol_hist_data(stocklist, timeframe="10y"):
    """
    Gets the stock price history for a user defined list of companys.
    Output is a consolidated dataframe for all stocks and it save everything as a csv.

    """
    tickerlist = list(stocklist["Ticker"])

    acc = []

    # Loop through each tickers in the ticketlist, getting the historical data
    for ticker in tickerlist:
        print(ticker)
        df = get_hist_data(ticker, timeframe)
        df["Ticker"] = ticker
        df["Date"] = df.index
        df["Name"] = stocklist[stocklist["Ticker"] == ticker]["Name"]
        acc.extend(df.values.tolist())
    col_names = [
        "Ticker",
        "Name",
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Dividends",
        "Stock Splits",
    ]
    output_hist = pd.DataFrame(acc, columns=col_names)
    output_hist["Date"] = pd.to_datetime(output_hist["Date"])
    output_hist.to_csv("stock_prices_10yrs.csv", index=False)
    return output_hist


df = generate_consol_hist_data(stocklist)


# How many tickers are in the dataset.
df.Ticker.nunique()

# General Information about the dataset
df.info()
# Shape of the dataframe
df.shape


# Filtering only the Apple Inc. Stock
apple = df[df["Ticker"] == "AAPL"]

# Check for duplicate rows
apple["Date"].duplicated().value_counts()

# Min / Max Closing price
apple.Close.min()
apple.Close.max()


# Check for null values and datatypes

apple.info()


# Plotting the Opening and Closing Prices
plt.figure(figsize=(15, 10))
plt.gcf().autofmt_xdate()
plt.plot(apple["Date"], apple["Close"], label="Close")
plt.xticks(fontsize=25)
plt.xlabel("Date", fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel("Closing Price [in USD]", fontsize=25)

# looking at the last 60 days
plt.figure(figsize=(15, 10))
plt.gcf().autofmt_xdate()
plt.plot(apple["Date"].iloc[-60:], apple["Close"].iloc[-60:], label="Close")
plt.xticks(fontsize=25)
plt.xlabel("Date", fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel("Closing Price [in USD]", fontsize=25)

# Generating the Dataset as csv
apple.to_csv("apple.csv",index = False)
