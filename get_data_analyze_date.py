from black import out
import numpy as np
import pandas as pd
import math
import yfinance as yf
import matplotlib.pyplot as plt

stocklist = pd.read_csv("ticker_full.csv")


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
        df = yf.Ticker(ticker).history(period=timeframe)
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
    output_hist.to_csv("stock_prices_10yrs.csv")
    return outpust_hist


df = generate_consol_hist[stocklist]


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
plt.plot(apple["Date"], apple["Close"], label="Close")
plt.xlabel("Date")
plt.ylabel("Closing Price [in USD]")

# looking at the last 60 days
plt.figure(figsize=(15, 10))
plt.plot(apple["Date"].iloc[-60:], apple["Close"].iloc[-60:], label="Close")
plt.xlabel("Date")
plt.ylabel("Closing Price [in USD]")
