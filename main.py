from typing import final
import numpy as np
import pandas as pd
import math
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math
from sklearn.preprocessing import MinMaxScaler

#%matplotlib inline


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


def get_consol_hist_data(tickerlist, timeframe="10y"):
    acc = []
    for ticker in tickerlist:
        print(ticker)
        df = get_hist_data(ticker, timeframe)
        df["Ticker"] = ticker
        df["Date"] = df.index
        acc.extend(df.values.tolist())
    col_names = [
        "Ticker",
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
    return outpust_hist





# Define the time window
window = 70

# Get all the data of GME which Yahoo offers
final_data = get_hist_data("AAPL")

# for the sake of convenience make all column names lower case
final_data = final_data.rename(columns=str.lower)
# reversing the dataframe to make the newest entry the first line
# final_data = final_data[::-1]
# generate date column from index
final_data["date"] = final_data.index
final_data = final_data.reset_index()


#  plotting the date vs the closing stock price
# plt.style.use('dark_background')
final_data.plot("date", "close", color="red")

short_data = final_data.head(window)

short_data.plot("date", "close", color="green")

plt.show()


# getting the closing price and making it to an array
df = final_data.filter(["close"]).values

# scale data to be between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# using a proportion of the data as training data
train_data_len = math.ceil(len(df) * 0.75)
train_data = scaled_data[0:train_data_len, :]

# the ggeneration of the predictor variable and the independent ones. e.g x1 = 1-60 day
# and y1 = 61, x2=2-61 and y2=62 and so on..

x_train_data = []
y_train_data = []
for i in range(window, len(train_data)):

    x_train_data.append(train_data[i - window : i, 0])
    y_train_data.append(train_data[i, 0])

    # c onverting the training data to numpy arrays
    x_train_data1, y_train_data1 = np.array(x_train_data), np.array(y_train_data)
    x_train_data2 = np.reshape(
        x_train_data1, (x_train_data1.shape[0], x_train_data1.shape[1], 1)
    )


model = Sequential()
model.add(
    LSTM(units=50, return_sequences=True, input_shape=(x_train_data2.shape[1], 1))
)
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mean_squared_error")

model.fit(x_train_data2, y_train_data1, batch_size=1, epochs=1)


# generating test data, beginning with the first data to predict after the train data lewn
test_data = scaled_data[train_data_len - window :, :]
x_test = []
y_test = df[train_data_len:, :]
for i in range(window, len(test_data)):
    x_test.append(test_data[i - window : i, 0])

# converting the training data to numpy arrays
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# making pred
pred = model.predict(x_test)
# translate the pred back to dollar values
pred = scaler.inverse_transform(pred)

rmse = np.sqrt(np.mean(((pred - y_test) ** 2)))
print(rmse)

train = final_data[:train_data_len]
test = final_data[train_data_len:]

test["Predictions"] = pred

plt.title("Model")
plt.xlabel("Date")
plt.ylabel("Close")

plt.plot(train["close"])
plt.plot(test[["close", "Predictions"]])

plt.legend(["Training", "Test", "Predictions"], loc="lower right")

plt.show()
