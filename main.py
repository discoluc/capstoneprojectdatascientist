import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math
from sklearn.preprocessing import MinMaxScaler

#%matplotlib inline


# Read in the previously generated Apple Stock Price File
apple_stock_prices = pd.read_csv("apple.csv")
apple_stock_prices["Date"] = pd.to_datetime(apple_stock_prices["Date"])


# Defining the target and the features
y_target = apple_stock_prices["Close"].values.reshape(-1, 1)
x_features = apple_stock_prices[["Open", "High", "Low"]].values


# Scale Data using the min max scaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler2 = MinMaxScaler(feature_range=(0, 1))
y_target_scaled = scaler.fit_transform(y_target)
x_features_scaled = scaler2.fit_transform(x_features)

data_scaled = np.concatenate((x_features_scaled, y_target_scaled), axis=1)


def test_train_set_generator1D(df, window_size, training_set_size=0.8):
    """
    Since we are building a LSTM we cannot simply split the dataset, we need to build our feature set
    going back a certain time e.g x = 0.day until 9.day and y = 10.day. So we can predict with the past 10 day
    the 11. day. This is for a 1D feature set (feature set equals the target).

    Input
    df... a 1D array containing the target/feature variable
    window size...how many days are we going back in time to predict the value
    training_set_size... percentage of the whole set which will be the training set

    Output:
    x_test, y_test ...the feature and target for the training set
    x_train, x_test... the feature and target for the test set
    train_data_len ... the size of the training set
    """

    x, y = [], []
    for i in range(window_size, len(df)):
        x.append(df[i - window_size : i])
        y.append(df[i])
    x, y = np.array(x), np.array(y)
    train_data_len = int(math.ceil(len(df) * training_set_size))
    x_train, x_test = x[:train_data_len], x[train_data_len - window_size :]
    y_train, y_test = y[:train_data_len], df[train_data_len:]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, x_test, y_train, y_test, train_data_len


def plot_model_performance(train, test, detail_days=60):
    """
    Plotting the predicted price vs the actual price. Including a second graph with a detail view
    (including detail_days last observations)

    Input
    train...the training dataset
    test ... the testing dataset
    detail_days...how many last observations do you want to have displayed
    """

    plt.xlabel("Date")
    plt.ylabel("Close Price [in USD]")
    plt.gcf().autofmt_xdate(rotation=45)
    plt.plot(train["Date"], train["Close"])
    plt.plot(test["Date"], test["Close"])
    plt.plot(test["Date"], test["Predictions"])

    plt.legend(["Training", "Test", "Predictions"], loc="lower right")

    plt.show()

    plt.xlabel("Date")
    plt.ylabel("Close Price [in USD]")
    plt.gcf().autofmt_xdate(rotation=45)

    plt.plot(test["Date"][-abs(detail_days) :], test["Close"][-abs(detail_days) :])
    plt.plot(
        test["Date"][-abs(detail_days) :], test["Predictions"][-abs(detail_days) :]
    )

    plt.legend(["Training", "Test", "Predictions"], loc="lower right")

    plt.show()

    print(test.tail(abs(detail_days)))


def simple_model(df, epochs=1, units=32, window_size=30, training_set_size=0.8):
    """
    Building a simple LSTM model, which only uses the closing price as a feature to
    predict the closing price

    df...Is a dataframe containing already scaled stock prices (Open, Close, Low, High)
    window_size...How many days is our model looking in the past
    training_set_size...What part of the data set is used to train the model
    units...the number of neurons
    epochs...How many times are you going through the model

    """

    target = df[:, -1]
    x_train, x_test, y_train, y_test, train_data_len = test_train_set_generator1D(
        target, window_size, training_set_size
    )

    model = Sequential()
    model.add(
        LSTM(
            units=units,
            activation="relu",
            return_sequences=False,
            input_shape=(x_train.shape[1], 1),
        )
    )
    model.add(Dense(units=1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()

    model.fit(x_train, y_train, batch_size=1, epochs=epochs)

    # making pred
    pred = model.predict(x_test)
    # translate the pred back to dollar values
    pred = scaler.inverse_transform(pred)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(np.mean(((pred - y_test) ** 2)))
    print("The root mean square erros is " + str(rmse))

    train = apple_stock_prices[:train_data_len]
    test = apple_stock_prices[train_data_len:]

    test["Predictions"] = pred

    plot_model_performance(train, test, -50)


def refined_model(df, epochs=1, units=32, window_size=30, training_set_size=0.8):
    """
    Building a simple LSTM model, which only uses the closing price as a feature to
    predict the closing price

    df...Is a dataframe containing already scaled stock prices (Open, Close, Low, High)
    window_size...How many days is our model looking in the past
    training_set_size...What part of the data set is used to train the model
    units...the number of neurons
    epochs...How many times are you going through the model

    """

    target = df[:, -1]
    x_train, x_test, y_train, y_test, train_data_len = test_train_set_generator1D(
        target, window_size, training_set_size
    )

    model = Sequential()
    model.add(
        LSTM(
            units=units,
            activation="relu",
            return_sequences=True,
            input_shape=(x_train.shape[1], 1),
        )
    )
    model.add(LSTM(units=units, activation="relu", return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()

    model.fit(x_train, y_train, batch_size=1, epochs=epochs)

    # making pred
    pred = model.predict(x_test)
    # translate the pred back to dollar values
    pred = scaler.inverse_transform(pred)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(np.mean(((pred - y_test) ** 2)))
    print("The root mean square erros is " + str(rmse))

    train = apple_stock_prices[:train_data_len]
    test = apple_stock_prices[train_data_len:]

    test["Predictions"] = pred

    plot_model_performance(train, test, -50)


# First call of the simple model
simple_model(data_scaled, epochs=1, units=10, window_size=30, training_set_size=0.8)

# Second call of the simple model with more refined parameters
simple_model(data_scaled, epochs=5, units=50, window_size=30, training_set_size=0.8)


# First call of the simple model
refined_model(data_scaled, epochs=1, units=10, window_size=30, training_set_size=0.8)

# Second call of the simple model with more refined parameters
refined_model(data_scaled, epochs=5, units=50, window_size=30, training_set_size=0.8)
