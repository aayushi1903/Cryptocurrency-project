# Covid Resistant Husky 3 - ADA price prediction

# import pip to install necessary libraries
import math

import pip
pip.main(['install', 'python-binance', 'pandas', 'scikit-learn', 'matplotlib', 'keras', 'tensorflow', 'plotly',
          'mplfinance'])
from keras.losses import mean_squared_error
from matplotlib.dates import date2num



# import the necessary libraries
import config
from binance.client import Client
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout, GRU
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import talib
from sklearn.metrics import r2_score
from mplfinance.original_flavor import candlestick_ohlc


# function call to split the data for training and test
def train_test_split(df, test_size=0.2):
    # split the data into 80 and 20
    split_row = len(df) - int(len(df) * test_size)
    # train data contains 80% of data
    train_data = df.iloc[:split_row]
    # test data contains 20% of the data
    test_data = df.iloc[split_row:]
    return train_data, test_data


# function to calcualte MACD, MA10,MA30 and RSI
def get_indicators(data2):
    # Get MACD
    data2["macd"], data2["macd_signal"], data2["macd_hist"] = talib.MACD(data2['Close'])

    # Get MA10 and MA30
    data2["ma10"] = talib.MA(data2["Close"], timeperiod=10)
    data2["ma30"] = talib.MA(data2["Close"], timeperiod=30)

    # Get RSI
    data2["rsi"] = talib.RSI(data2["Close"])
    return data2


# function call to initialise the RNN - GRU model
def Model_Initialisation_GRU(X_train, mod):
    model = Sequential()
    # Layer 1
    model.add(mod(70, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    # Dropout is set to 20%
    model.add(Dropout(0.2))
    # Layer 2
    model.add(mod(80, activation='relu'))
    # Dropout is set to 20%
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    print(model.summary())
    return model


# function call to initialise the RNN -LSTM model
def Model_Initialisation_LSTM(X_train, mod):
    model = Sequential()
    # Layer 1
    model.add(mod(300, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    # Dropout is set to 20%
    model.add(Dropout(0.2))
    # Layer 2
    model.add(mod(300, activation='relu'))
    # Dropout is set to 20%
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    print(model.summary())
    return model


# function call to plot the RSI, MACD, MA10 and MA30
def plot_chart(data, n, ticker):
    # Filter number of observations to plot
    data = data.iloc[-n:]

    # Create figure and set axes for subplots
    fig = plt.figure()
    fig.set_size_inches((30, 20))
    ax_candle = fig.add_axes((0, 0.72, 1, 0.32))
    ax_macd = fig.add_axes((0, 0.48, 1, 0.2), sharex=ax_candle)
    ax_rsi = fig.add_axes((0, 0.24, 1, 0.2), sharex=ax_candle)
    ax_vol = fig.add_axes((0, 0, 1, 0.2), sharex=ax_candle)

    # Format x-axis ticks as dates
    ax_candle.xaxis_date()

    # Get nested list of date, open, high, low and close prices
    ohlc = []
    for date, row in data.iterrows():
        openp, highp, lowp, closep = row[:4]
        ohlc.append([date2num(date), openp, highp, lowp, closep])

    # Plot candlestick chart
    ax_candle.plot(data.index, data["ma10"], label="MA10")
    ax_candle.plot(data.index, data["ma30"], label="MA30")
    candlestick_ohlc(ax_candle, ohlc, colorup="g", colordown="r", width=0.8)
    ax_candle.legend()

    # Plot MACD
    ax_macd.plot(data.index, data["macd"], label="macd")
    ax_macd.bar(data.index, data["macd_hist"] * 3, label="hist")
    ax_macd.plot(data.index, data["macd_signal"], label="signal")
    ax_macd.legend()

    # Plot RSI
    # Above 70% = overbought, below 30% = oversold
    ax_rsi.set_ylabel("(%)")
    ax_rsi.plot(data.index, [70] * len(data.index), label="overbought")
    ax_rsi.plot(data.index, [30] * len(data.index), label="oversold")
    ax_rsi.plot(data.index, data["rsi"], label="rsi")
    ax_rsi.legend()

    # Show volume in millions
    ax_vol.bar(data.index, data["Volume"] / 1000000)
    ax_vol.set_ylabel("(Million)")

    # Save the chart as PNG
    fig.savefig(ticker + ".png", bbox_inches="tight")

    plt.show()


# function call to train and predict the close values
def fit_model(train_set, test_set, X_train, y_train, model, scaler, model_name):
    model.compile(loss='mse', optimizer='adam')
    # fit the model with the training data
    model.fit(X_train, y_train, epochs=50, batch_size=32, shuffle=False)
    # storing the training data
    train_set = pd.DataFrame(train_set, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    # specifying the window as 10
    past_10_days = train_set.tail(10)
    df = past_10_days.append(test_set, ignore_index=True)
    print(df.head())
    # transforming the data
    inputs = scaler.transform(df)
    print(inputs)

    X_test = []
    y_test = []

    for i in range(10, inputs.shape[0]):
        X_test.append(inputs[i - 10:i])
        y_test.append(inputs[i, 0])
    # converting the data into array
    X_test, y_test = np.array(X_test), np.array(y_test)
    print(X_test.shape, y_test.shape)
    # predicting the close value on test data
    y_pred = model.predict(X_test)
    print(y_pred, y_test)

    # performing inverse of scalar transformation to get the actual value
    arr = scaler.scale_
    scale = 1 / arr[0]
    print(scale)
    y_pred = y_pred * scale
    y_test = y_test * scale
    # calculates the r2 value
    print("R2 score", round(r2_score(y_test, y_pred) * 100, 2))
    # calculates the RMSE value for training data
    trainScore = math.sqrt(mean_squared_error(y_train[0], y_pred[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    # calculates the RMSE value for test data
    testScore = math.sqrt(mean_squared_error(y_test[0], y_pred[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # plotting the graph to show predicted vs test value of the models
    plt.figure(figsize=(14, 5))
    plt.plot(y_test, color='red', label='ADA Price')
    plt.plot(y_pred, color='blue', label='Predicted ADA Price')
    plt.title('ADA Price Prediction')
    plt.xlabel('Time (in Hours)')
    plt.ylabel('ADA Price')
    plt.legend()
    plt.show()


# function to call to initialise the models and call the model to predict the data
def Model_Prediction(df):
    # split the data into 80:20 for training and testing
    train_set, test_set = train_test_split(data, test_size=0.2)
    scaler = MinMaxScaler()
    # transforming the data using Min, Max scaler
    train_set = scaler.fit_transform(train_set)
    print(train_set)
    # specifiying the window size as 10 and values are stored accordingly in train data
    X_train = []
    y_train = []
    for i in range(10, train_set.shape[0]):
        X_train.append(train_set[i - 10:i])
        y_train.append(train_set[i, 0])
    # converting the train data into array
    X_train, y_train = np.array(X_train), np.array(y_train)
    print(X_train.shape)
    # initialising RNN - GRU model
    model_GRU = Model_Initialisation_GRU(X_train, GRU)
    # initialising RNN - LSTM
    model_LSTM = Model_Initialisation_LSTM(X_train, LSTM)
    # training the LSTM model and predicting the close value
    fit_model(train_set, test_set, X_train, y_train, model_LSTM, scaler, 'LSTM')
    # training the GRU model and predicting the close value
    fit_model(train_set, test_set, X_train, y_train, model_GRU, scaler, 'GRU')


# Function call to plot the candelstick chart
def CandleStickChart(data):
    # using TA lib library for recognising the patterns in data
    candle_names = talib.get_function_groups()['Pattern Recognition']
    # Plotting the candelstick data
    fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'],
                                         close=data['Close'])])
    fig.update_layout(title='ADA price over the time', yaxis_title='Price')
    fig.show()


# function to plot log10 close price of ADA
def LogReturnsPlot(data):
    # calculates log10 of close price of ADA
    change = pd.DataFrame(data['Close']).apply(lambda x: np.log(x) - np.log(x.shift(1)))
    print(change)
    # plots the graph
    fig = go.Figure([go.Scatter(x=change.index, y=change['Close'])])
    fig.update_layout(title='log10 value of Closing price', xaxis_title='log10')
    fig.show()


# function call to download the dataset
def get_ADA_BinanceAPI():
    # connecting to the Binance API to fetch the data
    client = Client(config.binance_API, config.binance_Secret)
    # downloading the ADA data from Binance API for 1 day interval, 500 entries.
    ADA = client.get_klines(symbol='ADAUSDT', interval=Client.KLINE_INTERVAL_1DAY, limit=500)
    # converting the data into Dataframe and assigning columns
    ADA = pd.DataFrame(ADA, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                                     'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
                                     'Taker buy quote asset volume', 'Ignore'])
    # converting the open time column to datetime type
    ADA['Open time'] = pd.to_datetime(ADA['Open time'], unit='ms')
    # formatting the data in ascending order of time
    ADA.sort_values(by=['Open time'], inplace=True, ascending=True)
    # setting time column as index
    ADA.set_index('Open time', inplace=True)
    # converting the variables into float type
    ADA['Open'] = ADA['Open'].astype(float)
    ADA['High'] = ADA['High'].astype(float)
    ADA['Low'] = ADA['Low'].astype(float)
    ADA['Close'] = ADA['Close'].astype(float)
    ADA['Volume'] = ADA['Volume'].astype(float)
    # the 5 variables are used to predict close and are stored as dataframe and returned
    data = pd.DataFrame(ADA[['Open', 'High', 'Low', 'Close', 'Volume']])
    print(data.head(15))
    return data


# main function
if __name__ == '__main__':
    # function to download the data from Binance API
    data = get_ADA_BinanceAPI()
    # Function to plot log10 close price
    LogReturnsPlot(data)
    # fucntion call to calculate MACD, RSI, MA10 and MA30
    data2 = data.copy()
    data2 = get_indicators(data2)
    # fucntion call to plot MACD, RSI, MA10 and MA30
    plot_chart(data2, 150, "PLOTS")
    # function to plot the candelstick chart
    CandleStickChart(data)
    # model to predict the close price of ADA
    Model_Prediction(data)
