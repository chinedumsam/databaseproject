# LinearRegression is a machine learning library for linear regression
from sklearn.linear_model import LinearRegression

# pandas is used for data manipulation
import pandas as Panda

# matplotlib is used for plotting graphs
import matplotlib

# sets matplotlib to non interactive output
matplotlib.use('Agg', warn=False, force=True)
from matplotlib import pyplot as graph

# quandl is used to fetch data
import quandl

# save quandl API Key
quandl.ApiConfig.api_key = "ZnicxoqNgYQA4t56vi_a"

def GetStockPrediction(cpy, cutPoint = .8):
    print('getting data.....')
    # get Apple's stock price from 1995 to 2002
    Data = quandl.get(company, start_date='1995-01-01', end_date='2002-12-31')
    print('data collected....')
    # drops other columns except the closing price columns
    Data = Data[['Close']]

    # removes rows with missing values
    Data = Data.dropna()

    Data.Close.plot(figsize=(10, 5))

    graph.ylabel("AAPL price variations")

    graph.show()

    # gets the moving averages
    Data['Ma_3'] = Data['Close'].shift(1).rolling(window=3).mean()
    Data['Ma_9'] = Data['Close'].shift(1).rolling(window=9).mean()

    Data = Data.dropna()

    X = Data[['Ma_3', 'Ma_9']]

    y = Data['Close']

    cutOffPoint = int(cutPoint * len(Data))

    # Train dataset

    X_train = X[:cutOffPoint]

    y_train = y[:cutOffPoint]

    # Test dataset

    X_test = X[cutOffPoint:]

    y_test = y[cutOffPoint:]

    linear = LinearRegression().fit(X_train, y_train)

