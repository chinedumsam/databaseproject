# importing the libraries
import matplotlib
matplotlib.use('Agg')
import pandas as pd # importing pandas
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd # importing pandas
import datetime # importing datetime

## importing data downloaded from quandl
data=pd.read_csv("/home/ubuntu/EOD-AAPL1.csv")
data['Date']=pd.to_datetime(data['Date']) # converting to date
data=data.sort_values("Date") # sorting on basis of date
data=data.reset_index(drop=True) # resetting index
data.head()

# plotting using matplotlib
from matplotlib import pyplot as plt
data1=data[["Date","Adj_Close"]] # making a dataframe from data and adjusted closing price
data1.set_index("Date",inplace=True)
plt.plot(data1,c='green')
plt.xlabel('date')
plt.show()



########### creating training and test set
size_train=round(0.7*len(data))
date=data['Date']
data=data[["Adj_Close"]]
train=data[0:size_train]
test=data[(size_train-1):]
test_date=date[(size_train-1):]
y_train=data[0:(size_train+1)].shift(-1).dropna()


############# the neural network model
import keras
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
# adding the first layer
model.add(Dense(units = 10, activation = 'relu', input_dim = 1))
# Adding the second hidden layer
model.add(Dense(units= 1,activation = 'relu'))
# Adding the output layer
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.summary()

### fitting the model
model.fit(train, y_train, batch_size = 10, epochs = 20)

######### predicting on the test set
predicted=model.predict(test)
predicted=[j[0] for j in predicted]
predicted=predicted[0:(len(predicted)-1)]
test_and_predicted=pd.DataFrame()
test_and_predicted['test']=[j[0] for j in list(test[1:].values)]
test_and_predicted['predicted']=predicted
test_and_predicted.index=test_date[1:]

######## plotting the predicted vs test data
plt.plot(test_and_predicted['predicted'],c="red")
plt.plot(test_and_predicted['test'],c="green")
plt.rcParams["figure.figsize"] = (10,20)
plt.xlabel("date")
plt.ylabel("price")
plt.show()
print('saving graph')
plt.savefig('microsoft1.png')

#########printing the entire predicted vs test data 
print(test_and_predicted)

