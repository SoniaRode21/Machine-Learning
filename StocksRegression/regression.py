'''
A linear regression model to predict the close value of a stock a few days after. Currenlty the
code predicts stock close value after 10 days.
__author__='Soniya Rode'
__citation__='PythonProgramming'
'''


import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing ,svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

style.use('ggplot')

#
df=quandl.get('WIKI/GOOGL')
df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
#h-l: High - low percentage
df['h_l']=df['Adj. High']-df['Adj. Low'] /df['Adj. Low']

# O_C: Percentage change i.e open -close percentage
df['O_c']=df['Adj. Open']-df['Adj. Close'] /df['Adj. Close']

df=df[['h_l','O_c','Adj. Volume','Adj. Close']]

df.fillna(-99999,inplace=True)
#Column value to be predicted
predict='Adj. Close'

#value 10 days before
predicted_out=math.ceil(0.01*len(df))

#Label will have the value of predict column 1 day after
df['label']=df[predict].shift(-predicted_out)

df.dropna(inplace=True)
#x: features, Y=labels
x = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

#Scale all the features
x=preprocessing.scale(x)
#remaining will store stocks to predict their future values
remaining = x[-predicted_out:]
X = x[:-predicted_out]
df.dropna(inplace=True)

#Cross_vallidation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Define the linear Regression classifier
clf = LinearRegression(n_jobs=-1)

#Fit the data
clf.fit(x_train, y_train)

#get prediction accuracy
accuracy = clf.score(x_test, y_test)

forecast_set = clf.predict(remaining)
df['Forecast'] = np.nan

#Get the last date of known stock
last_date = df.iloc[-1].name
date = last_date.timestamp()

nextDate = date + 86400

#Get dates for predicted values
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(nextDate)
    nextDate += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

#Plot the known(adj.close) and predicted(forecast) values
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
