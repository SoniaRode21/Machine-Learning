'''
Applying K-Means algorithm to the Titanic dataset
__author__='Soniya Rode'
__citation__='PythonProgramming'
'''
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

def get_numerical_data(df):
    '''
    Function to convert non-numeric data to numeric
    :param df: Data frame
    :return: Modified dataframe having numerical values.
    '''

    #Get all the columns of the dataframe
    columns = df.columns.values

    for column in columns:

        #Dictionary to store numerical value for  qualitative columns
        numerical_values = {}

        #function returns numerical value of the qualitative column
        def convert_to_int(val):
            return numerical_values[val]

        #Check if column has non-numeric data
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            #get all non-numeric values and store them as a list in column_values
            column_values = df[column].values.tolist()
            #get the unique values
            unique_column_values = set(column_values)
            x = 0
            #set numeric value for each non-numeric value
            for unique in unique_column_values:
                if unique not in numerical_values:
                    numerical_values[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df


#Get titanic data
df = pd.read_excel('titanic.xls')
#drop unwanted columns
df.drop(['body','name'], 1, inplace=True)
df.fillna(0, inplace=True)

#Convert all non-numeric data to numeric
df = get_numerical_data(df)

#x-> Features, y-> label
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

cluster = KMeans(n_clusters=2)

cluster.fit(X)

#Get the accuracy
correct = 0
for i in range(len(X)):

    data = np.array(X[i].astype(float))
    data = data.reshape(-1, len(data))
    prediction = cluster.predict(data)
    if prediction[0] == y[i]:

        correct += 1

accuracy=correct/len(X)

print(accuracy)


