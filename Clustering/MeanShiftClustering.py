'''
Applying K-Means algorithm to the Titanic dataset
__author__='Soniya Rode'
__citation__='PythonProgramming'
'''
import numpy as np
from sklearn.cluster import MeanShift
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
dfCopy=df
#drop unwanted columns
df.drop(['body','name'], 1, inplace=True)
df.fillna(0, inplace=True)

#Convert all non-numeric data to numeric
df = get_numerical_data(df)

#x-> Features, y-> label
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

cluster = MeanShift()

cluster.fit(X)

#get all cluster labels
labels = cluster.labels_
#print(labels)
#get number of clusters
n_clusters_ = len(np.unique(labels))
print("number of clusters :",n_clusters_)
cluster_centers = cluster.cluster_centers_
#print(cluster_centers)

#Add new column to the original df
dfCopy['cluster_group']=np.nan
#Add label to each df column
for i in range(len(X)):
    dfCopy['cluster_group'].iloc[i] = labels[i]


survival_rates = {}
#for each cluster get survival rate
for i in range(n_clusters_):
    temp_df = dfCopy[ (dfCopy['cluster_group']==float(i)) ]

    survival_cluster = temp_df[  (temp_df['survived'] == 1) ]

    survival_rate = len(survival_cluster) / len(temp_df)
    #print(i,survival_rate)
    survival_rates[i] = survival_rate

print("Survival rates for each cluster : ",survival_rates)

