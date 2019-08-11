'''
An SVM classifier to classify the data point as a type of cancer:  Class: (2 for benign, 4 for malignant)
__author__='Soniya Rode'
__citation__='PythonProgramming'
Data Set : Breast Cancer Wisconsin from UCI

'''
import numpy as np
from sklearn import neighbors,svm
import pandas as pd
from sklearn import model_selection

#Read the cancer data
df=pd.read_csv("breast-cancer-wisconsin.txt")
df.replace("?",-99999,inplace=True)

#Drop the id column since id attribute has no relation to the class label
df.drop(['id'],1,inplace=True)

#Train data
x=np.array(df.drop(['class'],1))

#class labels
y=np.array(df['class'])

#cross validation
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2)

#SVM classifier
classifier=svm.SVC()


classifier.fit(x_train,y_train)
print("Accuracy of the SVM classifier is : ",classifier.score(x_test,y_test))

