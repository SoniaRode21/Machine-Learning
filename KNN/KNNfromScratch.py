'''
A Knn classifier from scratch using Euclidean distance to assign a class label to the data point
__author__='Soniya Rode'
__citation__='PythonProgramming'

'''
import numpy as np
from math import sqrt
from matplotlib import style
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
style.use('fivethirtyeight')


dataset={'class1':[[1,2],[1,3],[1,4],[2,3],[2,1],[2,2]],
         'class2':[[9,8],[7,8],[6,9],[8,8],[7,9],[7,7]]}
classify=[2,4]


def knnClassifier(data,classify,k=3):
    if len(data)>k:
        return("Value of k is < number of classes")

    distances = []
    for class_label in data:
        for features in data[class_label]:
            euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(classify)**2)))
            distances.append([euclidean_distance,class_label])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

result = knnClassifier(dataset, classify)
print(result)
