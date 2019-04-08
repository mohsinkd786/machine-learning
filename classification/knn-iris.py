import numpy as np
import random 
import scipy.stats as stat
import pandas as pd
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
cols = ['slength','swidth','plength','pwidth','flower']
irisData = pd.read_csv(url,names=cols)

# split data
# A with values
A = irisData.iloc[:,:-1].values
# B with values for flower
B = irisData.iloc[:,4].values

print(irisData.head())
#print(A)
#print(B)
# test data is 0.20 % of actual data
A_training,A_test,B_training,B_test = train_test_split(A,B,test_size = 0.20,random_state=11)

#
#print(A_training)
#print(A_test)
#print(B_training)
#print(B_test)

# choose the classifier

# uses euclidean distance by default
# now to choose the number of neighbors we have a hypothecal constant 
# referred as k 
# most commonly used value for k is 5
# larger the value more precision will be achieved

k = 5 # number of neighbors
kclassifier = KNeighborsClassifier(n_neighbors = k )
# run fitness model
kclassifier.fit(A_training,B_training)

# prediction of the categorized data
prediction = kclassifier.predict(A_test)

#print(prediction)

classificatn_report = classification_report(B_test,prediction)

#print(classificatn_report)

# K NN works on the concept of locating the nearesg neighbor
# Euclidean Distance
# 5.1,3.5 -> 4.9,3.0
# sqrt((5.1-3.5)2- (4.9-3.0)2)

# Manhattan Distance
# |x-y| |5.1-4.9|

# Hamilton Distance
