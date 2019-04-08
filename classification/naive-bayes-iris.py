import numpy as np
import random 
import scipy.stats as stat
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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

#########
# Naive Bayes
naiveBayesClassifier = GaussianNB()
naiveBayesClassifier.fit(A_training,B_training)

#print(naiveBayesClassifier)

naivePrediction = naiveBayesClassifier.predict(A_test)
naiveClassificatn_report = classification_report(B_test,naivePrediction)
print(naiveClassificatn_report)

confusion_matrixx = confusion_matrix(B_test,naivePrediction)

print(confusion_matrixx)