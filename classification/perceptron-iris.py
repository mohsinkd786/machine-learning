import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
cols = ['slength','swidth','plength','pwidth','flower']
irisData = pd.read_csv(url,names=cols)
A = irisData.iloc[:,:-1].values
# B with values for flower
B = irisData.iloc[:,4].values

#print(irisData.head())
# classification
perceptron = Perceptron(max_iter=20,tol=0.20)
perceptron.fit(A,B)

print(perceptron)
print(perceptron.score(A,B))

A_training,A_test,B_training,B_test = train_test_split(A,B,test_size = 0.20)
# after traing test
perceptron.fit(A_training,B_training)

print(perceptron)
print(perceptron.score(A_training,B_training))