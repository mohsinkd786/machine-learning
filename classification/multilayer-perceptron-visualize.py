import matplotlib.pyplot as plt
# sample data
from sklearn.datasets import fetch_openml
# multi layer perceptron classifier
from sklearn.neural_network import MLPClassifier
import pandas as pd

# https://www.openml.org/d/554
# name dataset - mnist_784

#X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
#X = X / 255

data = pd.read_csv('mnist.csv')
print(data.head())
X = data[:-1]
y = data[]
#X_train, X_test = X[:60000], X[60000:]
#y_train, y_test = y[:60000], y[60000:]

#mlp = MLPClassifier()
#mlp.fit(X_train,y_train)

#print(mlp.score(X_train,y_train))


