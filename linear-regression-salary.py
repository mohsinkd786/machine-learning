# Import the necessary libraries
import numpy
import matplotlib.pyplot as plot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('salaries.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 10)

linearRegressor = LinearRegression()
linearRegressor.fit(xTrain, yTrain)

# Predicting the test set results
yPrediction = linearRegressor.predict(xTest)

# Visualising the training set results
#plot.scatter(xTrain, yTrain, color = 'red')
#plot.plot(xTrain, linearRegressor.predict(xTrain), color = 'blue')
#plot.title('Salary vs Experience (Training set)')
#plot.xlabel('Years of Experience')
#plot.ylabel('Salary')
#plot.show()

# Visualising the test set results
#plot.scatter(xTest, yTest, color = 'red')
#plot.plot(xTrain, linearRegressor.predict(xTrain), color = 'blue')
#plot.title('Salary vs Experience (Test set)')
#plot.xlabel('Years of Experience')
#plot.ylabel('Salary')
#plot.show()