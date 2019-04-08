# list of libraries
# pip install numpy
# pip install scipy
# pip install scikit-learn
# pip install pandas
# pip install matplotlib
import sys
import scipy
import numpy
import pandas
import matplotlib
import sklearn

print('Python',sys.version)
print('scipy',scipy.__version__)
print('numpy',numpy.__version__)
print('sklearn',sklearn.__version__)
print('pandas',pandas.__version__)
print('matplot',matplotlib.__version__)


# plotting
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

# classification
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
# algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['s-length','s-width','p-length','p-width','flower']
dataset = pandas.read_csv(url,names=columns)
# gives the list of rows & columns
#print(dataset.shape)

# view the data / get top 50 rows with column infos
#print(dataset.head(50))

# getting statistical info / aggregate values + statistical data
#print(dataset.describe())

# class distribution / categorization
#print(dataset.groupby('flower').size())

# visualizing  / plots
# box plot
dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
# matplot
#plt.show()

# variance plots
# univariace
# dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
# matplot
# plt.show()

# histogramic view // gaussain distribution
#dataset.hist()
#plt.show() 

# 1. validating the data
# 2. Training the Model
# 3. Testing the Model
# 4. Identify the accuracy
# 100 how much %age we can keep for validation / reference data

array = dataset.values
x = array[:0,4] # variables
# push data data  create the subset / create a new series
x = array[:,0:4]  # values to be training
y = array[:,4] # variable

# print(y)
# how 
v_limit = 0.10
seed = 10
x_train,x_val,y_train,y_val = model_selection.train_test_split(
    + x,y,test_size = v_limit,random_state = seed)

# print(y_train)

# create Models / define models
models = []
models.append(('LR', LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('SVM',SVC(gamma ='auto')))
models.append(('KNN',KNeighborsClassifier()))

# result acheived via applying the algo
#result_data = []
#algos = []
#_scoring = 'accuracy'
# traverse via the models
for algoName,algoMethod in models:
    kf = model_selection.KFold(n_splits = 5, random_state = 5)
    result = model_selection.cross_val_score(algoMethod,x_train,y_train,cv = kf)
    print(result)
    #print('Mean is ',result.mean())


# define predictions 
knn = KNeighborsClassifier()

# training data set
knn.fit(x_train,y_train)
# validation data set
predictions = knn.predict(x_val)
#print(predictions)
map_predict_validation = list(zip(predictions,y_val))
for v in map_predict_validation:
    print(v)

# generate classification report with averages & predicts
#print(classification_report(y_val,predictions))

#print('Argument 1 ',sys.argv[1])
#print('Argument 2 ',sys.argv[2])

# true positive : no of possibilities in the validation data (t)
# false positive : no of possibilities in the predicted data (f)

val1 = input("Please enter the Ist value: ")
val2 = input("Please enter the 2nd value: ")
val3 = input("Please enter the 3rd value: ")
val4 = input("Please enter the 4th value: ")
z_vals = list([float(val1),float(val2),float(val3),float(val4)])
vals = [z_vals]

# predictions
for algoName,algoMethod in models:
    algoMethod.fit(x_train,y_train)
    custom_predictions = knn.predict(vals)
    print(algoName,custom_predictions)
    #custom_map_predict_validation = list(zip(custom_predictions,y_val))
