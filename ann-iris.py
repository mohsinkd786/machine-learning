import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Location of dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
# Read dataset to pandas dataframe
irisdata = pd.read_csv(url, names=names) 
#print(irisdata.head())

# Assign data from first four columns to X variable
X = irisdata.iloc[:, 0:4]
# Assign data from first fifth columns to y variable
#y = irisdata.iloc[:,-1]
y = irisdata.select_dtypes(include=[object])  
#print(y.head())

print(y.Class.unique())

# pre processor for label encoder 
# converting names to numbers
le = preprocessing.LabelEncoder()

y = y.apply(le.fit_transform)
#print(y)

# splitting into training & test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) 

# applying a scaler before the predictions are done
# so as to have uniformity in features
# scaling is only done of training data so as to ensure 
# test data is as real as possible

scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

# apply the multi layer neural network classifier
# hidden_layer_sizes specifies size of hidder layers
# (10,10,10) shall create 3 hidden layers with 10 nodes each
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)  
mlp.fit(X_train, y_train.values.ravel())

# apply predictions
predictions = mlp.predict(X_test)  

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
