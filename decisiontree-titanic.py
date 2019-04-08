import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from subprocess import call

df = pd.read_csv('titanic.csv', index_col='PassengerId')
print(df.head())

# The root node (the first decision node)
# partitions the data based on the most 
# influential feature partitioning
# There are 2 measures for this,
# Gini is Impurity
# and Entropy for Information gain

# The root node (the first decision node) partitions
# the data using the feature that provides
# the most information gain.

# Entropy
# Information gain tells us how important
# a given attribute of the feature vectors is.

# It is calculated as:

# Information Gain=entropy(parent)–[average entropy(children)]
# Where entropy is a common measure of target class impurity,
# given as:

# Entropy=Σi–pilog2pi
# where i is each of the target classes

# Gini Impurity
#Gini Impurity is another measure of impurity
# and is calculated as follows:

# Gini=1–Σip2i
# Gini impurity is computationally faster
# as it doesn’t require
# calculating logrithmic functions,
# though in reality which of the two methods
# is used rarely makes too much of a difference

# choose the specific columns
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]

# convert sex to integer
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# drop rows with incomplete / missing values 
df = df.dropna()

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=11)

# print(X_train)
# decision tree
model = DecisionTreeClassifier()
treeModel = model.fit(X_train, y_train)

prediction = model.predict(X_test)

a_score = accuracy_score(y_test, prediction)

print(a_score)

pd.DataFrame(
    confusion_matrix(y_test, prediction),
    columns=['Predicted Not Survival', 'Predicted Survival'],
    index=['True Not Survival', 'True Survival']
)

# http://www.graphviz.org/
# convert as a tree / node based
export_graphviz(treeModel, out_file='titanic.dot', feature_names=X.columns)

# convert dot to png
call(['dot', '-T', 'png', 'titanic.dot', '-o', 'titanic.png'])

# dot -Tpng titanic.dot -o titanic.png

# The impurity is the measure as given by Gini
# the samples are the number of observations
# remaining to classify and the value is the 
# how many samples are in class 0 (Did not survive)
# and how many samples are in class 1 (Survived)

# Let’s follow this part of the tree down,
# the nodes to the left are True and the
# nodes to the right are False:

# We see that we have 19 observations left
# to classify: 9 did not survive and 10 did

# From this point the most information gain
# is how many siblings (SibSp) were aboard.
# 9 out of the 10 samples with less than 2.5
# siblings survived.
# This leaves 10 observations left,
# 9 did not survive and 1 did.
# 6 of these children that only had one parent (Parch)
# aboard did not survive.
# None of the children aged > 3.5 survived
# Of the 2 remaining children,
# the one with > 4.5 siblings did not survive