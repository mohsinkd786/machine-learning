import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

cancerData = pd.read_csv('breast-cancer.csv')
#print(cancerData)
features = cancerData.drop(columns=['diagnosis'])
labels = pd.DataFrame ({'diagnosis':cancerData['diagnosis']})

#A = pd.DataFrame ({'radius': cancerData['radius_mean'] , 'texture': cancerData['texture_mean'] , 'perimeter' : cancerData['perimeter_mean'] , 'area' : cancerData['area_mean'] , 'smoothness' : cancerData['smoothness_mean'] ,'compactness' : cancerData['compactness_mean'],'concavity' : cancerData['concavity_mean'], 'convave' : cancerData['concave_points_mean'] ,'fractal' : cancerData['fractal_dimension_mean']})
#B = pd.DataFrame ({'diagnosis': cancerData['diagnosis']})
#print(A)
#print(B)
features_training,features_test, labels_training,labels_test = train_test_split(features,labels,test_size=0.20,random_state = 11)

#print(features_training)
#print(A_test)
#print(B_training)
#print(B_test)

k = 11
kclassifier = KNeighborsClassifier(n_neighbors = k)
model = kclassifier.fit(features_training,labels_training)
print(model)

prediction = kclassifier.predict(features_test)

print(prediction)

# classification report
#classificatn_report = classification_report(B_test,prediction)

#confusion_matrixx = confusion_matrix(B_test,prediction)

#print(classificatn_report)
#print(confusion_matrixx)