import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

cancerData = pd.read_csv('breast-cancer.csv')
#print(cancerData)
A = cancerData.drop(columns=['diagnosis'])
B = pd.DataFrame ({'diagnosis':cancerData['diagnosis']})

#A = pd.DataFrame ({'radius': cancerData['radius_mean'] , 'texture': cancerData['texture_mean'] , 'perimeter' : cancerData['perimeter_mean'] , 'area' : cancerData['area_mean'] , 'smoothness' : cancerData['smoothness_mean'] ,'compactness' : cancerData['compactness_mean'],'concavity' : cancerData['concavity_mean'], 'convave' : cancerData['concave_points_mean'] ,'fractal' : cancerData['fractal_dimension_mean']})
#B = pd.DataFrame ({'diagnosis': cancerData['diagnosis']})
#print(A)
#print(B)
A_training,A_test, B_training,B_test = train_test_split(A,B,test_size=0.20,random_state = 11)

#print(A_training)
#print(A_test)
#print(B_training)
#print(B_test)

k = 5
kclassifier = KNeighborsClassifier(n_neighbors = k)
kclassifier.fit(A_training,B_training)

prediction = kclassifier.predict(A_test)

#print(prediction)

# classification report
classificatn_report = classification_report(B_test,prediction)
confusion_matrixx = confusion_matrix(B_test,prediction)

print(classificatn_report)
print(confusion_matrixx)