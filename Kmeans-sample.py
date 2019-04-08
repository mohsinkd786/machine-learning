# k means 
import numpy as np
import random 
import scipy.stats as stat
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.cluster import KMeans

cancerData = pd.read_csv('classification/breast-cancer.csv')
A = cancerData.drop(columns=['diagnosis'])
A_arr = A.values
#B = pd.DataFrame ({'diagnosis':cancerData['diagnosis']})

# clasifcation we have both the data & classifying labels
# in clustering we only have data from which we need to classify

kmeans = KMeans(n_clusters = 2)

kfit = kmeans.fit(cancerData)
score = kfit.score(cancerData)
#print(kfit.score(B))
print(kfit.labels_)

#print(kmeans.fit_transform(A))

# idenfication of labels 
prediction = kfit.predict(cancerData)

print(prediction.values)

#print(prediction)
collaborated = np.concatenate(A_arr,prediction.values,axis=0)
print(collaborated)

#centroids = kmeans.cluster_centers_

#graph = plt.figure(figsize=(10,10))
#colors = map(lambda x: colmap[x+1],kfit.labels_)
# scatter 
#plt.scatter(A,B,color=colors)

#for ids,centroid in enumerate(centroids):
#    plt.scatter(*centroid,color=colmap[ids+1])
#plt.xlim(0,100)
#plt.ylim(0,100)
#plt.show()


#pca = PCA(n_components =1).fit(B)
#pca_d = pca.transform(B)
#pca_c = pca.transform(A)
#kmeans = KMeans(n_clusters=2)
#output_ =kmeans.fit(B)

#plt.plot(pca_c[:,0],pca_d[:,0],c=output_.labels_)
#plt.xlabel('Number of clusters')
#plt.ylabel('score')
#plt.show()