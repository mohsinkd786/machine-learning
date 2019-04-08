import pandas as pd
import numpy as np
import math
import operator
#https://gist.githubusercontent.com/gurchetan1000/ec90a0a8004927e57c24b20a6f8c8d35/raw/fcd83b35021a4c1d7f1f1d5dc83c07c8ffc0d3e2/iris.csv
data = pd.read_csv("https://gist.githubusercontent.com/gurchetan1000/ec90a0a8004927e57c24b20a6f8c8d35/raw/fcd83b35021a4c1d7f1f1d5dc83c07c8ffc0d3e2/iris.csv")
data.head()

def euclideanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)

def knn(trainingSet, testInstance, k):
 
    distances = {}
    sort = {}
 
    length = testInstance.shape[1]
    
    for x in range(len(trainingSet)):
        
        #### Start of STEP 3.1
        dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)

        distances[x] = dist[0]
   
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))
   
    neighbors = []
    
    for x in range(k):
        neighbors.append(sorted_d[x][0])
    classVotes = {}
    
    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]
 
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return(sortedVotes[0][0], neighbors)

testSet = [[7.2, 3.6, 5.1, 2.5]]
test = pd.DataFrame(testSet)
k = 1

# Running the Model
result,neighbor = knn(data, test, k)

# predicted class
print(result)
# nearest neighbor
print(neighbor)

# incase of 
# k = 3 
# Running KNN model 
# result,neighbor = knn(data, test, k) 
# Predicted class 
# print(result)
# print(neighbor)

# Using sklearn
# from sklearn.neighbors import KNeighborsClassifier
# neighbor = KNeighborsClassifier(n_neighbors=3)
# neighbor.fit(data.iloc[:,0:4], data['Name'])

# Predicted class
# print(neighbor.predict(test))
# 3 nearest neighbors
# print(neighbor.kneighbors(test)[1])

