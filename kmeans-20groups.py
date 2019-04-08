import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import HashingVectorizer,TfidfVectorizer
#from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
newsgroups =fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True)
#print(newsgroups.data[0])
data = pd.read_csv('20_newsgroup.csv')
#print(data.tail())

#vectorizer = HashingVectorizer()
vectorizer = TfidfVectorizer()
#X = vectorizer.fit_transform(newsgroups.data)
X = vectorizer.fit_transform(data)

print(X.shape)

# apply k means clustering
kmeans = KMeans(n_clusters = 5)
#kfit = kmeans.fit(X)
#PErceptron

# print(kfit)
# prediction
#kprediction = kmeans.predict(X)

#print(kfit)
#print(kprediction)