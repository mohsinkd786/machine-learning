import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from matplotlib import cm


titanic = sns.load_dataset('titanic')
titanic = titanic.copy()
titanic = titanic.dropna()
titanic['age'].plot.hist(
  bins = 50,
  title = "Histogram of the age variable"
)

titanic["age_zscore"] = zscore(titanic["age"])

print(titanic['age_zscore'])

titanic["is_outlier"] = titanic["age_zscore"].apply(
  lambda x: x <= -2.5 or x >= 2.5
)
print(titanic.head())
print(titanic.shape)
#print(len(titanic.columns))
# print([row for row in titanic if True == row[16]])

# titanic[titanic["is_outlier"]]

# DBSCAN — Density-Based 
# Spatial Clustering of Applications with Noise

# choose age & fare 
ageAndFare = titanic[["age", "fare"]]
ageAndFare.plot.scatter(x = "age", y = "fare")
scalar = MinMaxScaler()

# scaler
ageAndFare = scalar.fit_transform(ageAndFare)
ageAndFare = pd.DataFrame(ageAndFare, columns = ["age", "fare"])
scalar = ageAndFare.plot.scatter(x = "age", y = "fare")
plt.show()

# apply anomaly detection algo 
outlier_detection = DBSCAN(
  eps = 0.5, # distance b/w the 2 points 
  metric="euclidean",
  min_samples = 3,
  n_jobs = -1)

clusters = outlier_detection.fit_predict(ageAndFare)

print('### clusters')
print(clusters)

cmap = cm.get_cmap('Accent')
ageAndFare.plot.scatter(
  x = "age",
  y = "fare",
  c = clusters,
  cmap = cmap,
  colorbar = False
)
plt.show()

