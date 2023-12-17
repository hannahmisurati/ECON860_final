# import the relevant packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np

# suppress warnings
import warnings 
warnings.filterwarnings('ignore')


#define the dataset
dataset = pd.read_csv("c_results3.csv")
dataset = dataset.values
dataset = dataset[:,0:3]


# fit multiple k-means algorithms and store the values in an empty list to determine optimal number of clusters with help from internet (analytics vidyha)
SSE = []
for cluster in range(1,8):
    kmeans = KMeans(n_clusters = cluster, init='k-means++')
    kmeans.fit(dataset)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,8), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# use the elbow method to determine the optimum number of clusters to cross reference with above, and that value was 2 
kmeans = KMeans(n_clusters = 2, init='k-means++', random_state=1)
kmeans.fit(dataset)
pred_kmeans = kmeans.predict(dataset)

# value count of points 
frame = pd.DataFrame(dataset)
frame['cluster'] = pred_kmeans
print(frame['cluster'].value_counts())
print(silhouette_score(dataset, pred_kmeans, metric="euclidean"))



# scatter plot with cluster centroids
centroids = kmeans.cluster_centers_
plt.scatter(dataset[:, 0], dataset[:, 1], c=pred_kmeans)
plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="*", s=300)
plt.title('K-Means Clustering with Centroids')
plt.savefig("scatterplot_kmeans_with_centroids.png")
plt.show()


kmedo = KMedoids(n_clusters = 2, random_state=1)
kmedo.fit(dataset)

pred_kmedo = kmedo.predict(dataset)

#value count of points 
frame = pd.DataFrame(dataset)
frame['cluster'] = pred_kmedo
print(frame['cluster'].value_counts())
print('\n')
print(silhouette_score(dataset, pred_kmedo, metric="euclidean"))












