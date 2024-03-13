import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import cKDTree
import pandas 


#Data extraction from file

dataframe = pandas.read_csv('/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data.csv')

features = ['GroundTruth_X', 'Ground_Truth_Y', 'Ground_Truth_Yaw', 'Velocity_Linear_X', 'Velocity_Linear_Y', 'Velocity_Angular_Yaw']
target = ['Delta_X_X', 'Delta_X_Y', 'delta_X_Yaw']

X = dataframe[features].values
Y = dataframe[target].values

range_n_clusters = list(range(2, 20))
silhouette_scores = []

for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, init='k-means++', random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f'For n_clusters = {n_clusters} the average silhouette_score is: {silhouette_avg}')

# Plotting silhouette scores for different values of 'n_clusters'
best_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
print(f'Best number of clusters: {best_n_clusters}')

clusters = best_n_clusters

kmeans = KMeans(n_clusters=clusters, init='k-means++', n_init='auto').fit(X)

labels = kmeans.labels_

centers = kmeans.cluster_centers_

# For each cluster, select a sample (e.g., the one closest to the centroid)
samples = []
for i in range(clusters):
    
    cluster_points = X[labels == i]
    
    tree = cKDTree(cluster_points)

    k_nearest = 5

    _, idx = tree.query(centers[i], k=k_nearest)

    samples.append(cluster_points[idx])


samples = np.array(samples).reshape(-1, X.shape[1])

print("Selected samples from each cluster:\n", samples)

print(kmeans.cluster_centers_)