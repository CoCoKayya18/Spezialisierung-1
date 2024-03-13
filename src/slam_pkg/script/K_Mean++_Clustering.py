import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
import pandas 


#Data extraction from file

dataframe = pandas.read_csv('/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data.csv')

features = ['GroundTruth_X', 'Ground_Truth_Y', 'Ground_Truth_Yaw', 'Velocity_Linear_X', 'Velocity_Linear_Y', 'Velocity_Angular_Yaw']
target = ['Delta_X_X', 'Delta_X_Y', 'delta_X_Yaw']

X = dataframe[features].values
Y = dataframe[target].values

# Standarize the data and save to a csv


scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_standardized = scaler_X.fit_transform(X)
Y_standardized = scaler_Y.fit_transform(Y) 

standardizedData = np.hstack((X_standardized, Y_standardized))
standardizedColumns = features + target
standardizedDataFrame = pandas.DataFrame(standardizedData, columns=standardizedColumns)
csv_path = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/StandardizedData.csv'
standardizedDataFrame.to_csv(csv_path, index=False)


range_n_clusters = list(range(2, 40))
silhouette_scores = []
silhouette_scores_standardized = []

pandas.DataFrame()

for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, init='k-means++', random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f'In normal X: For n_clusters = {n_clusters} the average silhouette_score is: {silhouette_avg}')

for n_clusters_standardized in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters_standardized, init='k-means++', random_state=10)
    cluster_labels_standardized = clusterer.fit_predict(X_standardized)
    silhouette_avg_standardized = silhouette_score(X_standardized, cluster_labels_standardized)
    silhouette_scores_standardized.append(silhouette_avg_standardized)
    print(f'In Standardized X: For n_clusters = {n_clusters_standardized} the average silhouette_score is: {silhouette_avg_standardized}')

# Plotting silhouette scores for different values of 'n_clusters'
best_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
print(f'Best number of clusters: {best_n_clusters}')

best_n_clusters_standardized = range_n_clusters[np.argmax(silhouette_scores_standardized)]
print(f'Best number of clusters (standardized): {best_n_clusters_standardized}')

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