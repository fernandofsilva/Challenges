#%%
# Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Defining default Path to read the data
path = "/home/esssfff/Documents/Git/Challenges/Datasets/"

# Read data
data = pd.read_csv(path+"dados_Q5.csv")

# Convert the dato the a distance Matrix
distanceMatrix = pdist(data)

# Run the algorithm
comp_link = linkage(distanceMatrix, 'complete')

# Make a cut in the cluster in the distance 2.3
assignments = fcluster(comp_link, 2.3, 'distance')

# Function source:
# https://stackoverflow.com/questions/9362304/how-to-get-centroids-from-scipys-hierarchical-agglomerative-clustering
def to_codebook(X, part):
    """
    Calculates centroids according to flat cluster assignment

    Parameters
    ----------
    X : array, (n, d)
        The n original observations with d features

    part : array, (n)
        Partition vector. p[n]=c is the cluster assigned to observation n

    Returns
    -------
    codebook : array, (k, d)
        Returns a k x d codebook with k centroids
    """
    codebook = []

    for i in range(part.min(), part.max()+1):
        codebook.append(X[part == i].mean(0))

    return np.vstack(codebook)

centroids = to_codebook(data, assignments)

print(centroids)

# Plot a dendogram and check the results
dend = dendrogram(comp_link)
plt.show()

#%%
# Import Kmeans
from sklearn.cluster import KMeans

# Initiate the model and fiting
clf = kmeans = KMeans(n_clusters=5,
                      init=centroids,
                      n_init=1,
                      random_state=0, 
                      max_iter=10).fit(data)

# Clusters Centers
print(kmeans.cluster_centers_)