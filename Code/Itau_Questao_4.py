#%%
import numpy as np
from sklearn.decomposition import PCA

# Load Libraries
import pandas as pd

# Defining default Path
path = "/home/esssfff/Documents/Git/Challenges/Datasets/"

# Load Data
data = pd.read_csv(path+"dados_Q4.csv")
del path

# Initiate the PCA and fit
pca = PCA().fit(data)

# Print variance ratio of the second component
second_comp = pca.explained_variance_ratio_[1]*100
print("Variability of the seconf component {:.2f}".format(second_comp))

# Create a PCA that will retain 99% of the variance
pca = PCA(n_components=0.99, whiten=True)

# fit and transform the data
data_pca = pca.fit_transform(data)
min_comp = data_pca.shape[1]
print("Minimum components to explain 99% of the data is {:d}".format(min_comp))