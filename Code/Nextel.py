#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: esssfff
"""

# Importng pandas for data handling
import pandas as pd

# defining the default path of the files
path = "/home/esssfff/Documents/Github/Challenges/Datasets/"

# Load the data
df = pd.read_csv(path+"/data_scientist_test/house_sales.csv")

# Check the data
print(df.describe())

# Check the size of 
print(df.shape)

# Check the column tyoes
print(df.dtypes)

# Check if there is a null values
print(df.isnull().sum())

# Analysing the price column
print(df['price'].describe())

# Import librarie for visualization
import matplotlib.pyplot as plt

# the histogram of the price column
plt.hist(df['price'], bins=100, normed=1, alpha=0.75)
plt.xlabel('Price')
plt.ylabel('Counts')
plt.title('Histogram of Price Houses')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Sampling the data
# Defing steps for each range
blocks = 10000
intersect = (df['price'].max() - df['price'].min()) / blocks

# Create a list with each ranges
prange = []
for value in range(0, blocks+1):
    prange.append(df['price'].min() + value * intersect)
del blocks, intersect, value

sample_df = pd.DataFrame()
sample_size = 0.1

from itertools import tee

# function from itertools https://docs.python.org/3/library/itertools.html
# function picks pair of items in a list
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

for start, end in pairwise(prange):
    try:
        sample_df = sample_df.append(
            df[df['price'].between(start, end)].sample(
                frac=sample_size, replace=False, random_state=0))
    except:
        pass
del prange, sample_size, start, end

# Analysing the prince columns after sampling
plt.hist(sample_df['price'], bins=100, normed=1, alpha=0.75)
plt.xlabel('Price')
plt.ylabel('Counts')
plt.title('Histogram of Price Houses')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Starting modeling
from sklearn.model_selection import train_test_split

# Split the date between data and target
X = sample_df.drop("price", axis=1)
y = sample_df["price"]

# split the date between train and testing
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)

# Train the firt model with LogisticRegression and default parameters
from sklearn.linear_model import LinearRegression

lr = LinearRegression(fit_intercept=False)
lr.fit(Xtrain, ytrain)

# Score with default values
print(lr.score(Xtest, ytest))

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(lr).fit(Xtest, ytest)
eli5.show_weights(perm, feature_names = Xtest.columns.tolist())

# Analysing the importance of main features
from pdpbox import pdp

pdp_goals = pdp.pdp_isolate(model=lr, 
                            dataset=Xtest, 
                            model_features=Xtest.columns.tolist(), 
                            feature='size_house')
pdp.pdp_plot(pdp_goals, 'size_house')
plt.show()

pdp_goals = pdp.pdp_isolate(model=lr, 
                            dataset=Xtest, 
                            model_features=Xtest.columns.tolist(), 
                            feature='latitude')
pdp.pdp_plot(pdp_goals, 'latitude')
plt.show()

pdp_goals = pdp.pdp_isolate(model=lr, 
                            dataset=Xtest, 
                            model_features=Xtest.columns.tolist(), 
                            feature='num_bed')
pdp.pdp_plot(pdp_goals, 'num_bed')
plt.show()
del pdp_goals

# Modeling with Decision Tree
X = sample_df.drop("price", axis=1)
y = sample_df["price"]

# split the date between train and testing
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)

# Import DecistionTreeClasifier
from sklearn.tree import DecisionTreeRegressor

# We set random_state=0 for reproducibility 
tree = DecisionTreeRegressor(random_state=0)

# fit and predict
tree.fit(Xtrain, ytrain)
ypred = tree.predict(Xtest)

# Model Accuracy: how often is the classifier correct?
print("Accuracy on training set: {:.3f}".format(tree.score(Xtrain, ytrain))) 
print("Accuracy on test set: {:.3f}".format(tree.score(Xtest, ytest)))

# Modeling with Random Forest
X = sample_df.drop("price", axis=1)
y = sample_df["price"]

# split the date between train and testing
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)

# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

# We set random_state=0 for reproducibility 
forest = RandomForestRegressor(random_state=0)

# fit and predict
forest.fit(Xtrain, ytrain)
ypred = forest.predict(Xtest)

# Model Accuracy: how often is the classifier correct?
print("Accuracy on training set: {:.3f}".format(forest.score(Xtrain, ytrain))) 
print("Accuracy on test set: {:.3f}".format(forest.score(Xtest, ytest)))

# Feature Selection
from sklearn.feature_selection import RFE

forest = RandomForestRegressor(random_state=0)
rfe = RFE(forest, n_features_to_select=1).fit(X, y)

print("Num Features: %s" % (rfe.n_features_))
print("Selected Features: %s" % (rfe.support_))
print("Feature Ranking: %s" % (rfe.ranking_))

print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), X.columns)))

# Recursive Feature Elimination based on feature ranking
acc = []
rank = []

for value in range(1,max(rfe.ranking_)):
    
    # Split the date between data and target
    X = sample_df.drop("price", axis=1)
    y = sample_df["price"]
    
    features = rfe.ranking_ <= value

    names = X.columns
    names = list(names[features])
    X = X[names]

    # split the date between train and testing
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

    forest = RandomForestRegressor(random_state=0)
    forest.fit(Xtrain, ytrain)

    # Score with default values
    acc.append(forest.score(Xtest, ytest))
    rank.append(value)

del value, names, features

plt.plot(rank, acc)
plt.xlabel("Ranking")
plt.xlim(xmin = 0, xmax = max(rfe.ranking_))
plt.ylabel("Accuracy")
plt.ylim(ymin = 0, ymax = 1)
plt.title("Recursive Feature Elimination")
plt.show()

pd.merge(pd.DataFrame({"Features": sample_df.drop("price", axis=1).columns, 
    "Ranking": rfe.ranking_}), pd.DataFrame({"Accuracy": acc, 
    "Ranking": rank})).sort_values(by=['Ranking'])

frank = [x for x, y in zip(rank, acc) if y == max(acc)][0]
selected_features = list(X.columns[rfe.ranking_ <= frank])
print(selected_features)
    
del acc, rank

# Model tunning
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Defining datasets
X = sample_df.drop("price", axis=1)
X = X[selected_features]
y = sample_df["price"]

# We set random_state=0 for reproducibility
forest = RandomForestRegressor(random_state=0)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf_random = RandomizedSearchCV(estimator = forest, 
    param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, 
    random_state=42, n_jobs = -1)

rf_random.fit(Xtrain, ytrain)
xpred = rf_random.predict(Xtrain)
ypred = rf_random.predict(Xtest)

# Report the best parameters and the corresponding score
print("Best CV params", rf_random.best_params_)
print("Best CV accuracy", rf_random.best_score_)