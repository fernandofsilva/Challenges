#%%
# Load Libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error

# Defining default Path
path = "/home/esssfff/Documents/Git/Challenges/Datasets/"

data = pd.read_csv(path+"dados_Q2.csv")

# Defining target variable
target = "target"

# Defining Training variables
X = data.drop(target, axis=1)
y = data[target]

# Defining leave one out
loo = LeaveOneOut()

# Initialize the list for the validation
train = []
test = []

# Split the data between train and test 
for train_index, test_index in loo.split(X):
   X_train, X_test = X.loc[train_index], X.loc[test_index]
   y_train, y_test = y.loc[train_index], y.loc[test_index]

   svm = SVR(kernel="linear", C=0.1).fit(X_train, y_train)

   pred_train = svm.predict(X_train)
   pred_test = svm.predict(X_test)

   train.append(mean_absolute_error(y_train, pred_train))
   test.append(mean_absolute_error(y_test, pred_test))

print("Average MAE on train is {:.2f}".format(np.mean(train)))
print("Average MAE on test is {:.2f}".format(np.mean(test)))