# Load Libraries
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("dados_Q1.csv")

# Defining target variable
target = "target"

# Defining Training variables
X = data.drop(target, axis=1)
y = data[target]

# Defining Kfolds
kf = KFold(n_splits=10, random_state=0)

# Initialize the list for validation
train = []
test = []

# Split the data between train and test 
for train_index, test_index in kf.split(X):
   X_train, X_test = X.loc[train_index], X.loc[test_index]
   y_train, y_test = y.loc[train_index], y.loc[test_index]

   tree = DecisionTreeClassifier(random_state=0, 
                                 criterion="entropy").fit(X_train, y_train)

   train.append(tree.score(X_train, y_train))
   test.append(tree.score(X_test, y_test))

print("Average mean Score on train is {:.2f}".format(np.mean(train)))
print("Average mean Score on test is {:.2f}".format(np.mean(test)))