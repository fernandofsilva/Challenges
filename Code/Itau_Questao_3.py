# import numpy as np
import pandas as pd

from scipy import stats
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("dados_Q3.csv")

# Values less than 0.001 will be equal to 0
data = data.where(data > 0.001, 0)

# Defining target variable
target = "target"

# Defining X and y data
X = data.drop(target, axis=1)
y = data[target]

# Spliting the data between train and test
X_size, y_size  = X.shape[0] / 2,    y.shape[0] / 2
X_train, X_test = X.loc[0:X_size-1], X.loc[X_size:]
y_train, y_test = y.loc[0:y_size-1], y.loc[y_size:]

# Initiate the model and fiting
clf = LogisticRegression(penalty='l1', 
                        C=0.1, 
                        solver="liblinear").fit(X_train, y_train)
# Calculate F1 Score
f1 = f1_score(y_test, clf.predict(X_test))

# Print the result
print("F1 Score is {:.2f}".format(f1))

# Create a dataframe of the variable and their coeficients
df = pd.DataFrame({"variable": X_train.columns,
                   "coef"    : clf.coef_[0]}).sort_values(by="coef", 
                                                          ascending=False)
# Filter the significant coeficients
variables = df.loc[df["coef"] > 0.10]

# Print the results
print(variables["variable"].tolist())