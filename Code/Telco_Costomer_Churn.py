#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict behavior to retain customers. You can analyze all relevant customer 
data and develop focused customer retention programs." [IBM Sample Data Sets]

Each row represents a customer, each column contains customer’s attributes 
described on the column Metadata.

The raw data contains 7043 rows (customers) and 21 columns (features).

The “Churn” column is our target.

customerIDCustomer ID
genderCustomer gender (female, male)
SeniorCitizenWhether the customer is a senior citizen or not (1, 0)
PartnerWhether the customer has a partner or not (Yes, No)
DependentsWhether the customer has dependents or not (Yes, No)
tenureNumber of months the customer has stayed with the company
PhoneServiceWhether the customer has a phone service or not (Yes, No)
MultipleLinesWhether the customer has multiple lines or not (Yes, No, No phone service)
InternetServiceCustomer’s internet service provider (DSL, Fiber optic, No)
OnlineSecurityWhether the customer has online security or not (Yes, No, No internet service)
OnlineBackupWhether the customer has online backup or not (Yes, No, No internet service)
DeviceProtectionWhether the customer has device protection or not (Yes, No, No internet service)
TechSupportWhether the customer has tech support or not (Yes, No, No internet service)
StreamingTVWhether the customer has streaming TV or not (Yes, No, No internet service)
StreamingMoviesWhether the customer has streaming movies or not (Yes, No, No internet service)
ContractThe contract term of the customer (Month-to-month, One year, Two year)
PaperlessBillingWhether the customer has paperless billing or not (Yes, No)
PaymentMethodThe customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
MonthlyChargesThe amount charged to the customer monthly
TotalChargesThe total amount charged to the customer
ChurnWhether the customer churned or not (Yes or No)

@author: esssfff
"""

# Load pandas

import pandas as pd

path = "/home/esssfff/Documents/Github/Challenges/Datasets/"

df = pd.read_csv(path+"WA_Fn-UseC_-Telco-Customer-Churn.csv")
del path

# Check the size of 
print(df.shape)

# Check the column tyoes
print(df.dtypes)

# Check if there is a null values
print(df.isnull().sum())

# Check variabe values
for value in df.columns:
    print(value, "->", df[value].unique())
del value

# Reencode variables binaraies variables
bcols = ["Partner", "Dependents", "PhoneService", "MultipleLines", 
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", 
    "StreamingTV", "StreamingMovies", "PaperlessBilling", "Churn"]

enc_bin = lambda value: 1 if value=='Yes' else 0

enc_bcols = []

for col in bcols:
    df["{}_enc".format(col)] = df[col].apply(enc_bin)
    enc_bcols.append("{}_enc".format(col))
del col

# Reencode categorical variables 
ccols = ["gender", "InternetService", "Contract", "PaymentMethod"]

import re

def pattern(text):
    pattern = re.compile(r"[a-zA-Z]+")
    m = pattern.match(text)
    return m.group(0)
    
for col in ccols:
    df[col] = df[col].apply(pattern)
del col

# Get dummie variables
enc_ccols = pd.get_dummies(df[ccols])

# concat for the final dataframe encoded
df = pd.concat([df[enc_bcols], enc_ccols], axis=1, sort=False)
del bcols, ccols, enc_bcols, enc_ccols


# Starting modeling
from sklearn.model_selection import train_test_split

# Split the date between data and target
X = df.drop("Churn_enc", axis=1)
y = df["Churn_enc"]

# split the date between train and testing
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

# Train the firt model with LogisticRegression and default parameters
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(Xtrain, ytrain)

# Score with default values
lr.score(Xtest, ytest)

# Analising important features
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(lr).fit(Xtest, ytest)
eli5.show_weights(perm, feature_names = Xtest.columns.tolist())

# Spyder doesn't have support to HTML objects so the top features are write below
"""
Weight          Feature
0.0227 ± 0.0105	Contract_Month
0.0167 ± 0.0112	Contract_Two
0.0159 ± 0.0073	InternetService_Fiber
0.0131 ± 0.0089	InternetService_No
0.0080 ± 0.0097	OnlineSecurity_enc
"""

# Analysing the importance of main features

from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

pdp_goals = pdp.pdp_isolate(model=lr, 
                            dataset=Xtest, 
                            model_features=Xtest.columns.tolist(), 
                            feature='Contract_Month')
pdp.pdp_plot(pdp_goals, 'Contract_Month')
plt.show()

pdp_goals = pdp.pdp_isolate(model=lr, 
                            dataset=Xtest, 
                            model_features=Xtest.columns.tolist(), 
                            feature='Contract_Two')
pdp.pdp_plot(pdp_goals, 'Contract_Two')
plt.show()

pdp_goals = pdp.pdp_isolate(model=lr, 
                            dataset=Xtest, 
                            model_features=Xtest.columns.tolist(), 
                            feature='InternetService_Fiber')
pdp.pdp_plot(pdp_goals, 'InternetService_Fiber')
plt.show()

pdp_goals = pdp.pdp_isolate(model=lr, 
                            dataset=Xtest, 
                            model_features=Xtest.columns.tolist(), 
                            feature='InternetService_No')
pdp.pdp_plot(pdp_goals, 'InternetService_No')
plt.show()

pdp_goals = pdp.pdp_isolate(model=lr, 
                            dataset=Xtest, 
                            model_features=Xtest.columns.tolist(), 
                            feature='OnlineSecurity_enc')
pdp.pdp_plot(pdp_goals, 'OnlineSecurity_enc')
plt.show()

# Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np

test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, y)

# Summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)

features = fit.transform(X)

# Summarize selected features
print(features[0:5,:])
del features

# Import your necessary dependencies
from sklearn.feature_selection import RFE

model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))

from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X,y)

# A helper method for pretty-printing the coefficients
def pretty_print_coefs(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name) for coef, name in lst)


print ("Ridge model:", pretty_print_coefs(ridge.coef_))

# Subsetting the data with the selected features

names = X.columns
names = list(names[fit.support_])

X = X[names]

# Import Gridsearch
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

# split the date between train and testing
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

# We set random_state=0 for reproducibility 
linear_classifier = SGDClassifier(random_state=0)

# Instantiate the GridSearchCV object and run the search
parameters = {'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1], 
             'loss':['hinge', 'log'], 'penalty':['l1', 'l2']}
searcher = GridSearchCV(linear_classifier, parameters, cv=10)
searcher.fit(Xtrain, ytrain)
ypred = searcher.predict(Xtest)

# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(ytest, ypred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(ytest, ypred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(ytest, ypred))

"""
Best CV params {'alpha': 0.001, 'loss': 'hinge', 'penalty': 'l1'}
Best CV accuracy 0.762589928057554
Accuracy: 0.7620670073821693
Precision: 0.5491803278688525
Recall: 0.5738758029978587
"""

linear_classifier = LogisticRegression(random_state=0)

# Instantiate the GridSearchCV object and run the search
parameters = {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
searcher = GridSearchCV(linear_classifier, parameters, cv=10)
searcher.fit(Xtrain, ytrain)
ypred = searcher.predict(Xtest)

# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(ytest, ypred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(ytest, ypred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(ytest, ypred))

"""
Best CV params {'C': 0.1}
Best CV accuracy 0.762589928057554
Accuracy: 0.7620670073821693
Precision: 0.5491803278688525
Recall: 0.5738758029978587
"""
from sklearn.svm import SVC

# We set random_state=0 for reproducibility 
linear_classifier = SVC(random_state=0)

# Instantiate the GridSearchCV object and run the search
parameters = {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 
             'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(linear_classifier, parameters, cv=10)
searcher.fit(Xtrain, ytrain)
ypred = searcher.predict(Xtest)

# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(ytest, ypred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(ytest, ypred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(ytest, ypred))

"""
Best CV params {'C': 1, 'gamma': 0.1}
Best CV accuracy 0.762589928057554
Accuracy: 0.7620670073821693
Precision: 0.5491803278688525
Recall: 0.5738758029978587
"""


