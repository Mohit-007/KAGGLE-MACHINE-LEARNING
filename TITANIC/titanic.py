# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt # (it plot the graph)
import pandas as pd             # (it import the data)

dataset = pd.read_csv('train.csv')
X_train = dataset.iloc[:, [2,3,4,5,6,7,9,8,10,11]].values 
Y_train = dataset.iloc[:, 1].values

dataset_test = pd.read_csv('test.csv')
X_test = dataset_test.iloc[:, [1,2,3,4,5,6,8,7,9,10]].values


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:, [3]])
X_train[:, [3]] = imputer.transform(X_train[:, [3]])

from sklearn.preprocessing import Imputer
imputer_test = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer_test = imputer_test.fit(X_test[:, [3]])
X_test[:, [3]] = imputer_test.transform(X_test[:, [3]])

from sklearn.preprocessing import Imputer
imputer_test = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer_test = imputer_test.fit(X_test[:, [6]])
X_test[:, [6]] = imputer_test.transform(X_test[:, [6]])


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:, 1] = labelencoder_X.fit_transform(X_train[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1] )
X_train = onehotencoder.fit_transform(X_train).toarray()



X_test[:,[1,2]] = pd.get_dummies(X_test[:, 2])
X_train[:, [1,2]] = pd.get_dummies(X_train[:, 2])

X_test[:,[7,8,9]] = pd.get_dummies(X_test[:, 9])
X_train[:, [7,8,9]] = pd.get_dummies(X_train[:, 9])

df_test = pd.DataFrame(X_test)
df_train = pd.DataFrame(X_train)
df_dataset = pd.DataFrame(dataset)


from sklearn.preprocessing import StandardScaler
sc_X  = StandardScaler()
df_train = sc_X.fit_transform(df_train)
df_test = sc_X.transform(df_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(df_train, Y_train)

# Predicting the Test set results
np.any(np.isnan(df_test))
np.all(np.isfinite(df_test))
np.where(np.isnan(df_test))
y_pred = classifier.predict(df_test)


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = df_train, y = Y_train, cv = 10)
accuracies.mean()
accuracies.std()

from sklearn.decomposition import PCA
pca = PCA(n_components = 10)
df_train = pca.fit_transform(df_train)
df_test = pca.transform(df_test)
explained_variance = pca.explained_variance_ratio_


prediction = pd.DataFrame({"PassengerId" : dataset_test['PassengerId'],"Survived" : y_pred})
prediction.to_csv('prediction.csv',index=False)
prediction.head()
prediction = pd.read_csv('prediction.csv')