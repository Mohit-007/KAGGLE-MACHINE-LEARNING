

# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



dataset = pd.read_csv('train.csv')
X_train = dataset.iloc[:, 1:-1].values 
Y_train = dataset.iloc[:, 0:1].values

dataset_test = pd.read_csv('test.csv')
X_test = dataset_test.iloc[:, 0:-1].values


df_train = pd.DataFrame(X_train)
df_test = pd.DataFrame(X_test)
df_y = pd.DataFrame(Y_train)


df_test.isnull().sum()






# Fitting Naive Bayes to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, solver = "lbfgs")
classifier.fit(df_train, Y_train)
# accuracy = 97%

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(df_train, Y_train)


# Predicting the Test set results
y_pred = classifier.predict(df_test)

y_pred_0 = classifier.predict(df_test)

prediction = pd.DataFrame({"id" : X_test['id'],"target" : y_pred})
prediction.to_csv('prediction.csv',index=False)
prediction.head()
prediction = pd.read_csv('prediction.csv')

prediction_1 = pd.DataFrame({"id" : X_test['id'],"target" : y_pred_0})
prediction_1.to_csv('prediction_1.csv',index=False)
prediction_1.head()
prediction_1 = pd.read_csv('prediction_1.csv')




from sklearn.metrics import confusion_matrix
cm_1 = confusion_matrix(y_pred_0, y_pred)

cm = confusion_matrix(y_pred, y_pred_1)

