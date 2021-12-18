# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 21:30:55 2021

@author: RAMESHWAR LAL JI
"""

# import the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset
dataset = pd.read_csv('Car_Insurance_Claim.csv')
df_dataset = pd.DataFrame(dataset)

dataset['AGE'].unique()
dataset['GENDER'].unique()
dataset['RACE'].unique()        
dataset['DRIVING_EXPERIENCE'].unique()
dataset['EDUCATION'].unique()
dataset['INCOME'].unique()
dataset['VEHICLE_TYPE'].unique()
dataset['VEHICLE_YEAR'].unique()


len(dataset.columns)

dataset = dataset.iloc[:, 1:]
df_dataset = df_dataset.iloc[:, 1:]


count_1 = 0
count_0 = 0

for i in range(0, len(df_dataset)):
    if df_dataset['OUTCOME'][i] == 0.0:
        count_1 = count_1 + 1;
    else :    
        count_0 = count_0 + 1;

print(count_0)
print(count_1)

# independent and dependent variable
X = df_dataset.iloc[:, 0:-1].values
y = df_dataset.iloc[:, -1:].values

# dataframe of independent and dependent variable
df_X = pd.DataFrame(X)
df_y = pd.DataFrame(y)


        




# information of dataset
df_dataset.info()

# number of nan values in each attribute of dataset
df_dataset.isnull().sum()

# datatype of all cols of dataset
df_dataset.dtypes

# if missing value col object type then apply mode to fill missing value (A1, A4, A5, A6, A7)
# if missing value col int || float type then apply mean to fill missing value (A2, A14)



df_dataset['CREDIT_SCORE'] = df_dataset['CREDIT_SCORE'].fillna(df_dataset['CREDIT_SCORE'].astype(float).mean()) 
df_dataset['ANNUAL_MILEAGE'] = df_dataset['ANNUAL_MILEAGE'].fillna(df_dataset['ANNUAL_MILEAGE'].astype(float).mean()) 



import seaborn as sns
sns.set_style('darkgrid')
sns.countplot(x = 'OUTCOME', data = df_dataset)

import seaborn as sns
sns.set_style('darkgrid')
sns.scatterplot(x = 'A14', y = 'A16', data = df_dataset, palette = 'plasma', estimator = np.std)


df_dataset['A2'].plot(kind = 'line')
plt.xlabel('number of country')
plt.ylabel('number of immigrants')
plt.title('immigration')
plt.show()


# encoding of categorical variable


def encoding():
    df_final = df_dataset
    i=0
    for fields in df_dataset.columns:
        print(fields)
        if df_dataset[fields].dtype == 'O':
            df_1 = pd.get_dummies(df_dataset[fields] , drop_first = True)
        else:
            df_1 = df_dataset[fields].copy()
        df_dataset.drop([fields] , axis=1 , inplace=True)
        if i==0:
            df_final = df_1.copy()
        else:
            df_final = pd.concat([df_final,df_1] , axis=1)
        i = i+1
    df_final = pd.concat([df_dataset,df_final],axis=1)
    return df_final

df_data  = encoding()



# it will delete the duplicate cols if any 
df_data_new = df_data.loc[: , ~df_data.columns.duplicated()]

'''

df_1 = pd.DataFrame()
df_0 = pd.DataFrame()

for i in range(0, len(df_data_new)):
    if df_data_new['OUTCOME'][i] == 1:
        df_1 = pd.concat([df_1, df_data_new.iloc[i:i+1, :]], axis = 0)
    else:
        df_0 = pd.concat([df_0, df_data_new.iloc[i:i+1, :]], axis = 0)
        
       
dataset_2 = pd.DataFrame()
        
dataset_2 = pd.concat([dataset_2, df_1.iloc[:, :]], axis = 0)    
dataset_2 = pd.concat([dataset_2, df_0.iloc[int(len(df_0)/2)+1:int(len(df_0)), :]], axis = 0)
    

'''



# independent and dependent variable 
X = df_data.iloc[:, 0:-1].values
y = df_data.iloc[:, -1:].values

# train test split (75% 25%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from sklearn.model_selection import train_test_split
X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train, y_train, test_size = 0.25, random_state = 0)


# applying logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train_train, y_train_train)
# accuracy = 0.84 (83 + 63)/173

# y_pred vector (logistic regression)
y_pred_logistic_regression = classifier.predict(X_train_test)

'''

# accuracy matrix
from sklearn.metrics import confusion_matrix
cm_logistic_regression = confusion_matrix(y_test, y_pred_logistic_regression)


'''

# applying KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train_train, y_train_train)
# accuracy = 0.68 (76 + 42)/173

# y_pred vector (KNN)
y_pred_KNN = classifier.predict(X_train_test)


'''
# accuracy matrix
from sklearn.metrics import confusion_matrix
cm_KNN = confusion_matrix(y_test, y_pred_KNN)
'''

# applying naive bayes 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train_train, y_train_train)
# accuracy = 0.78 (84 + 52)/173

# y_pred vector (naive bayes)
y_pred_naive_bayes = classifier.predict(X_train_test)

'''
# accuracy matrix
from sklearn.metrics import confusion_matrix
cm_naive_bayes = confusion_matrix(y_test, y_pred_naive_bayes)
'''

X_final = pd.DataFrame()

X_final = pd.concat([pd.DataFrame(y_pred_logistic_regression), pd.DataFrame(y_pred_KNN), pd.DataFrame(y_pred_naive_bayes)], axis = 1)

# applying random forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_final, y_train_test)
# accuracy = 0.83 (87 + 57)/173


# y_pred vector (logistic regression)
y_pred_1 = classifier.predict(X_test)

# y_pred vector (KNN)
y_pred_2 = classifier.predict(X_test)


# y_pred vector (naive bayes)
y_pred_3 = classifier.predict(X_test)


X_final_test = pd.DataFrame()

X_final_test = pd.concat([pd.DataFrame(y_pred_1), pd.DataFrame(y_pred_2), pd.DataFrame(y_pred_3)], axis = 1)



# y_pred vector (random forest)
y_pred_random_forest = classifier.predict(X_final_test)

# accuracy matrix
from sklearn.metrics import confusion_matrix
cm_random_forest = confusion_matrix(y_test, y_pred_random_forest)

a = 547 / (547 + 247)

y_pred = y_pred_logistic_regression + y_pred_KNN + y_pred_random_forest + y_pred_naive_bayes
# accuracy = 0.86% (86 + 63)/173


for i in range(0, len(y_test)):
    if y_pred[i] >= 3:
       y_pred[i] = 1 
    elif y_pred[i] < 3:
       y_pred[i] = 0


# accuracy matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



