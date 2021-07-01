# import the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset
dataset = pd.read_csv('credit_card_approval.csv')
df_dataset = pd.DataFrame(dataset)

df_dataset.loc[240:275]


# independent and dependent variable
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1:].values

# dataframe of independent and dependent variable
df_X = pd.DataFrame(X)
df_y = pd.DataFrame(y)

# mapping the classes of dependent variable: '+' '-' => '1' '0'  
for i in range(0, len(df_dataset)):
    if df_dataset['A16'][i] is '+':
        df_dataset['A16'][i] = 1
    else:
        df_dataset['A16'][i] = 0

df_dataset.loc[240:275]

        
# mapping the missing values of independent variable: '?' => nan
for fields in df_dataset.columns:
    for i in range(0, len(df_dataset)):
        if df_dataset[fields][i] is '?':
            df_dataset[fields][i] = np.nan
        else:
            continue





# information of dataset
df_dataset.info()

# number of nan values in each attribute of dataset
df_dataset.isnull().sum()

# datatype of all cols of dataset
df_dataset.dtypes

# if missing value col object type then apply mode to fill missing value (A1, A4, A5, A6, A7)
# if missing value col int || float type then apply mean to fill missing value (A2, A14)

df_dataset.loc[240:275]


df_dataset['A1'] = df_dataset['A1'].fillna(df_dataset['A1'].mode()[0]) 

df_dataset['A2'] = df_dataset['A2'].fillna(df_dataset['A2'].astype(float).mean()) 
df_dataset['A2'] = df_dataset['A2'].astype(float)


df_dataset['A4'] = df_dataset['A4'].fillna(df_dataset['A4'].mode()[0]) 

df_dataset['A5'] = df_dataset['A5'].fillna(df_dataset['A5'].mode()[0]) 

df_dataset['A6'] = df_dataset['A6'].fillna(df_dataset['A6'].mode()[0]) 

df_dataset['A7'] = df_dataset['A7'].fillna(df_dataset['A7'].mode()[0]) 


df_dataset['A14'] = df_dataset['A14'].fillna(df_dataset['A14'].astype(float).mean()) 
df_dataset['A14'] = df_dataset['A14'].astype(float)

import seaborn as sns
sns.set_style('darkgrid')
sns.countplot(x = 'A1', data = df_dataset)

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

df_data.loc[240:275]


# it will delete the duplicate cols if any 
df_data_new = df_data.loc[: , ~df_data.columns.duplicated()]

# train test split
train_data_percentage = 0.8
test_data_percentage = 1 - train_data_percentage
n = len(df_dataset)

# independent and dependent variable 
X = df_data.iloc[:, 0:-1].values
y = df_data.iloc[:, -1:].values

# train test split (75% 25%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# applying logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
# accuracy = 0.84 (83 + 63)/173

# y_pred vector (logistic regression)
y_pred_logistic_regression = classifier.predict(X_test)

# accuracy matrix
from sklearn.metrics import confusion_matrix
cm_logistic_regression = confusion_matrix(y_test, y_pred_logistic_regression)


# applying KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
# accuracy = 0.68 (76 + 42)/173

# y_pred vector (KNN)
y_pred_KNN = classifier.predict(X_test)

# accuracy matrix
from sklearn.metrics import confusion_matrix
cm_KNN = confusion_matrix(y_test, y_pred_KNN)


# applying naive bayes 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# accuracy = 0.78 (84 + 52)/173

# y_pred vector (naive bayes)
y_pred_naive_bayes = classifier.predict(X_test)

# accuracy matrix
from sklearn.metrics import confusion_matrix
cm_naive_bayes = confusion_matrix(y_test, y_pred_naive_bayes)

# applying decision tree 
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# accuracy = 0.84 (85 + 61)/173

# y_pred vector (decision tree)
y_pred_decision_tree = classifier.predict(X_test)

# accuracy matrix
from sklearn.metrics import confusion_matrix
cm_decision_tree = confusion_matrix(y_test, y_pred_decision_tree)

# applying random forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# accuracy = 0.83 (87 + 57)/173

# y_pred vector (random forest)
y_pred_random_forest = classifier.predict(X_test)

# accuracy matrix
from sklearn.metrics import confusion_matrix
cm_random_forest = confusion_matrix(y_test, y_pred_random_forest)



y_pred = y_pred_logistic_regression + y_pred_decision_tree + y_pred_random_forest
# accuracy = 0.86% (86 + 63)/173


for i in range(0, len(y_test)):
    if y_pred[i] >= 2:
       y_pred[i] = 1 
    elif y_pred[i] < 2:
       y_pred[i] = 0


# accuracy matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



