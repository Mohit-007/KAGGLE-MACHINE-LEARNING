import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('mushrooms.csv')
df_dataset = pd.DataFrame(dataset)

X_train = dataset.iloc[:, 1:-1].values
y_train = dataset.iloc[:, 0:1].values

print(df_dataset['class'])

for i in range(0,8124):
    if df_dataset['class'][i] is 'p':
        df_dataset['class'][i] = 0
    else:
        df_dataset['class'][i] = 1
        

y_train = df_dataset['class']

X_train = dataset.iloc[:, 1:].values

df_dataset.info()

df_dataset.isnull().sum()

df_dataset.dtypes

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


df_data_new = df_data.loc[: , ~df_data.columns.duplicated()]

train_data_percentage = 0.8
test_data_percentage = 1 - train_data_percentage
n = len(df_dataset)


df_train = df_data.iloc[:int(n*train_data_percentage), :].values
df_test = df_data.iloc[int(n*train_data_percentage): , :].values

y = df_dataset['class']


y_train = y[:int(n*train_data_percentage) , ]
y_test = y[int(n*train_data_percentage): , ]

y_train = y_train.astype('int')
y_test = y_test.astype('int')

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(df_train, y_train)
# accuracy = 0.998 (1115 + 507)/1625

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(df_train, y_train)
# accuracy = 0.984 (1118 + 482)/1625

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(df_train, y_train)
# accuracy = 0.100 (1118 + 507)/1625

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(df_train, y_train)
# accuracy = 0.990 (1118 + 491)/1625

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(df_train, y_train)
# accuracy = 0.100 (1118 + 507)/1625

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(df_train, y_train)
# accuracy = 0.100 (1118 + 507)/1625

y_pred = classifier.predict(df_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
