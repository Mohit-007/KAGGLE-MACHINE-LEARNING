import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



class LinearRegression:

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
 

    def predict(self, X):
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated



# Importing the dataset
dataset = pd.read_csv('train.csv')
df_dataset = pd.DataFrame(dataset)
X_train = dataset.iloc[:, :-1].values
y_train = dataset.iloc[:, -1].values
X_test = pd.read_csv('test.csv').iloc[:, :].values

df_test = pd.DataFrame(X_test)
df_train = pd.DataFrame(X_train)


# number of null values in each 
df_train.isnull().sum()

# rows cols datatype information
df_dataset.info()

# filling the missing value
'''
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:, 0:81])
X_train[:, 0:81] = imputer.transform(X_train[:, 0:81])
'''
# int
df_train[3] = df_train[3].fillna(df_train[3].mean()) 
df_test[3] = df_test[3].fillna(df_test[3].mean())


# droping the col with a lot of null values
df_train.drop([0],axis=1,inplace=True)  
df_test.drop([0],axis=1,inplace=True)

df_train.drop([6],axis=1,inplace=True)  
df_test.drop([6],axis=1,inplace=True)

df_train.drop([59],axis=1,inplace=True)  
df_test.drop([59],axis=1,inplace=True)


df_train.drop([72],axis=1,inplace=True)  
df_test.drop([72],axis=1,inplace=True)

df_train.drop([73],axis=1,inplace=True)  
df_test.drop([73],axis=1,inplace=True)

df_train.drop([74],axis=1,inplace=True)  
df_test.drop([74],axis=1,inplace=True)

#df_train.drop([74],axis=1,inplace=True)  
#df_test.drop([74],axis=1,inplace=True)

# object (categorical data)

df_train[30] = df_train[30].fillna(df_train[30].mode()[0])
df_test[30] = df_test[30].fillna(df_test[30].mode()[0])

df_train[31] = df_train[31].fillna(df_train[31].mode()[0])
df_test[31] = df_test[31].fillna(df_test[31].mode()[0])

df_train[57] = df_train[57].fillna(df_train[57].mode()[0])
df_test[57] = df_test[57].fillna(df_test[57].mode()[0])

df_train[58] = df_train[58].fillna(df_train[58].mode()[0])
df_test[58] = df_test[58].fillna(df_test[58].mode()[0])

df_train[60] = df_train[60].fillna(df_train[60].mode()[0])
df_test[60] = df_test[60].fillna(df_test[60].mode()[0])

df_train[63] = df_train[63].fillna(df_train[63].mode()[0])
df_test[63] = df_test[63].fillna(df_test[63].mode()[0])

df_train[64] = df_train[64].fillna(df_train[64].mode()[0])
df_test[64] = df_test[64].fillna(df_test[64].mode()[0])

df_train.isnull().sum()

# categorical data (filling missing value)

df_train[25] = df_train[25].fillna(df_train[25].mode()[0])
df_test[25] = df_test[25].fillna(df_test[25].mode()[0])

df_train[26] = df_train[26].fillna(df_train[26].mode()[0])
df_test[26] = df_test[26].fillna(df_test[26].mode()[0])

df_train[32] = df_train[32].fillna(df_train[32].mode()[0])
df_test[32] = df_test[32].fillna(df_test[32].mode()[0])

df_train[33] = df_train[33].fillna(df_train[33].mode()[0])
df_test[33] = df_test[33].fillna(df_test[33].mode()[0])

df_train[35] = df_train[35].fillna(df_train[35].mode()[0])
df_test[35] = df_test[35].fillna(df_test[35].mode()[0])


df_train.isnull().sum()

# graph (visualization of null values)
import seaborn as sns
sns.heatmap( df_test.isnull() , yticklabels = False , cbar = False , cmap = "coolwarm" )
sns.heatmap( df_test.isnull() , yticklabels = False , cbar = False , cmap = "YlGnBu" )

df_test[47] = df_test[47].fillna(df_test[47].mean()) 
df_test[48] = df_test[48].fillna(df_test[48].mean())



# droping the col with null values
df_train.dropna(inplace = True)
df_test.dropna(inplace=True)

# concatenation of training and test data
df = pd.concat( [df_train,df_test] , axis = 0)


for fields in df.columns:
       if df_dataset[(df_dataset.columns)[fields]].dtype == 'O':
           continue
       else: 
           df[fields] = pd.to_numeric(df[fields])
        

# Encoding categorical data
def encoding():
    df_final = df
    i=0
    for fields in df.columns:
        print(fields)
        if df[fields].dtype == 'O':
            df_1 = pd.get_dummies(df[fields] , drop_first = True)
        else:
            df_1 = df[fields].copy()
        df.drop([fields] , axis=1 , inplace=True)
        if i==0:
            df_final = df_1.copy()
        else:
            df_final = pd.concat([df_final,df_1] , axis=1)
        i = i+1
    df_final = pd.concat([df,df_final],axis=1)
    return df_final    

df_data = encoding()
df_data = df_data.loc[: , ~df_data.columns.duplicated()] 

df[1].value_counts
np.unique(df[1])
df[25].dtype 

# Avoiding the Dummy Variable Trap
X = X[:, 1:]



# Splitting the dataset into the Training set and Test set
df_train = df_data.iloc[:1459, :].values
df_test = df_data.iloc[1459: , :].values

y_train = y_train[:1459 , ]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
df_train = sc_X.fit_transform(df_train)
df_test = sc_X.transform(df_test)


mean_value = sum(y_train)/len(y_train)
max_value = max(y_train)

y_train = (y_train - mean_value)/max_value

    

# Fitting Multiple Linear Regression to the Training set


#from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(df_train, y_train)




y_pred = regressor.predict(df_test)

y_pred = y_pred*max_value + mean_value

temp_test = pd.read_csv('test.csv').iloc[:, :]
df_temp_test = pd.DataFrame(temp_test)


df_pred = pd.DataFrame(y_pred)

df_pred[0] = df_pred[0].fillna(df_pred[0].mean()) 

y_pred = df_pred.iloc[:, -1].values


prediction = pd.DataFrame({"Id" : temp_test['Id'],"SalePrice" : y_pred})
prediction.to_csv('prediction.csv',index=False)
prediction.head()
prediction = pd.read_csv('prediction.csv')


