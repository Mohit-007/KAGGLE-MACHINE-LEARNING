# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the dataset
dataset_train = pd.read_csv('train_data.txt', delimiter = '\t', quoting = 1)
X_test = pd.read_csv('test_data.txt', delimiter = '\t', quoting = 1).iloc[:, [0, 1, 2, 3, 4]]
X_train = dataset_train.iloc[:, [2,3,4,5]]
y_train = dataset_train.iloc[:, [1]].values
df_train = pd.DataFrame(X_train)
df_test = pd.DataFrame(X_test)
df_y = pd.DataFrame(y_train)

y = y_train


df = pd.concat( [df_train,df_test] , axis = 0, join = 'outer', ignore_index = 'True')

df.isnull().sum()


# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

review = re.sub('[^a-zA-Z@]',' ',  df['Tweet'][0])
review = review.lower()
review = review.split()
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review = ' '.join(review)
corpus.append(review)



for i in range(0, 7365):
    review = re.sub('[^a-zA-Z@]', ' ', df['Tweet'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)



# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 15000)
X = cv.fit_transform(corpus).toarray()

X = np.append(X, values = df.iloc[:, [1, 2, 3]], axis = 1)

y = df_y.iloc[:, -1].values



# Splitting the dataset into the Training set and Test set
df_train = X[:5365, :]
df_test = X[5365: , :]

# Fitting Naive Bayes to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(df_train, y)
# accuracy = 97%

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(df_train, y)


# Predicting the Test set results
y_pred_0 = classifier.predict(df_test)

y_pred_1 = classifier.predict(df_test)

y_pred = classifier.predict(df_test)

prediction_1 = pd.DataFrame({"index" : X_test['index'],"User" : y_pred_1})
prediction_1.to_csv('prediction_1.csv',index=False)
prediction_1.head()
prediction_1 = pd.read_csv('prediction_1.csv')

prediction_0 = pd.DataFrame({"index" : X_test['index'],"User" : y_pred_0})
prediction_0.to_csv('prediction_0.csv',index=False)
prediction_0.head()
prediction_0 = pd.read_csv('prediction_0.csv')

prediction = pd.DataFrame({"index" : X_test['index'],"User" : y_pred})
prediction.to_csv('prediction.csv',index=False)
prediction.head()
prediction = pd.read_csv('prediction.csv')


y_pred_0 = pd.read_csv('prediction_0.csv')
y_pred_1 = pd.read_csv('prediction_1.csv')

y_pred_0 = pd.read_csv('prediction_0.csv').iloc[:, 1].values
y_pred_1 = pd.read_csv('prediction_1.csv').iloc[:, 1].values


from sklearn.metrics import confusion_matrix
cm_1 = confusion_matrix(y_pred_0, y_pred_1)

cm = confusion_matrix(y_pred, y_pred_1)

'''

# cross validation 

dataset_train = pd.read_csv('train_data.txt', delimiter = '\t', quoting = 1)
X_test = pd.read_csv('test_data.txt', delimiter = '\t', quoting = 1)
X_train = dataset_train.iloc[:, [2,3,4,5]]
y_train = dataset_train.iloc[:, [1]].values
df_train = pd.DataFrame(X_train)
df_test = pd.DataFrame(X_test)
df_y = pd.DataFrame(y_train)

df_data = pd.DataFrame(X_train)
y = y_train

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus_train = []

for i in range(0, 5365):
    review_train = re.sub('[^a-zA-Z@]', ' ', df_data['Tweet'][i])
    review_train = review_train.lower()
    review_train = review_train.split()
    ps = PorterStemmer()
    review_train = [ps.stem(word) for word in review_train if not word in set(stopwords.words('english'))]
    review_train = ' '.join(review_train)
    corpus_train.append(review_train)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 15000)
X_data = cv.fit_transform(corpus_train).toarray()



X_data = np.append(X_data, values = dataset_train.iloc[:, [3,4,5]], axis = 1)

y = df_y.iloc[:, -1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X_data, y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train_data = sc_X.fit_transform(X_train_data)
X_test_data = sc_X.transform(X_test_data)




from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver = 'liblinear')
classifier.fit(X_train_data, y_train_data)
# accuracy = 97%

# Predicting the Test set results
y_pred_validation = classifier.predict(X_test_data)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_data, y_pred_validation)

'''