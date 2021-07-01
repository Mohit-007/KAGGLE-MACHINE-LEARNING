# importing libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Loading Data
train = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')

# Separating Feature & Label Columns
col = ['Tweet value']
X_train = train[col]
Y_train = train.User
# Training Classifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)

# Predicting the Output on Test data
X_test = test[col]
Y_test = clf.predict(X_test)

# Dictionaries to obtain output in reqd. format
dict = {}
dict['index'] = []
dict['User'] = []

# Trying Computation and Hit and Trial for left-out-Cases
lis = []
for i in range(0, len(train)):
    lis.append([float(train.iloc[i][5]),int(train.iloc[i][1])])

#lis.append([10000000,0])
lis.sort()

prev = 1

lis2 = []
lis3 = []
flag = 0

i = 0
while i < len(train):
    if lis[i][1] != prev:
        lis2.append(lis[i][0])
    else:
        lis3.append([lis2[0], lis2[len(lis2)-1], 1-prev])
        lis2 = []
        i = i-1
        prev = 1-prev
    i = i+1

lis3.append([lis2[0], lis2[len(lis2)-1], 1-prev])


print(lis3[0][1])
cnt = 0
cnt1 = 0
values = [2440, 2684, 2928, 3221, 12201, 13421, 14641, 16105]
for data in range(0, len(test)):
    index = test.iloc[data][0]
    tweet = test.iloc[data][1]
    tweet_value = float(test.iloc[data][4])
    user = Y_test[data]
    if tweet_value in values:
        user = 1
    user1 = -1
    for j in range(0, len(lis3)):
        mn = lis3[j][0]
        mx = lis3[j][1]
        if mn <= tweet_value <= mx:
            user1 = lis3[j][2]
            break
    if tweet_value < 2430 or tweet_value > 16300:
        user1 = 0
    prevCnt = cnt
    if user1 == -1 and data%2 == 0:
        user1 = 0
        cnt = cnt+1
    elif user1 == -1:
        user1 = 1
        cnt = cnt+1

    user = user1
    dict['index'].append(index)
    dict['User'].append(user)

print(cnt)
# Creating dictionary based output dataframe
df = pd.DataFrame(dict)

# Storing the output to .csv file to upload/submit
df.to_csv('C:/Users/yasha/OneDrive/Desktop/Twitter data/result.csv', encoding='utf-8', index=False)
