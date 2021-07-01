import pandas as pd
from sklearn.ensemble import RandomForestRegressor

data_base=pd.read_csv("train_data.csv") #####importing train case
test=pd.read_csv("test_data.csv")   #####importing test case




features=['Retweet count','Likes count','Tweet value']   ######features for training
y=data_base.User
x=data_base[features] 
model=RandomForestRegressor()   ####### Using Random forest Regressor
model.fit(x,y)    
arr=model.predict(test[features])  #####prediction

arr=pd.DataFrame(arr)
arr.to_csv("C:\\Users\\asus\\Desktop\\sol.csv")   #######saveing
