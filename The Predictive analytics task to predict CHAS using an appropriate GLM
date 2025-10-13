import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LogisticRegression

df=pd.read_csv('boston_house_prices.csv')
X=df[['CRIM','ZN','INDUS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']]
Y=df['CHAS']
Y.unique()
df.head()
v=LogisticRegression()# 0,1
v.fit(X,Y)
print(v.coef_)
