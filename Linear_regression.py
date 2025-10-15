#Example 2.1: (linear regression): Use diamonds dataset, predict the price for the first sample of diamond in the dataset using GLM. For this purpose consider depth,table, length_mm, width_mm ,depth_mm as the input variables.
#apply statmodels.app to do the following tasks
#1. Parameter learning feature
#2. selection prediction of the last row to test your model model evaluation for full model versus reduced model in terms of RMSE

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

df=pd.read_csv('diamonds.csv')
data=df[['depth_mm','depth','length_mm','depth_mm','width_mm']]#input varibale,independent variables
beta=df['price']#output variable to predict price,dependent variable
data_train,data_test,beta_train,beta_test=train_test_split(data,beta,test_size=0.2,random_state=42)
v=LinearRegression()
v.fit(data_train,beta_train)
beta_pred=v.predict(data_test)# to predict prices based on new input features data_test
root_mean_squared_error=sqrt(mean_squared_error(beta_test,beta_pred))#calculates how far  (data_pred) are from the actual prices (beta_test) on average, with RMSE expressing this error in the same units as price.
print(root_mean_squared_error)

data_reduced=df[['length_mm','depth_mm']]
data_train_reduced,data_test_reduced,beta_train_reduced,beta_test_reduced=train_test_split(data,beta,test_size=0.2,random_state=42)
v_reduced=LinearRegression()
v_reduced.fit(data_train_reduced,beta_train_reduced)
beta_pred_reduced=(v_reduced.predict(data_test_reduced))
root_mean_squared_error_reduced=sqrt(mean_squared_error(beta_test_reduced,beta_pred_reduced))
print(root_mean_squared_error_reduced)

