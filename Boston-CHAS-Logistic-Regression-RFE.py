#Use Boston_house price dataset,
#Consider CHAS as the output variable. The remaining variables are inputs apply logistic regression to do the following tasks:
#1.parameter estimation
#2.attribute selection
#3.prediction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, accuracy_score


df=pd.read_csv('boston_house_prices.csv')
beta=df.drop('CHAS',axis=1)#removing CHAS from the remaning variables
data=df['CHAS']
beta_train,beta_test,data_train,data_test=train_test_split(beta,data,test_size=0.2,random_state=42)
b=LogisticRegression(max_iter=100000)
#Recursive Feature Elimination (RFE) is used because it systematically 
#identifies the most relevant features for a model by removing the least important ones iteratively. 

s=RFE(b,n_features_to_select=2)
s.fit(beta_train,data_train)
selected_features=beta.columns[s.support_]
selected=list(selected_features)
b.fit(beta_train [selected_features], data_train)
beta_pred=b.predict(beta_test[selected_features])
a=accuracy_score(data_test,beta_pred)
c=classification_report(data_test, beta_pred)
print(a)

print(c)



