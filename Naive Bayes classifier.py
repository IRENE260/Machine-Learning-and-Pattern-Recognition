#Data understanding of the dataset

#1#2Data prepration techniques

#3apply logistic regression, fit the model to predict rating, and make the prediction of last row in the dataset accordingly

#4apply Naive Bayes classifier, fit the model to predict rating, and make the prediction of last row in the dataset accordingly

#5evaluate logistic regression versus Naive Bayes classifier in terms of recall where trainset is 80% and testset is 20%

#6validate the results in 5 fold cross validation and deploy the best model accordingly

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


df=pd.read_csv('Affairs.csv')

#1.data understanding

print(df.describe())
print(df.info())

# 2. Prepare data: encode categorical variables


data=df.drop('rating',axis=1)
data=pd.get_dummies(data,columns=['occupation'],drop_first=True)#pd.get_dummies replaces a categorical column with multiple binary columns.so it converts occupation to 2 or 3
#columns. Occuaption is choosed because it is a categorical value,even if it is values in dataset
#drop_first=True removes the first dummy column to reduce redundancy.
beta=df['rating']

#To scale values for logisticregression

data_train, data_test, beta_train, beta_test = train_test_split(data, beta, test_size=0.2, random_state=42)

scalar=StandardScaler()
data_train_scaled=scalar.fit_transform(data_train)
data_test_scaled=scalar.transform(data_test)

v=LogisticRegression()
v.fit(data_train_scaled ,beta_train)
beta_pred=v.predict(data_train_scaled)


nb = GaussianNB()
nb.fit(data_train, beta_train)
beta_pred_nb = nb.predict(data_test)
