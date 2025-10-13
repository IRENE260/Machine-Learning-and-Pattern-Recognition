# Using Logistic Regression model,the variable rating

#Prepare the dataset
#Utlize the sklearn package to fit the logistic regression
#Evaluate this model in terms of accuracy where trainset is 80% and testset is 20%

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

df=pd.read_csv('Affairs.csv')
X=df[['rownames','affairs','gender','age','yearsmarried','children','religiousness','education','occupation']]
Y=df['rating']

X=pd.get_dummies(X,columns=['gender','children'],drop_first=True)
numeric_cols = ['affairs', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

v = LogisticRegression()
v.fit(X_train,Y_train)
Y_pred=v.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
print(f"Accuracy on test set: {accuracy:.3f}")
print("Confusion Matrix:")
print(conf_matrix)


