import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv('Seed_Data.csv')
data=df.drop('target',axis=1)
beta=df['target']
print(df.shape)

cfl1 = DecisionTreeClassifier(criterion='entropy')
cfl2=LogisticRegression()
cfl3=GaussianNB()

# bagging learner 1
bagging1 = BaggingClassifier(estimator=cfl1, n_estimators=50, max_samples=0.8, max_features=0.8)
# bagging learner 2
bagging2 = BaggingClassifier(estimator=cfl2, n_estimators=50, max_samples=0.8, max_features=0.8)

# bagging learner 3
bagging3 = BaggingClassifier(estimator=cfl3, n_estimators=50, max_samples=0.8, max_features=0.8)
RF = RandomForestClassifier(n_estimators=50, random_state=0)
print(bagging3)

import warnings
warnings.filterwarnings("ignore")

label = ['dt', 'LR','gnb', 'Bagging Tree', 'Bagging LR','Bagging NB','rf']
cfl_list = [clf1, clf2,clf3, bagging1, bagging2,bagging3,RF]


for cfl, label  in zip(cfl_list, label ):
    scores = cross_val_score(cfl, data, beta, cv=3, scoring='accuracy')
    print ("Accuracy:  ",(round(scores.mean(),4), round(scores.std(),3), label))
