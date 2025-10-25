import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('winequality-red.csv')
X = df.drop('quality', axis=1)
Y = df['quality']

acc_svc = []
acc_log = []

ker = ['linear', 'sigmoid', 'rbf']  # kernel types of SVM

for k in ker:
    mc_acc_svc = []
    mc_acc_lr = []
    for j in range(10):
       
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        # SVC
        model = SVC(kernel=k, C=1.1)
        model.fit(X_train, Y_train)
        pred_svc = model.predict(X_test)
        mc_acc_svc.append(accuracy_score(Y_test, pred_svc))

        # Logistic Regression
        model_lr = LogisticRegression(max_iter=1000)
        model_lr.fit(X_train, Y_train)
        pred_lr = model_lr.predict(X_test)
        mc_acc_lr.append(accuracy_score(Y_test, pred_lr))

    
    acc_svc.append(np.mean(mc_acc_svc))
    acc_log.append(np.mean(mc_acc_lr))
    print(f"{k} kernel SVC avg accuracy: {np.mean(mc_acc_svc):.4f}")
    print(f"{k} kernel Logistic Regression avg accuracy: {np.mean(mc_acc_lr):.4f}")

print( np.mean(acc_svc))
print(np.mean(acc_log))
