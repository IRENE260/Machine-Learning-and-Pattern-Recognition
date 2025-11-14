import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score

# Load data
df = pd.read_csv('Seed_Data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Base classifiers
svc = SVC(kernel='linear')
gnb = GaussianNB()

# Range of estimators for bagging
estimators_range = range(10, 110, 10)

# To store results
results = {'SVC': {'mean': [], 'std': []}, 'GNB': {'mean': [], 'std': []}}

for n in estimators_range:
    bagging_svc = BaggingClassifier(estimator=svc, n_estimators=n, max_samples=0.8, max_features=0.8, random_state=42)
    scores_svc = cross_val_score(bagging_svc, X, y, cv=5, scoring='accuracy')
    results['SVC']['mean'].append(scores_svc.mean())
    results['SVC']['std'].append(scores_svc.std())
    
    bagging_gnb = BaggingClassifier(estimator=gnb, n_estimators=n, max_samples=0.8, max_features=0.8, random_state=42)
    scores_gnb = cross_val_score(bagging_gnb, X, y, cv=5, scoring='accuracy')
    results['GNB']['mean'].append(scores_gnb.mean())
    results['GNB']['std'].append(scores_gnb.std())

# Plot error bars for both classifiers
plt.errorbar(estimators_range, results['SVC']['mean'], yerr=results['SVC']['std'], label='Bagging SVC', fmt='-o')
plt.errorbar(estimators_range, results['GNB']['mean'], yerr=results['GNB']['std'], label='Bagging GNB', fmt='-s')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('5-Fold CV Accuracy vs Number of Estimators')
plt.legend()
plt.show()

# Determine best bagging approach based on highest mean accuracy
best_svc_score = max(results['SVC']['mean'])
best_gnb_score = max(results['GNB']['mean'])

if best_svc_score > best_gnb_score:
    best_model = 'Bagging SVC'
    best_n = estimators_range[results['SVC']['mean'].index(best_svc_score)]
else:
    best_model = 'Bagging GNB'
    best_n = estimators_range[results['GNB']['mean'].index(best_gnb_score)]

print(f'Best bagging approach: {best_model} with {best_n} estimators achieving accuracy {max(best_svc_score, best_gnb_score):.4f}')
