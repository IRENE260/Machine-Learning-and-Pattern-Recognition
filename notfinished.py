import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Iris dataset using sklearn
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)  # Convert to DataFrame with column names
y = pd.Series(data.target, name='species')

# Create dictionary of models
models = {
    'Decision Tree Gini': DecisionTreeClassifier(criterion='gini'),
    'Decision Tree Entropy': DecisionTreeClassifier(criterion='entropy'),
    'Logistic Regression': LogisticRegression(max_iter=200),
    'SVM Linear': SVC(kernel='linear'),
    'SVM Poly': SVC(kernel='poly'),
    'SVM RBF': SVC(kernel='rbf'),
    'SVM Sigmoid': SVC(kernel='sigmoid')
}

# Monte Carlo evaluator with 10 iterations
n_splits = 10
results = {name: [] for name in models.keys()}

for _ in range(n_splits):
    # Random split of data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        # Predict test set
        y_pred = model.predict(X_test)
        # Calculate accuracy and store result
        accuracy = accuracy_score(y_test, y_pred)
        results[name].append(accuracy)

# Calculate mean accuracy for each model
mean_accuracies = {name: np.mean(acc) for name, acc in results.items()}

# Identify best-performing model
best_model_name = max(mean_accuracies, key=mean_accuracies.get)
best
