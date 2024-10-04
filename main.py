from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from Dataset import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from measures import measures

ds = Dataset('creditcard.csv')
# models = ['LR','SVM','RF']
models = ['SVM']

# Create train and test sets with a 70/30 split
ds.create_train_test(ratio=0.7, seed=42)

# Define the label 'Class' and use all other features for training
ds.define_label_features(label='Class')

# Now X_train, X_test, y_train, and y_test are available to use in your model
X_train, X_test = ds.train_features, ds.test_features
y_train, y_test = ds.train_label, ds.test_label

for model in models:
    if model == 'LR':
        model_trained = LogisticRegression()
    elif model == 'SVM':
        model_trained = SVC(kernel='rbf', C=1.0, gamma='scale',probability=True)
    elif model == 'RF':
        # Train a Random Forest model
        model_trained = RandomForestClassifier(n_estimators=100, random_state=42)

    model_trained.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = model_trained.predict(X_test)
    y_prob = model_trained.predict_proba(X_test)[:, 1]
    performance = measures(y_test, y_pred, y_prob)
    print(performance)

