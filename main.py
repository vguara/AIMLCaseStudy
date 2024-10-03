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
models = ['RF']

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
        model_trained.fit(ds.train_features, ds.train_label)  # Train the model using training features and labels

        y_pred = model_trained.predict(ds.test_features)
    elif model == 'SVM':
        model_trained = SVC(kernel='rbf', C=1.0, gamma='scale')
        model_trained.fit(ds.train_features, ds.train_label)  # Train the model using training features and labels

        y_pred = model_trained.predict(ds.test_features)
    elif model == 'RF':
        # Train a Random Forest model
        model_trained = RandomForestClassifier(n_estimators=100, random_state=42)
        model_trained.fit(X_train, y_train)

        # Make predictions and evaluate the model
        y_pred = model_trained.predict(X_test)
    else:
        print('Invalid model')
    performance = measures(y_test, y_pred)
    # conf_matrix = confusion_matrix(y_test, y_pred)
    #
    # print("Confusion Matrix:")
    # print(conf_matrix)
    #
    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred))
    #
    # print("\nAccuracy Score:")
    # print(accuracy_score(y_test, y_pred))