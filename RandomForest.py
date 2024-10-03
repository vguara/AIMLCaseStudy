from Dataset import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


ds = Dataset('creditcard.csv')

# Create train and test sets with a 70/30 split
ds.create_train_test(ratio=0.7, seed=42)

# Define the label 'Class' and use all other features for training
ds.define_label_features(label='Class')

# Now X_train, X_test, y_train, and y_test are available to use in your model
X_train, X_test = ds.train_features, ds.test_features
y_train, y_test = ds.train_label, ds.test_label

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = rf_model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))



