from Dataset import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

########### SCRIPT

ds = Dataset('creditcard.csv')

# Seed defines a specific value for the randomness so to say
# Basically if we use the same see, the same training and test sets will be created
# This is useful to reproduce the same results
# If we want a truly random data just run this without a seed (default is None)
ds.create_train_test(ratio=0.7, seed=42)

# 'Class' is how the fraud/legit binary is labeled in the dataset.
ds.define_label_features(label='Class')

# Using Logistic Regression from sklearn
model = LogisticRegression()
model.fit(ds.train_features, ds.train_label)  # Train the model using training features and labels

test_predictions = model.predict(ds.test_features)

# Asked chatGPT for ways to best view the results
accuracy = accuracy_score(ds.test_label, test_predictions)
conf_matrix = confusion_matrix(ds.test_label, test_predictions)
report = classification_report(ds.test_label, test_predictions)

# Print the results
print("Logistic Regression Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(report)

