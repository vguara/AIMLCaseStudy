from Dataset import Dataset
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from Dataset import find_best_model

old_ds = "creditcard.csv"
new_ds = "creditcard_2023.csv"

ds = Dataset(old_ds)

# Seed defines a specific value for the randomness so to say
# Basically if we use the same see, the same training and test sets will be created
# This is useful to reproduce the same results
# If we want a truly random data just run this without a seed (default is None)

if ds.fraud_rate < 0.1: #for the old dataset
    ds.create_train_test_set_fraud(fraud_data_ratio=0.3, test_data_ratio=0.005)
else: #for the new dataset
    ds.create_train_test(ratio=0.7)

if ds.fraud_rate > 0.1: #for the new data set, reducing the test data fraud rate to 0.5%
    ds.reduce_test_data_ratio(0.005)

ds.define_label_features(label="Class")

# 'Class' is how the fraud/legit binary is labeled in the dataset.

ratios = [0.02, 0.05, 0.1, 0.15]

ds.create_subsets(ratios=ratios)

# for ratio, subset in ds.subsets.items():
#     print(f"length of subset of ratio {ratio}: {len(subset)}")
#     fraud_cases = subset[subset['Class'] == 1]
#     total_cases = subset.shape[0]
#     print(f"number of fraud cases: {fraud_cases.shape[0]}")
#     print(f"number of total cases: {total_cases}")

# model = SVC(kernel='rbf', C=10, gamma=0.00000001)
model = RandomForestClassifier()
param_grid = {
    # 'C': [0.1, 1],
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [10,20]
}

bestm, params, score = find_best_model(model, param_grid, 5,'recall', ds.train_features, ds.train_label)

print(f"Best params {params}")
print(f"Best score {score}")

# print(f"Test data ratio: {Dataset.define_fraud_rate(ds.test_data)}")
#
#
# for ratio, subset in ds.subsets.items():
#     print(f"Subset {ratio}")
#     print(f"Subset size {len(subset[0])}")
#     features = subset[0]
#     label = subset[1]
#     grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='recall')
#     grid_search.fit(features, label)
#     # model.fit(train_features, train_label)  # Train the model using training features and labels
#     # test_predictions = model.predict(ds.test_features)
#     best_params = grid_search.best_params_
#     print(f"Best C value: {best_params['C']}")
#     best_model = grid_search.best_estimator_
#     test_predictions = best_model.predict(ds.test_features)
#
#     # Asked chatGPT for ways to best view the results
#     accuracy = accuracy_score(ds.test_label, test_predictions)
#     conf_matrix = confusion_matrix(ds.test_label, test_predictions)
#     report = classification_report(ds.test_label, test_predictions)
#
#     # Print the results
#     print(f"Support Vector Machine Model Performance for ratio {ratio} :")
#     print(f"Accuracy: {accuracy:.4f}")
#     print("Confusion Matrix:")
#     print(conf_matrix)
#     print("Classification Report:")
#     print(report)