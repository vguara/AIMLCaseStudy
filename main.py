from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from Dataset import Dataset
from sklearn.ensemble import RandomForestClassifier
from measures import measures


models = ['LR','SVM','RF']
# models = ['SVM']

old_ds = "creditcard.csv"
new_ds = "creditcard_2023.csv"

ds = Dataset(old_ds)

# Seed defines a specific value for the randomness so to say
# Basically if we use the same see, the same training and test sets will be created
# This is useful to reproduce the same results
# If we want a truly random data just run this without a seed (default is None)

if ds.fraud_rate < 0.1: #for the old dataset
    ds.create_train_test_set_fraud(fraud_data_ratio=0.3, test_data_ratio=0.005,seed=123)
else: #for the new dataset
    ds.create_train_test(ratio=0.7)

if ds.fraud_rate > 0.1: #for the new data set, reducing the test data fraud rate to 0.5%
    ds.reduce_test_data_ratio(0.005)

ds.define_label_features(label="Class")

# 'Class' is how the fraud/legit binary is labeled in the dataset.

ratios = [0.02, 0.05, 0.1, 0.15]

ds.create_subsets(ratios=ratios,seed=456)
X_test, y_test = ds.test_features, ds.test_label
for ratio, subset in ds.subsets.items():
    print(f"Subset {ratio}")
    print(f"Subset size {len(subset[0])}")

    X_train, y_train = subset[0],subset[1]

    for model in models:
        if model == 'LR':
            model_trained = LogisticRegression(solver='liblinear')
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

