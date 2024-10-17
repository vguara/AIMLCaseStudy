from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from Dataset import Dataset, find_best_model


def evaluate_models(models_with_params, cv, scoring, training_features, training_labels):
    results = []

    for model, param_grid in models_with_params:
        best_model, best_params, best_score = find_best_model(
            model=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            training_features=training_features,
            training_labels=training_labels
        )

        results.append({
            'model': best_model,
            'params': best_params,
            'score': best_score
        })

        print(model, param_grid)

    return results


dataset_path = "creditcard.csv"
ds = Dataset(dataset_path)

train_test_ratio = 0.7
ds.create_train_test(ratio=train_test_ratio, seed=42)

ds.define_label_features(label='Class')

X_train = ds.train_features
y_train = ds.train_label

print(X_train)
print(y_train)

models_with_params = [
    (LogisticRegression(), {
        'max_iter': [1000]
    }),
    (SVC(), { # kernel is set to 'rbf' by default
        'C': [0.1, 1]
    }),
    (RandomForestClassifier(), {
        'n_estimators': [50, 100]
    })
]

results = evaluate_models(
    models_with_params=models_with_params,
    cv=5,
    scoring='accuracy',
    training_features=X_train,
    training_labels=y_train
)

for result in results:
    print("Best Model:", result['model'])
    print("Best Parameters:", result['params'])
    print("Best Score:", result['score'])
    print("---")
