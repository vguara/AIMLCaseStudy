from sklearn.metrics import confusion_matrix

from Dataset import Dataset
from measures import train_and_evaluate_models, plot_performance, plot_fraud_capture_rate
from sklearn.preprocessing import StandardScaler


def main():
    """
    Main function to execute the fraud detection model training and evaluation process.

    :return: None
        The function does not return any value. It primarily performs I/O operations
        and visualizes results.
    """
    models = ['LR', 'SVM', 'RF']
    dataset_path = "creditcard.csv"
    # dataset_path = "creditcard_2023.csv"

    ds = Dataset(dataset_path)

    # Create train and test sets based on fraud rate
    if ds.fraud_rate < 0.1:
        ds.create_train_test_set_fraud(fraud_data_ratio=0.3, test_data_ratio=0.005, seed=123)
    else:
        ds.create_train_test(ratio=0.7)

    if ds.fraud_rate > 0.1:
        ds.reduce_test_data_ratio(0.005)

    ds.define_label_features(label="Class")

    ratios = [0.15, 0.1, 0.05, 0.02]
    ds.create_subsets(ratios=ratios, seed=456)

    X_test, y_test = ds.test_features, ds.test_label

    # Scale the features
    scaler = StandardScaler()
    ds.train_features = scaler.fit_transform(ds.train_features)  # Assuming ds has train_features
    X_test = scaler.transform(X_test)

    metrics = ['accuracy', 'specificity', 'sensitivity', 'precision', 'AUC', 'F', 'G-mean', 'wtdAcc',
               'fraud_capture_rate_1_percent', 'fraud_capture_rate_10_percent', 'fraud_capture_rate_30_percent']

    # Train and evaluate models
    performance = train_and_evaluate_models(ds, models, metrics, X_test, y_test)

    # Print confusion matrix for each model
    for model in models:
        y_pred = performance[model]['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix for {model}:\n{cm}")

    # Call function to plot the performance
    plot_performance(performance, ratios)

    # Plot the fraud capture rates
    plot_fraud_capture_rate(performance, ratios)


if __name__ == "__main__":
    main()
