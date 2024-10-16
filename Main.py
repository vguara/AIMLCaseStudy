from sklearn.metrics import confusion_matrix
from Dataset import Dataset
from Measures import train_and_evaluate_models, plot_performance, plot_fraud_capture_rate


def main():
    """
    Main function to execute the fraud detection model training and evaluation process.

    :return: None
        The function does not return any value. It primarily performs I/O operations
        and visualizes results.
    """
    models = ['LR', 'SVM', 'RF']
    is_old = True

    # Dataset selection
    data_path = "creditcard.csv" if is_old else "creditcard_2023.csv"
    dataset = Dataset(data_path)

    # Process the dataset
    if is_old:
        # For old dataset: use fraud-specific train/test split
        dataset.create_train_test_set_fraud(0.7, 0.005)
    else:
        # For new dataset: drop ID/time columns, split, and reduce data
        dataset.drop_id_or_time()
        dataset.create_train_test(0.7, seed=42)
        dataset.reduce_data_set(0.12, "Train")
        dataset.reduce_test_data_ratio(0.005)

    # Define the label and features
    dataset.define_label_features(label="Class")

    # Create subsets with specified ratios
    ratios = [0.15, 0.1, 0.05, 0.02]
    dataset.create_subsets(ratios=ratios, seed=456)

    # Extract test data
    X_test, y_test = dataset.test_features, dataset.test_label

    # Performance metrics to evaluate
    metrics = ['accuracy', 'specificity', 'sensitivity', 'precision', 'AUC', 'F', 'G-mean', 'wtdAcc',
               'fraud_capture_rate_1_percent', 'fraud_capture_rate_10_percent', 'fraud_capture_rate_30_percent']

    # Train and evaluate models
    performance = train_and_evaluate_models(dataset, models, metrics, X_test, y_test)

    # Print confusion matrix for each model
    for model in models:
        y_pred = performance[model]['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix for {model}:\n{cm}")

    # Plot the performance
    plot_performance(performance, ratios)

    # Plot fraud capture rates only for the old dataset
    if is_old:
        plot_fraud_capture_rate(performance, ratios)


if __name__ == "__main__":
    main()
