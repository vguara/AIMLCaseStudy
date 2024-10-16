from Dataset import Dataset
from measurestest import train_and_evaluate_models, plot_performance, plot_fraud_capture_rate
from sklearn.preprocessing import StandardScaler
import time

def main():
    """
    Main function to execute the fraud detection model training and evaluation process.

    :return: None
        The function does not return any value. It primarily performs I/O operations
        and visualizes results.+
    """
    start = time.time()

    models = ['LR', 'SVM', 'RF']

    old = False

    if old:
        data = "creditcard.csv"
    else:
        data = "creditcard_2023.csv"

    dstrain = Dataset(data)



    # Mark 1 as test and the other as train
    if not old:
        dstrain.drop_id_or_time()
        dstrain.create_train_test(0.6, 42)
        dstrain.reduce_data_set(0.25, "Train")
        dstrain.reduce_test_data_ratio(0.005)
    else:
        dstrain.create_train_test_set_fraud(0.7, 0.005)


    dstrain.define_label_features(label="Class")

    ratios = [0.15, 0.1, 0.05, 0.02]
    dstrain.create_subsets(ratios=ratios, seed=456)



    X_test, y_test = dstrain.test_features, dstrain.test_label

    metrics = ['accuracy', 'specificity', 'sensitivity', 'precision', 'AUC', 'F', 'G-mean', 'wtdAcc',
               'fraud_capture_rate_1_percent', 'fraud_capture_rate_10_percent', 'fraud_capture_rate_30_percent']

    # Train and evaluate models
    performance = train_and_evaluate_models(dstrain, models, metrics, X_test, y_test, old)

    # Call function to plot the performance
    plot_performance(performance, ratios)

    # Plot the fraud capture rates
    # if old:
    #     plot_fraud_capture_rate(performance, ratios)

    print(time.time() - start)

if __name__ == "__main__":
    main()
