import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def measures(y_true, y_pred, y_prob):
    """
    Calculates performance metrics for model predictions.

    :param y_true: array-like
        True labels of the dataset.

    :param y_pred: array-like
        Predicted labels from the model.

    :param y_prob: array-like
        Predicted probabilities from the model.

    :return: dict
        A dictionary containing calculated performance metrics including
        accuracy, sensitivity, specificity, precision, F1 score, G-mean,
        weighted accuracy, AUC, and fraud capture rates at specified thresholds.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    performance_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred),
        "specificity": calculate_specificity(tn, fp),
        "precision": calculate_precision(tp, fp),
        "f1_score": f1_score(y_true, y_pred),
        "g_mean": calculate_g_mean(recall_score(y_true, y_pred), calculate_specificity(tn, fp)),
        "wtd_acc": calculate_weighted_accuracy(tn, fp, fn, tp),
        "auc": roc_auc_score(y_true, y_prob),
        "fraud_capture_rate_1_percent": calculate_fraud_capture_rate(y_true, y_prob, 0.01),
        "fraud_capture_rate_10_percent": calculate_fraud_capture_rate(y_true, y_prob, 0.10),
        "fraud_capture_rate_30_percent": calculate_fraud_capture_rate(y_true, y_prob, 0.30),
    }

    return performance_metrics


def calculate_specificity(tn, fp):
    """
    Calculates specificity.

    :param tn: int
        True negatives from the confusion matrix.

    :param fp: int
        False positives from the confusion matrix.

    :return: float
        Specificity value, defined as TN / (TN + FP), or 0 if (TN + FP) is 0.
    """
    return tn / (tn + fp) if (tn + fp) != 0 else 0


def calculate_precision(tp, fp):
    """
    Calculates precision.

    :param tp: int
        True positives from the confusion matrix.

    :param fp: int
        False positives from the confusion matrix.

    :return: float
        Precision value, defined as TP / (TP + FP), or 0 if (TP + FP) is 0.
    """
    return tp / (tp + fp) if (tp + fp) != 0 else 0


def calculate_weighted_accuracy(tn, fp, fn, tp):
    """
    Calculates weighted accuracy.

    :param tn: int
        True negatives from the confusion matrix.

    :param fp: int
        False positives from the confusion matrix.

    :param fn: int
        False negatives from the confusion matrix.

    :param tp: int
        True positives from the confusion matrix.

    :return: float
        Weighted accuracy value, calculated as a weighted sum of sensitivity and specificity.
    """
    # sensitivity = recall_score([1] * tp + [0] * fn, [1] * tp + [0] * fn)  # Correctly use true and predicted labels
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    return 0.7 * sensitivity + 0.3 * specificity


def calculate_g_mean(sensitivity, specificity):
    """
    Calculates G-mean.

    :param sensitivity: float
        Sensitivity (True Positive Rate) of the model.

    :param specificity: float
        Specificity of the model.

    :return: float
        G-mean value, calculated as the square root of (sensitivity * specificity).
    """
    return np.sqrt(sensitivity * specificity)


def calculate_fraud_capture_rate(y_true, y_prob, threshold):
    """
    Calculates fraud capture rate at a given threshold.

    :param y_true: array-like
        True labels of the dataset.

    :param y_prob: array-like
        Predicted probabilities from the model.

    :param threshold: float
        Fraction of the top predicted probabilities to consider for capturing fraud.

    :return: float
        Fraud capture rate, defined as the proportion of actual fraud cases captured
        within the top fraction of predicted probabilities.
    """
    sorted_indices = np.argsort(y_prob)[::-1]
    sorted_y_true = y_true[sorted_indices]
    total_fraud_cases = sum(y_true)
    n_threshold = int(threshold * len(y_true))
    return sum(sorted_y_true[:n_threshold]) / total_fraud_cases if total_fraud_cases > 0 else 0


def plot_performance(performance, ratios):
    """
    Plots performance metrics of models across data subsets.

    :param performance: dict
        A dictionary where keys are model names ('LR', 'SVM', 'RF') and values are
        dictionaries of performance metrics (e.g., 'specificity', 'sensitivity',
        'precision', 'AUC', 'F1', 'G-mean', 'weighted_accuracy', 'accuracy'),
        each containing a list of values for different data subsets.

    :param ratios: list
        A list of data subset ratios (e.g., [0.02, 0.05, 0.1, 0.15]) corresponding to
        different data frames (DF1, DF2, DF3, DF4) in the plot.

    :return: None
        Displays a grid of subplots showing the performance of each model across all
        available metrics dynamically. The number of subplots adjusts based on the
        number of metrics, and any unused subplots are hidden.
    """
    # Sort the ratios to ensure correct plotting order
    ratios = sorted(ratios)

    metrics = ['accuracy', 'specificity', 'sensitivity', 'precision', 'AUC', 'F', 'G-mean', 'wtdAcc']

    num_metrics = len(metrics)
    rows = (num_metrics + 1) // 2
    fig, axs = plt.subplots(rows, 2, figsize=(10, 4 * rows))
    fig.tight_layout(pad=5.0)

    colour_map = {
        'LR': 'green',
        'SVM': 'blue',
        'RF': 'red'
    }

    # Loop through each metric and plot the performance
    for idx, metric in enumerate(metrics):
        row, col = divmod(idx, 2)
        for model in performance:
            marker_style = 's' if model == 'RF' else ('o' if model == 'SVM' else '^')
            axs[row, col].plot(ratios, performance[model][metric], label=model, marker=marker_style,
                               color=colour_map[model])

        axs[row, col].set_title(metric)
        axs[row, col].legend(loc='best')
        axs[row, col].set_xticks(ratios)
        axs[row, col].set_xticklabels(["DF1", "DF2", "DF3", "DF4"])
        axs[row, col].grid(True)

    # Hide any unused subplots
    for i in range(num_metrics, rows * 2):
        fig.delaxes(axs.flatten()[i])

    plt.show()


def plot_fraud_capture_rate(performance, ratios):
    """
    Plots fraud capture rates of models across data subsets.

    :param performance: dict
        A dictionary where keys are model names ('LR', 'SVM', 'RF') and values are
        dictionaries of performance metrics containing the fraud capture rates.

    :param ratios: list
        A list of data subset ratios (e.g., [0.02, 0.05, 0.1, 0.15]) corresponding to
        different data frames (DF1, DF2, DF3, DF4) in the plot.

    :return: None
        Displays a grid of subplots showing the fraud capture rates of each model.
    """
    # Sort the ratios to ensure correct plotting order
    ratios = sorted(ratios)

    capture_rates = ['fraud_capture_rate_1_percent', 'fraud_capture_rate_10_percent', 'fraud_capture_rate_30_percent']
    num_rates = len(capture_rates)
    rows = (num_rates + 1) // 2  # Calculate rows needed (2 rates per row)

    fig, axs = plt.subplots(rows, 2, figsize=(10, 4 * rows))
    fig.tight_layout(pad=5.0)

    colour_map = {
        'LR': 'green',
        'SVM': 'blue',
        'RF': 'red'
    }

    # Loop through each capture rate and plot the performance
    for idx, rate in enumerate(capture_rates):
        row, col = divmod(idx, 2)
        for model in performance:
            axs[row, col].plot(ratios, performance[model][rate], label=model, marker='o',
                               color=colour_map[model])

        percentage = rate.split('_')[-2]
        axs[row, col].set_title(f"Fraud capture rate in top {percentage}%")
        axs[row, col].legend(loc='best')
        axs[row, col].set_xticks(ratios)
        axs[row, col].set_xticklabels(["DF1", "DF2", "DF3", "DF4"])
        axs[row, col].grid(True)

    # Hide any unused subplots
    for i in range(num_rates, rows * 2):
        fig.delaxes(axs.flatten()[i])

    plt.show()


def initialize_performance_dict(models, metrics):
    """
    Initializes a performance dictionary with empty lists for each model and metric.

    :param models: list
        A list of model names to be included in the performance dictionary.

    :param metrics: list
        A list of metric names for which performance will be tracked.

    :return: dict
        A nested dictionary where keys are model names and values are dictionaries
        containing empty lists for each metric.
    """
    return {model: {metric: [] for metric in metrics} for model in models}


def train_and_evaluate_models(ds, models, metrics, X_test, y_test):
    """
    Trains and evaluates specified models on each data subset.

    :param ds: object
        An object containing data subsets and their respective labels.

    :param models: list
        A list of model names to be trained and evaluated.

    :param metrics: list
        A list of metrics to be calculated for each model's performance.

    :param X_test: array-like
        Test feature data used for predictions.

    :param y_test: array-like
        True labels corresponding to the test data.

    :return: dict
        A dictionary containing performance metrics for each model across different
        subsets.
    """
    performance = initialize_performance_dict(models, metrics)
    metric_key_mapping = {
        'accuracy': 'accuracy',
        'specificity': 'specificity',
        'sensitivity': 'sensitivity',
        'precision': 'precision',
        'AUC': 'auc',
        'F': 'f1_score',
        'G-mean': 'g_mean',
        'wtdAcc': 'wtd_acc',
        'fraud_capture_rate_1_percent': 'fraud_capture_rate_1_percent',
        'fraud_capture_rate_10_percent': 'fraud_capture_rate_10_percent',
        'fraud_capture_rate_30_percent': 'fraud_capture_rate_30_percent'
    }

    for ratio, subset in ds.subsets.items():
        print(f"Subset {ratio}")
        print(f"Subset size {len(subset[0])}")

        X_train, y_train = subset[0], subset[1]

        for model in models:
            model_trained = train_model(model, X_train, y_train)

            # Make predictions and evaluate the model
            X_test_scaled = StandardScaler().fit_transform(X_test)  # Scale test data
            y_pred = model_trained.predict(X_test_scaled)
            y_prob = model_trained.predict_proba(X_test_scaled)[:, 1]

            # Get the performance metrics
            results = measures(y_test, y_pred, y_prob)
            print(results)  # For debugging purposes

            # Add the results to performance dictionary for each model
            for metric in metrics:
                if metric_key_mapping[metric] in results:
                    performance[model][metric].append(results[metric_key_mapping[metric]])
                else:
                    print(f"Metric {metric} not found in results. Available metrics: {results.keys()}")

    return performance


def train_model(model_name, X_train, y_train):
    """
    Trains a model based on the model name.

    :param model_name: str
        The name of the model to train ('LR', 'SVM', 'RF').

    :param X_train: array-like
        The training feature data.

    :param y_train: array-like
        The training labels.

    :return: model
        The trained model.
    """
    if model_name == 'LR':
        model = LogisticRegression(max_iter=200)  # Increased max_iter for convergence
    elif model_name == 'SVM':
        model = SVC(probability=True, kernel='rbf', gamma=0.1, C=10)

        # grid_search = GridSearchCV(estimator=SVC(probability=True), param_grid={'C': [0.1, 1, 10]}, cv=5, scoring='sensitivity')
        # grid_search.fit(X_train, y_train)
        # best_params = grid_search.best_params_
        # print(best_params)
        # print(f"Best C value: {best_params['C']}")
        # model = grid_search.best_estimator_
    elif model_name == 'RF':
        model = RandomForestClassifier()
    else:
        raise ValueError("Model not recognized")

    # Scale data for better convergence
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model.fit(X_train_scaled, y_train)
    return model
