from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

# # Example: True labels and predicted labels from your fraud detection model
# y_true = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]  # Actual labels (0 = Non-fraud, 1 = Fraud)
# y_pred = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1]  # Predicted labels by the model
def measures(y_true, y_pred, y_prob):
    # 1. Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 2. Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # 3. Sensitivity (Recall)
    sensitivity = recall_score(y_true, y_pred)  # TP / (TP + FN)

    # 4. Specificity
    specificity = tn / (tn + fp)

    # 5. Precision
    if (tp + fp) == 0:
        precision = 0
    else:
        precision = precision_score(y_true, y_pred)  # TP / (TP + FP)

    # 6. F-measure (F1 Score)
    f_measure = f1_score(y_true, y_pred)

    # 7. G-mean
    g_mean = np.sqrt(sensitivity * specificity)

    # 8. Weighted Accuracy (wtdAcc)
    # Custom weight
    w = 0.7
    # Calculate weighted accuracy
    wtdAcc = w * sensitivity + (1 - w) * specificity

    # 9. AUC (Area Under the ROC Curve)
    auc = roc_auc_score(y_true, y_pred)

    #10.  
    # Sort the test set based on the predicted probability for fraud
    sorted_indices = np.argsort(y_prob)[::-1]  # Sort in descending order
    sorted_y_true = y_true.values[sorted_indices]     # Sort the true labels accordingly

    # Calculate the number of fraud cases
    total_fraud_cases = sum(y_true)

    # 1% and 10% depths
    n_1_percent = int(0.01 * len(y_true))  # Top 1% of the dataset
    n_10_percent = int(0.10 * len(y_true))  # Top 10% of the dataset
    n_30_percent = int(0.30 * len(y_true))  # Top 30% of the dataset

    # Fraud capture rate at 1% depth
    fraud_in_1_percent = sum(sorted_y_true[:n_1_percent])
    fraud_capture_rate_1_percent = fraud_in_1_percent / total_fraud_cases

    # Fraud capture rate at 10% depth
    fraud_in_10_percent = sum(sorted_y_true[:n_10_percent])
    fraud_capture_rate_10_percent = fraud_in_10_percent / total_fraud_cases

    # Fraud capture rate at 30% depth
    fraud_in_30_percent = sum(sorted_y_true[:n_30_percent])
    fraud_capture_rate_30_percent = fraud_in_30_percent / total_fraud_cases

    # # Display the results
    # print(f"Accuracy: {accuracy}")
    # print(f"Sensitivity (Recall): {sensitivity}")
    # print(f"Specificity: {specificity}")
    # print(f"Precision: {precision}")
    # print(f"F-measure (F1 Score): {f_measure}")
    # print(f"G-mean: {g_mean}")
    # print(f"Weighted Accuracy: {wtdAcc}")
    # print(f"AUC: {auc}")

    # Return all metrics as a dictionary
    performance_metrics = {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f_measure,
        "g_mean": g_mean,
        "wtd_acc": wtdAcc,
        "auc": auc,
        "fraud_capture_rate_1_percent":fraud_capture_rate_1_percent,
        "fraud_capture_rate_10_percent": fraud_capture_rate_10_percent,
        "fraud_capture_rate_30_percent": fraud_capture_rate_30_percent,
    }

    return performance_metrics
