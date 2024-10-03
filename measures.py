from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

# # Example: True labels and predicted labels from your fraud detection model
# y_true = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]  # Actual labels (0 = Non-fraud, 1 = Fraud)
# y_pred = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1]  # Predicted labels by the model
def measures(y_true, y_pred):
    # 1. Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 2. Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # 3. Sensitivity (Recall)
    sensitivity = recall_score(y_true, y_pred)  # TP / (TP + FN)

    # 4. Specificity
    specificity = tn / (tn + fp)

    # 5. Precision
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

    # Display the results
    print(f"Accuracy: {accuracy}")
    print(f"Sensitivity (Recall): {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Precision: {precision}")
    print(f"F-measure (F1 Score): {f_measure}")
    print(f"G-mean: {g_mean}")
    print(f"Weighted Accuracy: {wtdAcc}")
    print(f"AUC: {auc}")

    # Return all metrics as a dictionary
    performance_metrics = {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f_measure,
        "g_mean": g_mean,
        "wtd_acc": wtdAcc,
        "auc": auc
    }

    return performance_metrics
