import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # XGBoost predict() dapat mengembalikan float — pastikan int
    y_pred = y_pred.astype(int)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)
    report    = classification_report(y_test, y_pred, zero_division=0)

    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

    return {
        'accuracy':              accuracy,
        'precision':             precision,
        'recall':                recall,
        'f1':                    f1,
        'f1_per_class':          f1_per_class,
        'confusion_matrix':      cm,
        'classification_report': report,
        'y_pred':                y_pred,
    }
