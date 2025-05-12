import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_curve,
    roc_curve,
    auc
)


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
                   output_dir: str = '.') -> dict:
    os.makedirs(output_dir, exist_ok=True)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(
        model, 'predict_proba') else None

    # Calculate basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    print(f"Saved: {os.path.join(output_dir, 'confusion_matrix.png')}")

    metrics = {'accuracy': accuracy, 'classification_report': report}

    if hasattr(model[-1], 'feature_importances_'):
        importances = model[-1].feature_importances_
        feature_names = model[0].get_feature_names_out()

        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 8))
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i]
                   for i in indices], rotation=90)
        plt.xlim([-1, min(20, len(indices))])  # Only show top 20 features
        plt.tight_layout()
        plt.title('Feature Importance')
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        print(f"Saved: {os.path.join(output_dir, 'feature_importance.png')}")

        metrics['feature_importance'] = {
            feature_names[i]: importances[i] for i in indices
        }

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        print(f"Saved: {os.path.join(output_dir, 'roc_curve.png')}")

        precision, recall, _ = precision_recall_curve(y_test, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
        print(
            f"Saved: {os.path.join(output_dir, 'precision_recall_curve.png')}")

        metrics['roc_auc'] = roc_auc

    return metrics
