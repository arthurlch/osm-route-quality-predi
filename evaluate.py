import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import logging
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_curve,
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(model: Pipeline,
                   X_test: pd.DataFrame,
                   y_test: pd.Series,
                   output_dir: str = '.',
                   region_name: Optional[str] = None,
                   threshold: float = 0.5) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    prefix = ""
    if region_name:
        prefix = f"{region_name.replace(', ', '_').replace(' ', '_')}_"

    y_pred = model.predict(X_test)

    y_prob = None
    if hasattr(model, 'predict_proba'):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred_threshold = (y_prob >= threshold).astype(int)
        except Exception as e:
            logger.warning(f"Could not get prediction probabilities: {e}")

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))

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

    cm_path = os.path.join(output_dir, f"{prefix}confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Saved: {cm_path}")

    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'classification_report': report,
        'threshold_used': threshold
    }

    if hasattr(model[-1], 'feature_importances_'):
        try:
            importances = model[-1].feature_importances_

            if hasattr(model[0], 'get_feature_names_out'):
                feature_names = model[0].get_feature_names_out()
            else:
                feature_names = np.array(
                    [f"Feature {i}" for i in range(len(importances))])

            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(12, 8))
            plt.bar(range(len(indices)), importances[indices], align='center')
            plt.xticks(range(len(indices)), [
                       feature_names[i] for i in indices], rotation=90)
            plt.xlim([-1, min(20, len(indices))])  # Only show top 20 features
            plt.tight_layout()
            plt.title('Feature Importance')

            importance_path = os.path.join(
                output_dir, f"{prefix}feature_importance.png")
            plt.savefig(importance_path)
            plt.close()
            logger.info(f"Saved: {importance_path}")

            metrics['feature_importance'] = {
                feature_names[i]: importances[i] for i in indices
            }
        except Exception as e:
            logger.warning(f"Could not extract feature importances: {e}")

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

        roc_path = os.path.join(output_dir, f"{prefix}roc_curve.png")
        plt.savefig(roc_path)
        plt.close()
        logger.info(f"Saved: {roc_path}")

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')

        pr_path = os.path.join(
            output_dir, f"{prefix}precision_recall_curve.png")
        plt.savefig(pr_path)
        plt.close()
        logger.info(f"Saved: {pr_path}")

        metrics['roc_auc'] = roc_auc

    import json
    metrics_path = os.path.join(output_dir, f"{prefix}evaluation_metrics.json")

    def json_serialize(obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: json_serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [json_serialize(i) for i in obj]
        else:
            return obj

    with open(metrics_path, 'w') as f:
        json.dump(json_serialize(metrics), f, indent=2)
    logger.info(f"Saved metrics to: {metrics_path}")

    return metrics


def cross_region_evaluation(model: Pipeline,
                            test_regions: Dict[str, tuple],
                            output_dir: str = 'cross_region_eval') -> Dict[str, Dict[str, Any]]:

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for region_name, (X_test, y_test) in test_regions.items():
        logger.info(f"Evaluating model on region: {region_name}")

        region_dir = os.path.join(
            output_dir, region_name.replace(', ', '_').replace(' ', '_'))
        os.makedirs(region_dir, exist_ok=True)

        metrics = evaluate_model(
            model, X_test, y_test, output_dir=region_dir, region_name=region_name)
        results[region_name] = metrics

    plt.figure(figsize=(12, 8))
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    regions = list(results.keys())

    data = []
    for metric in metrics:
        metric_values = [results[region].get(metric, 0) for region in regions]
        data.append(metric_values)

    x = np.arange(len(regions))
    width = 0.2
    offsets = np.linspace(-0.3, 0.3, len(metrics))

    for i, (metric, values) in enumerate(zip(metrics, data)):
        plt.bar(x + offsets[i], values, width, label=metric)

    plt.xlabel('Region')
    plt.ylabel('Score')
    plt.title('Model Performance Across Regions')
    plt.xticks(x, regions, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

    comparison_path = os.path.join(output_dir, 'cross_region_comparison.png')
    plt.savefig(comparison_path)
    plt.close()
    logger.info(f"Saved cross-region comparison to: {comparison_path}")

    return results
