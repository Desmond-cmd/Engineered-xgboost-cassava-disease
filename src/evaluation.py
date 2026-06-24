import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from scipy.stats import wilcoxon


DISEASE_LABELS = {
    0: 'Bacterial Blight (CBB)',
    1: 'Brown Streak (CBSD)',
    2: 'Green Mottle (CGM)',
    3: 'Mosaic Disease (CMD)',
    4: 'Healthy'
}


def evaluate_model(y_true, y_pred, y_prob, model_name='Model'):
    """
    Compute all evaluation metrics for a trained model.

    Parameters
    ----------
    y_true : np.array
        True class labels
    y_pred : np.array
        Predicted class labels
    y_prob : np.array
        Predicted probabilities (n_samples, n_classes)
    model_name : str
        Name of the model for display

    Returns
    -------
    dict : all computed metrics
    """
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
        'auc_roc': roc_auc_score(
            y_true, y_prob, multi_class='ovr', average='macro'
        ),
        'auc_pr': average_precision_score(
            y_true,
            y_prob,
            average='macro'
        ),
    }

    print(f"\n{'='*50}")
    print(f"Results for: {model_name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != 'model':
            print(f"  {k:20s}: {v:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=list(DISEASE_LABELS.values())
    ))

    return metrics


def compare_models(baseline_scores, engineered_scores, metric_names=None):
    """
    Formally compare baseline vs E-XGBoost using Wilcoxon Signed-Rank Test.
    Reports delta, p-value, and significance for each metric.

    Parameters
    ----------
    baseline_scores : dict
        Dict of metric_name -> list of scores across folds
    engineered_scores : dict
        Dict of metric_name -> list of scores across folds
    metric_names : list
        Metrics to compare. Defaults to main 4 metrics.

    Returns
    -------
    pd.DataFrame : comparison table with p-values and deltas
    """
    if metric_names is None:
        metric_names = ['accuracy', 'macro_f1', 'auc_roc', 'auc_pr']

    rows = []
    for metric in metric_names:
        b_scores = np.array(baseline_scores[metric])
        e_scores = np.array(engineered_scores[metric])

        stat, p_value = wilcoxon(b_scores, e_scores)
        delta = np.mean(e_scores) - np.mean(b_scores)

        rows.append({
            'Metric': metric,
            'Baseline (mean±std)': f"{np.mean(b_scores):.4f}±{np.std(b_scores):.4f}",
            'E-XGBoost (mean±std)': f"{np.mean(e_scores):.4f}±{np.std(e_scores):.4f}",
            'Delta': f"{delta:+.4f}",
            'p-value': f"{p_value:.4f}",
            'Significant': 'Yes' if p_value < 0.05 else 'No',
            'Better': 'Yes' if delta > 0 else 'No'
        })

    df = pd.DataFrame(rows)
    print("\nBaseline vs E-XGBoost Comparison Table")
    print("="*80)
    print(df.to_string(index=False))
    return df


def plot_confusion_matrix(y_true, y_pred, model_name='Model',
                           save_path=None):
    """
    Plot normalised confusion matrix for cassava disease classes.
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f',
        cmap='Greens',
        xticklabels=list(DISEASE_LABELS.values()),
        yticklabels=list(DISEASE_LABELS.values())
    )
    plt.title(f'Normalised Confusion Matrix — {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


def save_metrics_to_csv(metrics_dict, filepath):
    """
    Save evaluation metrics to CSV for reproducibility.

    Parameters
    ----------
    metrics_dict : dict
        Output from evaluate_model()
    filepath : str
        Path to save CSV file
    """
    df = pd.DataFrame([metrics_dict])
    df.to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")
