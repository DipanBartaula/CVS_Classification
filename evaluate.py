"""
Evaluation metrics for CVS Classification.

Computes comprehensive metrics for multi-label binary classification
of the 3 CVS criteria (C1, C2, C3).
"""
import json
import logging
import os
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    hamming_loss,
    jaccard_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_curve,
    precision_recall_curve,
)

import config
from utils import plot_confusion_matrices

logger = logging.getLogger("CVS_Classification")


def compute_metrics(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    all_preds: np.ndarray,
    criterion_names: List[str] = config.CVS_CRITERIA_NAMES,
    threshold: float = config.CLASSIFICATION_THRESHOLD,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Compute comprehensive metrics for multi-label CVS classification.
    
    Args:
        all_labels: Ground truth labels, shape (N, 3)
        all_probs: Predicted probabilities, shape (N, 3)
        all_preds: Binary predictions, shape (N, 3)
        criterion_names: Names of the 3 CVS criteria
        threshold: Classification threshold
        save_dir: Directory to save metrics reports
    
    Returns:
        Dictionary with all computed metrics
    """
    if save_dir is None:
        save_dir = config.RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    metrics = {}
    
    # ──────────────────────────────────────────
    #  Overall Multi-Label Metrics
    # ──────────────────────────────────────────
    metrics["overall"] = {
        "hamming_loss": float(hamming_loss(all_labels, all_preds)),
        "exact_match_ratio": float(accuracy_score(all_labels, all_preds)),  # Subset accuracy
        "micro_f1": float(f1_score(all_labels, all_preds, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(all_labels, all_preds, average="weighted", zero_division=0)),
        "samples_f1": float(f1_score(all_labels, all_preds, average="samples", zero_division=0)),
        "micro_precision": float(precision_score(all_labels, all_preds, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
        "micro_recall": float(recall_score(all_labels, all_preds, average="micro", zero_division=0)),
        "macro_recall": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
        "jaccard_micro": float(jaccard_score(all_labels, all_preds, average="micro", zero_division=0)),
        "jaccard_macro": float(jaccard_score(all_labels, all_preds, average="macro", zero_division=0)),
    }
    
    # AUROC (requires probabilities)
    try:
        metrics["overall"]["micro_auroc"] = float(
            roc_auc_score(all_labels, all_probs, average="micro")
        )
        metrics["overall"]["macro_auroc"] = float(
            roc_auc_score(all_labels, all_probs, average="macro")
        )
        metrics["overall"]["weighted_auroc"] = float(
            roc_auc_score(all_labels, all_probs, average="weighted")
        )
    except ValueError as e:
        logger.warning(f"Could not compute AUROC: {e}")
        metrics["overall"]["micro_auroc"] = 0.0
        metrics["overall"]["macro_auroc"] = 0.0
        metrics["overall"]["weighted_auroc"] = 0.0
    
    # Average Precision (mAP)
    try:
        metrics["overall"]["micro_ap"] = float(
            average_precision_score(all_labels, all_probs, average="micro")
        )
        metrics["overall"]["macro_ap"] = float(
            average_precision_score(all_labels, all_probs, average="macro")
        )
    except ValueError as e:
        logger.warning(f"Could not compute AP: {e}")
        metrics["overall"]["micro_ap"] = 0.0
        metrics["overall"]["macro_ap"] = 0.0
    
    # CVS Achievement Rate (all 3 criteria met)
    cvs_achieved_gt = (all_labels.sum(axis=1) == 3).astype(int)
    cvs_achieved_pred = (all_preds.sum(axis=1) == 3).astype(int)
    metrics["cvs_achievement"] = {
        "accuracy": float(accuracy_score(cvs_achieved_gt, cvs_achieved_pred)),
        "precision": float(precision_score(cvs_achieved_gt, cvs_achieved_pred, zero_division=0)),
        "recall": float(recall_score(cvs_achieved_gt, cvs_achieved_pred, zero_division=0)),
        "f1": float(f1_score(cvs_achieved_gt, cvs_achieved_pred, zero_division=0)),
    }
    try:
        metrics["cvs_achievement"]["auroc"] = float(
            roc_auc_score(cvs_achieved_gt, all_probs.min(axis=1))
        )
    except ValueError:
        metrics["cvs_achievement"]["auroc"] = 0.0
    
    # ──────────────────────────────────────────
    #  Per-Criterion Metrics
    # ──────────────────────────────────────────
    metrics["per_criterion"] = {}
    confusion_matrices = {}
    
    for i, name in enumerate(criterion_names):
        y_true = all_labels[:, i]
        y_pred = all_preds[:, i]
        y_prob = all_probs[:, i]
        
        criterion_metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "specificity": float(_compute_specificity(y_true, y_pred)),
            "npv": float(_compute_npv(y_true, y_pred)),
        }
        
        # MCC
        try:
            criterion_metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        except Exception:
            criterion_metrics["mcc"] = 0.0
        
        # Cohen's Kappa
        try:
            criterion_metrics["kappa"] = float(cohen_kappa_score(y_true, y_pred))
        except Exception:
            criterion_metrics["kappa"] = 0.0
        
        # AUROC
        try:
            criterion_metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            criterion_metrics["auroc"] = 0.0
        
        # Average Precision
        try:
            criterion_metrics["average_precision"] = float(
                average_precision_score(y_true, y_prob)
            )
        except ValueError:
            criterion_metrics["average_precision"] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        criterion_metrics["confusion_matrix"] = cm.tolist()
        confusion_matrices[name] = cm
        
        # Class distribution
        criterion_metrics["support"] = {
            "positive": int(y_true.sum()),
            "negative": int(len(y_true) - y_true.sum()),
            "total": int(len(y_true)),
        }
        
        metrics["per_criterion"][name] = criterion_metrics
    
    # ──────────────────────────────────────────
    #  Plot Confusion Matrices
    # ──────────────────────────────────────────
    try:
        plot_confusion_matrices(confusion_matrices, save_dir=save_dir)
    except Exception as e:
        logger.warning(f"Failed to plot confusion matrices: {e}")
    
    # ──────────────────────────────────────────
    #  Save Metrics Report
    # ──────────────────────────────────────────
    _save_metrics_report(metrics, save_dir)
    
    return metrics


def _compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute specificity (true negative rate)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn = cm[0, 0]
    fp = cm[0, 1]
    if (tn + fp) > 0:
        return tn / (tn + fp)
    return 0.0


def _compute_npv(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Negative Predictive Value."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn = cm[0, 0]
    fn = cm[1, 0]
    if (tn + fn) > 0:
        return tn / (tn + fn)
    return 0.0


def compute_optimal_thresholds(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    criterion_names: List[str] = config.CVS_CRITERIA_NAMES,
) -> Dict[str, float]:
    """
    Find optimal classification threshold per criterion using Youden's J statistic.
    (maximizes sensitivity + specificity - 1)
    
    Returns:
        Dict mapping criterion name to optimal threshold.
    """
    optimal_thresholds = {}
    
    for i, name in enumerate(criterion_names):
        y_true = all_labels[:, i]
        y_prob = all_probs[:, i]
        
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            j_scores = tpr - fpr  # Youden's J
            best_idx = np.argmax(j_scores)
            optimal_thresholds[name] = float(thresholds[best_idx])
        except Exception:
            optimal_thresholds[name] = 0.5
    
    return optimal_thresholds


def _save_metrics_report(metrics: Dict, save_dir: str):
    """Save comprehensive metrics report as JSON and formatted text."""
    # Save JSON
    json_path = os.path.join(save_dir, "evaluation_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved to: {json_path}")
    
    # Save formatted text report
    report_path = os.path.join(save_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  VideoMAE CVS Classification — Evaluation Report\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-" * 40 + "\n")
        for key, val in metrics["overall"].items():
            f.write(f"  {key:<25s}: {val:.4f}\n")
        
        f.write("\n\nCVS ACHIEVEMENT (All 3 Criteria Met)\n")
        f.write("-" * 40 + "\n")
        for key, val in metrics["cvs_achievement"].items():
            f.write(f"  {key:<25s}: {val:.4f}\n")
        
        f.write("\n\nPER-CRITERION METRICS\n")
        f.write("-" * 40 + "\n")
        for name, cm in metrics["per_criterion"].items():
            f.write(f"\n  {name}\n")
            f.write(f"  {'~' * len(name)}\n")
            for key, val in cm.items():
                if key not in ("confusion_matrix", "support"):
                    f.write(f"    {key:<25s}: {val:.4f}\n")
            if "support" in cm:
                f.write(f"    {'positive count':<25s}: {cm['support']['positive']}\n")
                f.write(f"    {'negative count':<25s}: {cm['support']['negative']}\n")
    
    logger.info(f"Report saved to: {report_path}")


def print_metrics_summary(metrics: Dict):
    """Print a formatted summary of evaluation metrics to console."""
    print("\n" + "=" * 70)
    print("  📊 VideoMAE CVS Classification — Results Summary")
    print("=" * 70)
    
    # Overall
    overall = metrics["overall"]
    print("\n  ┌─ OVERALL METRICS ──────────────────────────────┐")
    print(f"  │  Exact Match Ratio:    {overall['exact_match_ratio']:.4f}                 │")
    print(f"  │  Hamming Loss:         {overall['hamming_loss']:.4f}                 │")
    print(f"  │  Macro F1:             {overall['macro_f1']:.4f}                 │")
    print(f"  │  Macro AUROC:          {overall.get('macro_auroc', 0):.4f}                 │")
    print(f"  │  Macro AP (mAP):       {overall.get('macro_ap', 0):.4f}                 │")
    print(f"  │  Micro Precision:      {overall['micro_precision']:.4f}                 │")
    print(f"  │  Micro Recall:         {overall['micro_recall']:.4f}                 │")
    print("  └─────────────────────────────────────────────────┘")
    
    # CVS Achievement
    cvs = metrics["cvs_achievement"]
    print("\n  ┌─ CVS ACHIEVEMENT (All 3 Criteria) ─────────────┐")
    print(f"  │  Accuracy:  {cvs['accuracy']:.4f}    Precision: {cvs['precision']:.4f}       │")
    print(f"  │  Recall:    {cvs['recall']:.4f}    F1:        {cvs['f1']:.4f}       │")
    print(f"  │  AUROC:     {cvs.get('auroc', 0):.4f}                             │")
    print("  └─────────────────────────────────────────────────┘")
    
    # Per criterion
    print("\n  ┌─ PER-CRITERION METRICS ─────────────────────────┐")
    for name, cm in metrics["per_criterion"].items():
        print(f"  │                                                  │")
        print(f"  │  {name:<48s}│")
        print(f"  │  Acc: {cm['accuracy']:.3f}  Prec: {cm['precision']:.3f}  "
              f"Rec: {cm['recall']:.3f}  F1: {cm['f1']:.3f}  │")
        print(f"  │  AUROC: {cm.get('auroc', 0):.3f}  AP: {cm.get('average_precision', 0):.3f}  "
              f"MCC: {cm.get('mcc', 0):.3f}  κ: {cm.get('kappa', 0):.3f} │")
    print("  └──────────────────────────────────────────────────┘")
    print()
