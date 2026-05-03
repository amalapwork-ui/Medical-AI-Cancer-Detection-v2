"""
Shared evaluation utilities for all medical AI training scripts.
Generates classification report, confusion matrices, confidence
distribution plots, and per-class metric bar charts.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


# ─────────────────────────────────────────────
# Main evaluation entry point
# ─────────────────────────────────────────────

def evaluate_model(model, dataset, class_names: list, output_dir: str) -> dict:
    """
    Full evaluation suite: collect predictions, print metrics, save plots.

    Returns a dict with keys: y_true, y_pred, y_pred_probs, report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred_probs = [], []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        y_pred_probs.extend(probs)
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report_str = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report_str)

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    _plot_confusion_matrix(y_true, y_pred, class_names, output_dir)
    _plot_confidence_distribution(y_true, y_pred, y_pred_probs, output_dir)
    _plot_per_class_metrics(report_dict, class_names, output_dir)

    overall_acc = float(report_dict["accuracy"])
    macro_f1    = float(report_dict["macro avg"]["f1-score"])
    print(f"\nOverall accuracy : {overall_acc:.4f}")
    print(f"Macro F1-score   : {macro_f1:.4f}")

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_probs": y_pred_probs,
        "report": report_dict,
        "accuracy": overall_acc,
        "macro_f1": macro_f1,
    }


# ─────────────────────────────────────────────
# Individual plot helpers
# ─────────────────────────────────────────────

def _plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    cm       = confusion_matrix(y_true, y_pred)
    cm_norm  = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", ax=axes[0])
    axes[0].set_title("Confusion Matrix — Counts")
    axes[0].set_ylabel("True"); axes[0].set_xlabel("Predicted")

    sns.heatmap(cm_norm, annot=True, fmt=".2%",
                xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", ax=axes[1])
    axes[1].set_title("Confusion Matrix — Normalized")
    axes[1].set_ylabel("True"); axes[1].set_xlabel("Predicted")

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def _plot_confidence_distribution(y_true, y_pred, y_pred_probs, output_dir):
    max_probs = np.max(y_pred_probs, axis=1)
    correct   = y_true == y_pred

    plt.figure(figsize=(10, 4))
    plt.hist(max_probs[correct],  bins=50, alpha=0.75,
             label=f"Correct ({correct.sum()})",    color="steelblue")
    plt.hist(max_probs[~correct], bins=50, alpha=0.75,
             label=f"Incorrect ({(~correct).sum()})", color="coral")
    plt.axvline(0.85, color="orange", ls="--", label="0.85 threshold")
    plt.axvline(0.70, color="red",    ls="--", label="0.70 threshold")
    plt.xlabel("Max Softmax Probability")
    plt.ylabel("Count")
    plt.title("Prediction Confidence Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_distribution.png", dpi=150)
    plt.show()
    plt.close()

    # Calibration summary
    for threshold in [0.50, 0.70, 0.80, 0.85, 0.90, 0.95]:
        mask    = max_probs >= threshold
        if mask.sum() == 0:
            continue
        acc_at  = (y_true[mask] == y_pred[mask]).mean()
        coverage = mask.mean()
        print(f"  Threshold ≥ {threshold:.2f}: coverage={coverage:.1%}, accuracy={acc_at:.4f}")


def _plot_per_class_metrics(report_dict, class_names, output_dir):
    classes    = [c for c in class_names if c in report_dict]
    precisions = [report_dict[c]["precision"] for c in classes]
    recalls    = [report_dict[c]["recall"]    for c in classes]
    f1s        = [report_dict[c]["f1-score"]  for c in classes]

    x     = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 2), 5))
    ax.bar(x - width, precisions, width, label="Precision", color="steelblue")
    ax.bar(x,         recalls,    width, label="Recall",    color="coral")
    ax.bar(x + width, f1s,        width, label="F1-Score",  color="mediumseagreen")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics")
    ax.axhline(0.90, color="red", ls="--", alpha=0.5, label="Target = 0.90")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "per_class_metrics.png", dpi=150)
    plt.show()
    plt.close()


# ─────────────────────────────────────────────
# Training curve plotter
# ─────────────────────────────────────────────

def plot_training_curves(h1, h2, output_dir: str, fine_tune_start: int = None):
    """
    Plot accuracy and loss curves across two training phases.
    h1 / h2 are keras History objects (h2 can be None for single-phase training).
    """
    output_dir = Path(output_dir)

    def _cat(key):
        a = h1.history.get(key, [])
        b = h2.history.get(key, []) if h2 else []
        return a + b

    acc     = _cat("accuracy")
    val_acc = _cat("val_accuracy")
    loss    = _cat("loss")
    val_loss= _cat("val_loss")
    ft      = fine_tune_start or (len(h1.history["accuracy"]) if h2 else None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(acc,     label="Train")
    axes[0].plot(val_acc, label="Validation")
    if ft:
        axes[0].axvline(ft, color="r", ls="--", label="Fine-tune start")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylim(0, 1.05); axes[0].legend()

    axes[1].plot(loss,     label="Train")
    axes[1].plot(val_loss, label="Validation")
    if ft:
        axes[1].axvline(ft, color="r", ls="--", label="Fine-tune start")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.show()
    plt.close()
