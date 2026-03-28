"""
02_evaluate.py  —  Full Model Evaluation
─────────────────────────────────────────────────────────────────────────────
Run after training to get a complete picture of model performance:
  - Per-class precision, recall, F1
  - Confusion matrix (heatmap)
  - ROC curves and AUC for each class
  - Precision-Recall curves
  - YOLOv8 detection metrics (mAP, IoU distribution)
  - Sample predictions with Grad-CAM overlays
  - Model inference speed benchmark

Usage:
    python notebooks/02_evaluate.py
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import time
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.classifier.dataset import get_dataloaders, CLASS_NAMES
from models.classifier.train_classifier import build_model, evaluate
from app.inference import load_classifier, load_detector, predict, DEVICE

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

C = cfg["classifier"]
P = cfg["paths"]
OUT = Path("notebooks/eval_outputs")
OUT.mkdir(parents=True, exist_ok=True)


# ── Load models and data ───────────────────────────────────────────────────

def load_everything():
    print("Loading models and test data...")

    clf_path = Path(P["classifier_output"]) / "best_model.pth"
    det_path = Path(P["detector_output"]) / "runs" / "weights" / "best.pt"

    if not clf_path.exists():
        print(f"ERROR: Classifier weights not found at {clf_path}")
        print("Run: python models/classifier/train_classifier.py")
        sys.exit(1)

    classifier = load_classifier(str(clf_path))
    detector   = load_detector(str(det_path)) if det_path.exists() else None

    _, _, test_loader, _ = get_dataloaders(
        data_dir   = P["iq_oth_nccd"],
        batch_size = C["batch_size"],
        image_size = C["image_size"],
        num_workers= 0      # safer for Windows
    )

    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()

    print("Running inference on test set...")
    _, test_acc, y_pred, y_true, y_prob = evaluate(
        classifier, test_loader, criterion, DEVICE
    )
    print(f"Test Accuracy: {test_acc:.4f}")

    return classifier, detector, y_pred, y_true, y_prob, test_loader


# ── Plot 1: Confusion Matrix ───────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred):
    print("\n[1/5] Plotting confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Confusion Matrix — Test Set", fontsize=14, fontweight="bold")

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[0], linewidths=0.5, linecolor="white")
    axes[0].set_xlabel("Predicted Class", fontweight="bold")
    axes[0].set_ylabel("True Class", fontweight="bold")
    axes[0].set_title("Raw Counts")

    # Normalised
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="RdYlGn",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[1], linewidths=0.5, linecolor="white",
                vmin=0, vmax=1)
    axes[1].set_xlabel("Predicted Class", fontweight="bold")
    axes[1].set_ylabel("True Class", fontweight="bold")
    axes[1].set_title("Normalised (Recall per Class)")

    plt.tight_layout()
    plt.savefig(OUT / "01_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 01_confusion_matrix.png")

    # Print classification report
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


# ── Plot 2: ROC Curves ────────────────────────────────────────────────────

def plot_roc_curves(y_true, y_prob):
    print("\n[2/5] Plotting ROC curves...")

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#059669", "#D97706", "#EF4444"]

    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        # One-vs-Rest: treat class i as positive, all others as negative
        y_bin  = (y_true == i).astype(int)
        y_sc   = y_prob[:, i]
        fpr, tpr, _ = roc_curve(y_bin, y_sc)
        auc = roc_auc_score(y_bin, y_sc)
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f"{cls}  (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random classifier")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curves — One vs Rest\n"
                 "EfficientNet-B4 on Test Set", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    # Overall OvR AUC
    overall_auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    ax.text(0.72, 0.08, f"Macro AUC = {overall_auc:.3f}",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="#EBF4FF", alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUT / "02_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Overall ROC-AUC (OvR macro): {overall_auc:.4f}")
    print(f"  Saved: 02_roc_curves.png")


# ── Plot 3: Precision-Recall Curves ───────────────────────────────────────

def plot_pr_curves(y_true, y_prob):
    print("\n[3/5] Plotting Precision-Recall curves...")
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#059669", "#D97706", "#EF4444"]

    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        y_bin = (y_true == i).astype(int)
        y_sc  = y_prob[:, i]
        prec, rec, _ = precision_recall_curve(y_bin, y_sc)
        ap = average_precision_score(y_bin, y_sc)
        ax.plot(rec, prec, color=color, linewidth=2.5,
                label=f"{cls}  (AP = {ap:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves\n"
                 "EfficientNet-B4 on Test Set", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "03_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 03_pr_curves.png")


# ── Plot 4: Confidence Distribution ───────────────────────────────────────

def plot_confidence_distribution(y_true, y_prob, y_pred):
    print("\n[4/5] Plotting confidence distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Confidence Analysis", fontsize=13, fontweight="bold")

    # Confidence on correct vs incorrect predictions
    max_probs = y_prob.max(axis=1)
    correct   = (y_pred == y_true)

    axes[0].hist(max_probs[correct],  bins=30, alpha=0.7,
                 color="#059669", label=f"Correct ({correct.sum()})", density=True)
    axes[0].hist(max_probs[~correct], bins=30, alpha=0.7,
                 color="#EF4444", label=f"Incorrect ({(~correct).sum()})", density=True)
    axes[0].set_title("Confidence: Correct vs Incorrect Predictions")
    axes[0].set_xlabel("Max Softmax Probability")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].axvline(0.75, color="black", linestyle="--",
                    label="75% threshold", linewidth=1.5)

    # Per-class confidence for correctly classified samples
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, ["#059669","#D97706","#EF4444"])):
        mask = (y_true == i) & correct
        if mask.sum() > 0:
            axes[1].hist(max_probs[mask], bins=20, alpha=0.6,
                         color=color, label=f"{cls} (n={mask.sum()})", density=True)

    axes[1].set_title("Confidence per Class (Correct Predictions)")
    axes[1].set_xlabel("Max Softmax Probability")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUT / "04_confidence_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 04_confidence_distribution.png")


# ── Benchmark 5: Inference Speed ──────────────────────────────────────────

def benchmark_inference_speed(classifier, detector, n_runs: int = 50):
    print(f"\n[5/5] Benchmarking inference speed ({n_runs} runs)...")

    # Create a dummy 512x512 RGB image
    from PIL import Image as PILImage
    dummy_img = PILImage.fromarray(
        np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    )

    # Warm up GPU
    for _ in range(5):
        _ = predict(dummy_img, classifier, detector)

    # Timed runs
    times = []
    for _ in tqdm(range(n_runs), desc="  Benchmarking"):
        t0 = time.perf_counter()
        _ = predict(dummy_img, classifier, detector)
        times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000  # convert to ms
    print(f"\n  Inference Speed on {DEVICE.type.upper()}:")
    print(f"  Mean:   {times.mean():.1f} ms")
    print(f"  Median: {np.median(times):.1f} ms")
    print(f"  P95:    {np.percentile(times, 95):.1f} ms")
    print(f"  Min:    {times.min():.1f} ms")
    print(f"  Max:    {times.max():.1f} ms")
    print(f"  Throughput: {1000/times.mean():.1f} scans/second")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(times, bins=20, color="#2563EB", alpha=0.8, edgecolor="none")
    ax.axvline(times.mean(), color="#EF4444", linestyle="--",
               linewidth=2, label=f"Mean = {times.mean():.1f}ms")
    ax.axvline(np.percentile(times, 95), color="#F59E0B", linestyle="--",
               linewidth=2, label=f"P95 = {np.percentile(times, 95):.1f}ms")
    ax.set_title(f"Inference Latency Distribution\n"
                 f"({n_runs} runs on {DEVICE.type.upper()})",
                 fontweight="bold")
    ax.set_xlabel("Inference Time (ms)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "05_inference_speed.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 05_inference_speed.png")


# ── Summary ───────────────────────────────────────────────────────────────

def print_eval_summary(y_true, y_pred, y_prob):
    print("\n" + "=" * 55)
    print("  EVALUATION SUMMARY")
    print("=" * 55)

    acc = (y_pred == y_true).mean()
    auc = roc_auc_score(y_true, y_prob, multi_class="ovr")

    print(f"\n  Overall Accuracy:  {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  ROC-AUC (macro):   {auc:.4f}")
    print(f"\n  Per-Class Metrics:")

    report = classification_report(y_true, y_pred,
                                   target_names=CLASS_NAMES,
                                   output_dict=True)
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*45}")
    for cls in CLASS_NAMES:
        m = report[cls]
        print(f"  {cls:<12} {m['precision']:>10.3f} "
              f"{m['recall']:>10.3f} {m['f1-score']:>10.3f}")

    print(f"\n  All plots saved to: notebooks/eval_outputs/")
    print("=" * 55)


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  MODEL EVALUATION")
    print("=" * 55)

    classifier, detector, y_pred, y_true, y_prob, test_loader = load_everything()

    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curves(y_true, y_prob)
    plot_pr_curves(y_true, y_prob)
    plot_confidence_distribution(y_true, y_prob, y_pred)
    benchmark_inference_speed(classifier, detector)
    print_eval_summary(y_true, y_pred, y_prob)


