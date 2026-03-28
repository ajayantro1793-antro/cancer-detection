"""
train_classifier.py  —  EfficientNet-B4 Classifier Training
─────────────────────────────────────────────────────────────────────────────
This script trains the EfficientNet-B4 classification model on the
IQ-OTH/NCCD dataset.

Key design decisions explained:
  1. Transfer learning — we start from ImageNet pretrained weights because
     the early layers already learn universal features (edges, textures,
     gradients) that are just as useful for CT scans as for natural images.
     We freeze those early layers and only fine-tune the last 3 blocks.

  2. Mixed precision (AMP) — we use torch.cuda.amp to run the forward pass
     in float16 and keep master weights in float32. This roughly halves VRAM
     usage on the RTX 4050 and speeds up training by ~30%.

  3. Label smoothing — instead of hard targets [0, 0, 1] we use soft targets
     like [0.033, 0.033, 0.933]. This prevents the model from becoming
     overconfident and improves calibration, which matters clinically.

  4. Early stopping — we monitor validation loss and stop training if it
     doesn't improve for `patience` epochs. This prevents overfitting and
     saves training time.

Usage:
    python models/classifier/train_classifier.py
─────────────────────────────────────────────────────────────────────────────
"""
import os
import sys
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import timm
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path so we can import from sibling packages
sys.path.append(str(Path(__file__).resolve().parents[2]))
from models.classifier.dataset import get_dataloaders, CLASS_NAMES

# ── Load config ────────────────────────────────────────────────────────────
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

C   = cfg["classifier"]
P   = cfg["paths"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── Model builder ──────────────────────────────────────────────────────────

def build_model(num_classes: int = 3,
                pretrained: bool = True,
                freeze_early: bool = True,
                unfreeze_blocks: int = 3) -> nn.Module:
    """
    Build EfficientNet-B4 for 3-class classification using the timm library.

    timm (PyTorch Image Models) is the standard library for pretrained
    vision models — it provides a clean API and up-to-date weights.

    Architecture overview for EfficientNet-B4:
        Input (3×512×512)
        → Stem conv
        → MBConv blocks (0–6) — we freeze 0 through (6-unfreeze_blocks)
        → Head conv
        → AdaptiveAvgPool → Flatten → Dropout(0.4) → FC(1792→num_classes)
    """
    model = timm.create_model(
        "efficientnet_b4",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=C["dropout"]       # dropout applied before final FC layer
    )

    # ── Freeze early layers to preserve general visual features ───────────
    if freeze_early:
        # Freeze everything first
        for param in model.parameters():
            param.requires_grad = False

        # Then unfreeze the last N MBConv blocks + head
        blocks = list(model.blocks)
        unfreeze_from = len(blocks) - unfreeze_blocks
        for block in blocks[unfreeze_from:]:
            for param in block.parameters():
                param.requires_grad = True

        # Always unfreeze the classification head
        for param in model.classifier.parameters():
            param.requires_grad = True
        for param in model.conv_head.parameters():
            param.requires_grad = True
        for param in model.bn2.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} "
              f"({100*trainable/total:.1f}%)")

    return model


# ── Training loop ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """
    Run one full pass through the training set.
    Returns average loss and accuracy for this epoch.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="  Training", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass — runs in float16 on GPU
        with autocast(enabled=C["mixed_precision"]):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        # Scaled backward pass — prevents underflow with float16 gradients
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on a validation or test DataLoader.
    Returns loss, accuracy, all predictions, and all true labels.
    The @torch.no_grad() decorator disables gradient tracking during
    evaluation — this saves memory and speeds things up.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="  Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=C["mixed_precision"]):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        probs  = torch.softmax(outputs, dim=1)
        preds  = probs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


# ── Plotting helpers ───────────────────────────────────────────────────────

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train Loss", color="#2563EB")
    ax1.plot(val_losses,   label="Val Loss",   color="#EF4444")
    ax1.set_title("Loss Curves");  ax1.set_xlabel("Epoch")
    ax1.legend();  ax1.grid(True, alpha=0.3)

    ax2.plot(train_accs, label="Train Acc", color="#2563EB")
    ax2.plot(val_accs,   label="Val Acc",   color="#EF4444")
    ax2.set_title("Accuracy Curves");  ax2.set_xlabel("Epoch")
    ax2.legend();  ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(save_dir) / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved training_curves.png")


def plot_confusion_matrix(y_true, y_pred, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted");  ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved confusion_matrix.png")


# ── Main training script ───────────────────────────────────────────────────

def main():
    torch.manual_seed(cfg["project"]["seed"])
    np.random.seed(cfg["project"]["seed"])

    save_dir = Path(P["classifier_output"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────
    print("\n[1/5] Loading datasets...")
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        data_dir   = P["iq_oth_nccd"],
        batch_size = C["batch_size"],
        image_size = C["image_size"],
        num_workers= 4
    )

    # ── Model ────────────────────────────────────────────────────────────
    print("\n[2/5] Building model...")
    model = build_model(
        num_classes    = C["num_classes"],
        pretrained     = C["pretrained"],
        freeze_early   = C["freeze_layers"],
        unfreeze_blocks= C["unfreeze_blocks"]
    ).to(DEVICE)

    # ── Loss & optimiser ─────────────────────────────────────────────────
    # Weighted cross-entropy with label smoothing.
    # class_weights penalise misclassification of the Malignant class more.
    weights   = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights,
                                     label_smoothing=C["label_smoothing"])

    # AdamW is Adam with decoupled weight decay — better regularisation
    # than standard Adam and recommended for fine-tuning pretrained models.
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=C["learning_rate"],
        weight_decay=C["weight_decay"]
    )

    # Cosine annealing scheduler: learning rate gradually decreases from
    # lr_max to lr_min following a cosine curve. This helps the model
    # converge to a better local minimum in later epochs.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=C["epochs"], eta_min=1e-6
    )

    # GradScaler manages the loss scaling required for mixed precision.
    scaler = GradScaler(enabled=C["mixed_precision"])

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n[3/5] Training for up to {C['epochs']} epochs...")
    best_val_loss   = float("inf")
    patience_count  = 0
    train_losses, val_losses = [], []
    train_accs, val_accs     = [], []

    for epoch in range(1, C["epochs"] + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE)
        vl_loss, vl_acc, _, _, _ = evaluate(
            model, val_loader, criterion, DEVICE)

        scheduler.step()
        elapsed = time.time() - t0

        train_losses.append(tr_loss);  val_losses.append(vl_loss)
        train_accs.append(tr_acc);     val_accs.append(vl_acc)

        print(f"Epoch {epoch:3d}/{C['epochs']} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | "
              f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.3f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Time: {elapsed:.1f}s")

        # ── Save best model ────────────────────────────────────────────
        if vl_loss < best_val_loss:
            best_val_loss  = vl_loss
            patience_count = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_loss":    vl_loss,
                "val_acc":     vl_acc,
                "class_names": CLASS_NAMES,
            }, save_dir / "best_model.pth")
            print(f"  ✓ Saved best model (val_loss={vl_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= C["early_stopping_patience"]:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {patience_count} epochs)")
                break

    # ── Final evaluation on test set ──────────────────────────────────────
    print("\n[4/5] Evaluating on test set...")
    checkpoint = torch.load(save_dir / "best_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    _, test_acc, y_pred, y_true, y_prob = evaluate(
        model, test_loader, criterion, DEVICE)

    print(f"\nTest Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # ROC-AUC (one-vs-rest for multiclass)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
        print(f"ROC-AUC (OvR): {auc:.4f}")
    except Exception:
        pass

    # ── Save plots ────────────────────────────────────────────────────────
    print("\n[5/5] Saving plots...")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir)
    plot_confusion_matrix(y_true, y_pred, save_dir)

    print(f"\nDone. Best model saved to {save_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()


