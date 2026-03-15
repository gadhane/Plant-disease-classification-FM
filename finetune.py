"""
Fine-tuning the Foundation Model for Plant Disease Classification
Loads the SSL pre-trained encoder (foundation_encoder.pt) and adds a
classification head on top, then trains on labelled plant disease images.

"""

import argparse
import csv
import os
import sys

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

matplotlib.use("Agg")   # non-interactive backend (works without a display)
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Make sure model.py is importable from the same directory
sys.path.append(os.path.dirname(__file__))
from model import Encoder

sys.stdout.reconfigure(encoding="utf-8")


# ============================================================================
# 1.  Classifier Model
# ============================================================================

class PlantDiseaseClassifier(nn.Module):
    """
    Foundation encoder  +  classification head.

    Three fine-tuning strategies are supported (set via --strategy):

      'frozen'   -- encoder weights locked, only head trains
                    Best when: few labelled samples, short training time needed
                    Risk: encoder features may not be perfectly aligned to task

      'partial'  -- only the last ResNet layer-4 + head unfreeze
                    Best when: moderate data, good balance of speed vs accuracy
                    Risk: lower layers may miss domain-specific low-level features

      'full'     -- entire encoder + head train together
                    Best when: ample labelled data, willing to train longer
                    Risk: overfitting if dataset is small
    """

    STRATEGIES = ("frozen", "partial", "full")

    def __init__(
        self,
        encoder:    Encoder,
        num_classes: int = 38,
        strategy:   str  = "partial",
        dropout:    float = 0.3,
    ):
        super().__init__()
        assert strategy in self.STRATEGIES, \
            f"strategy must be one of {self.STRATEGIES}, got '{strategy}'"

        self.encoder  = encoder
        self.strategy = strategy

        # --- Apply freezing strategy ---
        self._apply_strategy(strategy)

        # --- Classification head ---
        # Input dim = encoder embed_dim (256 by default)
        embed_dim = encoder.embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),   # raw logits — no softmax (CrossEntropyLoss handles it)
        )

    # ------------------------------------------------------------------
    def _apply_strategy(self, strategy: str):
        if strategy == "frozen":
            # Lock all encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False

        elif strategy == "partial":
            # Lock everything first
            for param in self.encoder.parameters():
                param.requires_grad = False
            # Unfreeze only layer4 (deepest ResNet block) + the embedding fc
            for param in self.encoder.backbone[-2].parameters():   # layer4
                param.requires_grad = True
            for param in self.encoder.fc.parameters():             # embedding fc
                param.requires_grad = True

        elif strategy == "full":
            # Unfreeze everything
            for param in self.encoder.parameters():
                param.requires_grad = True

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)          # (B, embed_dim)
        logits   = self.classifier(features) # (B, num_classes)
        return logits


# ============================================================================
# 2.  Data Loading
# ============================================================================

def get_transforms(image_size: int = 224):
    """
    Train: moderate augmentation (do NOT use SSL-style strong augmentations —
           those are for learning representations, not for classification).
    Val  : deterministic resize + centre crop only.
    """
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),   # slight oversize then centre crop
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_tf, val_tf


def get_dataloaders(
    data_dir:    str,
    batch_size:  int = 32,
    image_size:  int = 224,
    num_workers: int = 2,
):
    train_tf, val_tf = get_transforms(image_size)

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=train_tf,
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "valid"),
        transform=val_tf,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )

    class_names = train_dataset.classes   # sorted list of 38 class folder names
    print(f"[Data] Train samples : {len(train_dataset):,}")
    print(f"[Data] Val   samples : {len(val_dataset):,}")
    print(f"[Data] Classes found : {len(class_names)}")

    return train_loader, val_loader, class_names


# ============================================================================
# 3.  Metrics
# ============================================================================

def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """Fraction of samples where true label is in top-k predictions."""
    _, top_k_preds = logits.topk(k, dim=1)
    correct = top_k_preds.eq(targets.view(-1, 1).expand_as(top_k_preds))
    return correct.any(dim=1).float().mean().item()


def compute_metrics(all_targets, all_preds, all_logits, class_names):
    """
    Compute the full suite of classification metrics.

    Returns a dict with:
      accuracy, top5_accuracy,
      precision_macro, precision_weighted,
      recall_macro,    recall_weighted,
      f1_macro,        f1_weighted,
      per_class_accuracy  (list, one per class)
    """
    all_targets = np.array(all_targets)
    all_preds   = np.array(all_preds)
    all_logits  = torch.tensor(np.array(all_logits))  # (N, C)
    all_t       = torch.tensor(all_targets)

    accuracy     = (all_preds == all_targets).mean()
    top5_acc     = top_k_accuracy(all_logits, all_t, k=min(5, len(class_names)))

    precision_macro    = precision_score(all_targets, all_preds, average="macro",    zero_division=0)
    precision_weighted = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    recall_macro       = recall_score   (all_targets, all_preds, average="macro",    zero_division=0)
    recall_weighted    = recall_score   (all_targets, all_preds, average="weighted", zero_division=0)
    f1_macro           = f1_score       (all_targets, all_preds, average="macro",    zero_division=0)
    f1_weighted        = f1_score       (all_targets, all_preds, average="weighted", zero_division=0)

    # Per-class accuracy
    per_class_acc = []
    for cls_idx in range(len(class_names)):
        mask = all_targets == cls_idx
        if mask.sum() > 0:
            per_class_acc.append((all_preds[mask] == cls_idx).mean())
        else:
            per_class_acc.append(0.0)

    return {
        "accuracy":            float(accuracy),
        "top5_accuracy":       float(top5_acc),
        "precision_macro":     float(precision_macro),
        "precision_weighted":  float(precision_weighted),
        "recall_macro":        float(recall_macro),
        "recall_weighted":     float(recall_weighted),
        "f1_macro":            float(f1_macro),
        "f1_weighted":         float(f1_weighted),
        "per_class_accuracy":  per_class_acc,
    }


# ============================================================================
# 4.  Confusion Matrix Plot
# ============================================================================

def save_confusion_matrix(all_targets, all_preds, class_names, save_path):
    cm = confusion_matrix(all_targets, all_preds)
    # Normalise rows to percentage for readability across imbalanced classes
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    n = len(class_names)
    fig_size = max(16, n // 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix (row-normalised)", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[Metrics] Confusion matrix saved >> {save_path}")


# ============================================================================
# 5.  Train / Validate one epoch
# ============================================================================

def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    model.train() if is_train else model.eval()

    total_loss   = 0.0
    all_targets, all_preds, all_logits = [], [], []

    with torch.set_grad_enabled(is_train):
        for batch_idx, (images, targets) in enumerate(loader):
            images  = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss   = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1)

            total_loss   += loss.item()
            all_targets  += targets.cpu().tolist()
            all_preds    += preds.cpu().tolist()
            all_logits   += logits.detach().cpu().tolist()

            if is_train and (batch_idx + 1) % 500 == 0:
                print(f"    Batch {batch_idx+1}/{len(loader)}  loss={loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    return avg_loss, all_targets, all_preds, all_logits


# ============================================================================
# 6.  Metrics CSV Logger
# ============================================================================

def init_csv(save_path, fieldnames):
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_csv(save_path, row: dict):
    with open(save_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)


# ============================================================================
# 7.  Argument Parsing
# ============================================================================

def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune Plant Disease Classifier")

    parser.add_argument("--data_dir",        type=str,   default="Plant Disease/Images/Data/",
                        help="Root dir with train/ and valid/ subfolders")
    parser.add_argument("--checkpoint_path", type=str,   default="checkpoints/foundation_encoder.pt",
                        help="Path to the SSL pre-trained foundation_encoder.pt")
    parser.add_argument("--save_dir",        type=str,   default="finetune_checkpoints",
                        help="Where to save fine-tuning outputs")
    parser.add_argument("--strategy",        type=str,   default="partial",
                        choices=["frozen", "partial", "full"],
                        help="Freezing strategy: frozen | partial | full")
    parser.add_argument("--epochs",          type=int,   default=30)
    parser.add_argument("--batch_size",      type=int,   default=16)
    parser.add_argument("--image_size",      type=int,   default=224)
    parser.add_argument("--embed_dim",       type=int,   default=256,
                        help="Must match the embed_dim used during SSL pre-training")
    parser.add_argument("--num_classes",     type=int,   default=38)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--weight_decay",    type=float, default=1e-4)
    parser.add_argument("--dropout",         type=float, default=0.3)
    parser.add_argument("--num_workers",     type=int,   default=2)

    return parser.parse_args()


# ============================================================================
# 8.  Main
# ============================================================================

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Plant Disease Classifier -- Fine-tuning")
    print(f"{'='*55}")
    print(f"  Device       : {device}")
    print(f"  Data dir     : {args.data_dir}")
    print(f"  Foundation   : {args.checkpoint_path}")
    print(f"  Strategy     : {args.strategy}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  LR           : {args.lr}")
    print(f"{'='*55}\n")

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    assert len(class_names) == args.num_classes, (
        f"Found {len(class_names)} classes in data, expected {args.num_classes}. "
        "Check --num_classes or data folder structure."
    )

    # -----------------------------------------------------------------------
    # Load pre-trained encoder
    # -----------------------------------------------------------------------
    print(f"\n[Model] Loading foundation encoder from: {args.checkpoint_path}")
    encoder = Encoder(embed_dim=args.embed_dim)

    ckpt = torch.load(args.checkpoint_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder_state"])
    print(f"[Model] Encoder loaded successfully (embed_dim={ckpt['embed_dim']})")

    # -----------------------------------------------------------------------
    # Build classifier
    # -----------------------------------------------------------------------
    model = PlantDiseaseClassifier(
        encoder=encoder,
        num_classes=args.num_classes,
        strategy=args.strategy,
        dropout=args.dropout,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[Model] Strategy      : '{args.strategy}'")
    print(f"[Model] Trainable params : {trainable:,} / {total:,} total\n")

    # -----------------------------------------------------------------------
    # Loss, optimiser, scheduler
    # -----------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # smoothing helps 38-class generalisation

    # Only pass parameters that require gradients to the optimiser
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # -----------------------------------------------------------------------
    # CSV logger setup
    # -----------------------------------------------------------------------
    csv_path   = os.path.join(args.save_dir, "metrics_log.csv")
    csv_fields = [
        "epoch", "phase", "loss",
        "accuracy", "top5_accuracy",
        "precision_macro", "precision_weighted",
        "recall_macro",    "recall_weighted",
        "f1_macro",        "f1_weighted",
    ]
    init_csv(csv_path, csv_fields)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_val_acc   = 0.0
    best_val_f1    = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}  (lr={scheduler.get_last_lr()[0]:.6f})")
        print("-" * 45)

        # --- Train ---
        print("  [Train]")
        train_loss, t_targets, t_preds, t_logits = run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True
        )
        train_m = compute_metrics(t_targets, t_preds, t_logits, class_names)

        # --- Validate ---
        print("  [Val]")
        val_loss, v_targets, v_preds, v_logits = run_epoch(
            model, val_loader, criterion, None, device, is_train=False
        )
        val_m = compute_metrics(v_targets, v_preds, v_logits, class_names)

        scheduler.step()

        # --- Print epoch summary ---
        print(f"\n  {'Metric':<25} {'Train':>10} {'Val':>10}")
        print(f"  {'-'*47}")
        print(f"  {'Loss':<25} {train_loss:>10.4f} {val_loss:>10.4f}")
        print(f"  {'Accuracy':<25} {train_m['accuracy']:>10.4f} {val_m['accuracy']:>10.4f}")
        print(f"  {'Top-5 Accuracy':<25} {train_m['top5_accuracy']:>10.4f} {val_m['top5_accuracy']:>10.4f}")
        print(f"  {'Precision (macro)':<25} {train_m['precision_macro']:>10.4f} {val_m['precision_macro']:>10.4f}")
        print(f"  {'Precision (weighted)':<25} {train_m['precision_weighted']:>10.4f} {val_m['precision_weighted']:>10.4f}")
        print(f"  {'Recall (macro)':<25} {train_m['recall_macro']:>10.4f} {val_m['recall_macro']:>10.4f}")
        print(f"  {'Recall (weighted)':<25} {train_m['recall_weighted']:>10.4f} {val_m['recall_weighted']:>10.4f}")
        print(f"  {'F1 (macro)':<25} {train_m['f1_macro']:>10.4f} {val_m['f1_macro']:>10.4f}")
        print(f"  {'F1 (weighted)':<25} {train_m['f1_weighted']:>10.4f} {val_m['f1_weighted']:>10.4f}")

        # --- Log to CSV ---
        for phase, loss, m in [("train", train_loss, train_m), ("val", val_loss, val_m)]:
            append_csv(csv_path, {
                "epoch": epoch, "phase": phase, "loss": round(loss, 6),
                "accuracy":            round(m["accuracy"],            4),
                "top5_accuracy":       round(m["top5_accuracy"],       4),
                "precision_macro":     round(m["precision_macro"],     4),
                "precision_weighted":  round(m["precision_weighted"],  4),
                "recall_macro":        round(m["recall_macro"],        4),
                "recall_weighted":     round(m["recall_weighted"],     4),
                "f1_macro":            round(m["f1_macro"],            4),
                "f1_weighted":         round(m["f1_weighted"],         4),
            })

        # --- Save best model (by val accuracy) ---
        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            best_val_f1  = val_m["f1_macro"]
            path = os.path.join(args.save_dir, "best_model.pt")
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "val_accuracy":   best_val_acc,
                "val_f1_macro":   best_val_f1,
                "class_names":    class_names,
                "embed_dim":      args.embed_dim,
                "num_classes":    args.num_classes,
                "strategy":       args.strategy,
            }, path)
            print(f"\n  [Saved] Best model >> {path}  (val_acc={best_val_acc:.4f})")

        # --- Confusion matrix on val (every 5 epochs + final) ---
        if epoch % 5 == 0 or epoch == args.epochs:
            cm_path = os.path.join(args.save_dir, f"confusion_matrix_epoch_{epoch:03d}.png")
            save_confusion_matrix(v_targets, v_preds, class_names, cm_path)

            # Print per-class accuracy summary
            print(f"\n  Per-class Accuracy (val):")
            for cls_name, acc in zip(class_names, val_m["per_class_accuracy"]):
                bar = "#" * int(acc * 20)
                print(f"    {cls_name:<45} {acc:.3f}  |{bar:<20}|")

    # -----------------------------------------------------------------------
    # Save final model
    # -----------------------------------------------------------------------
    final_path = os.path.join(args.save_dir, "final_model.pt")
    torch.save({
        "epoch":        args.epochs,
        "model_state":  model.state_dict(),
        "class_names":  class_names,
        "embed_dim":    args.embed_dim,
        "num_classes":  args.num_classes,
        "strategy":     args.strategy,
    }, final_path)

    # Full classification report (precision/recall/f1 per class)
    print(f"\n{'='*55}")
    print(f"  Training complete!")
    print(f"  Best val accuracy : {best_val_acc:.4f}")
    print(f"  Best val F1 macro : {best_val_f1:.4f}")
    print(f"\n  Per-class Classification Report (final val epoch):")
    print(classification_report(v_targets, v_preds, target_names=class_names, zero_division=0))
    print(f"  Metrics log >> {csv_path}")
    print(f"  Best model  >> {os.path.join(args.save_dir, 'best_model.pt')}")
    print(f"  Final model >> {final_path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
