"""
SSL Pre-training Script  (Plant Disease Edition)
Trains the SimCLR foundation model on unlabelled plant disease images.
Saves checkpoints and the final encoder (ready for fine-tuning).
"""

import argparse
import os

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_ssl_dataloader
from loss import NTXentLoss
from model import SimCLRModel

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="SimCLR SSL Pre-training — Plant Disease")

    parser.add_argument("--data_dir",    type=str,   default="Plant Disease/Images/Data/train",
                        help="Root folder containing plant disease images")
    parser.add_argument("--epochs",      type=int,   default=15,
                        help="Number of training epochs")
    parser.add_argument("--batch_size",  type=int,   default=16,
                        help="Batch size (larger = more negatives = better SSL)")
    parser.add_argument("--image_size",  type=int,   default=224,
                        help="Images resized to image_size × image_size (224 for ResNet-18)")
    parser.add_argument("--embed_dim",   type=int,   default=256,
                        help="Encoder output dimension (256 suits 38-class downstream task)")
    parser.add_argument("--proj_dim",    type=int,   default=128,
                        help="Projection head output dimension")
    parser.add_argument("--lr",          type=float, default=3e-4,
                        help="Initial learning rate")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="NT-Xent loss temperature")
    parser.add_argument("--num_workers", type=int,   default=2,
                        help="DataLoader worker threads")
    parser.add_argument("--save_dir",    type=str,   default="checkpoints",
                        help="Where to save model checkpoints")
    parser.add_argument("--save_every",  type=int,   default=10,
                        help="Save a checkpoint every N epochs")
    parser.add_argument("--resume",      type=str,   default="checkpoints/checkpoint_epoch_005.pt",
                        help="Path to checkpoint to resume from")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Save / Load helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_dir):
    """Save full training state (so training can be resumed)."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"checkpoint_epoch_{epoch:03d}.pt")
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "loss":            loss,
    }, path)
    print(f"  [Checkpoint] Saved >> {path}")


def save_foundation_encoder(model, save_dir, embed_dim):
    """
    Save ONLY the encoder weights.
    This is the file used for fine-tuning downstream tasks.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "foundation_encoder.pt")
    torch.save({
        "encoder_state": model.encoder.state_dict(),
        "embed_dim":     embed_dim,
    }, path)
    print(f"  [Foundation Model] Encoder saved >> {path}")


def load_checkpoint(path, model, optimizer, scheduler, device):
    """Load a training checkpoint to resume training."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = ckpt["epoch"] + 1
    print(f"  [Resume] Loaded checkpoint from epoch {ckpt['epoch']} (loss={ckpt['loss']:.4f})")
    return start_epoch


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, loss_fn, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    for batch_idx, (view_a, view_b) in enumerate(loader):
        view_a = view_a.to(device)
        view_b = view_b.to(device)

        # Forward pass — get L2-normalised projections
        z_a, _ = model(view_a)
        z_b, _ = model(view_b)

        # NT-Xent contrastive loss
        loss = loss_fn(z_a, z_b)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 1000 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} "
                  f"| Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    return avg_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"  SimCLR SSL Pre-training — Plant Disease")
    print(f"{'='*50}")
    print(f"  Device      : {device}")
    print(f"  Data dir    : {args.data_dir}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Image size  : {args.image_size}×{args.image_size}")
    print(f"  Backbone    : ResNet-18 (pre-trained image)")
    print(f"  Embed dim   : {args.embed_dim}")
    print(f"  Proj dim    : {args.proj_dim}")
    print(f"  Temperature : {args.temperature}")
    print(f"{'='*50}\n")

    # --- Data ---
    loader = get_ssl_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )

    # --- Model ---
    model = SimCLRModel(embed_dim=args.embed_dim, proj_dim=args.proj_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}\n")

    # --- Loss, Optimizer, Scheduler ---
    loss_fn   = NTXentLoss(temperature=args.temperature)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # --- Optional resume ---
    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, device)

    # --- Training loop ---
    best_loss = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}  (lr={scheduler.get_last_lr()[0]:.6f})")

        avg_loss = train_one_epoch(model, loader, loss_fn, optimizer, device, epoch)
        scheduler.step()

        print(f"  >> Avg Loss: {avg_loss:.4f}")

        # Track best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_foundation_encoder(model, args.save_dir, args.embed_dim)

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, args.save_dir)

    # --- Final save ---
    print(f"\n{'='*50}")
    print(f"  Training complete! Best loss: {best_loss:.4f}")
    save_checkpoint(model, optimizer, scheduler, args.epochs, avg_loss, args.save_dir)
    save_foundation_encoder(model, args.save_dir, args.embed_dim)
    print(f"\n  Foundation encoder ready for fine-tuning:")
    print(f"  >> {os.path.join(args.save_dir, 'foundation_encoder.pt')}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
