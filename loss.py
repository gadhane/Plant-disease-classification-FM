"""
 NT-Xent Contrastive Loss
NT-Xent = Normalized Temperature-scaled Cross Entropy Loss
This is the loss function used in SimCLR.

Intuition
---------
Given a batch of N images, we have 2N views total (2 per image).
For each view, its "positive" is its paired augmentation.
All other 2(N-1) views in the batch are "negatives".

The loss pushes:
  ✅  positive pairs  →  high cosine similarity
  ❌  negative pairs  →  low cosine similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent Loss for SimCLR contrastive learning.

    Args:
        temperature : softmax temperature τ (default 0.5)
                      lower  → sharper distribution, harder negatives
                      higher → softer distribution, easier training
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_a : (N, D)  L2-normalised projections of view A
            z_b : (N, D)  L2-normalised projections of view B

        Returns:
            scalar loss
        """
        N = z_a.size(0)
        device = z_a.device

        # --- 1. Stack both views: shape (2N, D) ---
        z = torch.cat([z_a, z_b], dim=0)              # (2N, D)

        # --- 2. Cosine similarity matrix: shape (2N, 2N) ---
        # Since z is already L2-normalised, cosine_sim = z @ z^T
        sim = torch.mm(z, z.T) / self.temperature      # (2N, 2N)

        # --- 3. Build labels: positive pair indices ---
        # For index i in [0, N):     its positive is i+N
        # For index i in [N, 2N):   its positive is i-N
        labels = torch.cat([
            torch.arange(N, 2*N, device=device),       # first  N rows → pair is in second half
            torch.arange(0,   N, device=device),       # second N rows → pair is in first  half
        ])                                              # (2N,)

        # --- 4. Mask out self-similarity (diagonal) ---
        mask = torch.eye(2*N, dtype=torch.bool, device=device)
        sim.masked_fill_(mask, float('-inf'))           # prevents i==i from being selected

        # --- 5. Cross-entropy: for each view, treat its pair as the "class" ---
        loss = F.cross_entropy(sim, labels)

        return loss


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loss_fn = NTXentLoss(temperature=0.5)

    # Simulate a batch of 8 images → 8 pairs of projections
    z_a = F.normalize(torch.randn(8, 64), dim=1)
    z_b = F.normalize(torch.randn(8, 64), dim=1)

    loss = loss_fn(z_a, z_b)
    print(f"Loss (random embeddings) : {loss.item():.4f}")

    # If embeddings are identical (perfect), loss should approach 0
    loss_perfect = loss_fn(z_a, z_a.clone())
    print(f"Loss (perfect match)     : {loss_perfect.item():.4f}")
