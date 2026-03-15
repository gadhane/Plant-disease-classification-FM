"""
Foundation Model Architecture  (Plant Disease Edition)
Encoder + Projection Head for SimCLR-style SSL pre-training.
"""

import torch
import torch.nn as nn
import torchvision.models as models

# ---------------------------------------------------------------------------
# 1.  Encoder  (the part we KEEP after pre-training)
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    ResNet-18 based encoder for plant disease SSL pre-training.

    Why ResNet-18?
      - Residual connections allow deeper feature hierarchies without
        vanishing gradients — critical for subtle lesion/spot patterns
      - 11M params gives enough capacity for 38 fine-grained classes
      - Fast enough to train on a single GPU without ImageNet weights

    Input : (B, 3, 224, 224)  image batch
    Output: (B, embed_dim)    feature vectors
    """

    # ResNet-18's final avgpool output is always 512-dimensional
    RESNET18_FEAT_DIM = 512

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim

        # --- Load ResNet-18 with NO pretrained weights ---
        # SSL pre-training will learn representations from scratch
        # resnet = models.resnet18(weights=None)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # --- Strip the original classifier (fc layer) ---
        # Keep everything up to and including the global average pool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # Output shape: (B, 512, 1, 1)

        # --- Project ResNet features → embedding space ---
        self.fc = nn.Sequential(
            nn.Linear(self.RESNET18_FEAT_DIM, embed_dim),
            nn.BatchNorm1d(embed_dim),           # stabilises SSL training
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)              # (B, 512, 1, 1)
        features = features.flatten(1)           # (B, 512)
        embeddings = self.fc(features)           # (B, embed_dim)
        return embeddings


# ---------------------------------------------------------------------------
# 2.  Projection Head  (DISCARDED after pre-training)
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    3-layer MLP that maps encoder embeddings → SSL loss space.
    Only used during SSL pre-training; thrown away before fine-tuning.

    Upgraded to 3 layers (from 2) to match the richer ResNet-18 features.
    BatchNorm between layers improves training stability.

    Input : (B, embed_dim)
    Output: (B, proj_dim)
    """

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 512, proj_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),     # no BN/ReLU on final layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 3.  Full SimCLR Model  (Encoder + Projection Head bundled together)
# ---------------------------------------------------------------------------

class SimCLRModel(nn.Module):
    """
    Wraps Encoder + ProjectionHead for SSL pre-training.

    forward() returns:
      - z : L2-normalised projection vectors  → used for NT-Xent loss
      - h : raw encoder embeddings            → useful for inspection / fine-tuning
    """

    def __init__(self, embed_dim: int = 256, proj_dim: int = 128):
        super().__init__()
        self.encoder   = Encoder(embed_dim=embed_dim)
        self.projector = ProjectionHead(embed_dim=embed_dim, proj_dim=proj_dim)

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)                        # (B, embed_dim) — raw embedding
        z = self.projector(h)                      # (B, proj_dim)  — projection
        z = nn.functional.normalize(z, dim=1)      # L2-normalise   — needed for NT-Xent
        return z, h


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = SimCLRModel(embed_dim=256, proj_dim=128)
    dummy = torch.randn(8, 3, 224, 224)            # 224×224 — plant disease input size
    z, h  = model(dummy)
    print(f"Projection z : {z.shape}")             # (8, 128)
    print(f"Embedding  h : {h.shape}")             # (8, 256)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params : {total:,}")             # ~11.5M
