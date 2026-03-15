"""
tests/test_pipeline.py
-----------------------
Unit tests for model, loss, dataset, and classifier components.
All tests run on CPU and use tiny synthetic data — no real images needed.
These are run automatically by the CI workflow on every push/PR.
"""

import os
import sys
import tempfile

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model   import Encoder, ProjectionHead, SimCLRModel
from loss    import NTXentLoss
from dataset import PlantDiseaseSSLDataset, get_ssl_augmentation
from finetune import PlantDiseaseClassifier


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def dummy_batch():
    """A batch of 4 synthetic RGB images (3×64×64)."""
    return torch.randn(4, 3, 64, 64)


@pytest.fixture
def encoder():
    return Encoder(embed_dim=64)


@pytest.fixture
def simclr_model():
    return SimCLRModel(embed_dim=64, proj_dim=32)


@pytest.fixture
def tmp_data_dir():
    """
    Creates a temporary directory tree mimicking the plant disease structure:
        tmp/
          ClassA/  (3 synthetic PNG images)
          ClassB/  (3 synthetic PNG images)
    """
    with tempfile.TemporaryDirectory() as tmp:
        for cls in ["ClassA", "ClassB"]:
            cls_dir = os.path.join(tmp, cls)
            os.makedirs(cls_dir)
            for i in range(3):
                img = Image.fromarray(
                    (torch.rand(64, 64, 3) * 255).byte().numpy()
                )
                img.save(os.path.join(cls_dir, f"img_{i}.png"))
        yield tmp


# =============================================================================
# 1.  Encoder
# =============================================================================

class TestEncoder:

    def test_output_shape(self, encoder, dummy_batch):
        out = encoder(dummy_batch)
        assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"

    def test_embed_dim_attribute(self, encoder):
        assert encoder.embed_dim == 64

    def test_no_nan_in_output(self, encoder, dummy_batch):
        out = encoder(dummy_batch)
        assert not torch.isnan(out).any(), "Encoder output contains NaN"

    def test_gradient_flows(self, encoder, dummy_batch):
        dummy_batch.requires_grad_(True)
        out = encoder(dummy_batch)
        out.sum().backward()
        assert dummy_batch.grad is not None


# =============================================================================
# 2.  Projection Head
# =============================================================================

class TestProjectionHead:

    def test_output_shape(self):
        head  = ProjectionHead(embed_dim=64, hidden_dim=128, proj_dim=32)
        x     = torch.randn(4, 64)
        out   = head(x)
        assert out.shape == (4, 32)


# =============================================================================
# 3.  SimCLR Model
# =============================================================================

class TestSimCLRModel:

    def test_output_shapes(self, simclr_model, dummy_batch):
        z, h = simclr_model(dummy_batch)
        assert z.shape == (4, 32), f"Projection shape mismatch: {z.shape}"
        assert h.shape == (4, 64), f"Embedding shape mismatch: {h.shape}"

    def test_projection_is_l2_normalised(self, simclr_model, dummy_batch):
        z, _ = simclr_model(dummy_batch)
        norms = z.norm(dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5), \
            "Projections are not L2-normalised"

    def test_two_views_differ(self, simclr_model):
        """Two different inputs should produce different embeddings."""
        x1 = torch.randn(2, 3, 64, 64)
        x2 = torch.randn(2, 3, 64, 64)
        z1, _ = simclr_model(x1)
        z2, _ = simclr_model(x2)
        assert not torch.allclose(z1, z2), "Different inputs produced identical projections"


# =============================================================================
# 4.  NT-Xent Loss
# =============================================================================

class TestNTXentLoss:

    def test_loss_is_scalar(self):
        loss_fn = NTXentLoss(temperature=0.5)
        z_a = F.normalize(torch.randn(8, 32), dim=1)
        z_b = F.normalize(torch.randn(8, 32), dim=1)
        loss = loss_fn(z_a, z_b)
        assert loss.ndim == 0, "Loss should be a scalar"

    def test_loss_is_positive(self):
        loss_fn = NTXentLoss(temperature=0.5)
        z_a = F.normalize(torch.randn(8, 32), dim=1)
        z_b = F.normalize(torch.randn(8, 32), dim=1)
        loss = loss_fn(z_a, z_b)
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_perfect_match_lower_loss(self):
        """Identical positive pairs should yield lower loss than random pairs."""
        loss_fn   = NTXentLoss(temperature=0.5)
        z         = F.normalize(torch.randn(8, 32), dim=1)
        loss_same = loss_fn(z, z.clone())
        loss_rand = loss_fn(z, F.normalize(torch.randn(8, 32), dim=1))
        assert loss_same.item() < loss_rand.item(), \
            "Perfect match should yield lower loss than random pairs"

    def test_temperature_effect(self):
        """Lower temperature should produce higher (sharper) loss."""
        z_a = F.normalize(torch.randn(8, 32), dim=1)
        z_b = F.normalize(torch.randn(8, 32), dim=1)
        loss_low  = NTXentLoss(temperature=0.1)(z_a, z_b)
        loss_high = NTXentLoss(temperature=1.0)(z_a, z_b)
        assert loss_low.item() != loss_high.item(), \
            "Different temperatures should produce different loss values"

    def test_gradient_flows_through_loss(self):
        loss_fn = NTXentLoss(temperature=0.5)
        # F.normalize returns a non-leaf tensor, so .grad is always None on it.
        # Gradients must be checked on the raw leaf tensors before normalisation.
        raw_a = torch.randn(8, 32, requires_grad=True)
        raw_b = torch.randn(8, 32, requires_grad=True)
        z_a = F.normalize(raw_a, dim=1)
        z_b = F.normalize(raw_b, dim=1)
        loss = loss_fn(z_a, z_b)
        loss.backward()
        assert raw_a.grad is not None, "Gradient did not flow back to z_a"
        assert raw_b.grad is not None, "Gradient did not flow back to z_b"


# =============================================================================
# 5.  Dataset
# =============================================================================

class TestPlantDiseaseSSLDataset:

    def test_loads_images(self, tmp_data_dir):
        dataset = PlantDiseaseSSLDataset(data_dir=tmp_data_dir, image_size=64)
        assert len(dataset) == 6, f"Expected 6 images (2 classes x 3), got {len(dataset)}"

    def test_returns_two_views(self, tmp_data_dir):
        dataset = PlantDiseaseSSLDataset(data_dir=tmp_data_dir, image_size=64)
        view_a, view_b = dataset[0]
        assert view_a.shape == (3, 64, 64)
        assert view_b.shape == (3, 64, 64)

    def test_two_views_differ(self, tmp_data_dir):
        """Two views of the same image should differ due to random augmentation."""
        dataset = PlantDiseaseSSLDataset(data_dir=tmp_data_dir, image_size=64)
        # With random augmentation, views should almost always differ
        # (extremely unlikely to be identical by chance)
        differences = sum(
            not torch.allclose(dataset[0][0], dataset[0][1])
            for _ in range(5)
        )
        assert differences >= 4, "Views should differ with high probability"

    def test_empty_dir_raises(self, tmp_path):
        empty = str(tmp_path / "empty")
        os.makedirs(empty)
        with pytest.raises(ValueError, match="No images found"):
            PlantDiseaseSSLDataset(data_dir=empty)

    def test_augmentation_no_grayscale(self):
        """Confirm RandomGrayscale is NOT in the plant disease augmentation pipeline."""
        from torchvision import transforms
        aug = get_ssl_augmentation(image_size=64)
        transform_types = [type(t).__name__ for t in aug.transforms]
        assert "RandomGrayscale" not in transform_types, \
            "RandomGrayscale must be removed for plant disease (colour = disease signal)"


# =============================================================================
# 6.  Classifier
# =============================================================================

class TestPlantDiseaseClassifier:

    @pytest.fixture
    def classifier(self):
        enc = Encoder(embed_dim=64)
        return PlantDiseaseClassifier(encoder=enc, num_classes=10, strategy="frozen", dropout=0.0)

    def test_output_shape(self, classifier):
        x   = torch.randn(4, 3, 64, 64)
        out = classifier(x)
        assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"

    def test_frozen_strategy_no_encoder_grad(self, classifier):
        for param in classifier.encoder.parameters():
            assert not param.requires_grad, "Encoder params should be frozen"

    def test_partial_strategy_only_layer4_unfrozen(self):
        enc = Encoder(embed_dim=64)
        clf = PlantDiseaseClassifier(encoder=enc, num_classes=10, strategy="partial")
        # layer4 (backbone[-2]) params should be trainable
        layer4_trainable = any(p.requires_grad for p in enc.backbone[-2].parameters())
        assert layer4_trainable, "layer4 should be trainable under 'partial' strategy"

    def test_full_strategy_all_params_trainable(self):
        enc = Encoder(embed_dim=64)
        clf = PlantDiseaseClassifier(encoder=enc, num_classes=10, strategy="full")
        all_trainable = all(p.requires_grad for p in clf.parameters())
        assert all_trainable, "All params should be trainable under 'full' strategy"

    def test_invalid_strategy_raises(self):
        enc = Encoder(embed_dim=64)
        with pytest.raises(AssertionError):
            PlantDiseaseClassifier(encoder=enc, num_classes=10, strategy="invalid")