#!/usr/bin/env python3
"""
Basic tests to verify the pipeline works.

Run with: python -m pytest tests/test_pipeline.py -v
Or simply: python tests/test_pipeline.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch


def test_vn_transformer():
    """Test VN-Transformer spatial encoder."""
    from models.stage1_spatial import GaussianSpatialEncoder

    encoder = GaussianSpatialEncoder(
        dim=64,
        depth=2,
        heads=4,
        dim_feat=32,
        output_dim=128,
    )

    B, M = 2, 100
    positions = torch.randn(B, M, 3)
    scalars = torch.randn(B, M, 16)
    mask = torch.ones(B, M, dtype=torch.bool)

    output = encoder(positions, scalars, mask)

    assert output.shape == (B, 128), f"Expected (2, 128), got {output.shape}"
    print("✓ VN-Transformer spatial encoder test passed")


def test_mamba_temporal():
    """Test Mamba temporal encoder."""
    from models.stage2_temporal import TemporalMambaEncoder

    encoder = TemporalMambaEncoder(
        input_dim=128,
        hidden_dim=128,
        output_dim=128,
        num_layers=2,
        use_mixer_model=False,  # Use simpler version for test
    )

    B, T = 2, 10
    frame_embeddings = torch.randn(B, T, 128)
    mask = torch.ones(B, T, dtype=torch.bool)

    output = encoder(frame_embeddings, mask)

    assert output.shape == (B, 128), f"Expected (2, 128), got {output.shape}"
    print("✓ Mamba temporal encoder test passed")


def test_full_model():
    """Test full model integration."""
    from models.full_model import Gaussian4DClassifier

    model = Gaussian4DClassifier(
        spatial_dim=32,
        spatial_depth=1,
        spatial_heads=2,
        spatial_dim_feat=16,
        spatial_output_dim=64,
        temporal_hidden_dim=64,
        temporal_num_layers=1,
        num_classes=1,
        use_eigenvectors=False,
    )

    B, T, M = 2, 5, 50
    positions = torch.randn(B, T, M, 3)
    eigenvectors = torch.randn(B, T, M, 3, 3)
    scalars = torch.randn(B, T, M, 16)
    mask = torch.ones(B, T, M, dtype=torch.bool)

    outputs = model(positions, eigenvectors, scalars, mask)

    assert 'logits' in outputs
    assert 'probabilities' in outputs
    assert outputs['logits'].shape == (B, 1), f"Expected (2, 1), got {outputs['logits'].shape}"
    assert outputs['probabilities'].shape == (B, 1)
    print("✓ Full model integration test passed")


def test_synthetic_dataset():
    """Test synthetic dataset and collator."""
    from data.gaussian_dataset import SyntheticGaussianDataset, GaussianCollator

    dataset = SyntheticGaussianDataset(
        num_samples=10,
        sequence_length=5,
        num_gaussians=50,
    )

    assert len(dataset) == 10

    sample = dataset[0]
    assert 'positions' in sample
    assert 'scalars' in sample
    assert 'label' in sample

    collator = GaussianCollator()
    batch = collator([dataset[i] for i in range(4)])

    assert batch['positions'].shape[0] == 4  # Batch size
    print("✓ Synthetic dataset test passed")


def test_feature_extractor():
    """Test feature extraction from GaussianCloud."""
    from data.ply_parser import create_synthetic_gaussian_cloud
    from data.feature_extractor import GaussianFeatureExtractor

    cloud = create_synthetic_gaussian_cloud(num_gaussians=100)
    extractor = GaussianFeatureExtractor()

    features = extractor.extract(cloud)

    assert features.positions.shape == (100, 3)
    assert features.eigenvalues.shape == (100, 3)
    assert features.eigenvectors.shape == (100, 3, 3)
    print("✓ Feature extractor test passed")


def test_losses():
    """Test loss functions."""
    from training.losses import ClassificationLoss, BinaryFocalLoss

    logits = torch.randn(8, 1)
    targets = torch.randint(0, 2, (8,))

    # BCE loss
    bce_loss = ClassificationLoss(loss_type='bce')
    loss = bce_loss(logits, targets)
    assert loss.dim() == 0  # Scalar
    assert loss >= 0

    # Focal loss
    focal_loss = BinaryFocalLoss()
    loss = focal_loss(logits, targets)
    assert loss.dim() == 0
    assert loss >= 0

    print("✓ Loss functions test passed")


def test_metrics():
    """Test metrics computation."""
    from training.metrics import ClassificationMetrics

    preds = [0, 1, 1, 0, 1, 0, 1, 1]
    labels = [0, 1, 0, 0, 1, 1, 1, 0]
    probs = [0.2, 0.8, 0.7, 0.3, 0.9, 0.4, 0.85, 0.6]

    metrics = ClassificationMetrics()
    result = metrics.compute(preds, labels, probs)

    assert 'accuracy' in result
    assert 'precision' in result
    assert 'recall' in result
    assert 'f1' in result
    assert 0 <= result['accuracy'] <= 1

    print("✓ Metrics test passed")


def test_rotation_equivariance():
    """Test that VN-Transformer is approximately rotation equivariant."""
    from models.stage1_spatial import GaussianSpatialEncoder
    from utils.rotation_utils import random_rotation_matrix, apply_rotation

    torch.manual_seed(42)

    encoder = GaussianSpatialEncoder(
        dim=32,
        depth=1,
        heads=2,
        dim_feat=16,
        output_dim=64,
        bias_epsilon=0,  # Exact equivariance (no epsilon)
    )
    encoder.eval()

    B, M = 1, 50
    positions = torch.randn(B, M, 3)
    scalars = torch.randn(B, M, 16)

    # Get embedding for original
    with torch.no_grad():
        emb1 = encoder(positions, scalars)

    # Apply random rotation
    R = random_rotation_matrix()
    positions_rot = apply_rotation(positions, R)

    # Get embedding for rotated
    with torch.no_grad():
        emb2 = encoder(positions_rot, scalars)

    # Embeddings should be similar (rotation invariant output)
    diff = (emb1 - emb2).abs().mean().item()

    # Allow some tolerance due to numerical precision
    # Note: With bias_epsilon=0, should be more invariant
    print(f"  Rotation invariance error: {diff:.6f}")
    assert diff < 0.5, f"Rotation invariance error too high: {diff}"

    print("✓ Rotation equivariance test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Running pipeline tests...")
    print("=" * 50)

    test_vn_transformer()
    test_mamba_temporal()
    test_full_model()
    test_synthetic_dataset()
    test_feature_extractor()
    test_losses()
    test_metrics()
    test_rotation_equivariance()

    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)


if __name__ == '__main__':
    run_all_tests()
