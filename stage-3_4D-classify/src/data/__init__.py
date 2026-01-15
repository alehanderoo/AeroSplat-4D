"""
Data pipeline for 4D Gaussian Classifier.

Modules for loading, parsing, and preprocessing 3D Gaussian Splatting data.
"""

from .ply_parser import parse_ply, GaussianCloud, quaternion_to_rotation_matrix, create_synthetic_gaussian_cloud
from .gaussian_dataset import GaussianSequenceDataset, GaussianCollator, SyntheticGaussianDataset
from .feature_extractor import GaussianFeatureExtractor, GaussianFeatures, FastFeatureExtractor
from .augmentations import (
    TemporalAugmentation,
    SpatialAugmentation,
    GaussianAugmentation,
    ComposedAugmentation,
    get_train_augmentation,
)

__all__ = [
    'parse_ply',
    'GaussianCloud',
    'quaternion_to_rotation_matrix',
    'create_synthetic_gaussian_cloud',
    'GaussianSequenceDataset',
    'GaussianCollator',
    'SyntheticGaussianDataset',
    'GaussianFeatureExtractor',
    'GaussianFeatures',
    'FastFeatureExtractor',
    'TemporalAugmentation',
    'SpatialAugmentation',
    'GaussianAugmentation',
    'ComposedAugmentation',
    'get_train_augmentation',
]
