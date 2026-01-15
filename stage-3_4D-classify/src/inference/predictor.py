"""
Inference wrapper for deployment.
"""

import torch
from pathlib import Path
from typing import List, Dict, Union, Optional
import time
import logging

from ..data.ply_parser import parse_ply
from ..data.feature_extractor import GaussianFeatureExtractor
from ..models.full_model import Gaussian4DClassifier

logger = logging.getLogger(__name__)


class Predictor:
    """
    Inference wrapper for 4D Gaussian classifier.
    """

    def __init__(
        self,
        checkpoint_path: Path = None,
        model: Gaussian4DClassifier = None,
        config: Dict = None,
        device: str = 'cuda',
        max_gaussians: int = 50000,
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint (if model not provided)
            model: Pre-loaded model (if checkpoint_path not provided)
            config: Model configuration (required if model provided without checkpoint)
            device: Device to run inference on
            max_gaussians: Maximum Gaussians per frame (subsample if exceeded)
        """
        self.device = device
        self.max_gaussians = max_gaussians

        if model is not None:
            self.model = model.to(device)
            self.config = config or {}
        elif checkpoint_path is not None:
            self.model, self.config = self._load_model(checkpoint_path)
        else:
            raise ValueError("Either checkpoint_path or model must be provided")

        self.model.eval()

        # Feature extractor
        self.feature_extractor = GaussianFeatureExtractor()

    def _load_model(self, checkpoint_path: Path) -> tuple:
        """Load model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint.get('config', {})

        # Create model from config
        model_config = config.get('model', {})
        model = Gaussian4DClassifier(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        logger.info(f"Loaded model from {checkpoint_path}")
        return model, config

    @torch.no_grad()
    def predict_sequence(
        self,
        ply_paths: List[Path],
        return_embeddings: bool = False,
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """
        Predict class for a sequence of PLY files.

        Args:
            ply_paths: List of paths to PLY files (ordered temporally)
            return_embeddings: Whether to return intermediate embeddings

        Returns:
            Dictionary with:
                - 'probability': float, probability of drone class
                - 'prediction': int, 0=bird, 1=drone
                - 'latency_ms': float, inference time in milliseconds
                - 'frame_embeddings': optional, (T, D) tensor
                - 'temporal_embedding': optional, (D,) tensor
        """
        start_time = time.time()

        # Load and extract features
        clouds = []
        for p in ply_paths:
            try:
                cloud = parse_ply(p, device=self.device)
                clouds.append(cloud)
            except Exception as e:
                logger.warning(f"Failed to load {p}: {e}")

        if len(clouds) == 0:
            raise ValueError("No valid PLY files loaded")

        batch = self.feature_extractor.extract_batch(
            clouds,
            max_gaussians=self.max_gaussians,
            subsample_strategy='random',
        )

        # Add batch dimension and move to device
        positions = batch['positions'].unsqueeze(0).to(self.device)
        eigenvectors = batch['eigenvectors'].unsqueeze(0).to(self.device)
        scalars = batch['scalars'].unsqueeze(0).to(self.device)
        mask = batch['mask'].unsqueeze(0).to(self.device)

        # Forward pass
        outputs = self.model(positions, eigenvectors, scalars, mask)

        latency_ms = (time.time() - start_time) * 1000

        prob = outputs['probabilities'].item()
        result = {
            'probability': prob,
            'prediction': int(prob > 0.5),
            'confidence': abs(prob - 0.5) * 2,  # Distance from decision boundary
            'latency_ms': latency_ms,
        }

        if return_embeddings:
            result['frame_embeddings'] = outputs['frame_embeddings'].squeeze(0).cpu()
            result['temporal_embedding'] = outputs['temporal_embedding'].squeeze(0).cpu()

        return result

    @torch.no_grad()
    def predict_batch(
        self,
        sequences: List[List[Path]],
    ) -> List[Dict]:
        """
        Batch prediction for multiple sequences.

        Args:
            sequences: List of sequences, each a list of PLY paths

        Returns:
            List of prediction dictionaries
        """
        return [self.predict_sequence(seq) for seq in sequences]

    @torch.no_grad()
    def predict_from_tensors(
        self,
        positions: torch.Tensor,
        eigenvectors: torch.Tensor,
        scalars: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """
        Predict from pre-extracted tensors.

        Args:
            positions: (B, T, M, 3) or (T, M, 3) Gaussian positions
            eigenvectors: (B, T, M, 3, 3) or (T, M, 3, 3) Principal axes
            scalars: (B, T, M, D) or (T, M, D) Scalar features
            mask: Optional (B, T, M) or (T, M) validity mask

        Returns:
            Prediction dictionary
        """
        # Add batch dimension if needed
        if positions.dim() == 3:
            positions = positions.unsqueeze(0)
            eigenvectors = eigenvectors.unsqueeze(0)
            scalars = scalars.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        # Move to device
        positions = positions.to(self.device)
        eigenvectors = eigenvectors.to(self.device)
        scalars = scalars.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        # Forward pass
        outputs = self.model(positions, eigenvectors, scalars, mask)

        prob = outputs['probabilities'].squeeze().cpu()
        if prob.dim() == 0:
            prob = prob.item()
            pred = int(prob > 0.5)
        else:
            pred = (prob > 0.5).long().tolist()
            prob = prob.tolist()

        return {
            'probability': prob,
            'prediction': pred,
            'frame_embeddings': outputs['frame_embeddings'].cpu(),
            'temporal_embedding': outputs['temporal_embedding'].cpu(),
        }

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'total_parameters': num_params,
            'trainable_parameters': num_trainable,
            'device': str(self.device),
            'max_gaussians': self.max_gaussians,
            'config': self.config,
        }


class StreamingPredictor(Predictor):
    """
    Predictor optimized for streaming inference.

    Maintains a buffer of recent frames for efficient processing.
    """

    def __init__(
        self,
        buffer_size: int = 30,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.buffer_size = buffer_size
        self.frame_buffer = []

    def add_frame(self, ply_path: Path) -> Optional[Dict]:
        """
        Add a frame and optionally get prediction.

        Args:
            ply_path: Path to PLY file

        Returns:
            Prediction dict if buffer is full, else None
        """
        cloud = parse_ply(ply_path, device=self.device)
        self.frame_buffer.append(cloud)

        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)

        if len(self.frame_buffer) >= self.buffer_size:
            return self._predict_buffer()

        return None

    def _predict_buffer(self) -> Dict:
        """Predict from current buffer."""
        batch = self.feature_extractor.extract_batch(
            self.frame_buffer,
            max_gaussians=self.max_gaussians,
        )

        positions = batch['positions'].unsqueeze(0).to(self.device)
        eigenvectors = batch['eigenvectors'].unsqueeze(0).to(self.device)
        scalars = batch['scalars'].unsqueeze(0).to(self.device)
        mask = batch['mask'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(positions, eigenvectors, scalars, mask)

        prob = outputs['probabilities'].item()
        return {
            'probability': prob,
            'prediction': int(prob > 0.5),
            'confidence': abs(prob - 0.5) * 2,
        }

    def clear_buffer(self):
        """Clear the frame buffer."""
        self.frame_buffer = []
