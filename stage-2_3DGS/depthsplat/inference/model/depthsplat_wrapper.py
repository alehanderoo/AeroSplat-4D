"""
DepthSplat Model Wrapper for PyTorch Inference.

This module provides a wrapper around the DepthSplat model for direct
PyTorch inference.

Usage:
    from model.depthsplat_wrapper import DepthSplatWrapper, load_model

    # Load the model
    wrapper = load_model("/path/to/checkpoint.ckpt")

    # Run inference
    wrapper.eval()
    output = wrapper(input_tensor)
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, Union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the DepthSplat model."""
    
    # Input dimensions
    num_cameras: int = 5
    input_channels: int = 3
    input_height: int = 256
    input_width: int = 256
    
    # Output configuration
    output_positions: bool = True
    output_covariances: bool = True
    output_colors: bool = True
    output_opacities: bool = True
    
    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Get the input tensor shape."""
        return (1, self.num_cameras, self.input_channels, self.input_height, self.input_width)


class DepthSplatWrapper:
    """
    Wrapper for DepthSplat model.
    
    This wrapper:
    - Handles input reshaping if needed
    - Manages dictionary outputs
    - Provides consistent output tensor names
    
    Usage:
        wrapper = DepthSplatWrapper.from_checkpoint("/path/to/checkpoint.ckpt")
        wrapper.eval()
        
        # Run inference
        output = wrapper(input_tensor)
    """
    
    def __init__(self, model, config: Optional[ModelConfig] = None):
        """
        Initialize the wrapper.
        
        Args:
            model: The underlying PyTorch model (nn.Module)
            config: Model configuration
        """
        import torch
        import torch.nn as nn
        
        self.config = config or ModelConfig()
        self.model = model
        self._torch = torch
        self._nn = nn
        
    def forward(self, x):
        """
        Forward pass with clean tensor outputs.
        
        Args:
            x: Input tensor [B, num_cameras, C, H, W]
            
        Returns:
            Dictionary of output tensors
        """
        # Run the underlying model
        output = self.model(x)
        
        # Ensure outputs are contiguous tensors
        if isinstance(output, dict):
            result = {}
            for k, v in output.items():
                if self._torch.is_tensor(v):
                    result[k] = v.contiguous()
            return result
        elif self._torch.is_tensor(output):
            return {"gaussians": output.contiguous()}
        else:
            # Handle namedtuple or other structured outputs
            try:
                return {"output": self._torch.tensor(output).contiguous()}
            except Exception:
                return {"output": output}
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self
    
    def train(self, mode: bool = True):
        """Set model to training mode."""
        self.model.train(mode)
        return self
    
    def to(self, device):
        """Move model to device."""
        self.model = self.model.to(device)
        return self
    
    def __call__(self, x):
        """Call forward pass."""
        return self.forward(x)
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[ModelConfig] = None,
        model_class: Optional[type] = None,
        device: str = "cuda",
    ) -> "DepthSplatWrapper":
        """
        Load wrapper from a PyTorch checkpoint.
        
        The DepthSplat checkpoint is a PyTorch Lightning checkpoint containing
        a ModelWrapper with encoder and decoder. We extract these components
        and wrap them.
        
        Args:
            checkpoint_path: Path to the .ckpt file
            config: Model configuration
            model_class: Optional model class to instantiate
            device: Device to load model on
            
        Returns:
            DepthSplatWrapper instance
        """
        import torch
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading model from: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Get state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Create InferenceModel that extracts encoder/decoder from state dict
        model = cls._create_inference_model(state_dict, device)
        
        if model is None:
            raise ImportError(
                "Could not load DepthSplat model. "
                "Please provide model_class or ensure depthsplat is installed."
            )
        
        model.eval()
        model = model.to(device)
        
        wrapper = cls(model, config)
        logger.info("Model wrapper created successfully")
        
        return wrapper
    
    @staticmethod
    def _create_inference_model(state_dict: Dict, device: str):
        """
        Create an InferenceModel from the checkpoint state dict.
        
        The DepthSplat checkpoint contains a ModelWrapper (LightningModule)
        with encoder and decoder. We need to:
        1. Add depthsplat src to path
        2. Import the encoder/decoder classes with their configs
        3. Create instances and load the weights
        """
        import sys
        import torch
        import torch.nn as nn
        from pathlib import Path
        
        # Add depthsplat root to path
        depthsplat_root = Path(__file__).parent.parent.parent  # inference -> depthsplat
        if str(depthsplat_root) not in sys.path:
            sys.path.insert(0, str(depthsplat_root))
        
        logger.info(f"Added to path: {depthsplat_root}")
        
        # Analyze state dict to understand model structure
        encoder_keys = [k for k in state_dict.keys() if k.startswith('encoder.')]
        decoder_keys = [k for k in state_dict.keys() if k.startswith('decoder.')]
        
        logger.info(f"Found {len(encoder_keys)} encoder parameters, {len(decoder_keys)} decoder parameters")
        
        if not encoder_keys:
            logger.error("No encoder keys found in checkpoint")
            return None
        
        try:
            # Try to load the full ModelWrapper using hydra config
            from src.model.model_wrapper import ModelWrapper
            
            # The checkpoint may have hyper_parameters that we can use
            if 'hyper_parameters' in state_dict:
                logger.info("Found hyper_parameters in checkpoint")
            
            # Create a simple inference model that just holds encoder and decoder
            class DepthSplatInferenceModel(nn.Module):
                """
                Simplified inference model for PyTorch inference.
                
                This model wraps just the encoder for generating Gaussians.
                """
                
                def __init__(self, encoder, decoder=None):
                    super().__init__()
                    self.encoder = encoder
                    self.decoder = decoder
                
                def forward(self, images):
                    """
                    Forward pass for inference.
                    
                    Args:
                        images: Input tensor [B, V, C, H, W] where V is num_cameras
                        
                    Returns:
                        Gaussian parameters from the encoder
                    """
                    # Create context dict expected by encoder
                    B, V, C, H, W = images.shape
                    
                    # For ONNX export, we need to simplify this.
                    # This is a placeholder - actual conversion will need more work.
                    
                    # For now, just pass through and get Gaussians
                    context = {
                        'image': images,
                        # Dummy camera parameters (will be provided at inference time)
                        'extrinsics': torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, V, 4, 4).to(images.device),
                        'intrinsics': torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, V, 3, 3).to(images.device),
                        'near': torch.ones(B, V).to(images.device) * 0.1,
                        'far': torch.ones(B, V).to(images.device) * 100.0,
                    }
                    
                    gaussians = self.encoder(context, global_step=0, deterministic=True)
                    
                    if isinstance(gaussians, dict):
                        gaussians = gaussians['gaussians']
                    
                    # Return the Gaussian parameters as a dict
                    return {
                        'means': gaussians.means,
                        'covariances': gaussians.covariances,
                        'harmonics': gaussians.harmonics,
                        'opacities': gaussians.opacities,
                    }
            
            # Try to create encoder from config
            from src.model.encoder import get_encoder, EncoderDepthSplatCfg
            from src.model.decoder import get_decoder
            
            # We need the original config to create the encoder...
            # For now, log what we found and return None
            logger.warning(
                "Full model loading requires hydra config reconstruction. "
                "DepthSplat inference requires:\n"
                "  1. The encoder for generating Gaussian parameters\n"
                "  2. Proper camera intrinsics and extrinsics\n"
                "\nFor production inference, the pipeline loads the full model\n"
                "using hydra configuration from the checkpoint."
            )
            
            return None
            
        except ImportError as e:
            logger.error(f"Could not import DepthSplat modules: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating inference model: {e}")
            return None


def load_model(
    checkpoint_path: str,
    config: Optional[ModelConfig] = None,
) -> DepthSplatWrapper:
    """
    Load a DepthSplat model wrapper.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Model configuration
        
    Returns:
        Model wrapper instance
    """
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required")
    return DepthSplatWrapper.from_checkpoint(checkpoint_path, config)


def analyze_model(checkpoint_path: str, input_shape: Optional[Tuple] = None):
    """
    Analyze a DepthSplat model.
    
    Args:
        checkpoint_path: Path to the checkpoint
        input_shape: Optional input shape to test
        
    Returns:
        Dictionary with model analysis results
    """
    import torch
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    analysis = {
        "checkpoint_keys": list(checkpoint.keys()),
        "has_state_dict": "state_dict" in checkpoint,
        "has_model": "model" in checkpoint,
    }
    
    # Get state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    analysis["num_parameters"] = len(state_dict)
    analysis["parameter_names"] = list(state_dict.keys())[:20]  # First 20
    
    # Try to determine model architecture from keys
    layer_types = set()
    for key in state_dict.keys():
        parts = key.split('.')
        if len(parts) > 1:
            layer_types.add(parts[0])
    
    analysis["layer_types"] = list(layer_types)
    
    logger.info("Model Analysis:")
    for k, v in analysis.items():
        if k != "parameter_names":
            logger.info(f"  {k}: {v}")
    
    return analysis


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DepthSplat Model Wrapper")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/sandro/thesis/code/depthsplat/outputs/objaverse_white/checkpoints/epoch_0-step_65000.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze model structure",
    )
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_model(args.checkpoint)
    else:
        # Try to load the model
        try:
            wrapper = load_model(args.checkpoint)
            print(f"Loaded wrapper: {type(wrapper).__name__}")
        except Exception as e:
            print(f"Error loading model: {e}")
