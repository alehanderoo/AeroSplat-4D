"""
Configuration module for DepthSplat Inference Pipeline.

This module provides configuration loading and validation utilities.

Usage:
    from config import load_config, PipelineSettings
    
    config = load_config("config/pipeline_config.yaml")
    print(config.pipeline.name)
"""

import os
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class PipelineInfo:
    """Pipeline metadata."""
    name: str = "depthsplat-inference"
    version: str = "1.0.0"


@dataclass
class RTSPConfig:
    """RTSP source configuration."""
    urls: List[str] = field(default_factory=lambda: [
        "rtsp://localhost:8554/cam_01",
        "rtsp://localhost:8554/cam_02",
        "rtsp://localhost:8554/cam_03",
        "rtsp://localhost:8554/cam_04",
        "rtsp://localhost:8554/cam_05",
    ])
    latency: int = 100
    protocols: str = "tcp"
    retry_interval: int = 5
    max_retries: int = -1


@dataclass
class FileSourceConfig:
    """File source configuration."""
    directory: str = ""
    pattern: str = "cam_{cam_id}/rgb/rgb_{frame:04d}.png"
    fps: int = 30
    loop: bool = True


@dataclass
class SimulatorConfig:
    """Simulator configuration."""
    render_dir: str = ""
    host: str = "0.0.0.0"
    port: int = 8554
    fps: int = 30
    loop: bool = True


@dataclass
class SourcesConfig:
    """Source configuration."""
    mode: str = "rtsp"  # "rtsp", "file", "simulator"
    rtsp: RTSPConfig = field(default_factory=RTSPConfig)
    files: FileSourceConfig = field(default_factory=FileSourceConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)


@dataclass
class ModelInputConfig:
    """Model input specification."""
    num_cameras: int = 5
    channels: int = 3
    height: int = 256
    width: int = 256
    format: str = "RGB"


@dataclass
class NormalizationConfig:
    """Image normalization config."""
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""
    resize_mode: str = "bilinear"
    normalize: NormalizationConfig = field(default_factory=NormalizationConfig)


@dataclass
class ModelOutputConfig:
    """Model output specification."""
    positions: bool = True
    covariances: bool = True
    colors: bool = True
    opacities: bool = True


@dataclass
class ModelConfig:
    """Model configuration."""
    engine_path: str = "models/depthsplat_fp16.plan"
    checkpoint_path: str = ""
    input: ModelInputConfig = field(default_factory=ModelInputConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    output: ModelOutputConfig = field(default_factory=ModelOutputConfig)


@dataclass
class MuxerConfig:
    """DeepStream muxer configuration."""
    batch_size: int = 5
    width: int = 640
    height: int = 480
    batched_push_timeout: int = 33333
    sync_inputs: bool = True
    live_source: bool = True
    nvbuf_memory_type: int = 0


@dataclass
class InferenceConfig:
    """DeepStream inference configuration."""
    config_file: str = "config/deepstream/nvinfer_config.txt"
    interval: int = 0


@dataclass
class QueueConfig:
    """DeepStream queue configuration."""
    max_size_buffers: int = 2
    leaky: str = "downstream"


@dataclass
class DeepStreamConfig:
    """DeepStream configuration."""
    gpu_id: int = 0
    muxer: MuxerConfig = field(default_factory=MuxerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)


@dataclass
class BufferConfig:
    """Output buffer configuration."""
    max_frames: int = 120


@dataclass
class ExportConfig:
    """Export configuration."""
    enabled: bool = False
    format: str = "ply"
    directory: str = "output/gaussians"


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    enabled: bool = False
    render_fps: int = 30


@dataclass
class OutputConfig:
    """Output configuration."""
    buffer: BufferConfig = field(default_factory=BufferConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


@dataclass
class MetricsConfig:
    """Metrics configuration."""
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 9090


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    file: Optional[str] = None
    use_colors: bool = True


@dataclass
class PerformanceConfig:
    """Performance tracking configuration."""
    log_interval: int = 100
    track_memory: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


@dataclass
class PipelineSettings:
    """
    Complete pipeline settings container.

    This is the main configuration class that holds all settings.
    """
    pipeline: PipelineInfo = field(default_factory=PipelineInfo)
    sources: SourcesConfig = field(default_factory=SourcesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    deepstream: DeepStreamConfig = field(default_factory=DeepStreamConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Store raw dict for accessing non-dataclass config sections
    _raw_dict: Dict[str, Any] = field(default_factory=dict, repr=False)


# =============================================================================
# Configuration Loading
# =============================================================================

def _expand_env_vars(value: Any) -> Any:
    """
    Expand environment variables in string values.
    
    Supports:
        - ${VAR_NAME} - Required variable
        - ${VAR_NAME:default} - Variable with default
    """
    if isinstance(value, str):
        # Pattern for ${VAR} or ${VAR:default}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replace(match):
            var_name = match.group(1)
            default = match.group(2)
            env_value = os.environ.get(var_name)
            
            if env_value is not None:
                return env_value
            elif default is not None:
                return default
            else:
                logger.warning(f"Environment variable {var_name} not set and no default")
                return match.group(0)
        
        return re.sub(pattern, replace, value)
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    else:
        return value


def _dict_to_dataclass(cls, data: Dict) -> Any:
    """
    Recursively convert a dictionary to a dataclass instance.
    
    Handles nested dataclasses and missing/extra keys gracefully.
    """
    if data is None:
        return cls()
    
    # Get field types from dataclass
    import dataclasses
    if not dataclasses.is_dataclass(cls):
        return data
    
    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    
    kwargs = {}
    for field_name, field_type in field_types.items():
        if field_name in data:
            value = data[field_name]
            
            # Get the actual type (handle Optional, etc.)
            actual_type = field_type
            if hasattr(field_type, '__origin__'):
                # Handle Optional, List, etc.
                if field_type.__origin__ is Union:
                    # Get non-None type from Optional
                    args = [a for a in field_type.__args__ if a is not type(None)]
                    if args:
                        actual_type = args[0]
                elif field_type.__origin__ is list:
                    # Keep as list
                    actual_type = list
            
            # Recurse for nested dataclasses
            if dataclasses.is_dataclass(actual_type) and isinstance(value, dict):
                kwargs[field_name] = _dict_to_dataclass(actual_type, value)
            elif actual_type is list and isinstance(value, list):
                # Handle lists (don't recurse into list items by default)
                kwargs[field_name] = value
            else:
                kwargs[field_name] = value
    
    return cls(**kwargs)


def load_config(
    config_path: Union[str, Path],
    expand_env: bool = True,
) -> PipelineSettings:
    """
    Load pipeline configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        expand_env: Whether to expand environment variables
        
    Returns:
        PipelineSettings instance with all configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required. Install with: pip install pyyaml")
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from: {config_path}")
    
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    if raw_config is None:
        raw_config = {}
    
    # Expand environment variables
    if expand_env:
        raw_config = _expand_env_vars(raw_config)

    # Convert to dataclass
    settings = _dict_to_dataclass(PipelineSettings, raw_config)

    # Store raw dict for accessing non-dataclass sections (like visualization)
    settings._raw_dict = raw_config

    logger.info(f"Loaded configuration: {settings.pipeline.name} v{settings.pipeline.version}")

    return settings


def load_config_dict(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration as a raw dictionary.
    
    Useful when you need access to the raw config without type conversion.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required. Install with: pip install pyyaml")
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Main functions
    "load_config",
    "load_config_dict",
    
    # Settings classes
    "PipelineSettings",
    "PipelineInfo",
    "SourcesConfig",
    "RTSPConfig",
    "FileSourceConfig",
    "SimulatorConfig",
    "ModelConfig",
    "ModelInputConfig",
    "PreprocessingConfig",
    "NormalizationConfig",
    "ModelOutputConfig",
    "DeepStreamConfig",
    "MuxerConfig",
    "InferenceConfig",
    "QueueConfig",
    "OutputConfig",
    "BufferConfig",
    "ExportConfig",
    "VisualizationConfig",
    "MonitoringConfig",
    "MetricsConfig",
    "LoggingConfig",
    "PerformanceConfig",
]
