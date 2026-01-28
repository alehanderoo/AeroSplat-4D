"""
DepthSplat Inference Service.

This module provides a standalone inference service for 3D Gaussian Splatting
reconstruction from multi-view images. It is designed to be used as a backend
for various frontends (Gradio, REST API, CLI, etc.).

Example usage:
    from inference_backend import InferenceService, InputContext, RenderSettings

    # Initialize service
    service = InferenceService.from_checkpoint("/path/to/checkpoint.ckpt")

    # Run inference
    result = service.reconstruct(
        images=[img1, img2, img3, img4, img5],
        extrinsics=camera_extrinsics,
        intrinsics=camera_intrinsics,
        render_settings=RenderSettings(azimuth=45, elevation=30, distance=1.0),
    )

    # Access outputs
    print(f"PLY saved to: {result.ply_path}")
    print(f"Rendered image: {result.rendered_image_path}")
"""

import os
import sys
import uuid
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange, repeat

# Add depthsplat to path for imports
BACKEND_ROOT = Path(__file__).parent
DEPTHSPLAT_ROOT = BACKEND_ROOT.parent
if str(DEPTHSPLAT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPTHSPLAT_ROOT))

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from src.config import load_typed_root_config
from src.global_cfg import set_cfg
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder
from src.model.ply_export import export_ply
from src.dataset.data_module import get_data_shim
from src.misc.image_io import save_video as save_video_frames

from .types import (
    InputContext,
    RenderSettings,
    VideoSettings,
    InferenceResult,
    DepthAnalysisResult,
)
from .config import ServiceConfig, ModelConfig, InferenceConfig
from .camera_utils import (
    create_target_camera,
    create_360_video_cameras,
    compute_mean_camera_elevation,
)
from .depth_analysis import (
    apply_colormap,
    apply_turbo_colormap,
    normalize_depth_for_display,
    compute_depth_metrics,
    scale_and_shift_pred,
    format_metrics_for_display,
)


class DepthAnythingV2Wrapper:
    """
    Wrapper for standalone Depth Anything V2 inference.

    This runs the pretrained monocular depth model independently of the
    DepthSplat pipeline to evaluate raw monocular depth quality.
    """

    def __init__(self, model_type: str = "vitb", device: str = "cuda"):
        self.device = device
        self.model_type = model_type
        self.model = None

        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

    def _load_model(self):
        """Lazy load the model on first use."""
        if self.model is not None:
            return

        print(f"Loading Depth Anything V2 ({self.model_type})...")

        depth_anything_path = DEPTHSPLAT_ROOT.parent / "Depth-Anything-V2"
        if depth_anything_path.exists():
            if str(depth_anything_path) not in sys.path:
                sys.path.insert(0, str(depth_anything_path))

            from depth_anything_v2.dpt import DepthAnythingV2

            self.model = DepthAnythingV2(**self.model_configs[self.model_type])

            weight_path = DEPTHSPLAT_ROOT / "pretrained" / f"depth_anything_v2_{self.model_type}.pth"
            if weight_path.exists():
                self.model.load_state_dict(torch.load(weight_path, map_location='cpu'))
                print(f"  Loaded weights from {weight_path}")
            else:
                print(f"  Warning: Weights not found at {weight_path}")

            self.model = self.model.to(self.device).eval()
        else:
            print(f"  Warning: Depth-Anything-V2 not found at {depth_anything_path}")
            self.model = None

    @torch.no_grad()
    def infer(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Run depth estimation on a list of images."""
        self._load_model()

        if self.model is None:
            return [np.zeros((img.shape[0], img.shape[1])) for img in images]

        depths = []
        for img in images:
            depth = self.model.infer_image(img)
            depths.append(depth)

        return depths


class InferenceService:
    """
    DepthSplat Inference Service.

    Provides a clean API for 3D Gaussian Splatting reconstruction from
    multi-view images.
    """

    def __init__(self, config: ServiceConfig):
        """
        Initialize the inference service.

        Args:
            config: Service configuration
        """
        self.config = config
        self.device = config.model.device

        # Load model
        print(f"Loading config: {config.model.config_name}")
        self.cfg_dict = self._load_config(config.model.config_name)
        self.cfg = load_typed_root_config(self.cfg_dict)
        set_cfg(self.cfg_dict)

        print("Building model...")
        self.encoder, self.encoder_visualizer = get_encoder(self.cfg.model.encoder)
        self.decoder = get_decoder(self.cfg.model.decoder, self.cfg.dataset)

        print(f"Loading checkpoint from: {config.model.checkpoint_path}")
        checkpoint = torch.load(config.model.checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
        decoder_state = {k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder.')}

        self.encoder.load_state_dict(encoder_state, strict=False)
        self.decoder.load_state_dict(decoder_state, strict=False)

        self.encoder = self.encoder.to(self.device).eval()
        self.decoder = self.decoder.to(self.device).eval()

        # Get data shim for preprocessing
        self.data_shim = get_data_shim(self.encoder)

        # Store config values
        self.image_shape = self.cfg.dataset.image_shape
        self.near = self.cfg.dataset.near
        self.far = self.cfg.dataset.far
        self.background_color = self.cfg.dataset.background_color

        # Initialize depth analysis module
        if config.inference.enable_depth_analysis:
            self.depth_anything = DepthAnythingV2Wrapper(
                model_type=self.cfg.model.encoder.monodepth_vit_type,
                device=self.device,
            )
        else:
            self.depth_anything = None

        print("Model loaded successfully!")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_name: str = "objaverse_white",
        device: str = "cuda",
    ) -> "InferenceService":
        """
        Create inference service from checkpoint path.

        Args:
            checkpoint_path: Path to model checkpoint
            config_name: Experiment config name
            device: Compute device

        Returns:
            InferenceService instance
        """
        config = ServiceConfig(
            model=ModelConfig(
                checkpoint_path=checkpoint_path,
                config_name=config_name,
                device=device,
            ),
            inference=InferenceConfig(),
        )
        return cls(config)

    def _load_config(self, config_name: str) -> DictConfig:
        """Load configuration using Hydra compose API."""
        config_dir = str(DEPTHSPLAT_ROOT / "config")

        GlobalHydra.instance().clear()

        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="main",
                overrides=[f"+experiment={config_name}", "mode=test"],
            )

        return cfg

    def preprocess_images(
        self,
        images: List[np.ndarray],
        target_size: Tuple[int, int] = None,
    ) -> torch.Tensor:
        """
        Preprocess input images.

        Args:
            images: List of numpy arrays (H, W, 3) in [0, 255]
            target_size: Target size (H, W) for resizing

        Returns:
            Tensor of shape [1, V, 3, H, W] in [0, 1]
        """
        if target_size is None:
            target_size = self.image_shape

        processed = []
        for img in images:
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0

            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

            if img_tensor.shape[1:] != tuple(target_size):
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0),
                    size=target_size,
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)

            processed.append(img_tensor)

        images_tensor = torch.stack(processed, dim=0).unsqueeze(0)
        return images_tensor.to(self.device)

    def build_batch(
        self,
        images: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        near: float = None,
        far: float = None,
    ) -> dict:
        """
        Build batch dictionary for the encoder.

        Args:
            images: [1, V, 3, H, W] tensor
            extrinsics: [1, V, 4, 4] camera-to-world matrices
            intrinsics: [1, V, 3, 3] normalized intrinsics
            near: Near plane distance
            far: Far plane distance

        Returns:
            Batch dictionary with context and target
        """
        b, v, _, h, w = images.shape

        if near is None:
            near = self.near
        if far is None:
            far = self.far

        context = {
            'image': images,
            'extrinsics': extrinsics.to(self.device),
            'intrinsics': intrinsics.to(self.device),
            'near': torch.full((b, v), near, device=self.device),
            'far': torch.full((b, v), far, device=self.device),
            'index': torch.arange(v, device=self.device).unsqueeze(0),
        }

        target = {
            'image': images[:, :1],
            'extrinsics': extrinsics[:, :1].to(self.device),
            'intrinsics': intrinsics[:, :1].to(self.device),
            'near': torch.full((b, 1), near, device=self.device),
            'far': torch.full((b, 1), far, device=self.device),
            'index': torch.zeros(b, 1, dtype=torch.long, device=self.device),
        }

        batch = {
            'context': context,
            'target': target,
            'scene': ['inference_input'],
        }

        return batch

    @torch.no_grad()
    def reconstruct(
        self,
        images: List[np.ndarray],
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        render_settings: RenderSettings = None,
        video_settings: VideoSettings = None,
        output_dir: str = None,
    ) -> InferenceResult:
        """
        Run 3D Gaussian Splatting reconstruction.

        This is the main entry point for inference. It takes multi-view images
        and camera parameters, reconstructs a 3D Gaussian Splat model, and
        renders novel views.

        Args:
            images: List of RGB images as numpy arrays [H, W, 3] in uint8
            extrinsics: [V, 4, 4] camera-to-world matrices (OpenCV convention)
            intrinsics: [V, 3, 3] normalized intrinsics matrices
            render_settings: Settings for novel view rendering (optional)
            video_settings: Settings for 360 video generation (optional)
            output_dir: Directory to save outputs (optional, uses temp if None)

        Returns:
            InferenceResult with paths to all generated outputs
        """
        torch.cuda.empty_cache()

        # Set defaults
        if render_settings is None:
            render_settings = RenderSettings()
        if video_settings is None:
            video_settings = VideoSettings()

        # Create output directory
        if output_dir is None:
            output_dir = os.path.join(
                self.config.inference.output_dir,
                f"output_{uuid.uuid4().hex[:8]}"
            )
        os.makedirs(output_dir, exist_ok=True)

        # Preprocess images
        images_tensor = self.preprocess_images(images)
        b, v, c, h, w = images_tensor.shape

        # Convert camera parameters to tensors
        extrinsics_tensor = torch.from_numpy(extrinsics).float().unsqueeze(0).to(self.device)
        intrinsics_tensor = torch.from_numpy(intrinsics).float().unsqueeze(0).to(self.device)

        # Build batch
        batch = self.build_batch(images_tensor, extrinsics_tensor, intrinsics_tensor)
        batch = self.data_shim(batch)
        context = batch['context']

        # Update h, w after potential cropping by data shim
        h, w = context['image'].shape[-2:]

        # Run encoder
        print("Running encoder...")
        visualization_dump = {}
        gaussians = self.encoder(context, global_step=0, deterministic=True, visualization_dump=visualization_dump)

        if isinstance(gaussians, dict):
            gaussians = gaussians['gaussians']

        # Create target camera from render settings
        target_extrinsic = create_target_camera(
            azimuth=render_settings.azimuth,
            elevation=render_settings.elevation,
            distance=render_settings.distance,
            base_radius=self.config.inference.target_radius,
        )
        target_extrinsics_tensor = torch.from_numpy(target_extrinsic[np.newaxis, ...]).float().unsqueeze(0).to(self.device)
        target_intrinsics_tensor = intrinsics_tensor[:, :1]

        target_near = torch.full((1, 1), self.near, device=self.device)
        target_far = torch.full((1, 1), self.far, device=self.device)

        # Render from target viewpoint
        print("Rendering from target viewpoint...")
        output = self.decoder.forward(
            gaussians,
            target_extrinsics_tensor,
            target_intrinsics_tensor,
            target_near,
            target_far,
            (h, w),
            depth_mode="depth",
        )

        rendered_images = output.color[0]
        rendered_depth = output.depth[0] if output.depth is not None else None
        rendered_alpha = output.alpha[0] if output.alpha is not None else None

        # Convert to numpy
        rendered_np = (rendered_images.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

        # Save rendered image
        result_image_path = os.path.join(output_dir, "rendered.png")
        Image.fromarray(rendered_np[0]).save(result_image_path)

        # Save depth visualization
        depth_image_path = None
        if rendered_depth is not None:
            depth_np = rendered_depth[0].cpu().numpy()
            depth_normalized = normalize_depth_for_display(depth_np)
            depth_colored = apply_turbo_colormap(depth_normalized)
            depth_image_path = os.path.join(output_dir, "depth.png")
            Image.fromarray(depth_colored).save(depth_image_path)

        # Save silhouette
        silhouette_image_path = None
        if rendered_alpha is not None:
            alpha_np = rendered_alpha[0].cpu().numpy()
            alpha_np = np.clip(alpha_np, 0, 1)
            silhouette_np = (alpha_np * 255).astype(np.uint8)
            silhouette_image_path = os.path.join(output_dir, "silhouette.png")
            Image.fromarray(silhouette_np, mode='L').save(silhouette_image_path)

        # Generate videos
        video_rgb_path = None
        video_depth_path = None
        video_silhouette_path = None
        if video_settings.enabled and video_settings.num_frames > 0:
            video_rgb_path, video_depth_path, video_silhouette_path = self._generate_videos(
                gaussians=gaussians,
                base_intrinsics=intrinsics_tensor[:, 0],
                output_dir=output_dir,
                video_settings=video_settings,
                h=h,
                w=w,
            )

        # Export PLY
        ply_path = None
        if self.config.inference.enable_ply_export and 'scales' in visualization_dump:
            ply_path = os.path.join(output_dir, "gaussians.ply")
            self._export_ply(gaussians, visualization_dump, context, ply_path, v, h, w)

        # Compute depth analysis
        depth_analysis = None
        if self.config.inference.enable_depth_analysis:
            depth_analysis = self._compute_depth_analysis(
                images=images,
                visualization_dump=visualization_dump,
                output_dir=output_dir,
            )

        return InferenceResult(
            ply_path=ply_path,
            rendered_image_path=result_image_path,
            depth_image_path=depth_image_path,
            silhouette_image_path=silhouette_image_path,
            video_rgb_path=video_rgb_path,
            video_depth_path=video_depth_path,
            video_silhouette_path=video_silhouette_path,
            output_dir=output_dir,
            depth_analysis=depth_analysis.to_dict() if depth_analysis else None,
            metadata={
                'num_views': v,
                'num_gaussians': gaussians.means.shape[1],
                'render_settings': {
                    'azimuth': render_settings.azimuth,
                    'elevation': render_settings.elevation,
                    'distance': render_settings.distance,
                },
            },
        )

    @torch.no_grad()
    def render_novel_view(
        self,
        gaussians,
        context: dict,
        render_settings: RenderSettings,
        intrinsics: torch.Tensor,
        h: int,
        w: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Render a novel view from existing Gaussians.

        Args:
            gaussians: Gaussian model from encoder
            context: Context dictionary with camera parameters
            render_settings: Render settings
            intrinsics: [1, 3, 3] intrinsics tensor
            h, w: Output image dimensions

        Returns:
            Tuple of (rgb_image, depth_image, alpha_image) as numpy arrays
        """
        target_extrinsic = create_target_camera(
            azimuth=render_settings.azimuth,
            elevation=render_settings.elevation,
            distance=render_settings.distance,
            base_radius=self.config.inference.target_radius,
        )
        target_extrinsics_tensor = torch.from_numpy(target_extrinsic[np.newaxis, ...]).float().unsqueeze(0).to(self.device)
        target_intrinsics_tensor = intrinsics.unsqueeze(1)

        target_near = torch.full((1, 1), self.near, device=self.device)
        target_far = torch.full((1, 1), self.far, device=self.device)

        output = self.decoder.forward(
            gaussians,
            target_extrinsics_tensor,
            target_intrinsics_tensor,
            target_near,
            target_far,
            (h, w),
            depth_mode="depth",
        )

        rgb = (output.color[0, 0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        depth = output.depth[0, 0].cpu().numpy() if output.depth is not None else None
        alpha = output.alpha[0, 0].cpu().numpy() if output.alpha is not None else None

        return rgb, depth, alpha

    def _generate_videos(
        self,
        gaussians,
        base_intrinsics: torch.Tensor,
        output_dir: str,
        video_settings: VideoSettings,
        h: int,
        w: int,
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """Generate 360-degree rotation videos."""
        print(f"Generating {video_settings.num_frames}-frame videos...")

        # Generate camera trajectory
        video_extrinsics = create_360_video_cameras(
            num_frames=video_settings.num_frames,
            radius=self.config.inference.target_radius * video_settings.distance,
            elevation_angle=video_settings.elevation,
        )

        video_extrinsics_tensor = torch.from_numpy(video_extrinsics).float().unsqueeze(0).to(self.device)
        video_intrinsics_tensor = base_intrinsics.unsqueeze(1).expand(-1, video_settings.num_frames, -1, -1)
        video_near = torch.full((1, video_settings.num_frames), self.near, device=self.device)
        video_far = torch.full((1, video_settings.num_frames), self.far, device=self.device)

        # Render all frames
        output = self.decoder.forward(
            gaussians,
            video_extrinsics_tensor,
            video_intrinsics_tensor,
            video_near,
            video_far,
            (h, w),
            depth_mode="depth" if video_settings.include_depth else None,
        )

        video_frames_rgb = output.color[0]
        video_frames_depth = output.depth[0] if output.depth is not None else None
        video_frames_alpha = output.alpha[0] if output.alpha is not None else None

        # Save RGB video
        video_rgb_path = os.path.join(output_dir, "video_rgb.mp4")
        frames_list = [frame for frame in video_frames_rgb]
        save_video_frames(frames_list, video_rgb_path, fps=video_settings.fps)

        # Save depth video
        video_depth_path = None
        if video_settings.include_depth and video_frames_depth is not None:
            depth_frames = []
            for i in range(video_frames_depth.shape[0]):
                depth_np = video_frames_depth[i].cpu().numpy()
                depth_normalized = normalize_depth_for_display(depth_np)
                depth_colored = apply_turbo_colormap(depth_normalized)
                depth_tensor = torch.from_numpy(depth_colored).permute(2, 0, 1).float() / 255.0
                depth_frames.append(depth_tensor)
            video_depth_path = os.path.join(output_dir, "video_depth.mp4")
            save_video_frames(depth_frames, video_depth_path, fps=video_settings.fps)

        # Save silhouette video
        video_silhouette_path = None
        if video_settings.include_silhouette and video_frames_alpha is not None:
            silhouette_frames = []
            for i in range(video_frames_alpha.shape[0]):
                alpha_np = video_frames_alpha[i].cpu().numpy()
                alpha_np = np.clip(alpha_np, 0, 1)
                silhouette_tensor = torch.from_numpy(alpha_np).float().unsqueeze(0).expand(3, -1, -1)
                silhouette_frames.append(silhouette_tensor)
            video_silhouette_path = os.path.join(output_dir, "video_silhouette.mp4")
            save_video_frames(silhouette_frames, video_silhouette_path, fps=video_settings.fps)

        return video_rgb_path, video_depth_path, video_silhouette_path

    def _export_ply(
        self,
        gaussians,
        visualization_dump: dict,
        context: dict,
        ply_path: str,
        v: int,
        h: int,
        w: int,
    ):
        """Export Gaussians to PLY file."""
        from scipy.spatial.transform import Rotation as R
        print(f"Exporting PLY to: {ply_path}")

        scales = visualization_dump['scales']
        rotations = visualization_dump['rotations']

        means = rearrange(
            gaussians.means, "() (v h w spp) xyz -> h w spp v xyz", v=v, h=h, w=w
        )

        mask = torch.zeros_like(means[..., 0], dtype=torch.bool)
        GAUSSIAN_TRIM = self.config.inference.gaussian_trim_border
        mask[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = 1

        def trim(element):
            element = rearrange(
                element, "() (v h w spp) ... -> h w spp v ...", v=v, h=h, w=w
            )
            return element[mask][None]

        # Convert rotations from camera space to world space
        cam_rotations = trim(rotations)[0]
        c2w_mat = repeat(
            context["extrinsics"][0, :, :3, :3],
            "v a b -> h w spp v a b",
            h=h,
            w=w,
            spp=1,
        )
        c2w_mat = c2w_mat[mask]

        cam_rotations_np = R.from_quat(
            cam_rotations.detach().cpu().numpy()
        ).as_matrix()
        world_mat = c2w_mat.detach().cpu().numpy() @ cam_rotations_np
        world_rotations = R.from_matrix(world_mat).as_quat()
        world_rotations = torch.from_numpy(world_rotations).to(scales)

        export_ply(
            context["extrinsics"][0, 0],
            trim(gaussians.means)[0],
            trim(scales)[0],
            world_rotations,
            trim(gaussians.harmonics)[0],
            trim(gaussians.opacities)[0],
            Path(ply_path),
        )

    def _compute_depth_analysis(
        self,
        images: List[np.ndarray],
        visualization_dump: dict,
        output_dir: str,
    ) -> DepthAnalysisResult:
        """Compute comprehensive depth analysis."""
        depth_dir = os.path.join(output_dir, "depth_analysis")
        os.makedirs(depth_dir, exist_ok=True)

        result = DepthAnalysisResult()

        # Run standalone Depth Anything V2
        if self.depth_anything is not None:
            print("Running standalone Depth Anything V2...")
            standalone_depths = self.depth_anything.infer(images)

            for i, depth in enumerate(standalone_depths):
                viz = apply_colormap(depth, 'plasma')
                path = os.path.join(depth_dir, f"standalone_da_view_{i}.png")
                Image.fromarray(viz).save(path)
                result.standalone_da_paths.append(path)

        # Extract coarse cost-volume depth
        if visualization_dump and 'coarse_mv_depth' in visualization_dump:
            coarse_mv = visualization_dump['coarse_mv_depth'][0].cpu().numpy()
            for i in range(coarse_mv.shape[0]):
                depth = 1.0 / np.clip(coarse_mv[i], 1e-6, None)
                viz = apply_colormap(depth, 'plasma')
                path = os.path.join(depth_dir, f"coarse_mv_view_{i}.png")
                Image.fromarray(viz).save(path)
                result.coarse_mv_paths.append(path)

        # Extract DPT residual
        if visualization_dump and 'mono_depth' in visualization_dump:
            residual = visualization_dump['mono_depth'][0].cpu().numpy()
            for i in range(residual.shape[0]):
                viz = apply_colormap(residual[i], 'coolwarm')
                path = os.path.join(depth_dir, f"residual_view_{i}.png")
                Image.fromarray(viz).save(path)
                result.residual_paths.append(path)

        # Compute final fused depth
        if visualization_dump and 'depth' in visualization_dump:
            fused = visualization_dump['depth'][0].squeeze(-1).squeeze(-1).cpu().numpy()
            for i in range(fused.shape[0]):
                depth = 1.0 / np.clip(fused[i], 1e-6, None)
                viz = apply_colormap(depth, 'plasma')
                path = os.path.join(depth_dir, f"final_fused_view_{i}.png")
                Image.fromarray(viz).save(path)
                result.final_fused_paths.append(path)

        return result
