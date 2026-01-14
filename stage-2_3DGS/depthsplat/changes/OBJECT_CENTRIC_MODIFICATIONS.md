# Object-Centric DepthSplat Modifications

This document details all modifications made to the original DepthSplat codebase ([cvg/depthsplat](https://github.com/cvg/depthsplat)) to enable object-centric 3D Gaussian Splatting training on Objaverse data.

---

## Table of Contents
1. [Overview](#overview)
2. [New Files](#new-files)
3. [Architectural Modifications](#architectural-modifications)
4. [Loss Functions](#loss-functions)
5. [Cost Volume Modifications](#cost-volume-modifications)
6. [Dataset and View Sampling](#dataset-and-view-sampling)
7. [Visualization Changes](#visualization-changes)
8. [Configuration Changes](#configuration-changes)

---

## Overview

The original DepthSplat is designed for **scene-level reconstruction** from video sequences (RealEstate10K, DL3DV). Our modifications adapt it for **object-centric reconstruction** from multi-view renders (Objaverse), requiring:

| Aspect | Original (Scene-Level) | Modified (Object-Centric) |
|--------|------------------------|---------------------------|
| Depth bounds | 0.1-1000m (room/outdoor scale) | 0.5-5.0m (object scale) |
| View sampling | Sequential frames (temporal) | 360° views (spatial) |
| Background | Scene content | Textureless (black/white) |
| Supervision | RGB only | RGB + Silhouette + Depth |
| Cost volume far plane | Scene far plane | Object background |

---

## New Files

### 1. Loss Functions (3 new files)

#### `src/loss/loss_silhouette.py`
**Purpose**: Silhouette consistency loss for object-centric reconstruction.

```python
L_silhouette = BCE(predicted_alpha, GT_mask)
```

**Key Features**:
- Estimates alpha from rendered color by comparing to background color
- Supports both black and white backgrounds
- Provides geometric supervision independent of texture
- Handles textureless regions where photometric losses fail

**Configuration**:
```python
@dataclass
class LossSilhouetteCfg:
    weight: float = 0.1
    method: str = "alpha_from_color"  # or "alpha_from_depth"
    background_color: tuple = (0.0, 0.0, 0.0)  # or (1.0, 1.0, 1.0) for white
    alpha_threshold: float = 0.01
```

#### `src/loss/loss_depth.py`
**Purpose**: Direct depth supervision using ground truth depth from Blender renders.

**Key Features**:
- L1 loss on valid foreground pixels only
- Background depth regularization (pushes background to far plane)
- Configurable depth validity range

**Configuration**:
```python
@dataclass
class LossDepthCfg:
    weight: float = 0.1
    min_depth: float = 0.1
    max_depth: float = 10.0
    background_weight: float = 0.1  # regularization for background
```

#### `src/loss/loss_mask.py`
**Purpose**: Direct alpha/mask supervision (literature-standard approach from LGM, GRM, TriplaneGaussian).

**Key Features**:
- Supervises rendered alpha against GT segmentation mask
- Decouples alpha supervision from RGB loss
- Supports both MSE and BCE loss variants

---

### 2. Dataset Loader

#### `src/dataset/dataset_objaverse.py`
**Purpose**: Dataset loader for Objaverse multi-view renders.

**Key Differences from RE10K**:
- Object-centric depth bounds (0.5-5.0m vs 0.1-1000m)
- Loads segmentation masks (`use_masks: bool`)
- Loads ground truth depth maps (`use_depth: bool`)
- No temporal ordering assumption

**Data Format**:
```python
result = {
    "context": {
        "image": [...],       # RGB images
        "mask": [...],        # Segmentation masks (NEW)
        "depth": [...],       # GT depth maps (NEW)
        "extrinsics": [...],
        "intrinsics": [...],
        "near": 0.5,          # Tighter bounds
        "far": 5.0,
    },
    ...
}
```

---

### 3. View Sampler

#### `src/dataset/view_sampler/view_sampler_object_centric.py`
**Purpose**: View sampling for 360° object captures.

**Sampling Strategies**:
1. **Farthest Point Sampling**: Greedily selects views maximizing angular coverage
2. **Uniform Sampling**: Views at uniform angular intervals
3. **Random Sampling**: Random view selection

**Key Differences from Bounded Sampler**:
- No temporal/sequential frame assumption
- Camera position-based selection (not frame indices)
- Maximizes spatial coverage around object

```python
def _farthest_point_sampling(self, positions, num_samples):
    # Compute pairwise camera distances
    dist_matrix = torch.cdist(positions, positions)

    # Greedily select farthest views
    selected = [random_start]
    for _ in range(num_samples - 1):
        farthest_idx = min_distances.argmax()
        selected.append(farthest_idx)
        min_distances = torch.minimum(min_distances, dist_matrix[farthest_idx])
    return selected
```

---

## Architectural Modifications

### Multi-View Depth Estimation (`src/model/encoder/unimatch/mv_unimatch.py`)

#### Change 1: Mask-Aware Cost Volume
**Purpose**: Set cost volume values to background for masked (non-object) regions.

**Original**:
```python
def forward(self, images, ...):
    # No mask handling
    cost_volume = (ref_features * warped_tgt_features).sum(2).mean(1)
```

**Modified**:
```python
def forward(self, images, ..., mask=None):
    # Process mask
    if mask is not None:
        mask = rearrange(mask, "b v h w -> (b v) 1 h w")
        mask = F.interpolate(mask, size=(ori_h, ori_w), mode="nearest")

    # Apply mask to cost volume
    cost_volume = (ref_features * warped_tgt_features).sum(2).mean(1)

    if mask is not None:
        curr_mask = F.interpolate(mask, size=(h, w), mode="nearest")
        cost_volume = cost_volume * curr_mask  # Zero out background
```

**Rationale**: Background pixels should not contribute to multi-view matching since they have no meaningful correspondence.

#### Change 2: Background Depth Regularization
**Purpose**: Push background depth to far plane (inverse depth = min_depth).

**Added Code** (lines 544-548):
```python
if mask is not None:
    # Set background depth to min_depth (far plane in inverse depth)
    bg_val = min_depth.view(-1, 1, 1, 1)
    depth = depth * curr_mask + bg_val * (1 - curr_mask)
    match_prob = match_prob * curr_mask
```

**Rationale**: Prevents arbitrary depth predictions in background regions, which would create floating Gaussians.

#### Change 3: Monocular Depth Output
**Purpose**: Store DPT upsampler output for visualization/debugging.

**Added Code** (lines 582-586):
```python
# Store monocular depth component
mono_depth = residual_depth.squeeze(1)
mono_depth = rearrange(mono_depth, "(b v) ... -> b v ...", b=b, v=v)
results_dict.update({"mono_depth": mono_depth})
```

---

### Encoder Interface (`src/model/encoder/encoder_depthsplat.py`)

#### Change 1: Mask Propagation
**Purpose**: Pass masks from dataset to depth predictor.

**Original** (line ~162):
```python
results_dict = self.depth_predictor(
    context["image"],
    ...
)
```

**Modified** (line ~170):
```python
results_dict = self.depth_predictor(
    context["image"],
    ...
    mask=context.get("mask"),  # NEW: Pass mask to depth predictor
)
```

#### Change 2: Interpolation Mode
**Purpose**: Smoother match probability upsampling.

**Original** (line ~223):
```python
match_prob = F.interpolate(match_prob, size=depth.shape[-2:], mode='nearest')
```

**Modified**:
```python
match_prob = F.interpolate(match_prob, size=depth.shape[-2:], mode='bilinear', align_corners=True)
```

---

### Training Wrapper (`src/model/model_wrapper.py`)

#### Change 1: Foreground Mask Loss Weighting
**Purpose**: Compute loss only on foreground pixels to prevent collapse to background.

**Added Code** (lines 243-249):
```python
valid_depth_mask = None
if "mask" in batch["target"]:
    target_mask = batch["target"]["mask"]
    if target_mask.dim() == 5:
        target_mask = target_mask.squeeze(2)
    # valid_depth_mask=True means "exclude this pixel"
    valid_depth_mask = (target_mask < 0.5).unsqueeze(2).expand_as(output.color)
```

**Rationale**: Background comprises ~90% of pixels in object-centric data. Without masking, the model would learn to predict uniform background color.

#### Change 2: Depth Loss Integration
**Purpose**: Pass predicted depths to depth supervision loss.

**Added Code** (lines 262-271):
```python
elif loss_fn.name == "depth":
    loss = loss_fn.forward(
        output,
        batch,
        gaussians,
        self.global_step,
        pred_depths=pred_depths,  # NEW: Pass encoder's depth prediction
        valid_depth_mask=valid_depth_mask,
    )
```

#### Change 3: Silhouette Visualization
**Purpose**: Visualize predicted vs GT silhouettes during validation.

**Added Code** (lines 731-779):
```python
if "mask" in batch["context"]:
    # Render from context viewpoints
    context_render = self.decoder.forward(...)

    # Compute predicted silhouette from color
    if is_white_bg:
        pred_silhouette = 1.0 - context_rendered.min(dim=1)[0]
    else:
        pred_silhouette = context_rendered.max(dim=1)[0]

    # Log comparison: context images, GT masks, predicted silhouettes
    silhouette_comparison = vcat(...)
    self.logger.log_image("silhouette", ...)
```

---

## Loss Functions

### Modified: `src/loss/loss_mse.py`

**Added Configuration**:
```python
@dataclass
class LossMseCfg:
    weight: float
    use_foreground_mask: bool = False  # NEW: Optional mask-based loss
```

**Added Logic**:
```python
# Only apply foreground masking if configured
if self.cfg.use_foreground_mask:
    if valid_depth_mask is not None:
        delta = delta[~valid_depth_mask]  # Exclude background
```

### Modified: `src/loss/loss_lpips.py`

**Same pattern**: Added `use_foreground_mask` config option.

### Modified: `src/loss/__init__.py`

**Original**:
```python
LOSSES = {
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
}
```

**Modified**:
```python
LOSSES = {
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossSilhouetteCfgWrapper: LossSilhouette,  # NEW
    LossDepthCfgWrapper: LossDepth,            # NEW
    LossMaskCfgWrapper: LossMask,              # NEW
}
```

---

## Cost Volume Modifications

### How Cost Volume Far Plane is Set to Object Background

The key insight is that the **cost volume depth range** must be adapted for object-centric data where the "far plane" should correspond to the object's background, not distant scene geometry.

#### Mechanism:

1. **Dataset provides tighter depth bounds**:
   ```python
   # DatasetObjaverse
   near: float = 0.5   # Object surface starts here
   far: float = 5.0    # Background/empty space
   ```

2. **Depth bounds flow to cost volume**:
   ```python
   # encoder_depthsplat.py
   results_dict = self.depth_predictor(
       context["image"],
       min_depth=1. / context["far"],   # Inverse depth: 1/5.0 = 0.2
       max_depth=1. / context["near"],  # Inverse depth: 1/0.5 = 2.0
       ...
   )
   ```

3. **Cost volume samples depth candidates within bounds**:
   ```python
   # mv_unimatch.py
   depth_candidates = min_depth + linear_space * (max_depth - min_depth)
   # For object-centric: depth candidates span 0.5m to 5.0m only
   ```

4. **Mask zeros out background cost volume**:
   ```python
   if mask is not None:
       cost_volume = cost_volume * curr_mask  # Background → 0
   ```

5. **Background depth forced to far plane**:
   ```python
   if mask is not None:
       depth = depth * curr_mask + bg_val * (1 - curr_mask)
       # Background pixels get depth = far (e.g., 5.0m)
   ```

---

## Visualization Changes

### `src/visualization/validation_in_3d.py`

#### Change 1: Percentile-Based Bounds for Projections
**Purpose**: Exclude outlier Gaussians (floaters) from projection visualization.

**Original**:
```python
def render_projections(gaussians, resolution, margin=0.1, ...):
    minima = gaussians.means.min(dim=1).values
    maxima = gaussians.means.max(dim=1).values
```

**Modified**:
```python
def render_projections(gaussians, resolution, margin=0.1, percentile_crop=0.0, ...):
    if percentile_crop > 0:
        # Use 2nd-98th percentile for tighter bounds
        minima = torch.quantile(means, percentile_crop, dim=0)
        maxima = torch.quantile(means, 1.0 - percentile_crop, dim=0)
    else:
        minima = gaussians.means.min(dim=1).values
        maxima = gaussians.means.max(dim=1).values
```

#### Change 2: Configurable Frustum Scale
**Purpose**: Smaller camera frustums for object-scale visualization.

**Original**:
```python
def render_cameras(batch, resolution):
    return draw_cameras(..., frustum_scale=0.05)  # Hardcoded
```

**Modified**:
```python
def render_cameras(batch, resolution, frustum_scale=0.05):
    return draw_cameras(..., frustum_scale=frustum_scale)  # Configurable
```

---

## Configuration Changes

### Recommended Object-Centric Config

```yaml
# config/experiment/objaverse.yaml

dataset:
  name: objaverse
  near: 0.5
  far: 5.0
  use_masks: true
  use_depth: true

model:
  encoder:
    gaussian_adapter:
      gaussian_scale_max: 0.1  # Smaller Gaussians for objects (vs 3.0 for scenes)

loss:
  - mse:
      weight: 1.0
      use_foreground_mask: false  # Compute on full image
  - lpips:
      weight: 0.05
      use_foreground_mask: false
  - silhouette:
      weight: 0.1
      background_color: [1.0, 1.0, 1.0]  # White background
  - mask:
      weight: 0.1

dataset/view_sampler:
  name: object_centric
  num_context_views: 2
  num_target_views: 1
  sampling_strategy: farthest_point
```

---

## Summary of Key Modifications

| Component | Modification | Purpose |
|-----------|--------------|---------|
| `mv_unimatch.py` | Mask-aware cost volume | Zero background matching |
| `mv_unimatch.py` | Background depth regularization | Push background to far plane |
| `encoder_depthsplat.py` | Pass mask to depth predictor | Enable mask-aware depth |
| `model_wrapper.py` | Foreground mask loss | Prevent collapse to background |
| `loss_silhouette.py` | NEW | Geometric supervision |
| `loss_depth.py` | NEW | Depth supervision |
| `loss_mask.py` | NEW | Alpha supervision |
| `dataset_objaverse.py` | NEW | Object-centric data loading |
| `view_sampler_object_centric.py` | NEW | 360° view sampling |

---

## References

- Original DepthSplat: https://github.com/cvg/depthsplat
- LGM (mask supervision approach): https://arxiv.org/abs/2402.05054
- GRM (object-centric Gaussians): https://arxiv.org/abs/2403.14621




---

# Do we need silhouette?

now that you know everything about the changes made to depthsplat. Do we actually need the silhouette if we have the alpha/mask supervision? just write your answer here, don't modify any file                                                         
                                                                                                                                                                                                                                                          
● Good question. Looking at the two losses:                                                                                                                                                                                                               
                                                                                                                                                                                                                                                          
  loss_mask.py (Direct Alpha Supervision):                                                                                                                                                                                                                
  - Requires prediction.alpha from the decoder                                                                                                                                                                                                            
  - Direct supervision: MSE(predicted_alpha, GT_mask)                                                                                                                                                                                                     
  - This is the literature-standard approach (LGM, GRM, TriplaneGaussian)                                                                                                                                                                                 
                                                                                                                                                                                                                                                          
  loss_silhouette.py (Heuristic Silhouette):                                                                                                                                                                                                              
  - Does NOT require alpha from decoder                                                                                                                                                                                                                   
  - Estimates alpha from rendered color using heuristics:                                                                                                                                                                                                 
    - Black background: alpha = max(R, G, B)                                                                                                                                                                                                              
    - White background: alpha = 1 - min(R, G, B)                                                                                                                                                                                                          
  - Then applies BCE against GT mask                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                          
  Answer: No, you likely don't need both.                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                          
  If your decoder outputs alpha (check if prediction.alpha is not None), use only loss_mask because:                                                                                                                                                      
  1. It's direct supervision, not heuristic-based                                                                                                                                                                                                         
  2. The silhouette heuristic can be noisy (e.g., dark objects on black background would have low estimated alpha even in foreground)                                                                                                                     
  3. Double-supervising the same objective with different methods could cause conflicting gradients                                                                                                                                                       
                                                                                                                                                                                                                                                          
  The silhouette loss was likely added as a workaround when the decoder didn't output alpha. Looking at loss_mask.py line 67-68, it gracefully handles missing alpha by returning zero loss.                                                              
                                                                                                                                                                                                                                                          
  Recommendation:                                                                                                                                                                                                                                         
  - If decoder outputs alpha → use loss_mask only, remove loss_silhouette                                                                                                                                                                                 
  - If decoder doesn't output alpha → modify decoder to output alpha, then use loss_mask                                                                                                                                                                  
                                                                                                                                                                                                                                                          
  The silhouette loss is essentially a "poor man's mask loss" for when you can't access the actual rendered opacity. 