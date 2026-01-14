# Fix: Augmentation Now Applies to Mask and Depth

**Date:** 2025-12-23
**File:** `src/dataset/shims/augmentation_shim.py`

## Problem

The horizontal flip augmentation (`augment: true` in config) was only being applied to:
- `image` tensors
- `extrinsics` (camera poses)

However, the Objaverse dataset also includes `mask` and `depth` data in the views. These were **not** being flipped, causing a mismatch between the flipped images and un-flipped masks/depths during training.

This would cause the model to learn incorrect correspondences between pixels and their masks/depths when augmentation was active.

## Solution

Modified `reflect_views()` to also flip `mask` and `depth` tensors when present:

```python
def reflect_views(views: AnyViews) -> AnyViews:
    result = {
        **views,
        "image": views["image"].flip(-1),
        "extrinsics": reflect_extrinsics(views["extrinsics"]),
    }
    # Also flip mask and depth if present (for object-centric datasets)
    if "mask" in views:
        result["mask"] = views["mask"].flip(-1)
    if "depth" in views:
        result["depth"] = views["depth"].flip(-1)
    return result
```

## Impact

- **Silhouette loss** (`loss_silhouette.py`): Now receives correctly aligned masks
- **Depth supervision**: Now receives correctly aligned depth maps
- **Backward compatible**: Uses conditional checks, so datasets without mask/depth are unaffected

## Affected Datasets

- `DatasetObjaverse` - primary dataset using masks and depths
- Any future datasets that include `mask` or `depth` fields in views
