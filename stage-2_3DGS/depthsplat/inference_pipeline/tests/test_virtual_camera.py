#!/usr/bin/env python3
"""
Test script for virtual camera transformation.

Verifies that the camera calibration service correctly computes
virtual camera extrinsics and intrinsics for cropped regions.
"""

import sys
from pathlib import Path

# Add inference directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def test_compute_crop_rotation():
    """Test the crop rotation computation."""
    from services.camera_calibration_service import compute_crop_rotation

    # Test case 1: No offset (crop center = principal point)
    # Should return identity rotation
    R = compute_crop_rotation(cx=1280, cy=720, fx=1000, fy=1000,
                              crop_center_x=1280, crop_center_y=720)
    expected = np.eye(3)
    np.testing.assert_allclose(R, expected, atol=1e-6,
                               err_msg="Identity rotation failed")
    print("[PASS] Test 1: No offset returns identity")

    # Test case 2: Horizontal offset
    # Crop center to the right of principal point
    R = compute_crop_rotation(cx=1280, cy=720, fx=1000, fy=1000,
                              crop_center_x=1380, crop_center_y=720)
    # Should rotate around Y axis (negative, since we rotate camera to look right)
    assert R[0, 0] < 1.0, "Expected rotation around Y axis"
    assert R[0, 2] < 0, "Expected negative X-Z coupling for rightward rotation"
    print(f"[PASS] Test 2: Horizontal offset - R[0,2]={R[0,2]:.4f}")

    # Test case 3: Vertical offset
    # Crop center below principal point
    R = compute_crop_rotation(cx=1280, cy=720, fx=1000, fy=1000,
                              crop_center_x=1280, crop_center_y=820)
    # Should rotate around X axis
    assert R[1, 1] < 1.0, "Expected rotation around X axis"
    print(f"[PASS] Test 3: Vertical offset - R[1,1]={R[1,1]:.4f}")

    # Test case 4: Combined offset
    R = compute_crop_rotation(cx=1280, cy=720, fx=1000, fy=1000,
                              crop_center_x=1380, crop_center_y=820)
    # Should be a valid rotation matrix (orthonormal, det=1)
    det = np.linalg.det(R)
    np.testing.assert_allclose(det, 1.0, atol=1e-6,
                               err_msg="Rotation matrix determinant should be 1")
    RRT = R @ R.T
    np.testing.assert_allclose(RRT, np.eye(3), atol=1e-6,
                               err_msg="R @ R.T should be identity")
    print("[PASS] Test 4: Combined offset - valid rotation matrix")


def test_virtual_camera_intrinsics():
    """Test that virtual camera intrinsics have centered principal point."""
    from services.camera_calibration_service import CameraCalibrationService

    # Create a mock calibration service
    json_path = Path(__file__).parent.parent.parent / "renders" / "5cams_10-01-26" / "drone_camera_observations.json"

    # Check if test data exists
    if not json_path.exists():
        # Use a path relative to home
        json_path = Path.home() / "thesis/renders/5cams_10-01-26/drone_camera_observations.json"

    if not json_path.exists():
        print(f"[SKIP] Test data not found at {json_path}")
        return

    service = CameraCalibrationService(json_path)

    # Test with a crop region
    crop_region = (1000, 500, 1256, 756)  # 256x256 crop

    # Get intrinsics with virtual camera mode
    intr_virtual = service.get_normalized_intrinsics(
        "cam_01", crop_region=crop_region, use_virtual_camera=True
    )

    # Principal point should be centered (0.5, 0.5)
    np.testing.assert_allclose(intr_virtual[0, 2], 0.5, atol=1e-6,
                               err_msg="cx should be 0.5 with virtual camera")
    np.testing.assert_allclose(intr_virtual[1, 2], 0.5, atol=1e-6,
                               err_msg="cy should be 0.5 with virtual camera")
    print(f"[PASS] Virtual camera intrinsics: cx={intr_virtual[0,2]:.3f}, cy={intr_virtual[1,2]:.3f}")

    # Get intrinsics without virtual camera mode (legacy)
    intr_legacy = service.get_normalized_intrinsics(
        "cam_01", crop_region=crop_region, use_virtual_camera=False
    )

    # Principal point should NOT be centered
    assert intr_legacy[0, 2] != 0.5 or intr_legacy[1, 2] != 0.5, \
        "Legacy mode should have non-centered principal point"
    print(f"[PASS] Legacy intrinsics: cx={intr_legacy[0,2]:.3f}, cy={intr_legacy[1,2]:.3f}")

    # Focal lengths should be the same in both modes
    np.testing.assert_allclose(intr_virtual[0, 0], intr_legacy[0, 0], atol=1e-6,
                               err_msg="fx should be same in both modes")
    print(f"[PASS] Focal length preserved: fx={intr_virtual[0,0]:.3f}")


def test_virtual_camera_extrinsics():
    """Test that virtual camera extrinsics are correctly transformed."""
    from services.camera_calibration_service import CameraCalibrationService

    json_path = Path.home() / "thesis/renders/5cams_10-01-26/drone_camera_observations.json"

    if not json_path.exists():
        print(f"[SKIP] Test data not found at {json_path}")
        return

    service = CameraCalibrationService(json_path)

    # Get original extrinsics
    ext_original = service.get_extrinsics("cam_01")

    # Crop at principal point (should give same extrinsics)
    resolution = service.get_resolution("cam_01")
    intrinsics = service.get_intrinsics("cam_01")
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Crop centered on principal point
    crop_region = (int(cx - 128), int(cy - 128), int(cx + 128), int(cy + 128))
    ext_virtual = service.get_virtual_extrinsics("cam_01", crop_region)

    # Should be approximately the same (small numerical differences)
    np.testing.assert_allclose(ext_virtual, ext_original, atol=1e-5,
                               err_msg="Centered crop should preserve extrinsics")
    print("[PASS] Centered crop preserves extrinsics")

    # Crop off-center (should rotate extrinsics)
    crop_region_offset = (500, 300, 756, 556)  # Off-center crop
    ext_offset = service.get_virtual_extrinsics("cam_01", crop_region_offset)

    # Rotation part should be different
    R_orig = ext_original[:3, :3]
    R_offset = ext_offset[:3, :3]

    # Check that rotation changed
    diff = np.abs(R_orig - R_offset).max()
    assert diff > 0.01, f"Expected rotation change, got max diff {diff}"
    print(f"[PASS] Off-center crop changes rotation (max diff: {diff:.4f})")

    # Translation should remain the same
    t_orig = ext_original[:3, 3]
    t_offset = ext_offset[:3, 3]
    np.testing.assert_allclose(t_offset, t_orig, atol=1e-6,
                               err_msg="Translation should not change")
    print("[PASS] Translation preserved")

    # Extrinsics should still be a valid SE(3) matrix
    R = ext_offset[:3, :3]
    det = np.linalg.det(R)
    np.testing.assert_allclose(det, 1.0, atol=1e-5,
                               err_msg="Rotation determinant should be 1")
    print("[PASS] Virtual extrinsics is valid SE(3)")


def test_tensor_methods():
    """Test the tensor methods with virtual camera."""
    from services.camera_calibration_service import CameraCalibrationService

    json_path = Path.home() / "thesis/renders/5cams_10-01-26/drone_camera_observations.json"

    if not json_path.exists():
        print(f"[SKIP] Test data not found at {json_path}")
        return

    service = CameraCalibrationService(json_path)

    # Create crop regions for all cameras (256x256 - small crop)
    crop_regions_small = [
        (1000, 500, 1256, 756),
        (800, 400, 1056, 656),
        (1100, 600, 1356, 856),
        (900, 450, 1156, 706),
        (950, 480, 1206, 736),
    ]

    # Get tensors with virtual camera (small crop)
    intrinsics_small = service.get_intrinsics_tensor(
        crop_regions=crop_regions_small,
        use_virtual_camera=True,
    )

    print("--- Small crop (256x256) ---")
    print(f"Intrinsics shape: {intrinsics_small.shape}")

    # All intrinsics should have centered principal point
    for i in range(5):
        cx = intrinsics_small[0, i, 0, 2].item()
        cy = intrinsics_small[0, i, 1, 2].item()
        assert abs(cx - 0.5) < 1e-5, f"Camera {i}: cx should be 0.5, got {cx}"
        assert abs(cy - 0.5) < 1e-5, f"Camera {i}: cy should be 0.5, got {cy}"

    print("[PASS] All cameras have centered principal point (0.5, 0.5)")

    # Print focal lengths for inspection
    print("\nFocal lengths (normalized) with 256x256 crop:")
    fx_small = intrinsics_small[0, 0, 0, 0].item()
    print(f"  fx={fx_small:.3f} (expected ~7.6 for 256 crop)")

    # Now test with larger crop (1024x1024) - this should give reasonable focal length
    crop_regions_large = [
        (628, 208, 1652, 1232),   # 1024x1024 centered differently per camera
        (500, 100, 1524, 1124),
        (700, 300, 1724, 1324),
        (550, 150, 1574, 1174),
        (600, 200, 1624, 1224),
    ]

    intrinsics_large = service.get_intrinsics_tensor(
        crop_regions=crop_regions_large,
        use_virtual_camera=True,
    )
    extrinsics_large = service.get_extrinsics_tensor(
        crop_regions=crop_regions_large,
        use_virtual_camera=True,
    )

    print("\n--- Large crop (1024x1024) ---")
    print(f"Intrinsics shape: {intrinsics_large.shape}")
    print(f"Extrinsics shape: {extrinsics_large.shape}")

    # Print focal lengths for inspection
    print("\nFocal lengths (normalized) with 1024x1024 crop:")
    fx_large = intrinsics_large[0, 0, 0, 0].item()
    print(f"  fx={fx_large:.3f} (expected ~1.9 for 1024 crop)")

    # Verify focal length is in reasonable range for training
    assert fx_large < 3.0, f"Focal length {fx_large} still too high for training distribution"
    print(f"[PASS] Focal length {fx_large:.3f} is in reasonable range (<3.0)")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Virtual Camera Transformation")
    print("=" * 60)

    print("\n--- Test: Crop Rotation Computation ---")
    test_compute_crop_rotation()

    print("\n--- Test: Virtual Camera Intrinsics ---")
    test_virtual_camera_intrinsics()

    print("\n--- Test: Virtual Camera Extrinsics ---")
    test_virtual_camera_extrinsics()

    print("\n--- Test: Tensor Methods ---")
    test_tensor_methods()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
