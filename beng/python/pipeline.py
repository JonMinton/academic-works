"""
Full End-to-End Pipeline
=========================

Combines the three algorithm stages:
  Stage 1a -- Marker detection (markers.py)
  Stage 1b -- Camera angle estimation via calibration (calibration.py)
  Stage 2  -- Object displacement via DIC (dic.py)

Thesis reference: Chapter 6 -- System Integration

The full measurement workflow:
1. Capture a reference image and a displaced image.
2. Detect markers in the reference image to determine camera-plane angle.
3. Use the calibration polynomial to convert the marker ratio to an angle.
4. Run DIC on the greyscale image pair to get pixel displacements.
5. Apply the rigid-body compensation to isolate object deformation.
6. (Optional) Convert pixel displacements to physical units using the
   calibrated angle and known geometry.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import NamedTuple

from .dic import (
    compute_displacement_map,
    load_greyscale,
    subtract_dc,
    image_correlate,
    find_rigid_displacement,
    DispMap,
)
from .markers import detect_markers, MarkerDetectionResult
from .calibration import (
    relative_coords,
    plane_angles,
    CalibrationResult,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class PipelineResult(NamedTuple):
    """Complete result from the end-to-end pipeline."""
    marker_result: MarkerDetectionResult
    estimated_angle: float | None       # degrees, from calibration poly
    marker_ratio: float
    displacement_map: DispMap
    rigid_dx: int
    rigid_dy: int


# ---------------------------------------------------------------------------
# Angle estimation from calibration
# ---------------------------------------------------------------------------

def estimate_angle(
    marker_ratio: float,
    calib: CalibrationResult,
) -> float | None:
    """Estimate the camera angle from a marker ratio using the calibration polynomial.

    Inverts the polynomial: given ratio, solve for angle.
    Since the polynomial maps angle -> ratio, we find the angle
    where poly(angle) = ratio by root-finding.

    Parameters
    ----------
    marker_ratio : float
        Measured right_vert / left_vert ratio.
    calib : CalibrationResult
        Fitted calibration data.

    Returns
    -------
    float or None
        Estimated angle in the same units as the calibration angles,
        or None if no valid root is found.
    """
    if np.any(np.isnan(calib.poly_coeffs)):
        return None

    # Solve poly(x) - ratio = 0
    shifted_coeffs = calib.poly_coeffs.copy()
    shifted_coeffs[-1] -= marker_ratio
    roots = np.roots(shifted_coeffs)

    # Filter for real roots within the calibration range
    real_roots = roots[np.isreal(roots)].real
    angle_min = calib.angles.min()
    angle_max = calib.angles.max()

    valid = real_roots[(real_roots >= angle_min - 1) &
                       (real_roots <= angle_max + 1)]

    if len(valid) == 0:
        return None
    # Return the root closest to the middle of the range
    mid = (angle_min + angle_max) / 2
    return float(valid[np.argmin(np.abs(valid - mid))])


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    colour_img_path: str | Path,
    grey_img1_path: str | Path,
    grey_img2_path: str | Path,
    calib: CalibrationResult | None = None,
    n_blocks: int = 3,
    use_red_plane: bool = False,
    marker_physical: np.ndarray | None = None,
) -> PipelineResult:
    """Run the complete measurement pipeline.

    Parameters
    ----------
    colour_img_path : path-like
        Colour image for marker detection (Stage 1).
    grey_img1_path, grey_img2_path : path-like
        Greyscale image pair for DIC (Stage 2).
    calib : CalibrationResult or None
        Pre-computed calibration.  If None, angle estimation is skipped.
    n_blocks : int
        Sub-image block count for DIC.
    use_red_plane : bool
        Use exclusive colour plane for marker detection.
    marker_physical : np.ndarray or None
        Physical marker coordinates.

    Returns
    -------
    PipelineResult
    """
    if marker_physical is None:
        marker_physical = np.array([
            [0, 0], [70, 0], [0, 50], [70, 50],
        ], dtype=np.float64)

    # Stage 1: Marker detection and angle estimation
    marker_result = detect_markers(colour_img_path,
                                   use_red_plane=use_red_plane)

    rel_image, _ = relative_coords(marker_result.coords.coords,
                                   marker_physical)
    left_vert, right_vert = plane_angles(rel_image)
    marker_ratio = right_vert / left_vert if left_vert != 0 else float("nan")

    estimated_angle = None
    if calib is not None:
        estimated_angle = estimate_angle(marker_ratio, calib)

    # Stage 2: DIC displacement mapping
    disp = compute_displacement_map(grey_img1_path, grey_img2_path,
                                    n_blocks=n_blocks)

    return PipelineResult(
        marker_result=marker_result,
        estimated_angle=estimated_angle,
        marker_ratio=marker_ratio,
        displacement_map=disp,
        rigid_dx=disp.rigid.dx,
        rigid_dy=disp.rigid.dy,
    )
