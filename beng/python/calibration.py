"""
Angle Calibration and Polynomial Fitting
==========================================

Port of MATLAB files:
  CalibMain.m       -- Outer loop over calibration sets
  CalibSubMain.m    -- Inner loop over images within a set
  RelativeCoords.m  -- Normalise marker coords relative to top-left
  PlaneAngles.m     -- Compute vertical marker separations
  LoadImage.m       -- Calibration image loader (set/angle naming)

Thesis reference: Chapter 5 -- Calibration

Algorithm summary
-----------------
1. Define real-world marker coordinates (known physical positions,
   e.g. 70mm x 50mm rectangle).
2. For each calibration image (17 angles x 10 sets = 170 images):
   a. Detect the four markers using the marker pipeline.
   b. Normalise image coordinates relative to the top-left marker
      and scale to [0, 1].
   c. Compute the left and right vertical separations:
        left_vert  = image_coords[bottom_left].y - image_coords[top_left].y
        right_vert = image_coords[bottom_right].y - image_coords[top_right].y
   d. Compute the ratio right_vert / left_vert.
3. Average ratios across all sets for each angle.
4. Fit a 3rd-degree polynomial to the angle-vs-ratio data.
5. The polynomial can then be used to estimate camera angle from
   any new image's marker ratio.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import NamedTuple

from .markers import detect_markers, find_marker_coords, create_mask, load_colour


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class CalibrationResult(NamedTuple):
    """Result of the full calibration process."""
    angles: np.ndarray          # Known angles (degrees)
    ratios: np.ndarray          # Measured vert ratios per angle (averaged over sets)
    poly_coeffs: np.ndarray     # Polynomial coefficients (highest power first)
    poly_degree: int            # Degree of fitted polynomial
    raw_ratios: np.ndarray      # shape (n_angles, n_sets) -- individual measurements
    marker_verts: np.ndarray    # shape (n_angles, n_sets, 2) -- left_vert, right_vert


# ---------------------------------------------------------------------------
# Coordinate normalisation  (ports RelativeCoords.m)
# ---------------------------------------------------------------------------

def relative_coords(
    image_coords: np.ndarray,
    marker_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalise coordinates relative to the top-left marker.

    Ports ``RelativeCoords.m``.  Subtracts the top-left marker position
    from all markers, then scales by the maximum coordinate value.

    Parameters
    ----------
    image_coords : np.ndarray
        Shape (4, 2), detected marker positions in pixels.
    marker_coords : np.ndarray
        Shape (4, 2), known physical marker positions.

    Returns
    -------
    (rel_image, rel_marker) : tuple of np.ndarray
        Both normalised to [0, 1] range.
    """
    rel_image = image_coords.copy().astype(np.float64)
    rel_marker = marker_coords.copy().astype(np.float64)

    # Subtract top-left origin
    origin = rel_image[0].copy()
    rel_image -= origin

    # Scale to [0, 1]
    max_val = np.max(np.abs(rel_image))
    if max_val > 0:
        rel_image /= max_val

    max_marker = np.max(np.abs(rel_marker))
    if max_marker > 0:
        rel_marker /= max_marker

    return rel_image, rel_marker


# ---------------------------------------------------------------------------
# Plane angle computation  (ports PlaneAngles.m)
# ---------------------------------------------------------------------------

def plane_angles(
    image_coords: np.ndarray,
) -> tuple[float, float]:
    """Compute the left and right vertical marker separations.

    Ports ``PlaneAngles.m``.

    Parameters
    ----------
    image_coords : np.ndarray
        Shape (4, 2), normalised marker coords.
        Row order: top-left, top-right, bottom-left, bottom-right.
        Columns: (x, y).

    Returns
    -------
    (left_vert, right_vert) : tuple of float
    """
    left_vert = image_coords[2, 1] - image_coords[0, 1]   # BL.y - TL.y
    right_vert = image_coords[3, 1] - image_coords[1, 1]  # BR.y - TR.y
    return float(left_vert), float(right_vert)


# ---------------------------------------------------------------------------
# Calibration image loading  (ports LoadImage.m)
# ---------------------------------------------------------------------------

def calibration_image_path(
    base_dir: str | Path,
    set_idx: int,
    angle_idx: int,
) -> Path:
    """Construct the path to a calibration image.

    Ports ``LoadImage.m``.  The naming convention is:
        {set}_{angle:02d}.BMP
    where set is 0-9 and angle is 01-17.

    Parameters
    ----------
    base_dir : path-like
        Directory containing the calibration images.
    set_idx : int
        Set index (0-9).
    angle_idx : int
        Angle index within the set (0-16, mapped to files 01-17).

    Returns
    -------
    Path
    """
    filename = f"{set_idx}_{angle_idx + 1:02d}.BMP"
    path = Path(base_dir) / filename
    # Handle case-insensitive file systems
    if not path.exists():
        # Try lowercase
        for ext in [".BMP", ".bmp"]:
            alt = Path(base_dir) / f"{set_idx}_{angle_idx + 1:02d}{ext}"
            if alt.exists():
                return alt
    return path


# ---------------------------------------------------------------------------
# Single-image calibration measurement
# ---------------------------------------------------------------------------

def measure_single_image(
    img_path: str | Path,
    marker_physical: np.ndarray,
    use_red_plane: bool = False,
) -> tuple[float, float, float]:
    """Detect markers and compute the vertical ratio for one image.

    Parameters
    ----------
    img_path : path-like
        Path to the calibration image.
    marker_physical : np.ndarray
        Known physical marker coordinates, shape (4, 2).
    use_red_plane : bool
        Whether to use exclusive colour plane extraction.

    Returns
    -------
    (left_vert, right_vert, ratio) : tuple of float
    """
    img = load_colour(img_path)
    mask, _ = create_mask(img, use_red_plane=use_red_plane)
    coords = find_marker_coords(mask)

    rel_image, _ = relative_coords(coords.coords, marker_physical)
    left_vert, right_vert = plane_angles(rel_image)

    ratio = right_vert / left_vert if left_vert != 0 else float("nan")
    return left_vert, right_vert, ratio


# ---------------------------------------------------------------------------
# Full calibration pipeline  (ports CalibMain.m + CalibSubMain.m)
# ---------------------------------------------------------------------------

def run_calibration(
    image_dir: str | Path,
    n_angles: int = 17,
    n_sets: int = 10,
    poly_degree: int = 3,
    marker_physical: np.ndarray | None = None,
    use_red_plane: bool = False,
    verbose: bool = True,
) -> CalibrationResult:
    """Run the full calibration across all image sets and angles.

    Ports ``CalibMain.m`` and ``CalibSubMain.m``.

    Parameters
    ----------
    image_dir : path-like
        Directory containing calibration images named {set}_{angle:02d}.BMP.
    n_angles : int
        Number of angle positions (default 17).
    n_sets : int
        Number of repeated sets (default 10).
    poly_degree : int
        Degree of polynomial to fit (default 3).
    marker_physical : np.ndarray or None
        Known physical marker coordinates.  Default is the thesis values:
        [(0,0), (70,0), (0,50), (70,50)] mm.
    use_red_plane : bool
        Use exclusive colour plane extraction.
    verbose : bool
        Print progress.

    Returns
    -------
    CalibrationResult
    """
    if marker_physical is None:
        marker_physical = np.array([
            [0, 0],
            [70, 0],
            [0, 50],
            [70, 50],
        ], dtype=np.float64)

    raw_ratios = np.full((n_angles, n_sets), np.nan)
    marker_verts = np.full((n_angles, n_sets, 2), np.nan)

    for s in range(n_sets):
        for a in range(n_angles):
            img_path = calibration_image_path(image_dir, s, a)
            if not img_path.exists():
                if verbose:
                    print(f"  Skipping missing: {img_path.name}")
                continue

            try:
                lv, rv, ratio = measure_single_image(
                    img_path, marker_physical, use_red_plane
                )
                raw_ratios[a, s] = ratio
                marker_verts[a, s, 0] = lv
                marker_verts[a, s, 1] = rv
                if verbose:
                    print(f"  Set {s}, Angle {a}: ratio={ratio:.4f}")
            except Exception as e:
                if verbose:
                    print(f"  Error at set={s}, angle={a}: {e}")

    # Average ratios across sets for each angle
    mean_ratios = np.nanmean(raw_ratios, axis=1)

    # The angles themselves -- the MATLAB code doesn't store them explicitly.
    # From the thesis, 17 positions correspond to angles that we can infer
    # from the image numbering.  Without explicit angle labels, we use
    # indices 0..16 as placeholders.  If known angles are available they
    # should be supplied externally.
    angles = np.arange(n_angles, dtype=np.float64)

    # Fit polynomial (only on non-NaN values)
    valid = ~np.isnan(mean_ratios)
    if valid.sum() >= poly_degree + 1:
        coeffs = np.polyfit(angles[valid], mean_ratios[valid], poly_degree)
    else:
        coeffs = np.full(poly_degree + 1, np.nan)

    return CalibrationResult(
        angles=angles,
        ratios=mean_ratios,
        poly_coeffs=coeffs,
        poly_degree=poly_degree,
        raw_ratios=raw_ratios,
        marker_verts=marker_verts,
    )


# ---------------------------------------------------------------------------
# Convenience: calibration from a small subset of images
# ---------------------------------------------------------------------------

def run_calibration_subset(
    image_paths: list[str | Path],
    angles: np.ndarray,
    poly_degree: int = 3,
    marker_physical: np.ndarray | None = None,
    use_red_plane: bool = False,
) -> CalibrationResult:
    """Run calibration on an explicit list of images with known angles.

    Useful for demonstrations with a small number of images.

    Parameters
    ----------
    image_paths : list of path-like
        Paths to calibration images.
    angles : np.ndarray
        Corresponding angles in degrees (same length as image_paths).
    poly_degree : int
        Polynomial degree.
    marker_physical : np.ndarray or None
        Physical marker positions.

    Returns
    -------
    CalibrationResult
    """
    if marker_physical is None:
        marker_physical = np.array([
            [0, 0],
            [70, 0],
            [0, 50],
            [70, 50],
        ], dtype=np.float64)

    n = len(image_paths)
    ratios = np.full(n, np.nan)
    verts = np.full((n, 1, 2), np.nan)

    for idx, path in enumerate(image_paths):
        try:
            lv, rv, ratio = measure_single_image(
                path, marker_physical, use_red_plane
            )
            ratios[idx] = ratio
            verts[idx, 0, 0] = lv
            verts[idx, 0, 1] = rv
        except Exception as e:
            print(f"  Error processing {path}: {e}")

    valid = ~np.isnan(ratios)
    if valid.sum() >= poly_degree + 1:
        coeffs = np.polyfit(angles[valid], ratios[valid], poly_degree)
    else:
        coeffs = np.full(poly_degree + 1, np.nan)

    return CalibrationResult(
        angles=angles,
        ratios=ratios,
        poly_coeffs=coeffs,
        poly_degree=poly_degree,
        raw_ratios=ratios.reshape(-1, 1),
        marker_verts=verts,
    )
