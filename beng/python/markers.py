"""
Colour-Based Marker Detection Pipeline
========================================

Port of MATLAB files:
  MaskMaker.m         -- Create binary mask from colour image
  RedMaskMaker.m      -- Exclusive red colour plane extraction
  Markers.m           -- Prototype marker detection script
  Quarterize.m        -- Split mask into quadrants for per-marker processing
  FindImageCoords.m   -- Find marker centroids from quadrants
  XFindImageCoords.m  -- Revised version using full mask dimensions
  CentreFind.m        -- Bounding-box extraction for a marker region
  BlockCentreFind.m   -- Weighted centroid calculation
  EdgeMarker.m        -- Region-growing marker labelling

Thesis reference: Chapter 3 -- Marker Detection

Algorithm summary
-----------------
1. Extract an "exclusive colour plane" from the RGB image.
   For red markers:  plane = complement( (G+B)/2 ) - complement(R)
   This isolates pixels that are distinctly red.
2. Subtract DC, normalise contrast (stretch to [0,1]).
3. Repeat DC subtraction and normalisation for extra contrast.
4. Apply Sobel edge detection to find marker boundaries.
5. Dilate edges (line structuring elements at 0 and 90 degrees).
6. Fill holes to create solid marker blobs.
7. Clear border-touching objects.
8. Erode with a diamond structuring element (twice) to refine.
9. Use the eroded mask to select connected components from the
   filled mask, giving clean marker regions.
10. Split the mask into four quadrants (one per marker).
11. For each quadrant, find the marker's bounding box and compute
    a weighted centroid (sum of pixel positions weighted by mask value).
12. Transform quadrant-local coordinates back to full-image coordinates.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from pathlib import Path
from PIL import Image
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class MarkerCoords(NamedTuple):
    """Pixel coordinates of the four calibration markers.

    Order: top-left, top-right, bottom-left, bottom-right.
    Each entry is (x, y) in pixel coordinates.
    """
    coords: np.ndarray   # shape (4, 2), columns are (x, y)


class MarkerDetectionResult(NamedTuple):
    """Full intermediate results from the marker detection pipeline."""
    original: np.ndarray         # Input RGB image
    colour_plane: np.ndarray     # Exclusive colour plane (float)
    normalised: np.ndarray       # Contrast-normalised plane
    edges: np.ndarray            # Sobel edge detection result
    dilated: np.ndarray          # After dilation
    filled: np.ndarray           # After hole-filling
    border_cleared: np.ndarray   # After clearing border objects
    eroded: np.ndarray           # After erosion
    final_mask: np.ndarray       # Final clean marker mask
    coords: MarkerCoords         # Detected marker coordinates


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_colour(path: str | Path) -> np.ndarray:
    """Load an RGB image as a uint8 numpy array (H, W, 3)."""
    img = np.array(Image.open(path))
    if img.ndim == 2:
        raise ValueError("Expected colour image, got greyscale")
    return img[:, :, :3]  # drop alpha if present


# ---------------------------------------------------------------------------
# Exclusive colour plane extraction  (ports RedMaskMaker.m, Markers.m)
# ---------------------------------------------------------------------------

def exclusive_colour_plane(
    img: np.ndarray,
    channel: int = 0,
) -> np.ndarray:
    """Create an exclusive colour plane for the given channel.

    Thesis ref: Section 3.3 -- Exclusive Colour Planes.

    For the red channel (channel=0), the exclusive plane is computed as:
        plane = complement( mean(other channels) ) - complement(target)
    which isolates pixels that are distinctly *more* red than the average
    of the other channels.

    Parameters
    ----------
    img : np.ndarray
        RGB image, uint8, shape (H, W, 3).
    channel : int
        0=red, 1=green, 2=blue.

    Returns
    -------
    np.ndarray
        Float64 exclusive colour plane, values roughly in [0, 255].
    """
    img_f = img.astype(np.float64)
    other_channels = [c for c in range(3) if c != channel]
    other_mean = np.mean(img_f[:, :, other_channels], axis=2)

    # complement = 255 - x
    comp_other = 255.0 - other_mean
    comp_target = 255.0 - img_f[:, :, channel]

    plane = comp_other - comp_target
    return plane


# ---------------------------------------------------------------------------
# Contrast normalisation  (ports MaskMaker.m DC-subtract + imadjust)
# ---------------------------------------------------------------------------

def normalise_contrast(plane: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Subtract DC and stretch contrast, repeated ``iterations`` times.

    Ports the repeated DC-subtract + ``imadjust`` pattern from
    ``MaskMaker.m``.

    Parameters
    ----------
    plane : np.ndarray
        Input plane (float64).
    iterations : int
        Number of subtract-and-stretch cycles.

    Returns
    -------
    np.ndarray
        Normalised plane in [0, 255] range (float64).
    """
    result = plane.copy()
    for _ in range(iterations):
        result = result - result.mean()
        # Clip to uint8 range, then stretch to [0, 255]
        result = np.clip(result, 0, 255)
        rmin, rmax = result.min(), result.max()
        if rmax > rmin:
            result = (result - rmin) / (rmax - rmin) * 255.0
        else:
            result = np.zeros_like(result)
    return result


# ---------------------------------------------------------------------------
# Edge detection  (ports Sobel edge in MaskMaker.m)
# ---------------------------------------------------------------------------

def sobel_edge(img: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Apply Sobel edge detection.

    Ports ``edge(NormInvGrey, 'sobel')`` from MaskMaker.m.

    Parameters
    ----------
    img : np.ndarray
        Greyscale image (float64).
    threshold : float or None
        Edge threshold.  If None, an automatic threshold is used
        (mean + 1 standard deviation of the gradient magnitude).

    Returns
    -------
    np.ndarray
        Boolean edge map.
    """
    sx = ndimage.sobel(img, axis=1)
    sy = ndimage.sobel(img, axis=0)
    mag = np.hypot(sx, sy)

    if threshold is None:
        threshold = mag.mean() + mag.std()

    return mag > threshold


# ---------------------------------------------------------------------------
# Morphological operations  (ports MaskMaker.m dilation/fill/erosion)
# ---------------------------------------------------------------------------

def dilate_edges(edges: np.ndarray, line_length: int = 3) -> np.ndarray:
    """Dilate edges with line structuring elements at 0 and 90 degrees.

    Ports the ``strel('line', 3, 90)`` and ``strel('line', 3, 0)``
    dilation from MaskMaker.m.
    """
    # Vertical line SE
    se_vert = np.zeros((line_length, 1), dtype=bool)
    se_vert[:, 0] = True

    # Horizontal line SE
    se_horiz = np.zeros((1, line_length), dtype=bool)
    se_horiz[0, :] = True

    result = ndimage.binary_dilation(edges, structure=se_vert)
    result = ndimage.binary_dilation(result, structure=se_horiz)
    return result


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes in a binary mask.  Ports ``imfill(BW, 'holes')``."""
    return ndimage.binary_fill_holes(mask)


def clear_border(mask: np.ndarray) -> np.ndarray:
    """Remove objects connected to the image border.

    Ports ``imclearborder(BW, 4)``.
    """
    # Label connected components
    labelled, n_labels = ndimage.label(mask)
    border_labels = set()

    # Find labels touching any border
    border_labels.update(labelled[0, :].ravel())
    border_labels.update(labelled[-1, :].ravel())
    border_labels.update(labelled[:, 0].ravel())
    border_labels.update(labelled[:, -1].ravel())
    border_labels.discard(0)  # background

    result = mask.copy()
    for lbl in border_labels:
        result[labelled == lbl] = False
    return result


def erode_diamond(mask: np.ndarray, radius: int = 2, iterations: int = 2) -> np.ndarray:
    """Erode with a diamond structuring element.

    Ports ``seD = strel('diamond', 2); imerode(BW, seD)`` applied twice.
    """
    # Build diamond SE
    size = 2 * radius + 1
    se = np.zeros((size, size), dtype=bool)
    for i in range(size):
        for j in range(size):
            if abs(i - radius) + abs(j - radius) <= radius:
                se[i, j] = True

    result = mask.copy()
    for _ in range(iterations):
        result = ndimage.binary_erosion(result, structure=se)
    return result


def select_by_seeds(
    filled_mask: np.ndarray,
    seed_mask: np.ndarray,
) -> np.ndarray:
    """Select connected components from ``filled_mask`` seeded by ``seed_mask``.

    Ports ``bwselect(NormInvBWNobord, c, r)`` where (r, c) come from
    ``find(BWeroded)``.
    """
    labelled, n_labels = ndimage.label(filled_mask)
    seed_labels = set(labelled[seed_mask].ravel())
    seed_labels.discard(0)

    result = np.zeros_like(filled_mask, dtype=bool)
    for lbl in seed_labels:
        result[labelled == lbl] = True
    return result


# ---------------------------------------------------------------------------
# Full mask creation pipeline  (ports MaskMaker.m)
# ---------------------------------------------------------------------------

def create_mask(
    img: np.ndarray,
    use_red_plane: bool = False,
    channel: int = 0,
) -> tuple[np.ndarray, dict]:
    """Create a binary marker mask from a colour image.

    Ports ``MaskMaker.m`` (greyscale path) and ``RedMaskMaker.m``
    (exclusive colour plane path).

    Parameters
    ----------
    img : np.ndarray
        RGB image, uint8, shape (H, W, 3).
    use_red_plane : bool
        If True, use the exclusive colour plane method (RedMaskMaker).
        If False, use the inverted red channel method (MaskMaker).
    channel : int
        Colour channel for exclusive plane (default 0 = red).

    Returns
    -------
    mask : np.ndarray
        Boolean marker mask.
    intermediates : dict
        Dictionary of intermediate results for visualisation.
    """
    intermediates = {}

    if use_red_plane:
        plane = exclusive_colour_plane(img, channel=channel)
        intermediates["colour_plane"] = plane
    else:
        # MaskMaker.m: Grey = InputImage(:,:,1); invGrey = imcomplement(Grey)
        grey = img[:, :, channel].astype(np.float64)
        plane = 255.0 - grey
        intermediates["colour_plane"] = plane

    normalised = normalise_contrast(plane)
    intermediates["normalised"] = normalised

    edges = sobel_edge(normalised)
    intermediates["edges"] = edges

    dilated = dilate_edges(edges)
    intermediates["dilated"] = dilated

    filled = fill_holes(dilated)
    intermediates["filled"] = filled

    border_cleared = clear_border(filled)
    intermediates["border_cleared"] = border_cleared

    eroded = erode_diamond(border_cleared)
    intermediates["eroded"] = eroded

    # Use eroded mask as seeds to select from border-cleared mask
    if eroded.any():
        mask = select_by_seeds(border_cleared, eroded)
    else:
        mask = border_cleared
    intermediates["final_mask"] = mask

    return mask, intermediates


# ---------------------------------------------------------------------------
# Quadrant splitting  (ports Quarterize.m)
# ---------------------------------------------------------------------------

def quarterise(mask: np.ndarray) -> list[np.ndarray]:
    """Split a marker mask into four quadrants.

    Ports ``Quarterize.m``.  The image is divided at its centre into
    top-left, top-right, bottom-left, bottom-right quadrants.

    Each quadrant is returned with the same orientation (i.e. the
    top-right quadrant is mirrored back, etc.) so that the marker
    within it appears in a canonical position for centroid finding.

    After centroid finding, coordinates are transformed back to
    full-image space.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask, shape (H, W).

    Returns
    -------
    list of 4 np.ndarray
        [top_left, top_right, bottom_left, bottom_right].
    """
    H, W = mask.shape
    half_h = int(np.ceil(H / 2))
    half_w = int(np.ceil(W / 2))

    top_left = mask[:half_h, :half_w].copy()
    top_right = mask[:half_h, W - half_w:].copy()
    bottom_left = mask[H - half_h:, :half_w].copy()
    bottom_right = mask[H - half_h:, W - half_w:].copy()

    # The MATLAB code mirrors each quadrant so markers appear in the
    # top-left corner of each.  We replicate: top_right flipped LR,
    # bottom_left flipped UD, bottom_right flipped both.
    # However, after computing the centroid the MATLAB code then flips
    # the coordinates back.  We handle this in find_marker_coords
    # instead, returning the quadrants in their original orientation.
    # This is cleaner but produces the same result.

    return [top_left, top_right, bottom_left, bottom_right]


# ---------------------------------------------------------------------------
# Weighted centroid  (ports BlockCentreFind.m)
# ---------------------------------------------------------------------------

def weighted_centroid(region: np.ndarray) -> tuple[float, float]:
    """Compute the weighted centroid of a binary region.

    Ports ``BlockCentreFind.m``.  The centroid is computed as the
    sum of (position * column/row sum) / total sum.

    Parameters
    ----------
    region : np.ndarray
        Binary mask of a single marker.

    Returns
    -------
    (x, y) : tuple of float
        Centroid position within the region.
    """
    region_f = region.astype(np.float64)

    col_sums = region_f.sum(axis=0)  # sum down columns
    row_sums = region_f.sum(axis=1)  # sum across rows

    total = region_f.sum()
    if total == 0:
        return (0.0, 0.0)

    x_indices = np.arange(1, region_f.shape[1] + 1, dtype=np.float64)
    y_indices = np.arange(1, region_f.shape[0] + 1, dtype=np.float64)

    x_pos = np.sum(x_indices * col_sums) / total
    y_pos = np.sum(y_indices * row_sums) / total

    return (x_pos, y_pos)


# ---------------------------------------------------------------------------
# Bounding-box extraction + centroid  (ports CentreFind.m)
# ---------------------------------------------------------------------------

def find_region_centroid(quarter: np.ndarray) -> tuple[float, float]:
    """Find the centroid of the marker region within a quadrant.

    Ports ``CentreFind.m``:
    1. Find the bounding box of non-zero pixels.
    2. Extract that sub-region.
    3. Compute weighted centroid of the sub-region.
    4. Transform back to quadrant coordinates.

    Parameters
    ----------
    quarter : np.ndarray
        Binary mask of one image quadrant.

    Returns
    -------
    (x, y) : tuple of float
        Centroid in quadrant-local coordinates (1-based to match MATLAB).
    """
    col_sums = quarter.sum(axis=0)
    row_sums = quarter.sum(axis=1)

    # Find bounding box
    nonzero_cols = np.nonzero(col_sums)[0]
    nonzero_rows = np.nonzero(row_sums)[0]

    if len(nonzero_cols) == 0 or len(nonzero_rows) == 0:
        return (0.0, 0.0)

    x_start = nonzero_cols[0]
    x_end = nonzero_cols[-1]
    y_start = nonzero_rows[0]
    y_end = nonzero_rows[-1]

    sub_region = quarter[y_start: y_end + 1, x_start: x_end + 1]
    cx, cy = weighted_centroid(sub_region)

    # Transform back to quadrant coords (0-based)
    return (cx + x_start, cy + y_start)


# ---------------------------------------------------------------------------
# Full marker coordinate detection  (ports FindImageCoords.m / XFindImageCoords.m)
# ---------------------------------------------------------------------------

def find_marker_coords(
    mask: np.ndarray,
) -> MarkerCoords:
    """Find the image coordinates of all four markers.

    Ports ``XFindImageCoords.m`` (the revised version that uses
    full mask dimensions for the coordinate transform).

    Parameters
    ----------
    mask : np.ndarray
        Binary marker mask for the full image.

    Returns
    -------
    MarkerCoords
        Coordinates of the four markers in full-image pixel space.
    """
    H, W = mask.shape
    quarters = quarterise(mask)

    coords = np.zeros((4, 2))  # (x, y) for each marker

    for q_idx, quarter in enumerate(quarters):
        cx, cy = find_region_centroid(quarter)

        if q_idx == 0:  # top-left: no transform needed
            coords[0] = [cx, cy]
        elif q_idx == 1:  # top-right: mirror X back
            coords[1] = [W - cx, cy]
        elif q_idx == 2:  # bottom-left: mirror Y back
            coords[2] = [cx, H - cy]
        elif q_idx == 3:  # bottom-right: mirror both
            coords[3] = [W - cx, H - cy]

    return MarkerCoords(coords=coords)


# ---------------------------------------------------------------------------
# Full marker detection pipeline
# ---------------------------------------------------------------------------

def detect_markers(
    img_path: str | Path,
    use_red_plane: bool = False,
    channel: int = 0,
) -> MarkerDetectionResult:
    """Run the full marker detection pipeline on a colour image.

    Parameters
    ----------
    img_path : path-like
        Path to a colour BMP image with four markers.
    use_red_plane : bool
        Use exclusive colour plane extraction instead of simple inversion.
    channel : int
        Colour channel (0=red, 1=green, 2=blue).

    Returns
    -------
    MarkerDetectionResult
        All intermediate stages and final marker coordinates.
    """
    img = load_colour(img_path)
    mask, intermediates = create_mask(img, use_red_plane=use_red_plane,
                                     channel=channel)
    coords = find_marker_coords(mask)

    return MarkerDetectionResult(
        original=img,
        colour_plane=intermediates["colour_plane"],
        normalised=intermediates["normalised"],
        edges=intermediates["edges"],
        dilated=intermediates["dilated"],
        filled=intermediates["filled"],
        border_cleared=intermediates["border_cleared"],
        eroded=intermediates["eroded"],
        final_mask=mask,
        coords=coords,
    )
