"""
Digital Image Correlation (DIC) via 2D FFT Cross-Correlation
=============================================================

Port of MATLAB files:
  DIC/ImageWork.m      -- Core FFT cross-correlation
  DIC/RigidBodyFind.m  -- Find rigid-body displacement from NCC peak
  DIC/SubImage.m       -- Sub-image extraction with rigid-body offset
  DIC/SubImageWork.m   -- Full sub-image correlation workflow
  DIC/ImCor2_05.m      -- Main DIC pipeline

Thesis reference: Chapter 4 -- Digital Image Correlation

Algorithm summary
-----------------
1. Load a pair of greyscale images.
2. Subtract the DC component (mean) from each to enhance correlation contrast.
3. Compute the 2D FFT cross-correlation:
       CC  = |IFFT2( FFT2(f1) * conj(FFT2(f2)) )|   (shifted)
       NCC = CC / sqrt( sum(f1^2) * sum(f2^2) )
4. Find the peak of the NCC to determine rigid-body displacement.
5. Divide the images into overlapping sub-image blocks.
6. For each sub-image pair (offset by the rigid-body displacement),
   compute the local NCC and find the residual displacement.
7. Assemble a displacement map from the sub-image results.
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from pathlib import Path
from PIL import Image
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class CorrelationResult(NamedTuple):
    """Result of a single cross-correlation between two images."""
    cc: np.ndarray       # Cross-correlation (absolute, shifted)
    ncc: np.ndarray      # Normalised cross-correlation
    f1: np.ndarray       # Trimmed image 1
    f2: np.ndarray       # Trimmed image 2


class RigidDisplacement(NamedTuple):
    """Rigid-body displacement found from NCC peak."""
    dx: int   # Horizontal displacement (pixels)
    dy: int   # Vertical displacement (pixels)
    magnitude: float


class SubImageDisplacement(NamedTuple):
    """Displacement found within a single sub-image pair."""
    dx: int
    dy: int
    magnitude: float


class DispMap(NamedTuple):
    """Full displacement map assembled from sub-image blocks."""
    disp_x: np.ndarray     # X-component displacement grid
    disp_y: np.ndarray     # Y-component displacement grid
    disp_mag: np.ndarray   # Magnitude displacement grid
    rigid: RigidDisplacement


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_greyscale(path: str | Path) -> np.ndarray:
    """Load an image and convert to float64 greyscale.

    If the image has 3 channels, the first (red) channel is used --
    consistent with the original MATLAB code which used ``imread``
    returning an (H, W) matrix for greyscale BMPs, or ``InputImage(:,:,1)``
    for colour images.
    """
    img = np.array(Image.open(path))
    if img.ndim == 3:
        img = img[:, :, 0]  # red channel
    return img.astype(np.float64)


def subtract_dc(img: np.ndarray) -> np.ndarray:
    """Subtract the DC (mean) component to enhance correlation contrast.

    Thesis ref: Section 4.2 -- the mean is subtracted so that the
    cross-correlation is dominated by spatial texture rather than
    overall brightness.
    """
    return img - img.mean()


# ---------------------------------------------------------------------------
# Core cross-correlation  (ports DIC/ImageWork.m)
# ---------------------------------------------------------------------------

def image_correlate(m1: np.ndarray, m2: np.ndarray) -> CorrelationResult:
    """Compute cross-correlation and normalised cross-correlation.

    Ports ``ImageWork.m``.  Both images should already have had their
    DC component subtracted.

    The MATLAB code trims both images to (N-1, M-1) before computing
    the FFT.  We replicate this for fidelity, though in practice the
    trim has negligible effect on 320x240 images.

    Parameters
    ----------
    m1, m2 : np.ndarray
        DC-subtracted greyscale images of the same shape.

    Returns
    -------
    CorrelationResult
        Named tuple of (cc, ncc, f1, f2).
    """
    N, M = m1.shape
    f1 = m1[: N - 1, : M - 1]
    f2 = m2[: N - 1, : M - 1]

    # Cross-correlation via Fourier domain multiplication
    F1 = fft2(f1)
    F2 = fft2(f2)
    cc = fftshift(np.abs(ifft2(F1 * np.conj(F2))))

    # Normalised cross-correlation
    denominator = np.sqrt(np.sum(f1 ** 2) * np.sum(f2 ** 2))
    if denominator == 0:
        ncc = np.zeros_like(cc)
    else:
        ncc = cc / denominator

    return CorrelationResult(cc=cc, ncc=ncc, f1=f1, f2=f2)


# ---------------------------------------------------------------------------
# Rigid-body displacement  (ports DIC/RigidBodyFind.m)
# ---------------------------------------------------------------------------

def find_rigid_displacement(ncc: np.ndarray) -> RigidDisplacement:
    """Find the rigid-body displacement from the NCC peak position.

    Ports ``RigidBodyFind.m``.

    The NCC has been ``fftshift``-ed so that zero displacement
    corresponds to the centre of the array.  The peak position
    relative to centre gives the rigid-body shift.

    Parameters
    ----------
    ncc : np.ndarray
        Normalised cross-correlation matrix (fftshift-ed).

    Returns
    -------
    RigidDisplacement
    """
    N, M = ncc.shape

    # In MATLAB: [MaxVal, RigidY] = max(max(NCC))
    #            [MaxVal2, RigidX] = max(max(NCC'))
    # max(NCC) operates column-wise, giving a row vector; max of that
    # gives the column index.  max(NCC') gives the row index.
    peak_idx = np.unravel_index(np.argmax(ncc), ncc.shape)
    rigid_x = peak_idx[0] - int(np.ceil(N / 2))  # row index -> Y in image
    rigid_y = peak_idx[1] - int(np.ceil(M / 2))  # col index -> X in image

    # MATLAB naming: RigidX was from max(NCC') which is the row (Y) direction.
    # To keep the variable semantics consistent with the MATLAB code:
    #   RigidX = row_peak - ceil(N/2)   (vertical offset)
    #   RigidY = col_peak - ceil(M/2)   (horizontal offset)
    # We expose these as dx (horizontal) and dy (vertical) for clarity.
    dx = rigid_y  # horizontal
    dy = rigid_x  # vertical
    magnitude = np.sqrt(float(dx ** 2 + dy ** 2))

    return RigidDisplacement(dx=dx, dy=dy, magnitude=magnitude)


# ---------------------------------------------------------------------------
# Sub-image extraction  (ports DIC/SubImage.m, DIC/SubImageWork.m)
# ---------------------------------------------------------------------------

def extract_subimage_pair(
    m1: np.ndarray,
    m2: np.ndarray,
    i: int,
    j: int,
    sub_w: int,
    sub_h: int,
    rigid_dx: int,
    rigid_dy: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a corresponding sub-image pair, offset by rigid-body displacement.

    Ports the sub-image extraction logic from ``SubImage.m``.

    Parameters
    ----------
    m1, m2 : np.ndarray
        Full DC-subtracted images.
    i, j : int
        Block indices (0-based; the MATLAB code used 1-based).
    sub_w, sub_h : int
        Width and height of each sub-image block.
    rigid_dx, rigid_dy : int
        Rigid-body displacement to compensate for.

    Returns
    -------
    (si1, si2) : tuple of np.ndarray
    """
    # Horizontal start positions
    if rigid_dx >= 0:
        m1_x_start = rigid_dx + (sub_w * i) // 2
        m2_x_start = (sub_w * i) // 2
    else:
        m1_x_start = (sub_w * i) // 2
        m2_x_start = (sub_w * i) // 2 - rigid_dx

    # Vertical start positions
    if rigid_dy >= 0:
        m1_y_start = rigid_dy + (sub_h * j) // 2
        m2_y_start = (sub_h * j) // 2
    else:
        m1_y_start = (sub_h * j) // 2
        m2_y_start = (sub_h * j) // 2 - rigid_dy

    si1 = m1[m1_y_start: m1_y_start + sub_h + 1,
             m1_x_start: m1_x_start + sub_w + 1]
    si2 = m2[m2_y_start: m2_y_start + sub_h + 1,
             m2_x_start: m2_x_start + sub_w + 1]

    return si1, si2


def subimage_displacement(
    m1: np.ndarray,
    m2: np.ndarray,
    i: int,
    j: int,
    sub_w: int,
    sub_h: int,
    rigid_dx: int,
    rigid_dy: int,
) -> SubImageDisplacement:
    """Compute displacement for a single sub-image pair.

    Ports ``SubImageWork.m`` / ``SubImage.m``.

    Parameters
    ----------
    m1, m2 : np.ndarray
        Full DC-subtracted images.
    i, j : int
        Block indices (0-based).
    sub_w, sub_h : int
        Sub-image block dimensions.
    rigid_dx, rigid_dy : int
        Rigid-body displacement.

    Returns
    -------
    SubImageDisplacement
    """
    si1, si2 = extract_subimage_pair(m1, m2, i, j, sub_w, sub_h,
                                     rigid_dx, rigid_dy)

    # Correlate the sub-images
    corr = image_correlate(si1, si2)

    # Find the sub-image displacement from the NCC peak
    N_sub, M_sub = corr.ncc.shape
    peak = np.unravel_index(np.argmax(corr.ncc), corr.ncc.shape)
    sub_dy = peak[0] - int(np.ceil(N_sub / 2))
    sub_dx = peak[1] - int(np.ceil(M_sub / 2))
    mag = np.sqrt(float(sub_dx ** 2 + sub_dy ** 2))

    return SubImageDisplacement(dx=sub_dx, dy=sub_dy, magnitude=mag)


# ---------------------------------------------------------------------------
# Full DIC pipeline  (ports DIC/ImCor2_05.m)
# ---------------------------------------------------------------------------

def compute_displacement_map(
    img1_path: str | Path,
    img2_path: str | Path,
    n_blocks: int = 3,
) -> DispMap:
    """Run the full DIC pipeline on an image pair.

    Ports the main loop from ``ImCor2_05.m``:
    1. Load greyscale image pair and subtract DC.
    2. Compute whole-image NCC to find rigid-body displacement.
    3. Divide images into overlapping sub-image blocks.
    4. For each block, compute residual (non-rigid) displacement.
    5. Return displacement maps.

    Parameters
    ----------
    img1_path, img2_path : path-like
        Paths to the two greyscale BMP images.
    n_blocks : int
        Number of sub-image blocks per axis (default 3, giving a
        5x5 grid with half-block overlaps).

    Returns
    -------
    DispMap
    """
    m1_raw = load_greyscale(img1_path)
    m2_raw = load_greyscale(img2_path)

    if m1_raw.shape != m2_raw.shape:
        raise ValueError(
            f"Image sizes differ: {m1_raw.shape} vs {m2_raw.shape}"
        )

    m1 = subtract_dc(m1_raw)
    m2 = subtract_dc(m2_raw)

    # Whole-image correlation for rigid-body displacement
    corr = image_correlate(m1, m2)
    rigid = find_rigid_displacement(corr.ncc)

    H, W = m1.shape

    # Sub-image grid dimensions (accounting for rigid-body offset)
    sub_x_total = W - abs(rigid.dx) - 1
    sub_y_total = H - abs(rigid.dy) - 1

    sub_w = sub_x_total // n_blocks
    sub_h = sub_y_total // n_blocks

    # Grid includes overlap points: 2*n_blocks - 1 in each direction
    grid_size = 2 * n_blocks - 1
    disp_x = np.zeros((grid_size, grid_size))
    disp_y = np.zeros((grid_size, grid_size))
    disp_mag = np.zeros((grid_size, grid_size))

    for gi in range(grid_size):
        for gj in range(grid_size):
            try:
                sd = subimage_displacement(
                    m1, m2, gi, gj, sub_w, sub_h,
                    rigid.dx, rigid.dy,
                )
                disp_x[gj, gi] = sd.dx
                disp_y[gj, gi] = sd.dy
                disp_mag[gj, gi] = sd.magnitude
            except (ValueError, IndexError):
                # Sub-image may extend beyond image boundary at edges
                disp_x[gj, gi] = np.nan
                disp_y[gj, gi] = np.nan
                disp_mag[gj, gi] = np.nan

    return DispMap(
        disp_x=disp_x,
        disp_y=disp_y,
        disp_mag=disp_mag,
        rigid=rigid,
    )
