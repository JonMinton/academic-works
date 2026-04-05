"""
Interactive Demonstrations with Plotly
=======================================

This module provides interactive visualisations of each stage of the
BEng machine vision calibration system.  Run as a script to generate
all demonstrations, or import individual functions.

Usage::

    python -m beng.python.demo

Or from the beng/python directory::

    python demo.py

Requires: numpy, scipy, Pillow, plotly, matplotlib
"""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Handle running as script or as module
if __name__ == "__main__":
    # Add parent directory to path so imports work
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from beng.python.dic import (
    load_greyscale,
    subtract_dc,
    image_correlate,
    find_rigid_displacement,
    compute_displacement_map,
    extract_subimage_pair,
)
from beng.python.markers import (
    load_colour,
    exclusive_colour_plane,
    normalise_contrast,
    sobel_edge,
    dilate_edges,
    fill_holes,
    clear_border,
    erode_diamond,
    select_by_seeds,
    create_mask,
    find_marker_coords,
    detect_markers,
    weighted_centroid,
)
from beng.python.calibration import (
    relative_coords,
    plane_angles,
    measure_single_image,
    run_calibration_subset,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TEST_DIR = Path(__file__).resolve().parent / "test_images"


# ---------------------------------------------------------------------------
# Demo 1: DIC Cross-Correlation
# ---------------------------------------------------------------------------

def demo_dic_correlation(
    img1_path: str | Path | None = None,
    img2_path: str | Path | None = None,
    show: bool = True,
) -> go.Figure:
    """Demonstrate 2D FFT cross-correlation on a greyscale image pair.

    Shows:
    - The two input images (DC-subtracted)
    - The cross-correlation surface (CC)
    - The normalised cross-correlation (NCC) with peak marked
    - The rigid-body displacement vector

    Thesis ref: Chapter 4, Section 4.2-4.3.
    """
    if img1_path is None:
        img1_path = TEST_DIR / "Im1Grey.bmp"
    if img2_path is None:
        img2_path = TEST_DIR / "Im2Grey.bmp"

    m1_raw = load_greyscale(img1_path)
    m2_raw = load_greyscale(img2_path)
    m1 = subtract_dc(m1_raw)
    m2 = subtract_dc(m2_raw)

    corr = image_correlate(m1, m2)
    rigid = find_rigid_displacement(corr.ncc)

    N, M = corr.ncc.shape

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Image 1 (DC-subtracted)",
            "Image 2 (DC-subtracted)",
            "Cross-Correlation (CC)",
            f"Normalised CC (NCC) -- peak at dx={rigid.dx}, dy={rigid.dy}",
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # Image 1
    fig.add_trace(
        go.Heatmap(z=corr.f1[::-1], colorscale="gray", showscale=False,
                    hovertemplate="Row: %{y}<br>Col: %{x}<br>Value: %{z:.1f}<extra>Image 1</extra>"),
        row=1, col=1,
    )

    # Image 2
    fig.add_trace(
        go.Heatmap(z=corr.f2[::-1], colorscale="gray", showscale=False,
                    hovertemplate="Row: %{y}<br>Col: %{x}<br>Value: %{z:.1f}<extra>Image 2</extra>"),
        row=1, col=2,
    )

    # CC surface
    fig.add_trace(
        go.Heatmap(z=corr.cc[::-1], colorscale="jet", showscale=True,
                    colorbar=dict(x=0.45, y=0.18, len=0.4),
                    hovertemplate="Row: %{y}<br>Col: %{x}<br>CC: %{z:.2f}<extra>CC</extra>"),
        row=2, col=1,
    )

    # NCC surface with peak marker
    peak_row, peak_col = np.unravel_index(np.argmax(corr.ncc), corr.ncc.shape)
    fig.add_trace(
        go.Heatmap(z=corr.ncc[::-1], colorscale="jet", showscale=True,
                    colorbar=dict(x=1.02, y=0.18, len=0.4),
                    hovertemplate="Row: %{y}<br>Col: %{x}<br>NCC: %{z:.4f}<extra>NCC</extra>"),
        row=2, col=2,
    )
    # Mark the peak
    fig.add_trace(
        go.Scatter(x=[peak_col], y=[N - 1 - peak_row],
                   mode="markers",
                   marker=dict(size=12, color="red", symbol="x"),
                   name=f"Peak (dx={rigid.dx}, dy={rigid.dy})",
                   showlegend=True),
        row=2, col=2,
    )

    fig.update_layout(
        title="DIC: 2D FFT Cross-Correlation<br>"
              f"<sup>Rigid displacement: dx={rigid.dx}px, dy={rigid.dy}px, "
              f"magnitude={rigid.magnitude:.1f}px</sup>",
        height=800,
        width=1000,
    )

    # Set axes to equal aspect ratio
    for i in range(1, 5):
        row = (i - 1) // 2 + 1
        col = (i - 1) % 2 + 1
        fig.update_xaxes(scaleanchor=f"y{i}" if i > 1 else "y",
                         constrain="domain", row=row, col=col)

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# Demo 2: DIC Displacement Map
# ---------------------------------------------------------------------------

def demo_dic_displacement_map(
    img1_path: str | Path | None = None,
    img2_path: str | Path | None = None,
    n_blocks: int = 3,
    show: bool = True,
) -> go.Figure:
    """Demonstrate the full DIC displacement map.

    Shows a 3-panel view: X displacement, Y displacement, and magnitude.

    Thesis ref: Chapter 4, Section 4.4-4.5.
    """
    if img1_path is None:
        img1_path = TEST_DIR / "Im1Grey.bmp"
    if img2_path is None:
        img2_path = TEST_DIR / "Im2Grey.bmp"

    disp = compute_displacement_map(img1_path, img2_path, n_blocks=n_blocks)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            f"X Displacement (rigid dx={disp.rigid.dx})",
            f"Y Displacement (rigid dy={disp.rigid.dy})",
            "Displacement Magnitude",
        ],
        horizontal_spacing=0.08,
    )

    for idx, (data, name) in enumerate([
        (disp.disp_x, "X Disp"),
        (disp.disp_y, "Y Disp"),
        (disp.disp_mag, "Magnitude"),
    ]):
        fig.add_trace(
            go.Heatmap(
                z=data[::-1],
                colorscale="jet",
                showscale=True,
                colorbar=dict(x=0.3 + idx * 0.35, len=0.8),
                hovertemplate=f"Block row: %{{y}}<br>Block col: %{{x}}<br>{name}: %{{z:.2f}}px<extra></extra>",
            ),
            row=1, col=idx + 1,
        )

    grid_size = 2 * n_blocks - 1
    fig.update_layout(
        title=f"DIC Sub-Image Displacement Map ({grid_size}x{grid_size} blocks)<br>"
              f"<sup>Rigid body: dx={disp.rigid.dx}px, dy={disp.rigid.dy}px</sup>",
        height=450,
        width=1100,
    )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# Demo 3: NCC 3D Surface
# ---------------------------------------------------------------------------

def demo_ncc_surface(
    img1_path: str | Path | None = None,
    img2_path: str | Path | None = None,
    show: bool = True,
) -> go.Figure:
    """Show the NCC as an interactive 3D surface plot.

    The peak of this surface corresponds to the rigid-body displacement.

    Thesis ref: Chapter 4, Section 4.3.
    """
    if img1_path is None:
        img1_path = TEST_DIR / "Im1Grey.bmp"
    if img2_path is None:
        img2_path = TEST_DIR / "Im2Grey.bmp"

    m1 = subtract_dc(load_greyscale(img1_path))
    m2 = subtract_dc(load_greyscale(img2_path))
    corr = image_correlate(m1, m2)
    rigid = find_rigid_displacement(corr.ncc)

    N, M = corr.ncc.shape
    # Downsample for performance if large
    step = max(1, min(N, M) // 150)
    ncc_ds = corr.ncc[::step, ::step]

    ax_x = np.linspace(-M / 2, M / 2 - 1, ncc_ds.shape[1])
    ax_y = np.linspace(-N / 2, N / 2 - 1, ncc_ds.shape[0])

    fig = go.Figure(data=[
        go.Surface(
            z=ncc_ds,
            x=ax_x,
            y=ax_y,
            colorscale="jet",
            hovertemplate="dx: %{x:.0f}px<br>dy: %{y:.0f}px<br>NCC: %{z:.4f}<extra></extra>",
        ),
    ])

    fig.update_layout(
        title=f"NCC 3D Surface -- Peak at dx={rigid.dx}, dy={rigid.dy}<br>"
              f"<sup>Rotate to explore the correlation landscape</sup>",
        scene=dict(
            xaxis_title="Horizontal offset (px)",
            yaxis_title="Vertical offset (px)",
            zaxis_title="NCC",
        ),
        height=700,
        width=900,
    )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# Demo 4: Marker Detection Pipeline
# ---------------------------------------------------------------------------

def demo_marker_detection(
    img_path: str | Path | None = None,
    use_red_plane: bool = False,
    show: bool = True,
) -> go.Figure:
    """Step-by-step visualisation of the marker detection pipeline.

    Shows all intermediate stages: colour plane, normalisation, edges,
    dilation, hole-filling, border clearing, erosion, and final mask
    with detected centroids overlaid.

    Thesis ref: Chapter 3, Sections 3.2-3.5.
    """
    if img_path is None:
        img_path = TEST_DIR / "Markers.BMP"

    result = detect_markers(img_path, use_red_plane=use_red_plane)

    stages = [
        ("Original (Red channel)", result.original[:, :, 0]),
        ("Exclusive Colour Plane" if use_red_plane else "Inverted Red Channel",
         result.colour_plane),
        ("Contrast Normalised", result.normalised),
        ("Sobel Edges", result.edges.astype(float)),
        ("Dilated", result.dilated.astype(float)),
        ("Holes Filled", result.filled.astype(float)),
        ("Border Cleared", result.border_cleared.astype(float)),
        ("Eroded", result.eroded.astype(float)),
        ("Final Mask + Centroids", result.final_mask.astype(float)),
    ]

    n = len(stages)
    cols = 3
    rows = (n + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[s[0] for s in stages],
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
    )

    for idx, (title, data) in enumerate(stages):
        r = idx // cols + 1
        c = idx % cols + 1

        cscale = "gray" if "Edge" not in title and "Mask" not in title else "greys"
        if "Mask" in title or "Dilated" in title or "Filled" in title or "Cleared" in title or "Eroded" in title:
            cscale = "gray"

        fig.add_trace(
            go.Heatmap(
                z=data[::-1],
                colorscale=cscale,
                showscale=False,
                hovertemplate=f"Row: %{{y}}<br>Col: %{{x}}<br>Value: %{{z:.1f}}<extra>{title}</extra>",
            ),
            row=r, col=c,
        )

    # Overlay marker centroids on the final mask panel
    coords = result.coords.coords
    final_r = (n - 1) // cols + 1
    final_c = (n - 1) % cols + 1
    H = result.final_mask.shape[0]

    labels = ["TL", "TR", "BL", "BR"]
    colours = ["red", "blue", "green", "orange"]
    for i, (label, colour) in enumerate(zip(labels, colours)):
        x, y = coords[i]
        fig.add_trace(
            go.Scatter(
                x=[x], y=[H - 1 - y],
                mode="markers+text",
                marker=dict(size=10, color=colour, symbol="cross"),
                text=[label],
                textposition="top center",
                textfont=dict(color=colour, size=10),
                name=f"{label} ({x:.1f}, {y:.1f})",
                showlegend=True,
            ),
            row=final_r, col=final_c,
        )

    fig.update_layout(
        title="Marker Detection Pipeline<br>"
              "<sup>Each panel shows one stage of the morphological processing</sup>",
        height=300 * rows,
        width=1100,
        legend=dict(x=0.75, y=0.02),
    )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# Demo 5: Marker Overlay on Original Image
# ---------------------------------------------------------------------------

def demo_marker_overlay(
    img_path: str | Path | None = None,
    use_red_plane: bool = False,
    show: bool = True,
) -> go.Figure:
    """Show detected markers overlaid on the original colour image.

    Thesis ref: Chapter 3, Section 3.6.
    """
    if img_path is None:
        img_path = TEST_DIR / "Markers.BMP"

    result = detect_markers(img_path, use_red_plane=use_red_plane)
    img = result.original
    H, W = img.shape[:2]
    coords = result.coords.coords

    fig = go.Figure()

    # Show the RGB image using px.imshow-style approach
    fig.add_trace(go.Image(z=img))

    # Overlay marker positions
    labels = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
    colours = ["red", "cyan", "lime", "yellow"]
    for i, (label, colour) in enumerate(zip(labels, colours)):
        x, y = coords[i]
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=14, color=colour, symbol="cross",
                        line=dict(width=2, color="black")),
            text=[label],
            textposition="bottom center",
            textfont=dict(color=colour, size=11),
            name=f"{label} ({x:.1f}, {y:.1f})",
        ))

    # Draw lines between markers to show the quadrilateral
    order = [0, 1, 3, 2, 0]
    xs = [coords[i][0] for i in order]
    ys = [coords[i][1] for i in order]
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines",
        line=dict(color="yellow", width=2, dash="dash"),
        name="Marker quadrilateral",
    ))

    fig.update_layout(
        title="Detected Markers Overlaid on Original Image",
        height=600,
        width=800,
        xaxis=dict(range=[0, W], constrain="domain"),
        yaxis=dict(range=[H, 0], scaleanchor="x"),
    )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# Demo 6: Calibration Polynomial Fit
# ---------------------------------------------------------------------------

def demo_calibration(
    image_paths: list[str | Path] | None = None,
    angles: np.ndarray | None = None,
    show: bool = True,
) -> go.Figure:
    """Demonstrate the calibration polynomial fitting.

    If no images/angles are provided, uses the test calibration images
    (5 images from set 0, at assumed evenly-spaced angles).

    Thesis ref: Chapter 5, Sections 5.2-5.3.
    """
    if image_paths is None:
        image_paths = [
            TEST_DIR / "0_01.BMP",
            TEST_DIR / "0_05.BMP",
            TEST_DIR / "0_09.BMP",
            TEST_DIR / "0_13.BMP",
            TEST_DIR / "0_17.BMP",
        ]
    if angles is None:
        # Assume evenly spaced from -40 to +40 degrees (17 positions)
        # We have images 01, 05, 09, 13, 17 => indices 0, 4, 8, 12, 16
        all_angles = np.linspace(-40, 40, 17)
        indices = [0, 4, 8, 12, 16]
        angles = all_angles[indices]

    # Filter to only existing images
    valid_paths = []
    valid_angles = []
    for p, a in zip(image_paths, angles):
        if Path(p).exists():
            valid_paths.append(p)
            valid_angles.append(a)

    if len(valid_paths) < 2:
        print("Not enough calibration images found. Generating synthetic demo.")
        return _demo_calibration_synthetic(show=show)

    valid_angles = np.array(valid_angles)

    # Use degree 2 since we only have 5 points
    degree = min(3, len(valid_paths) - 1)
    calib = run_calibration_subset(
        valid_paths, valid_angles, poly_degree=degree
    )

    fig = go.Figure()

    # Scatter of measured points
    valid_mask = ~np.isnan(calib.ratios)
    fig.add_trace(go.Scatter(
        x=calib.angles[valid_mask],
        y=calib.ratios[valid_mask],
        mode="markers",
        marker=dict(size=10, color="blue"),
        name="Measured ratios",
        hovertemplate="Angle: %{x:.1f} deg<br>Ratio: %{y:.4f}<extra></extra>",
    ))

    # Fitted polynomial curve
    if not np.any(np.isnan(calib.poly_coeffs)):
        x_fit = np.linspace(calib.angles.min() - 5, calib.angles.max() + 5, 200)
        y_fit = np.polyval(calib.poly_coeffs, x_fit)

        fig.add_trace(go.Scatter(
            x=x_fit, y=y_fit,
            mode="lines",
            line=dict(color="red", width=2),
            name=f"Polynomial fit (degree {calib.poly_degree})",
        ))

        # Show the equation
        terms = []
        for i, c in enumerate(calib.poly_coeffs):
            power = calib.poly_degree - i
            if power == 0:
                terms.append(f"{c:.6f}")
            elif power == 1:
                terms.append(f"{c:.6f}x")
            else:
                terms.append(f"{c:.6f}x^{power}")
        equation = " + ".join(terms)

        fig.add_annotation(
            x=0.5, y=0.95,
            xref="paper", yref="paper",
            text=f"y = {equation}",
            showarrow=False,
            font=dict(size=10),
        )

    fig.update_layout(
        title=f"Calibration: Marker Ratio vs Camera Angle<br>"
              f"<sup>Polynomial degree {calib.poly_degree}, "
              f"{valid_mask.sum()} valid measurements</sup>",
        xaxis_title="Camera Angle (degrees)",
        yaxis_title="Vertical Ratio (right/left)",
        height=500,
        width=800,
    )

    if show:
        fig.show()
    return fig


def _demo_calibration_synthetic(show: bool = True) -> go.Figure:
    """Fallback: demonstrate calibration with synthetic data."""
    np.random.seed(42)
    angles = np.linspace(-40, 40, 17)
    # Simulate ratios: at 0 degrees ratio ~1.0, changes with angle
    true_coeffs = [1e-6, -2e-4, 0.005, 1.0]
    ratios = np.polyval(true_coeffs, angles) + np.random.normal(0, 0.01, 17)

    coeffs = np.polyfit(angles, ratios, 3)
    x_fit = np.linspace(-45, 45, 200)
    y_fit = np.polyval(coeffs, x_fit)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=angles, y=ratios,
        mode="markers",
        marker=dict(size=10, color="blue"),
        name="Synthetic measurements",
    ))
    fig.add_trace(go.Scatter(
        x=x_fit, y=y_fit,
        mode="lines",
        line=dict(color="red", width=2),
        name="3rd-degree polynomial fit",
    ))

    fig.update_layout(
        title="Calibration Demo (Synthetic Data)<br>"
              "<sup>Simulated marker ratio vs camera angle</sup>",
        xaxis_title="Camera Angle (degrees)",
        yaxis_title="Vertical Ratio (right/left)",
        height=500, width=800,
    )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# Demo 7: Sub-Image Grid Visualisation
# ---------------------------------------------------------------------------

def demo_subimage_grid(
    img1_path: str | Path | None = None,
    img2_path: str | Path | None = None,
    n_blocks: int = 3,
    show: bool = True,
) -> go.Figure:
    """Visualise the sub-image grid overlaid on the original image.

    Shows how the image is divided into overlapping blocks for
    local displacement measurement.

    Thesis ref: Chapter 4, Section 4.4.
    """
    if img1_path is None:
        img1_path = TEST_DIR / "Im1Grey.bmp"
    if img2_path is None:
        img2_path = TEST_DIR / "Im2Grey.bmp"

    m1_raw = load_greyscale(img1_path)
    m2_raw = load_greyscale(img2_path)
    m1 = subtract_dc(m1_raw)
    m2 = subtract_dc(m2_raw)
    corr = image_correlate(m1, m2)
    rigid = find_rigid_displacement(corr.ncc)

    H, W = m1.shape
    sub_x_total = W - abs(rigid.dx) - 1
    sub_y_total = H - abs(rigid.dy) - 1
    sub_w = sub_x_total // n_blocks
    sub_h = sub_y_total // n_blocks
    grid_size = 2 * n_blocks - 1

    fig = go.Figure()

    # Show image 1
    fig.add_trace(go.Heatmap(
        z=m1_raw[::-1],
        colorscale="gray",
        showscale=False,
        hovertemplate="Row: %{y}<br>Col: %{x}<br>Value: %{z:.0f}<extra></extra>",
    ))

    # Draw sub-image grid
    colours = px.colors.qualitative.Set3
    for gi in range(grid_size):
        for gj in range(grid_size):
            x_start = (sub_w * gi) // 2 + (abs(rigid.dx) if rigid.dx >= 0 else 0)
            y_start = (sub_h * gj) // 2 + (abs(rigid.dy) if rigid.dy >= 0 else 0)

            colour = colours[(gi + gj) % len(colours)]
            fig.add_shape(
                type="rect",
                x0=x_start, y0=H - 1 - y_start,
                x1=x_start + sub_w, y1=H - 1 - (y_start + sub_h),
                line=dict(color=colour, width=1),
                fillcolor=colour,
                opacity=0.15,
            )

    fig.update_layout(
        title=f"Sub-Image Grid ({grid_size}x{grid_size} blocks, {sub_w}x{sub_h}px each)<br>"
              f"<sup>Overlapping blocks with half-block stride</sup>",
        height=600,
        width=800,
        xaxis=dict(constrain="domain"),
        yaxis=dict(scaleanchor="x"),
    )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# Demo 8: Displacement Vector Field
# ---------------------------------------------------------------------------

def demo_displacement_vectors(
    img1_path: str | Path | None = None,
    img2_path: str | Path | None = None,
    n_blocks: int = 3,
    show: bool = True,
) -> go.Figure:
    """Show the displacement field as a quiver/arrow plot.

    Thesis ref: Chapter 4, Section 4.5.
    """
    if img1_path is None:
        img1_path = TEST_DIR / "Im1Grey.bmp"
    if img2_path is None:
        img2_path = TEST_DIR / "Im2Grey.bmp"

    disp = compute_displacement_map(img1_path, img2_path, n_blocks=n_blocks)
    grid_size = 2 * n_blocks - 1

    fig = go.Figure()

    # Background: displacement magnitude as heatmap
    fig.add_trace(go.Heatmap(
        z=disp.disp_mag[::-1],
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="Magnitude (px)"),
        hovertemplate="Block (%{x}, %{y})<br>Magnitude: %{z:.2f}px<extra></extra>",
    ))

    # Quiver arrows
    scale = 0.3  # Arrow length scale factor
    for j in range(grid_size):
        for i in range(grid_size):
            dx = disp.disp_x[j, i]
            dy = disp.disp_y[j, i]
            if np.isnan(dx) or np.isnan(dy):
                continue
            fig.add_annotation(
                x=i, y=grid_size - 1 - j,
                ax=i + dx * scale,
                ay=grid_size - 1 - j - dy * scale,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor="white",
            )

    fig.update_layout(
        title=f"Displacement Vector Field ({grid_size}x{grid_size})<br>"
              f"<sup>Arrows show local displacement after rigid-body removal "
              f"(rigid: dx={disp.rigid.dx}, dy={disp.rigid.dy})</sup>",
        height=600,
        width=700,
        xaxis_title="Block column",
        yaxis_title="Block row",
    )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# Run all demos
# ---------------------------------------------------------------------------

def run_all_demos(show: bool = True) -> dict[str, go.Figure]:
    """Run all demonstrations and return the figures.

    Parameters
    ----------
    show : bool
        Whether to open each figure in the browser.

    Returns
    -------
    dict mapping demo name to Plotly figure.
    """
    print("=" * 60)
    print("BEng Machine Vision Calibration -- Python Port Demos")
    print("=" * 60)

    figures = {}

    print("\n[1/8] DIC Cross-Correlation...")
    try:
        figures["dic_correlation"] = demo_dic_correlation(show=show)
        print("      Done.")
    except Exception as e:
        print(f"      Error: {e}")

    print("\n[2/8] DIC Displacement Map...")
    try:
        figures["dic_displacement"] = demo_dic_displacement_map(show=show)
        print("      Done.")
    except Exception as e:
        print(f"      Error: {e}")

    print("\n[3/8] NCC 3D Surface...")
    try:
        figures["ncc_surface"] = demo_ncc_surface(show=show)
        print("      Done.")
    except Exception as e:
        print(f"      Error: {e}")

    print("\n[4/8] Marker Detection Pipeline...")
    try:
        figures["marker_pipeline"] = demo_marker_detection(show=show)
        print("      Done.")
    except Exception as e:
        print(f"      Error: {e}")

    print("\n[5/8] Marker Overlay...")
    try:
        figures["marker_overlay"] = demo_marker_overlay(show=show)
        print("      Done.")
    except Exception as e:
        print(f"      Error: {e}")

    print("\n[6/8] Calibration Polynomial...")
    try:
        figures["calibration"] = demo_calibration(show=show)
        print("      Done.")
    except Exception as e:
        print(f"      Error: {e}")

    print("\n[7/8] Sub-Image Grid...")
    try:
        figures["subimage_grid"] = demo_subimage_grid(show=show)
        print("      Done.")
    except Exception as e:
        print(f"      Error: {e}")

    print("\n[8/8] Displacement Vectors...")
    try:
        figures["displacement_vectors"] = demo_displacement_vectors(show=show)
        print("      Done.")
    except Exception as e:
        print(f"      Error: {e}")

    print("\n" + "=" * 60)
    print(f"Complete. {len(figures)} of 8 demos generated successfully.")
    print("=" * 60)

    return figures


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all_demos(show=True)
