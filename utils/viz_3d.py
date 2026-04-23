"""
utils/viz_3d.py
===============
3-D volume visualization using Maximum Intensity Projections (MIP).

This module provides high-speed projection views of 3-D volumes (Axial, Coronal, 
and Sagittal) with support for model comparisons and anchor prompt overlays.
"""

from __future__ import annotations

import io
from typing import Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_mip_views(
    volumes: Dict[str, np.ndarray],
    colors:  Optional[Dict[str, str]] = None,
    anchor_prompt: Optional[np.ndarray] = None,
    dpi: int = 100,
    suptitle: str = "",
) -> bytes:
    """
    Render Maximum Intensity Projections (axial / coronal / sagittal) for each
    volume.

    Projection Behavior
    -------------------
    This function "squashes" the 3-D volume into three 2-D planes by taking the
    maximum value along each axis. It does NOT choose a single slice; every
    pixel in the output represents the highest value encountered across the
    entire depth of the volume for that projection.

    This is much faster than marching cubes — no ``scikit-image`` required.
    Good sanity-check alternative when meshes are too noisy.

    Parameters
    ----------
    volumes : dict[str, ndarray]
        Label → binary (X, Y, Z) volume.
    colors : dict[str, str] or None
        Matplotlib colour per label.
    anchor_prompt : ndarray or None
        Optional (X, Y, Z) binary volume containing the initial prompt.
        If provided, it will be overlaid as a yellow contour on all panels.
    dpi : int
    suptitle : str

    Returns
    -------
    bytes
        PNG image bytes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(volumes.keys())
    axes_names = ["Axial (↑)", "Coronal (→)", "Sagittal (·)"]
    n_cols  = len(labels)
    n_rows  = 3    # one row per projection axis

    _default_colors = ["#3a86ff", "#ff6b6b", "#6bcb77", "#ffd166", "#adb5bd"]
    if colors is None:
        colors = {lbl: _default_colors[i % len(_default_colors)]
                  for i, lbl in enumerate(labels)}

    # Pre-calculate anchor prompt MIPs once
    p_mips = [None, None, None]
    if anchor_prompt is not None:
        ap = anchor_prompt.astype(np.float32)
        p_mips = [ap.max(axis=0), ap.max(axis=1), ap.max(axis=2)]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.8 * n_cols, 2.8 * n_rows), dpi=dpi)
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)

    if n_cols == 1:
        axes = [[ax] for ax in axes]   # ensure 2-D indexing

    for col, lbl in enumerate(labels):
        vol = volumes[lbl].astype(np.float32)
        mips = [vol.max(axis=0), vol.max(axis=1), vol.max(axis=2)]
        for row, (mip, ax_name, p_mip) in enumerate(zip(mips, axes_names, p_mips)):
            ax = axes[row][col]
            ax.imshow(mip, cmap="Greys_r", vmin=0, vmax=1)
            
            # Label contour
            ax.contour(mip, levels=[0.5], colors=[colors.get(lbl, "red")],
                       linewidths=0.8)
            
            # Anchor prompt overlay (Yellow)
            if p_mip is not None and np.any(p_mip > 0.5):
                ax.contour(p_mip, levels=[0.5], colors=["yellow"],
                           linewidths=1.2)

            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(lbl, fontsize=9)
            if col == 0:
                ax.set_ylabel(ax_name, fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.read()
