"""
visualize.py
============
Utilities for visualising packing episodes and training curves.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects


# ---------------------------------------------------------------------------
# Single-episode packing plot
# ---------------------------------------------------------------------------


def plot_packing(
    placed_items: List[Dict[str, Any]],
    remaining_items: List[tuple],
    zone_w: int,
    zone_h: int,
    title: str = "Packing Result",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Draw the packed zone with all placed rectangles annotated.

    Parameters
    ----------
    placed_items   : list of dicts with keys x, y, w, h, color
    remaining_items: list of (w, h) tuples for un-placed items
    zone_w, zone_h : zone dimensions
    title          : figure title
    save_path      : if given, the figure is saved here
    ax             : existing Axes to draw into (optional)
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 9))
    else:
        fig = ax.figure

    # Zone
    ax.add_patch(
        patches.FancyBboxPatch(
            (0, 0),
            zone_w,
            zone_h,
            boxstyle="square,pad=0",
            linewidth=2.5,
            edgecolor="#222",
            facecolor="#fafafa",
        )
    )

    # Grid lines (light)
    for x in range(zone_w + 1):
        ax.axvline(x, color="#ddd", linewidth=0.3, zorder=0)
    for y in range(zone_h + 1):
        ax.axhline(y, color="#ddd", linewidth=0.3, zorder=0)

    # Placed items
    for item in placed_items:
        color = item.get("color", "#4c72b0")
        rect = patches.Rectangle(
            (item["x"], item["y"]),
            item["w"],
            item["h"],
            linewidth=1.2,
            edgecolor="#222",
            facecolor=color,
            alpha=0.80,
            zorder=2,
        )
        ax.add_patch(rect)
        cx = item["x"] + item["w"] / 2
        cy = item["y"] + item["h"] / 2
        # white halo for readability
        ax.text(
            cx,
            cy,
            f"{item['w']}×{item['h']}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
            path_effects=[
                matplotlib.patheffects.withStroke(linewidth=2, foreground="black")
            ],
            zorder=3,
        )

    # Stats
    placed_area = sum(i["w"] * i["h"] for i in placed_items)
    zone_area = zone_w * zone_h
    utilisation = placed_area / zone_area * 100

    ax.set_xlim(-0.5, zone_w + 0.5)
    ax.set_ylim(-0.5, zone_h + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cells)", fontsize=11)
    ax.set_ylabel("y (cells)", fontsize=11)

    n_total = len(placed_items) + len(remaining_items)
    ax.set_title(
        f"{title}\n"
        f"Placed {len(placed_items)}/{n_total} items  ·  "
        f"Zone utilisation {utilisation:.1f}%",
        fontsize=13,
    )

    # Remaining items sidebar
    if remaining_items:
        rem_str = "Un-packed: " + "  ".join(f"{w}×{h}" for w, h in remaining_items[:12])
        if len(remaining_items) > 12:
            rem_str += " …"
        ax.text(
            0.01,
            0.01,
            rem_str,
            transform=ax.transAxes,
            fontsize=8,
            color="#888",
            verticalalignment="bottom",
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    return fig, ax


# ---------------------------------------------------------------------------
# Multi-episode grid
# ---------------------------------------------------------------------------


def plot_episode_grid(
    episodes: List[Dict[str, Any]],
    cols: int = 3,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot several episodes in a grid layout.

    Each entry in *episodes* is a dict:
      {placed_items, remaining_items, zone_w, zone_h, title (optional)}
    """
    rows = (len(episodes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 6))
    axes_flat = np.array(axes).flatten()

    for i, ep in enumerate(episodes):
        plot_packing(
            placed_items=ep["placed_items"],
            remaining_items=ep.get("remaining_items", []),
            zone_w=ep["zone_w"],
            zone_h=ep["zone_h"],
            title=ep.get("title", f"Episode {i + 1}"),
            ax=axes_flat[i],
        )

    # Hide unused axes
    for j in range(len(episodes), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Metrics dashboard
# ---------------------------------------------------------------------------


def plot_metrics(
    metrics: Dict[str, List[float]],
    smoothing: int = 20,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training metrics (rewards, utilisation, placement rate, etc.).

    Parameters
    ----------
    metrics   : dict mapping metric name → list of per-episode values
    smoothing : moving-average window size
    save_path : optional path to save the figure
    """
    n = len(metrics)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    def _smooth(x, w):
        if len(x) < w:
            return x
        kernel = np.ones(w) / w
        return np.convolve(x, kernel, mode="valid")

    colors = plt.cm.tab10.colors
    for i, (name, values) in enumerate(metrics.items()):
        ax = axes_flat[i]
        raw = np.array(values, dtype=float)
        x_raw = np.arange(len(raw)) + 1
        ax.plot(x_raw, raw, alpha=0.25, color=colors[i % 10], linewidth=0.8)
        if len(raw) >= smoothing:
            smooth = _smooth(raw, smoothing)
            x_sm = np.arange(len(smooth)) + smoothing // 2 + 1
            ax.plot(
                x_sm, smooth, color=colors[i % 10], linewidth=2, label=f"MA-{smoothing}"
            )
            ax.legend(fontsize=9)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)

    for j in range(len(metrics), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Training Metrics", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    return fig
