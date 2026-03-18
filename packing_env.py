"""
Rectangle Packing Environment
==============================
A custom Gymnasium environment for the 2D rectangle bin-packing problem.

Zone  : One large rectangle (the bin).
Items : A list of randomly sized small rectangles to pack into the zone.
Agent : Learns to place each item by choosing (x, y, rotation).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, List, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _can_place(grid: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
    """Return True if a w×h block can be placed at (x, y) without overlap."""
    zone_h, zone_w = grid.shape
    if x + w > zone_w or y + h > zone_h:
        return False
    return not np.any(grid[y : y + h, x : x + w])


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class RectanglePackingEnv(gym.Env):
    """
    2-D Rectangle Bin-Packing Environment.

    Parameters
    ----------
    zone_w, zone_h : int
        Dimensions of the zone (bin) in grid cells.
    n_items : int
        Number of small rectangles to pack per episode.
    min_item_size, max_item_size : int
        Min/max size (in grid cells) of each item dimension.
    max_invalid_ratio : float
        Episode terminates early when invalid actions exceed
        max_invalid_ratio * zone_w * zone_h.
    seed : int | None
        Optional RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        zone_w: int = 20,
        zone_h: int = 15,
        n_items: int = 10,
        min_item_size: int = 2,
        max_item_size: int = 5,
        max_invalid_ratio: float = 3.0,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.zone_w = zone_w
        self.zone_h = zone_h
        self.n_items = n_items
        self.min_item_size = min_item_size
        self.max_item_size = max_item_size
        self.max_invalid_steps = int(max_invalid_ratio * zone_w * zone_h)

        # ── Action space ────────────────────────────────────────────────────
        # Flat index = rotation * (zone_w * zone_h) + y * zone_w + x
        # rotation ∈ {0, 1}  →  0 = original, 1 = 90° CCW
        # Last action: skip current item
        self.n_positions = zone_w * zone_h
        self.skip_action_idx = 2 * self.n_positions  # Index of skip action
        self.action_space = spaces.Discrete(2 * self.n_positions + 1)  # +1 for skip

        # ── Observation space ────────────────────────────────────────────────
        # [ grid (zone_w*zone_h floats, 0/1)
        #   | current_w / zone_w, current_h / zone_h   (2 floats)
        #   | remaining / n_items                       (1 float) ]
        obs_dim = zone_w * zone_h + 3
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Internal state (initialised in reset())
        self.grid: np.ndarray = np.zeros((zone_h, zone_w), dtype=np.int8)
        self.items: List[Tuple[int, int]] = []
        self.current_idx: int = 0
        self.placed_items: List[Dict[str, Any]] = []
        self.invalid_count: int = 0
        self._item_colors: np.ndarray = np.empty((0, 4))

        # Seeding
        self._np_rng = np.random.default_rng(seed)

        # Render figure handle
        self._fig: Optional[plt.Figure] = None
        self._ax: Optional[plt.Axes] = None

    # ── Internal helpers ────────────────────────────────────────────────────

    def _generate_items(self) -> List[Tuple[int, int]]:
        lo, hi = self.min_item_size, self.max_item_size
        return [
            (
                int(self._np_rng.integers(lo, hi + 1)),
                int(self._np_rng.integers(lo, hi + 1)),
            )
            for _ in range(self.n_items)
        ]

    @property
    def _current_item(self) -> Optional[Tuple[int, int]]:
        if self.current_idx < len(self.items):
            return self.items[self.current_idx]
        return None

    def _build_obs(self) -> np.ndarray:
        item = self._current_item
        if item is not None:
            nw = item[0] / self.zone_w
            nh = item[1] / self.zone_h
        else:
            nw = nh = 0.0
        remaining = max(0, len(self.items) - self.current_idx) / len(self.items)
        return np.concatenate(
            [self.grid.flatten().astype(np.float32), [nw, nh, remaining]]
        ).astype(np.float32)

    # ── Action mask (external helper for MaskablePPO) ───────────────────────

    def action_masks(self) -> np.ndarray:
        """
        Boolean mask over all actions.
        Used by sb3_contrib.MaskablePPO.
        """
        mask = np.zeros(2 * self.n_positions + 1, dtype=bool)  # +1 for skip
        item = self._current_item

        # Skip action is always allowed (unless all items packed)
        if item is not None:
            mask[self.skip_action_idx] = True

        if item is None:
            return mask

        for rot in range(2):
            w, h = (item[0], item[1]) if rot == 0 else (item[1], item[0])
            base = rot * self.n_positions
            for y in range(self.zone_h):
                for x in range(self.zone_w):
                    if _can_place(self.grid, x, y, w, h):
                        mask[base + y * self.zone_w + x] = True
        # If every placement action is invalid, allow all (avoids empty-mask crash)
        # Skip action is still available in the mask
        if not np.any(mask[:-1]):  # Check only placement actions, not skip
            mask[:-1] = True
        return mask

    # ── Gymnasium API ────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)

        self.grid = np.zeros((self.zone_h, self.zone_w), dtype=np.int8)
        self.items = self._generate_items()
        # sort items by area (largest first) to make learning easier
        # self.items.sort(key=lambda x: x[0] * x[1], reverse=True)
        self.current_idx = 0
        self.placed_items = []
        self.invalid_count = 0
        cmap = plt.cm.get_cmap("tab20", self.n_items)
        self._item_colors = [cmap(i) for i in range(self.n_items)]

        return self._build_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        item = self._current_item

        # Episode already done
        if item is None:
            return self._build_obs(), 0.0, True, False, {}

        # ── Check if skip action ────────────────────────────────────────────
        if int(action) == self.skip_action_idx:
            # Skip current item and move to next
            # reward = minus area of skipped item to encourage packing more items
            reward = -0.5 * item[0] * item[1]
            self.current_idx += 1

            terminated = self.current_idx >= len(self.items)
            info = dict(
                placed=len(self.placed_items),
                total=len(self.items),
                utilisation=float(self.grid.sum()) / (self.zone_w * self.zone_h),
                action_type="skip",
            )
            return self._build_obs(), reward, terminated, False, info

        # ── Decode placement action ──────────────────────────────────────────
        rot = int(action) // self.n_positions  # 0 or 1
        pos = int(action) % self.n_positions
        x = pos % self.zone_w
        y = pos // self.zone_w
        w, h = (item[0], item[1]) if rot == 0 else (item[1], item[0])

        # ── Try to place ─────────────────────────────────────────────────────
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        if _can_place(self.grid, x, y, w, h):
            # Valid placement
            self.grid[y : y + h, x : x + w] = 1
            self.placed_items.append(
                dict(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    orig_idx=self.current_idx,
                    color=self._item_colors[self.current_idx],
                )
            )
            reward = float(w * h)  # area reward
            self.current_idx += 1

            # if grid is full
            if self.grid.sum() == self.zone_w * self.zone_h:
                terminated = True
                reward += self.zone_w * self.zone_h  # bonus: all items packed!
                info["all_packed"] = True
        else:
            # Invalid placement
            reward = -2.0
            self.invalid_count += 1
            if self.invalid_count >= self.max_invalid_steps:
                truncated = True
                info["truncated_reason"] = "too_many_invalid"

        info.update(
            placed=len(self.placed_items),
            total=len(self.items),
            utilisation=float(self.grid.sum()) / (self.zone_w * self.zone_h),
        )
        return self._build_obs(), reward, terminated, truncated, info

    # ── Rendering ────────────────────────────────────────────────────────────

    def render(self) -> Optional[np.ndarray]:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Zone background
        ax.add_patch(
            patches.Rectangle(
                (0, 0),
                self.zone_w,
                self.zone_h,
                linewidth=2,
                edgecolor="black",
                facecolor="#f0f0f0",
            )
        )

        # Placed items
        for item in self.placed_items:
            rect = patches.Rectangle(
                (item["x"], item["y"]),
                item["w"],
                item["h"],
                linewidth=1,
                edgecolor="black",
                facecolor=item["color"],
                alpha=0.75,
            )
            ax.add_patch(rect)
            cx = item["x"] + item["w"] / 2
            cy = item["y"] + item["h"] / 2
            ax.text(
                cx,
                cy,
                f"{item['w']}×{item['h']}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

        util = self.grid.sum() / (self.zone_w * self.zone_h) * 100
        ax.set_xlim(-0.5, self.zone_w + 0.5)
        ax.set_ylim(-0.5, self.zone_h + 0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("x (grid cells)")
        ax.set_ylabel("y (grid cells)")
        ax.set_title(
            f"Rectangle Packing — {len(self.placed_items)}/{len(self.items)} items"
            f"  |  Utilisation {util:.1f}%",
            fontsize=13,
        )

        # Legend: remaining items
        remaining = self.items[self.current_idx :]
        if remaining:
            legend_txt = "Remaining: " + ", ".join(f"{w}×{h}" for w, h in remaining[:6])
            if len(remaining) > 6:
                legend_txt += " …"
            ax.text(
                0.01,
                -0.06,
                legend_txt,
                transform=ax.transAxes,
                fontsize=8,
                color="gray",
            )

        plt.tight_layout()
        return fig, ax

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
