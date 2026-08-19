"""
Server-side matplotlib renderer — reproduces Model_3's park_single v2 chart.
Input is an OptimizeResponse (already containing ball scatter points, three position sets,
park boundary, stats); output is PNG bytes, with an appearance matching the thesis figures.
"""
import io
from typing import TypedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes

from .schemas import OptimizeResponse


class MarkerStyle(TypedDict):
    color: str
    marker: str
    size: int
    label: str
    offset: int


# ── Colors and styles (aligned with Model_3) ────────────────────────────────────
_STYLES: dict[str, MarkerStyle] = {
    "league_avg": dict(color="#1565C0", marker="D", size=170, label="League Avg",       offset=-17),
    "no_park":    dict(color="#C0392B", marker="o", size=200, label="RE24 Opt (no park)", offset=-34),
    "with_park":  dict(color="#7B2FBE", marker="*", size=480, label="RE24 Opt (park={park})", offset=16),
    "custom":     dict(color="#7B2FBE", marker="*", size=480, label="Selected Fielders", offset=16),
}
_WALL_COLOR: str = "#FF6B00"


def _draw_field(ax: Axes) -> None:
    bases = np.array([(0, 0), (63.64, 63.64), (0, 127.28), (-63.64, 63.64), (0, 0)])
    ax.plot(bases[:, 0], bases[:, 1], color="black", lw=2, zorder=2)
    ax.scatter(bases[:-1, 0], bases[:-1, 1], c="white", edgecolors="black", s=80, zorder=3)
    ax.plot([0,  250], [0, 250], color="black", lw=1.5, zorder=1)
    ax.plot([0, -250], [0, 250], color="black", lw=1.5, zorder=1)
    ax.add_patch(patches.Arc(
        (0, 0), 800, 800, theta1=45, theta2=135,
        linestyle="--", color="gray", lw=2, zorder=1,
    ))


def _add_markers(ax: Axes, pts: dict[str, dict[str, float]],
                 style: MarkerStyle, park: str) -> None:
    color  = style["color"]
    label  = style["label"].format(park=park)
    offset = style["offset"]
    arr = np.array([[pts["LF"]["x"], pts["LF"]["y"]],
                    [pts["CF"]["x"], pts["CF"]["y"]],
                    [pts["RF"]["x"], pts["RF"]["y"]]])
    ax.scatter(arr[:, 0], arr[:, 1], c=color, s=style["size"], marker=style["marker"],
               zorder=8, edgecolors="white", linewidth=1.5, label=label)
    for position_code, (x, y) in zip(["LF", "CF", "RF"], arr):
        ax.text(x, y + offset, position_code, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white", zorder=9,
                bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="white", lw=0.8, alpha=0.95))


def render_plot(resp: OptimizeResponse) -> bytes:
    """resp: OptimizeResponse pydantic object → PNG bytes"""
    positions = resp.positions
    stats = resp.stats
    park  = stats.home_team or ""

    balls = resp.balls
    ball_x = np.array([b.x for b in balls])
    ball_y = np.array([b.y for b in balls])
    is_wall_ball = np.array([b.is_wall_ball for b in balls], dtype=bool)
    catch_prob = np.array([b.catch_prob for b in balls])

    fig, ax = plt.subplots(figsize=(10, 9))
    _draw_field(ax)

    # ── Blue KDE density background ──────────────────────────────────────
    if len(ball_x) > 5:
        try:
            sns.kdeplot(x=ball_x, y=ball_y, fill=True, cmap="Blues", ax=ax,
                        alpha=0.28, levels=10, zorder=0, warn_singular=False)
            sns.kdeplot(x=ball_x, y=ball_y, fill=False, color="#4a72c4", ax=ax,
                        alpha=0.15, levels=10, linewidths=0.4, zorder=1,
                        warn_singular=False)
        except Exception:
            pass

    # ── park boundary ─────────────────────────────────────────
    if resp.park_boundary:
        park_x = [c.x for c in resp.park_boundary]
        park_y = [c.y for c in resp.park_boundary]
        ax.plot(park_x, park_y, color="#00CC55", lw=2.2, alpha=0.9, zorder=3, label="Park Boundary")

    # ── Ball scatter points ────────────────────────────────────────────────
    scatter_plot = ax.scatter(ball_x[~is_wall_ball], ball_y[~is_wall_ball], c=catch_prob[~is_wall_ball],
                              cmap="RdYlGn", vmin=0, vmax=1,
                              s=30, alpha=0.85, edgecolors="gray", linewidths=0.3, zorder=4)
    cbar = plt.colorbar(scatter_plot, ax=ax, fraction=0.040, pad=0.02, shrink=0.78)
    cbar.set_label("Catch Probability", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    if is_wall_ball.any():
        ax.scatter(ball_x[is_wall_ball], ball_y[is_wall_ball], c=_WALL_COLOR, s=55, marker="*",
                   edgecolors="black", linewidths=0.4, zorder=5,
                   label=f"Wall Ball ({int(is_wall_ball.sum())})")

    # ── Position markers (custom > with_park > no_park; league average is kept only for numeric comparison, not drawn) ──
    if "custom" in positions:
        draw_keys = ["custom"]
    elif "with_park" in positions:
        draw_keys = ["with_park"]
    else:
        draw_keys = ["no_park"]
    for key in draw_keys:
        if key in positions:
            position_set = positions[key]
            _add_markers(ax, {"LF": position_set.LF.model_dump(), "CF": position_set.CF.model_dump(),
                              "RF": position_set.RF.model_dump()}, _STYLES[key], park)

    # ── Axes ────────────────────────────────────────────────
    ax.set_xlim(-280, 280)
    ax.set_ylim(-10, 450)
    ax.set_aspect("equal")
    ax.set_xlabel("x coordinate (ft)", fontsize=11)
    ax.set_ylabel("y coordinate (ft)", fontsize=11)
    ax.tick_params(labelsize=9)

    # ── Title ──────────────────────────────────────────────────
    fig.suptitle(
        f"{resp.title}\n"
        f"Situation: {resp.situation}   |   n = {stats.n_balls}   |   n_wall = {stats.n_wall_balls}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ── Legend (bottom left) ────────────────────────────────────────────
    handles, labels = ax.get_legend_handles_labels()
    # Sort per requirements: Park Boundary, Wall Ball, League Avg, no_park, with_park
    order_key = ["Park Boundary", "Wall Ball", "League Avg", "RE24 Opt (no park)", "RE24 Opt (park"]
    def _rank(label: str) -> int:
        for i, k in enumerate(order_key):
            if label.startswith(k):
                return i
        return len(order_key)
    pairs = sorted(zip(handles, labels), key=lambda handle_label: _rank(handle_label[1]))
    ax.legend([h for h, _ in pairs], [l for _, l in pairs],
              loc="lower left", fontsize=9.5, framealpha=0.9)

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

