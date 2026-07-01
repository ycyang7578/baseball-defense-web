"""
Server-side matplotlib renderer — 重現 Model_3 的 park_single v2 圖。
輸入為 OptimizeResponse（已含球散點、三組站位、park boundary、stats），
輸出 PNG bytes，外觀與論文圖一致。
"""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns

# ── 顏色與樣式（對齊 Model_3）────────────────────────────────────
_STYLES = {
    "league_avg": dict(color="#1565C0", marker="D", size=170, label="League Avg",       offset=-17),
    "no_park":    dict(color="#C0392B", marker="o", size=200, label="RE24 Opt (no park)", offset=-34),
    "with_park":  dict(color="#7B2FBE", marker="*", size=480, label="RE24 Opt (park={park})", offset=16),
    "custom":     dict(color="#7B2FBE", marker="*", size=480, label="Selected Fielders", offset=16),
}
_WALL_COLOR = "#FF6B00"
_ORDER = ["league_avg", "no_park", "with_park"]


def _draw_field(ax):
    bases = np.array([(0, 0), (63.64, 63.64), (0, 127.28), (-63.64, 63.64), (0, 0)])
    ax.plot(bases[:, 0], bases[:, 1], color="black", lw=2, zorder=2)
    ax.scatter(bases[:-1, 0], bases[:-1, 1], c="white", edgecolors="black", s=80, zorder=3)
    ax.plot([0,  250], [0, 250], color="black", lw=1.5, zorder=1)
    ax.plot([0, -250], [0, 250], color="black", lw=1.5, zorder=1)
    ax.add_patch(patches.Arc(
        (0, 0), 800, 800, theta1=45, theta2=135,
        linestyle="--", color="gray", lw=2, zorder=1,
    ))


def _add_markers(ax, pts, style, park):
    color  = style["color"]
    label  = style["label"].format(park=park)
    offset = style["offset"]
    arr = np.array([[pts["LF"]["x"], pts["LF"]["y"]],
                    [pts["CF"]["x"], pts["CF"]["y"]],
                    [pts["RF"]["x"], pts["RF"]["y"]]])
    ax.scatter(arr[:, 0], arr[:, 1], c=color, s=style["size"], marker=style["marker"],
               zorder=8, edgecolors="white", linewidth=1.5, label=label)
    for code, (cx, cy) in zip(["LF", "CF", "RF"], arr):
        ax.text(cx, cy + offset, code, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white", zorder=9,
                bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="white", lw=0.8, alpha=0.95))


def render_plot(resp) -> bytes:
    """resp: OptimizeResponse pydantic 物件 → PNG bytes"""
    pos   = resp.positions
    stats = resp.stats
    park  = stats.home_team or ""

    balls = resp.balls
    bx = np.array([b.x for b in balls])
    by = np.array([b.y for b in balls])
    wall = np.array([b.is_wall_ball for b in balls], dtype=bool)
    cp = np.array([b.catch_prob for b in balls])

    fig, ax = plt.subplots(figsize=(10, 9))
    _draw_field(ax)

    # ── 藍色 KDE 密度背景 ──────────────────────────────────────
    if len(bx) > 5:
        try:
            sns.kdeplot(x=bx, y=by, fill=True, cmap="Blues", ax=ax,
                        alpha=0.28, levels=10, zorder=0, warn_singular=False)
            sns.kdeplot(x=bx, y=by, fill=False, color="#4a72c4", ax=ax,
                        alpha=0.15, levels=10, linewidths=0.4, zorder=1,
                        warn_singular=False)
        except Exception:
            pass

    # ── park boundary ─────────────────────────────────────────
    if resp.park_boundary:
        px = [c.x for c in resp.park_boundary]
        py = [c.y for c in resp.park_boundary]
        ax.plot(px, py, color="#00CC55", lw=2.2, alpha=0.9, zorder=3, label="Park Boundary")

    # ── 球散點 ────────────────────────────────────────────────
    sc = ax.scatter(bx[~wall], by[~wall], c=cp[~wall], cmap="RdYlGn", vmin=0, vmax=1,
                    s=30, alpha=0.85, edgecolors="gray", linewidths=0.3, zorder=4)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.040, pad=0.02, shrink=0.78)
    cbar.set_label("Catch Probability", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    if wall.any():
        ax.scatter(bx[wall], by[wall], c=_WALL_COLOR, s=55, marker="*",
                   edgecolors="black", linewidths=0.4, zorder=5,
                   label=f"Wall Ball ({int(wall.sum())})")

    # ── 站位標記（custom > with_park > 全部）────────────────────
    if "custom" in pos:
        draw_keys = ["custom"]
    elif "with_park" in pos:
        draw_keys = ["with_park"]
    else:
        draw_keys = _ORDER
    for key in draw_keys:
        if key in pos:
            ps = pos[key]
            _add_markers(ax, {"LF": ps.LF.model_dump(), "CF": ps.CF.model_dump(),
                              "RF": ps.RF.model_dump()}, _STYLES[key], park)

    # ── 座標軸 ────────────────────────────────────────────────
    ax.set_xlim(-280, 280)
    ax.set_ylim(-10, 450)
    ax.set_aspect("equal")
    ax.set_xlabel("x coordinate (ft)", fontsize=11)
    ax.set_ylabel("y coordinate (ft)", fontsize=11)
    ax.tick_params(labelsize=9)

    # ── 標題 ──────────────────────────────────────────────────
    fig.suptitle(
        f"{resp.title}\n"
        f"Situation: {resp.situation}   |   n = {stats.n_balls}   |   n_wall = {stats.n_wall_balls}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ── 圖例（左下）────────────────────────────────────────────
    handles, labels = ax.get_legend_handles_labels()
    # 依需求排序：Park Boundary, Wall Ball, League Avg, no_park, with_park
    order_key = ["Park Boundary", "Wall Ball", "League Avg", "RE24 Opt (no park)", "RE24 Opt (park"]
    def _rank(lbl):
        for i, k in enumerate(order_key):
            if lbl.startswith(k):
                return i
        return len(order_key)
    pairs = sorted(zip(handles, labels), key=lambda hl: _rank(hl[1]))
    ax.legend([h for h, _ in pairs], [l for _, l in pairs],
              loc="lower left", fontsize=9.5, framealpha=0.9)

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


_ZONE_COLORS = {"LF": "#3B82F6", "CF": "#10B981", "RF": "#F97316"}
_ZONE_MARKERS = {"LF": ("o", "#1D4ED8"), "CF": ("D", "#065F46"), "RF": ("o", "#C2410C")}


def render_coverage_map(
    positions: dict,
    grid: dict,
    home_team: str | None = None,
    title: str = "Coverage Map",
    park_boundary=None,
) -> bytes:
    """
    Render full-outfield joint catch probability heatmap.
    positions: {"LF": (x,y), "CF": (x,y), "RF": (x,y)}
    grid: output of compute_coverage_grid (xs, ys, joint, per_pos)
    """
    xs = grid["xs"]
    ys = grid["ys"]
    ZZ = grid["joint"]
    per_pos = grid["per_pos"]

    fig, ax = plt.subplots(figsize=(8, 8))

    # 接殺機率熱圖（主背景）
    c = ax.pcolormesh(xs, ys, ZZ, cmap="RdYlGn", vmin=0, vmax=1,
                      shading="nearest", alpha=0.90, zorder=0)
    cbar = plt.colorbar(c, ax=ax, fraction=0.036, pad=0.02)
    cbar.set_label("Joint Catch Probability", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # 各位置守備區域邊界（等高線）
    p_stack = np.stack([per_pos[pos] for pos in ("LF", "CF", "RF")], axis=0)
    dominant = np.argmax(p_stack, axis=0)    # 0=LF, 1=CF, 2=RF
    for i, pos in enumerate(("LF", "CF", "RF")):
        mask = (dominant == i).astype(float)
        try:
            ax.contour(xs, ys, mask, levels=[0.5],
                       colors=[_ZONE_COLORS[pos]], linewidths=1.8, alpha=0.75, zorder=2)
        except Exception:
            pass

    # Park boundary overlay
    if park_boundary:
        bx = [c[0] for c in park_boundary]
        by = [c[1] for c in park_boundary]
        ax.plot(bx, by, color="#00CC55", lw=2.0, alpha=0.9, zorder=3, label="Park Boundary")

    # 守備員位置
    for pos, (fx, fy) in positions.items():
        marker, color = _ZONE_MARKERS[pos]
        ax.scatter(fx, fy, s=250, c=color, marker=marker, zorder=8,
                   edgecolors="white", linewidths=1.8)
        ax.text(fx, fy - 18, pos, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white", zorder=9,
                bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="white", lw=0.8, alpha=0.95))

    # 位置圖例（色點）
    for pos, color in _ZONE_COLORS.items():
        ax.plot([], [], color=color, lw=2.5, label=f"{pos} zone")

    _draw_field(ax)

    ax.set_xlim(-250, 250)
    ax.set_ylim(0, 445)
    ax.set_aspect("equal")
    ax.set_xlabel("x coordinate (ft)", fontsize=10)
    ax.set_ylabel("y coordinate (ft)", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_title(title, fontsize=13, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
