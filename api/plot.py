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

    # ── 站位標記（custom > with_park > no_park；聯盟平均只留數字比較，圖上不畫）──
    if "custom" in pos:
        draw_keys = ["custom"]
    elif "with_park" in pos:
        draw_keys = ["with_park"]
    else:
        draw_keys = ["no_park"]
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

