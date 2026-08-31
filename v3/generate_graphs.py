"""
VYREX Training Graphs — For community sharing
================================================
Generates publication-quality training charts from WandB data.

Usage:
    python generate_graphs.py
    python generate_graphs.py --output ./graphs
    python generate_graphs.py --dpi 200

Output:
    1. entropy_journey.png    — Entropy descent with version markers
    2. metrics_dashboard.png  — 4-panel: Speed, Boost, Aerial Touch, Demos
    3. ppo_health.png         — Clip fraction + entropy overlay
    4. version_impact.png     — Before/after bar chart per version
"""

import argparse
import math
import os
import sys
import numpy as np

try:
    import wandb
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.patches import FancyBboxPatch
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install wandb matplotlib")
    sys.exit(1)

# ============================================================================
# CONFIG
# ============================================================================

WANDB_ENTITY = "ogmatrixai"
WANDB_PROJECT = "rlgym-ppo"
WANDB_RUN_ID = "9ue9rans"

MAX_ENTROPY = math.log(90)  # ln(90) = 4.4998

# Version boundaries (step, label, color)
VERSIONS = [
    (2_380_000_000, "v3.5", "#666666"),
    (2_860_000_000, "v3.6", "#888888"),
    (3_130_000_000, "v3.7\nThe Sharpening", "#e74c3c"),
    (4_030_000_000, "v3.8\nThe Coordinator", "#3498db"),
    (5_005_000_000, "v3.9\nThe Fuel Line", "#2ecc71"),
    (5_720_000_000, "v3.10\nThe Clean Hit", "#f39c12"),
]

# Color palette
BG_COLOR = "#0d1117"
PANEL_COLOR = "#161b22"
TEXT_COLOR = "#e6edf3"
GRID_COLOR = "#21262d"
ACCENT_CYAN = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_RED = "#f85149"
ACCENT_ORANGE = "#d29922"
ACCENT_PURPLE = "#bc8cff"

# ============================================================================
# DATA LOADING
# ============================================================================

def load_wandb_data(samples=3000):
    """Pull training metrics from WandB using run.history() (same as analyze.py)."""
    print("Loading WandB data...")
    api = wandb.Api()
    run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{WANDB_RUN_ID}")

    # Step interpolation — needed to convert internal _step to env steps
    step_rows = list(run.history(keys=["Cumulative Timesteps"], samples=500, pandas=False))
    step_rows = [r for r in step_rows if r.get("Cumulative Timesteps") is not None]
    step_pairs = sorted([(r["_step"], int(r["Cumulative Timesteps"])) for r in step_rows])
    print(f"  Step interpolation points: {len(step_pairs)}")

    def interp_step(internal_step):
        """Convert WandB internal _step → environment cumulative timesteps."""
        if not step_pairs:
            return internal_step
        if internal_step <= step_pairs[0][0]:
            return step_pairs[0][1]
        if internal_step >= step_pairs[-1][0]:
            return step_pairs[-1][1]
        for i in range(len(step_pairs) - 1):
            s0, t0 = step_pairs[i]
            s1, t1 = step_pairs[i + 1]
            if s0 <= internal_step <= s1:
                frac = (internal_step - s0) / max(s1 - s0, 1)
                return int(t0 + frac * (t1 - t0))
        return step_pairs[-1][1]

    # Game metrics (use exact WandB key names from rlgym-ppo)
    game_keys = [
        "vyrex/airborne_frac", "vyrex/avg_boost", "vyrex/avg_speed",
        "vyrex/avg_teammate_dist", "vyrex/goal_diff_blue",
        "vyrex/touches_per_step", "vyrex/demos_per_step", "vyrex/aerial_touch_rate",
    ]
    game_rows = list(run.history(keys=game_keys, samples=samples, pandas=False))
    game_rows = [r for r in game_rows if any(r.get(k) is not None for k in game_keys)]
    # Add interpolated env steps
    for r in game_rows:
        r["env_step"] = interp_step(r.get("_step", 0))
    print(f"  Game rows: {len(game_rows)}")

    # PPO metrics (use exact WandB key names)
    ppo_keys = ["Policy Entropy", "SB3 Clip Fraction", "Mean KL Divergence",
                "Value Function Loss", "Policy Reward"]
    ppo_rows = list(run.history(keys=ppo_keys, samples=samples, pandas=False))
    ppo_rows = [r for r in ppo_rows if any(r.get(k) is not None for k in ppo_keys)]
    for r in ppo_rows:
        r["env_step"] = interp_step(r.get("_step", 0))
    print(f"  PPO rows: {len(ppo_rows)}")

    return game_rows, ppo_rows


def bucket_data(rows, key, bucket_size=50_000_000):
    """Bucket raw data into fixed-size bins, returning (centers, means, stds)."""
    if not rows:
        return np.array([]), np.array([]), np.array([])

    buckets = {}
    for r in rows:
        step = r.get("env_step", r.get("_step", 0))
        val = r.get(key)
        if val is None or (isinstance(val, str)):
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue
        b = (step // bucket_size) * bucket_size + bucket_size // 2
        buckets.setdefault(b, []).append(val)

    if not buckets:
        return np.array([]), np.array([]), np.array([])

    centers = sorted(buckets.keys())
    means = [np.mean(buckets[c]) for c in centers]
    stds = [np.std(buckets[c]) for c in centers]

    return np.array(centers), np.array(means), np.array(stds)


def step_to_b(steps):
    """Convert steps to billions for axis labels."""
    return np.array(steps) / 1e9

# ============================================================================
# STYLE HELPERS
# ============================================================================

def apply_dark_style(fig, axes):
    """Apply consistent dark theme to figure and axes."""
    fig.patch.set_facecolor(BG_COLOR)
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax in axes:
        ax.set_facecolor(PANEL_COLOR)
        ax.tick_params(colors=TEXT_COLOR, which="both")
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID_COLOR)
        ax.spines["bottom"].set_color(GRID_COLOR)
        ax.grid(True, alpha=0.15, color=GRID_COLOR, linestyle="--")
        ax.set_xlabel("Training Steps (Billions)", fontsize=9, color=TEXT_COLOR)


def add_version_markers(ax, y_range=None, labels=True):
    """Add vertical version boundary lines."""
    for step, label, color in VERSIONS:
        ax.axvline(x=step / 1e9, color=color, alpha=0.5, linestyle="--", linewidth=0.8)
        if labels and y_range is not None:
            ax.text(
                step / 1e9 + 0.02, y_range[1] - (y_range[1] - y_range[0]) * 0.05,
                label.split("\n")[0], fontsize=6, color=color, alpha=0.7,
                rotation=90, va="top", ha="left",
            )

# ============================================================================
# GRAPH 1: ENTROPY JOURNEY
# ============================================================================

def plot_entropy(ppo_rows, output_dir):
    """The complete entropy descent story — the hero chart."""
    print("Generating entropy journey...")

    centers, means, stds = bucket_data(ppo_rows, "Policy Entropy", bucket_size=25_000_000)
    if len(centers) == 0:
        print("  No entropy data!")
        return

    x = step_to_b(centers)
    pct = means / MAX_ENTROPY * 100

    fig, ax = plt.subplots(figsize=(14, 6))
    apply_dark_style(fig, ax)

    # Fill between for std dev band
    ax.fill_between(x, (means - stds) / MAX_ENTROPY * 100, (means + stds) / MAX_ENTROPY * 100,
                     alpha=0.15, color=ACCENT_CYAN)

    # Main entropy line
    ax.plot(x, pct, color=ACCENT_CYAN, linewidth=1.5, zorder=3)

    # Entropy walls — annotate the plateaus
    walls = [
        (3.63, 3.13, 4.03, "#f85149", "Wall #1\n3.63 (80.7%)\n500M stuck"),
        (3.51, 4.03, 5.00, "#f85149", "Wall #2\n3.51 (78.0%)\n825M stuck"),
        (3.25, 5.35, 5.72, "#d29922", "Wall #3\n3.25 (72.2%)\n375M..."),
    ]
    for level, start_b, end_b, color, label in walls:
        level_pct = level / MAX_ENTROPY * 100
        ax.hlines(level_pct, start_b, end_b, colors=color, linestyles=":", linewidth=1.5, alpha=0.7)
        ax.annotate(label, xy=(end_b, level_pct), fontsize=7, color=color, alpha=0.9,
                     ha="left", va="center",
                     xytext=(end_b + 0.05, level_pct),
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_COLOR, edgecolor=color, alpha=0.8))

    # ent_coef changes — key events
    events = [
        (0.400, "ent 0.01→0.005", ACCENT_GREEN),
        (0.897, "ent 0.005→0.003", ACCENT_GREEN),
        (3.130, "ent 0.003→0.001\n(v3.7)", "#e74c3c"),
        (4.030, "ent→0.0008\n(v3.8)", "#3498db"),
        (5.005, "ent→0.0006\n(v3.9)", ACCENT_GREEN),
        (5.720, "ent→0.00048\n(v3.10)", ACCENT_ORANGE),
    ]
    for step_b, label, color in events:
        # Find nearest entropy value
        idx = np.argmin(np.abs(x - step_b))
        if idx < len(pct):
            ax.annotate("", xy=(step_b, pct[idx]), xytext=(step_b, pct[idx] + 3),
                         arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
            ax.text(step_b, pct[idx] + 3.5, label, fontsize=6.5, color=color,
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=BG_COLOR, edgecolor=color, alpha=0.7))

    # Reference lines
    ax.axhline(y=100, color=ACCENT_RED, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.text(0.05, 100.5, "Random (100%)", fontsize=7, color=ACCENT_RED, alpha=0.5)

    ax.set_xlim(0, max(x) + 0.3)
    ax.set_ylim(65, 102)
    ax.set_ylabel("Entropy (% of maximum)", fontsize=11, color=TEXT_COLOR)
    ax.set_title("VYREX — Entropy Journey: From Random to Intentional",
                  fontsize=14, fontweight="bold", color=ACCENT_CYAN, pad=15)

    # Current stats box
    current_ent = means[-1]
    current_pct = current_ent / MAX_ENTROPY * 100
    current_steps = centers[-1] / 1e9
    stats_text = (f"Current: {current_ent:.3f} ({current_pct:.1f}%)\n"
                  f"Steps: {current_steps:.2f}B\n"
                  f"Action Space: 90 actions\n"
                  f"Max Entropy: {MAX_ENTROPY:.3f}")
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=8, color=TEXT_COLOR, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, alpha=0.9))

    fig.tight_layout()
    path = os.path.join(output_dir, "entropy_journey.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# GRAPH 2: METRICS DASHBOARD (4-panel)
# ============================================================================

def plot_metrics_dashboard(game_rows, output_dir):
    """4-panel dashboard: Speed, Boost, Aerial Touch, Demos."""
    print("Generating metrics dashboard...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    apply_dark_style(fig, axes)

    configs = [
        ("vyrex/avg_speed", "Avg Speed (uu/s)", ACCENT_CYAN, "Speed — Ground Movement"),
        ("vyrex/avg_boost", "Avg Boost", ACCENT_GREEN, "Boost — The Bottleneck"),
        ("vyrex/aerial_touch_rate", "Aerial Touch Rate", ACCENT_PURPLE, "Aerial Touches — Air Game"),
        ("vyrex/demos_per_step", "Demos/Step", ACCENT_RED, "Demolitions — Aggression"),
    ]

    for ax, (key, ylabel, color, title) in zip(axes.flatten(), configs):
        centers, means, stds = bucket_data(game_rows, key, bucket_size=100_000_000)
        if len(centers) == 0:
            continue

        x = step_to_b(centers)
        ax.fill_between(x, means - stds, means + stds, alpha=0.15, color=color)
        ax.plot(x, means, color=color, linewidth=1.5)

        # Trend line (last 2B)
        mask = x > (max(x) - 2.0)
        if sum(mask) > 3:
            z = np.polyfit(x[mask], means[mask], 1)
            trend_x = np.linspace(x[mask].min(), x[mask].max(), 50)
            ax.plot(trend_x, np.polyval(z, trend_x), "--", color=color, alpha=0.4, linewidth=1)
            direction = "↑" if z[0] > 0 else "↓"
            ax.text(0.95, 0.05, f"Trend: {direction} {abs(z[0]):.4f}/B",
                    transform=ax.transAxes, fontsize=7, color=color, ha="right", va="bottom")

        # Version markers (no labels to avoid clutter)
        for step, label, vcolor in VERSIONS[2:]:  # Only v3.7+
            ax.axvline(x=step / 1e9, color=vcolor, alpha=0.3, linestyle="--", linewidth=0.6)

        ax.set_ylabel(ylabel, fontsize=9, color=TEXT_COLOR)
        ax.set_title(title, fontsize=10, fontweight="bold", color=color, pad=8)

        # Current value annotation
        if len(means) > 0:
            ax.text(0.95, 0.95, f"Current: {means[-1]:.4f}",
                    transform=ax.transAxes, fontsize=8, color=TEXT_COLOR,
                    ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_COLOR, edgecolor=GRID_COLOR))

    fig.suptitle("VYREX — Training Metrics Dashboard @ 5.72B Steps",
                 fontsize=14, fontweight="bold", color=ACCENT_CYAN, y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "metrics_dashboard.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# GRAPH 3: PPO HEALTH
# ============================================================================

def plot_ppo_health(ppo_rows, output_dir):
    """Entropy + clip fraction dual-axis chart."""
    print("Generating PPO health...")

    fig, ax1 = plt.subplots(figsize=(14, 5))
    apply_dark_style(fig, ax1)

    # Entropy on left axis
    ec, em, es = bucket_data(ppo_rows, "Policy Entropy", bucket_size=25_000_000)
    if len(ec) > 0:
        x = step_to_b(ec)
        ax1.plot(x, em, color=ACCENT_CYAN, linewidth=1.2, label="Entropy")
        ax1.fill_between(x, em - es, em + es, alpha=0.1, color=ACCENT_CYAN)
        ax1.set_ylabel("Entropy", fontsize=10, color=ACCENT_CYAN)
        ax1.tick_params(axis="y", labelcolor=ACCENT_CYAN)

    # Clip fraction on right axis
    ax2 = ax1.twinx()
    cc, cm, cs = bucket_data(ppo_rows, "SB3 Clip Fraction", bucket_size=25_000_000)
    if len(cc) > 0:
        x2 = step_to_b(cc)
        ax2.plot(x2, cm, color=ACCENT_ORANGE, linewidth=1.0, alpha=0.8, label="Clip Fraction")
        ax2.fill_between(x2, cm - cs, cm + cs, alpha=0.08, color=ACCENT_ORANGE)
        ax2.set_ylabel("Clip Fraction", fontsize=10, color=ACCENT_ORANGE)
        ax2.tick_params(axis="y", labelcolor=ACCENT_ORANGE)
        ax2.spines["right"].set_color(ACCENT_ORANGE)
        ax2.spines["right"].set_visible(True)

        # Healthy range band
        ax2.axhspan(0.005, 0.05, alpha=0.05, color=ACCENT_GREEN)
        ax2.text(0.02, 0.045, "Healthy clip range", fontsize=7, color=ACCENT_GREEN, alpha=0.5,
                 transform=ax2.get_yaxis_transform())

    # Version markers
    for step, label, color in VERSIONS[2:]:
        ax1.axvline(x=step / 1e9, color=color, alpha=0.4, linestyle="--", linewidth=0.8)
        ax1.text(step / 1e9, ax1.get_ylim()[1], label.split("\n")[0],
                 fontsize=7, color=color, alpha=0.7, rotation=90, va="top", ha="right")

    ax1.set_title("VYREX — PPO Training Health",
                   fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=12)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               fontsize=8, facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    fig.tight_layout()
    path = os.path.join(output_dir, "ppo_health.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# GRAPH 4: VERSION IMPACT BAR CHART
# ============================================================================

def plot_version_impact(game_rows, ppo_rows, output_dir):
    """Side-by-side bar chart showing metric changes per version."""
    print("Generating version impact chart...")

    version_ranges = {
        "v3.7": (3_130_000_000, 4_030_000_000),
        "v3.8": (4_030_000_000, 5_005_000_000),
        "v3.9": (5_005_000_000, 5_720_000_000),
    }

    def avg_in_range(rows, key, start, end):
        vals = [float(r[key]) for r in rows
                if r.get("env_step", r.get("_step", 0)) >= start
                and r.get("env_step", r.get("_step", 0)) < end
                and r.get(key) is not None and not isinstance(r.get(key), str)]
        return np.mean(vals) if vals else 0

    metrics = [
        ("Policy Entropy", ppo_rows, "Entropy", ACCENT_CYAN),
        ("vyrex/avg_boost", game_rows, "Avg Boost", ACCENT_GREEN),
        ("vyrex/avg_speed", game_rows, "Speed (÷10)", ACCENT_ORANGE),
        ("vyrex/aerial_touch_rate", game_rows, "Aerial Touch (×100)", ACCENT_PURPLE),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    apply_dark_style(fig, axes)

    versions = list(version_ranges.keys())
    colors_v = ["#e74c3c", "#3498db", "#2ecc71"]

    for ax, (key, rows, label, color) in zip(axes, metrics):
        values = []
        for v in versions:
            start, end = version_ranges[v]
            val = avg_in_range(rows, key, start, end)
            # Normalize for display
            if "Speed" in label:
                val /= 10
            elif "Aerial" in label:
                val *= 100
            values.append(val)

        bars = ax.bar(versions, values, color=colors_v, alpha=0.8, edgecolor="none", width=0.6)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)

        ax.set_title(label, fontsize=10, fontweight="bold", color=color, pad=8)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)

    fig.suptitle("VYREX — Metric Averages by Version",
                 fontsize=14, fontweight="bold", color=ACCENT_CYAN, y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "version_impact.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate VYREX training graphs")
    parser.add_argument("--output", default="./graphs", help="Output directory")
    parser.add_argument("--dpi", type=int, default=200, help="Image DPI")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    game_rows, ppo_rows = load_wandb_data()

    plot_entropy(ppo_rows, args.output)
    plot_metrics_dashboard(game_rows, args.output)
    plot_ppo_health(ppo_rows, args.output)
    plot_version_impact(game_rows, ppo_rows, args.output)

    print(f"\nAll graphs saved to {args.output}/")
    print("Files:")
    for f in sorted(os.listdir(args.output)):
        if f.endswith(".png"):
            size = os.path.getsize(os.path.join(args.output, f)) / 1024
            print(f"  {f} ({size:.0f} KB)")


if __name__ == "__main__":
    main()
