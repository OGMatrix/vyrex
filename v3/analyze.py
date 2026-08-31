"""
VYREX Training Analyzer — Comprehensive WandB Data Analysis
===========================================================
Deep analysis tool for the AI manager. Pulls all training data,
computes trajectories, detects plateaus, compares versions, and
generates actionable diagnostics.

Usage:
    python analyze.py                          # Full analysis at current step
    python analyze.py --since 4.5              # Analyze from 4.5B onward
    python analyze.py --version v3.8           # Analyze specific version
    python analyze.py --compare v3.7 v3.8      # Compare two versions
    python analyze.py --health                 # PPO health check only
    python analyze.py --entropy-detail         # Fine-grained entropy analysis
    python analyze.py --match path/to/match.json  # Analyze match file
    python analyze.py --all                    # Everything

Author: VYREX AI Manager
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================================================
# CONFIGURATION — Version boundaries and WandB project info
# ============================================================================

WANDB_ENTITY = "ogmatrixai"
WANDB_PROJECT = "rlgym-ppo"
WANDB_RUN_ID = "9ue9rans"

# Action space
MAX_ACTIONS = 90
MAX_ENTROPY = math.log(MAX_ACTIONS)  # ln(90) = 4.4998

# Version history with step boundaries
VERSIONS = {
    "v3.5": (2_380_000_000, 2_860_000_000),
    "v3.6": (2_860_000_000, 3_130_000_000),
    "v3.7": (3_130_000_000, 4_030_000_000),
    "v3.8": (4_030_000_000, 5_005_000_000),
    "v3.9": (5_005_000_000, 5_720_000_000),
    "v3.10": (5_720_000_000, 6_500_000_000),
    "v3.11": (6_500_000_000, 7_200_000_000),
    "v3.12": (7_200_000_000, 7_845_000_000),
    "v3.13": (7_845_000_000, 8_565_000_000),
    "v3.14": (8_565_000_000, 11_000_000_000),
    "v3.15": (11_000_000_000, 11_200_000_000),
    "v3.16": (11_200_000_000, 11_400_000_000),
    "v3.17": (11_400_000_000, 11_960_000_000),
    "v3.18": (11_960_000_000, 12_500_000_000),
    "v3.19": (12_500_000_000, 14_000_000_000),
    "v3.20": (14_000_000_000, 14_700_000_000),
    "v3.21": (14_700_000_000, 15_400_000_000),
    "v3.22": (15_400_000_000, 16_000_000_000),
    "v3.23": (16_000_000_000, 16_500_000_000),
    "v3.24": (16_500_000_000, 17_000_000_000),
    "v3.25": (17_000_000_000, None),  # None = ongoing
}

# Key hyperparameters per version for context
VERSION_PARAMS = {
    "v3.7": {"ent_coef": 0.001, "batch": 200_000, "spacing": 0.012, "components": 12},
    "v3.8": {"ent_coef": 0.0008, "batch": 200_000, "spacing": 0.025, "components": 12},
    "v3.9": {"ent_coef": 0.0006, "batch": 200_000, "spacing": 0.025, "components": 12},
    "v3.10": {"ent_coef": 0.00048, "batch": 200_000, "spacing": 0.025, "components": 12},
    "v3.11": {"ent_coef": 0.00038, "batch": 200_000, "spacing": 0.025, "components": 11},
    "v3.12": {"ent_coef": 0.00038, "batch": 200_000, "spacing": 0.025, "components": 13},
    "v3.13": {"ent_coef": 0.00038, "batch": 200_000, "spacing": 0.025, "components": 13},
    "v3.14": {"ent_coef": 0.00038, "batch": 200_000, "spacing": 0.025, "components": 15},
    "v3.17": {"ent_coef": 0.0005, "batch": 200_000, "spacing": 0.025, "components": 17},
    "v3.18": {"ent_coef": 0.0005, "batch": 200_000, "spacing": 0.025, "components": 18},
    "v3.19": {"ent_coef": 0.0005, "batch": 200_000, "spacing": 0.025, "components": 18},
    "v3.20": {"ent_coef": 0.0005, "batch": 200_000, "spacing": 0.04, "components": 18},
    "v3.21": {"ent_coef": 0.0005, "batch": 200_000, "spacing": 0.04, "components": 19},
    "v3.22": {"ent_coef": 0.0005, "batch": 200_000, "spacing": 0.04, "components": 20},
    "v3.23": {"ent_coef": 0.0005, "batch": 200_000, "spacing": 0.04, "components": 19},
    "v3.24": {"ent_coef": 0.0005, "batch": 200_000, "spacing": 0.04, "components": 18},
    "v3.25": {"ent_coef": 0.0005, "batch": 200_000, "spacing": 0.04, "components": 19},
}

# ============================================================================
# METRIC DEFINITIONS
# ============================================================================

# WandB metric keys for game metrics (logged by metrics_logger.py)
GAME_METRICS = {
    "airborne":    "vyrex/airborne_frac",
    "boost":       "vyrex/avg_boost",
    "speed":       "vyrex/avg_speed",
    "team_dist":   "vyrex/avg_teammate_dist",
    "goal_diff":   "vyrex/goal_diff_blue",
    "touches":     "vyrex/touches_per_step",
    "demos":       "vyrex/demos_per_step",
    "aerial_touch":"vyrex/aerial_touch_rate",
    # Enhanced v3.10+ metrics
    "zero_boost":  "vyrex/zero_boost_frac",
    "supersonic":  "vyrex/supersonic_frac",
    "own_half":    "vyrex/own_half_frac",
    "near_wall":   "vyrex/near_wall_frac",
    "has_flip":    "vyrex/has_flip_frac",
    "boosting":    "vyrex/boosting_frac",
    "ball_dist":   "vyrex/avg_ball_dist",
    "ball_speed":  "vyrex/ball_speed",
    "dbl_commit":  "vyrex/double_commit_frac",
    "boost_coll":  "vyrex/boost_collected_per_step",
    "touch_speed": "vyrex/touch_ball_speed",
    # v3.14 metrics
    "aerial_seq":  "vyrex/aerial_seq_touches",
}

# WandB metric keys for PPO health
PPO_METRICS = {
    "entropy":     "Policy Entropy",
    "clip":        "SB3 Clip Fraction",
    "kl":          "Mean KL Divergence",
    "vf_loss":     "Value Function Loss",
    "reward":      "Policy Reward",
}

# Desired directions for each metric (for trend assessment)
METRIC_DESIRED_DIR = {
    "airborne":     "stable",    # We want ~30-35%, not too high or low
    "boost":        "up",        # Higher is better (was critically low)
    "speed":        "up",        # Faster = better
    "team_dist":    "stable",    # 2500-3500 ideal range
    "goal_diff":    "up",        # Winning more
    "touches":      "up",        # More engagement
    "demos":        "up",        # More aggression
    "aerial_touch": "up",        # Better connection rate
    "entropy":      "down",      # More decisive policy
    "clip":         "stable",    # 0.01-0.03 healthy
    "kl":           "stable",    # Low = stable updates
    "vf_loss":      "stable",    # Lower = better value estimates
    # Enhanced metrics
    "zero_boost":   "down",      # Less time with no boost = better boost management
    "supersonic":   "up",        # More supersonic time = better speed utilization
    "own_half":     "stable",    # ~50% ideal, too high = passive, too low = overcommitting
    "near_wall":    "stable",    # Some wall play is good, too much = hugging walls
    "has_flip":     "stable",    # Flip economy — using flips but not wasting them
    "boosting":     "stable",    # Active boost usage — not over/under-boosting
    "ball_dist":    "stable",    # Balanced — not ball-chasing, not too passive
    "ball_speed":   "up",        # Higher ball speed = more dynamic play
    "dbl_commit":   "down",      # Less double-committing = better rotation
    "boost_coll":   "up",        # More pad collection = better boost acquisition
    "touch_speed":  "up",        # Harder hits = more effective touches
    "aerial_seq":   "up",        # Longer aerial sequences = better aerial technique
}

# Healthy ranges for stability checks
HEALTHY_RANGES = {
    "clip":       (0.005, 0.050),
    "kl":         (0.0, 0.01),
    "airborne":   (0.20, 0.45),
    "boost":      (5.0, 30.0),
    "entropy":    (2.5, 4.2),
    # Enhanced metrics
    "zero_boost": (0.0, 0.30),    # If >30% of agents have zero boost, starvation
    "own_half":   (0.35, 0.65),   # Balanced rotation between halves
    "dbl_commit": (0.0, 0.25),    # More than 25% double-commits is problematic
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_wandb_data(samples_game=1500, samples_ppo=2000, samples_steps=500):
    """Load all relevant data from WandB API."""
    import wandb
    api = wandb.Api()
    run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{WANDB_RUN_ID}")

    print("  Loading WandB data...")

    # Current summary
    summary = dict(run.summary)

    # Step interpolation data
    step_rows = list(run.history(
        keys=["Cumulative Timesteps"],
        samples=samples_steps,
        pandas=False
    ))
    step_rows = [r for r in step_rows if r.get("Cumulative Timesteps") is not None]
    step_pairs = sorted([
        (r["_step"], int(r["Cumulative Timesteps"]))
        for r in step_rows
    ])

    # Game metrics — split into core (always available) and enhanced (v3.10+ only)
    # Querying WandB with keys that don't exist yet causes empty results,
    # so we query core keys first, then try enhanced separately and merge.
    core_game_keys = [
        "vyrex/airborne_frac", "vyrex/avg_boost", "vyrex/avg_speed",
        "vyrex/avg_teammate_dist", "vyrex/goal_diff_blue",
        "vyrex/touches_per_step", "vyrex/demos_per_step", "vyrex/aerial_touch_rate",
    ]
    enhanced_game_keys = [v for v in GAME_METRICS.values() if v not in core_game_keys]

    game_rows = list(run.history(keys=core_game_keys, samples=samples_game, pandas=False))
    game_rows = [r for r in game_rows if any(r.get(k) is not None for k in core_game_keys)]

    # Try to merge enhanced metrics if they exist in WandB
    if enhanced_game_keys:
        try:
            enh_rows = list(run.history(keys=enhanced_game_keys, samples=samples_game, pandas=False))
            enh_map = {r["_step"]: r for r in enh_rows if r.get("_step") is not None
                       and any(r.get(k) is not None for k in enhanced_game_keys)}
            if enh_map:
                for r in game_rows:
                    ext = enh_map.get(r["_step"])
                    if ext:
                        for ek in enhanced_game_keys:
                            if ek in ext and ext[ek] is not None:
                                r[ek] = ext[ek]
                print(f"  Enhanced metrics: {len(enh_map)} rows merged")
            else:
                print(f"  Enhanced metrics: not yet available in WandB (deploy new logger first)")
        except Exception:
            print(f"  Enhanced metrics: query failed (not yet logged)")

    # PPO metrics — query core pair first; they log on every update step
    ppo_core_keys = ["Policy Entropy", "SB3 Clip Fraction"]
    ppo_extra_keys = [k for k in PPO_METRICS.values() if k not in ppo_core_keys]
    ppo_all_keys = ppo_core_keys + ppo_extra_keys

    ppo_rows = list(run.history(keys=ppo_core_keys, samples=samples_ppo, pandas=False))
    ppo_rows = [r for r in ppo_rows if any(r.get(k) is not None for k in ppo_core_keys)]

    # Merge in extra metrics if available (they may be on same _step)
    if ppo_rows:
        extra_rows = list(run.history(keys=ppo_extra_keys, samples=samples_ppo, pandas=False))
        extra_map = {r["_step"]: r for r in extra_rows if r.get("_step") is not None}
        for r in ppo_rows:
            ext = extra_map.get(r["_step"])
            if ext:
                for ek in ppo_extra_keys:
                    if ek in ext and ext[ek] is not None:
                        r[ek] = ext[ek]

    # Interpolate estimated steps for all rows
    def interp(step):
        if not step_pairs:
            return step
        if step <= step_pairs[0][0]:
            return step_pairs[0][1]
        if step >= step_pairs[-1][0]:
            return step_pairs[-1][1]
        for i in range(len(step_pairs) - 1):
            if step_pairs[i][0] <= step <= step_pairs[i + 1][0]:
                f = (step - step_pairs[i][0]) / (step_pairs[i + 1][0] - step_pairs[i][0])
                return int(step_pairs[i][1] + f * (step_pairs[i + 1][1] - step_pairs[i][1]))
        return step

    for r in game_rows:
        r["_es"] = interp(r["_step"])
    for r in ppo_rows:
        r["_es"] = interp(r["_step"])

    print(f"  Loaded: {len(game_rows)} game rows, {len(ppo_rows)} PPO rows, "
          f"{len(step_pairs)} step anchors")
    print(f"  Step range: {step_pairs[0][1]/1e9:.3f}B to {step_pairs[-1][1]/1e9:.3f}B")

    return {
        "summary": summary,
        "game_rows": game_rows,
        "ppo_rows": ppo_rows,
        "step_pairs": step_pairs,
    }


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def bucket_data(rows, bucket_size, min_step=0, max_step=None):
    """Group rows into step-based buckets."""
    buckets = defaultdict(list)
    for r in rows:
        es = r.get("_es", 0)
        if es < min_step:
            continue
        if max_step and es > max_step:
            continue
        b = (es // bucket_size) * bucket_size
        buckets[b].append(r)
    return dict(sorted(buckets.items()))


def compute_stats(values):
    """Compute comprehensive statistics for a list of values."""
    if not values:
        return {"n": 0, "mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
    arr = np.array(values)
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def extract_metric(rows, wandb_key):
    """Extract non-None values of a metric from rows."""
    return [r[wandb_key] for r in rows if r.get(wandb_key) is not None]


def detect_plateau(values, steps, window=5, threshold=0.005):
    """
    Detect if a metric has plateaued.
    Returns (is_plateau, plateau_duration_steps, plateau_level, slope).
    """
    if len(values) < window * 2:
        return False, 0, 0, 0

    # Compute rolling average
    recent = values[-window:]
    earlier = values[-window * 2:-window]

    recent_mean = np.mean(recent)
    earlier_mean = np.mean(earlier)
    recent_std = np.std(recent)

    # Slope: change per unit
    delta = recent_mean - earlier_mean
    rel_change = abs(delta) / (abs(earlier_mean) + 1e-10)

    # Plateau if relative change is below threshold AND variance is low
    is_plateau = rel_change < threshold and recent_std < abs(recent_mean) * 0.02

    # Find how far back the plateau extends
    plateau_start = len(values) - 1
    for i in range(len(values) - 1, 0, -1):
        if abs(values[i] - recent_mean) > recent_mean * 0.015:
            plateau_start = i + 1
            break

    plateau_duration = 0
    if is_plateau and plateau_start < len(values) and len(steps) == len(values):
        plateau_duration = steps[-1] - steps[plateau_start]

    return is_plateau, plateau_duration, float(recent_mean), float(delta)


def compute_trend(bucket_means, n_recent=3):
    """
    Compute trend direction and strength from bucketed means.
    Returns: (direction, slope, r_squared)
    direction: 'rising', 'falling', 'stable', 'volatile'
    """
    if len(bucket_means) < 3:
        return "insufficient_data", 0, 0

    x = np.arange(len(bucket_means))
    y = np.array(bucket_means)

    # Linear regression
    coeffs = np.polyfit(x, y, 1)
    slope = coeffs[0]

    # R-squared
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Recent trend (last n buckets)
    if len(bucket_means) >= n_recent:
        recent = bucket_means[-n_recent:]
        recent_slope = (recent[-1] - recent[0]) / (n_recent - 1) if n_recent > 1 else 0
    else:
        recent_slope = slope

    # Classify
    rel_slope = abs(slope) / (abs(np.mean(y)) + 1e-10)
    if r_squared < 0.3 and rel_slope > 0.005:
        direction = "volatile"
    elif rel_slope < 0.002:
        direction = "stable"
    elif slope > 0:
        direction = "rising"
    else:
        direction = "falling"

    return direction, float(slope), float(r_squared)


def assess_health(metric_name, value, trend_dir):
    """Assess if a metric is healthy, concerning, or critical."""
    if metric_name in HEALTHY_RANGES:
        low, high = HEALTHY_RANGES[metric_name]
        if value < low or value > high:
            return "CRITICAL"

    desired = METRIC_DESIRED_DIR.get(metric_name, "stable")
    if desired == "up" and trend_dir == "falling":
        return "CONCERNING"
    if desired == "down" and trend_dir == "rising":
        return "CONCERNING"
    if desired == "stable" and trend_dir == "volatile":
        return "CONCERNING"

    return "OK"


# ============================================================================
# REPORT GENERATORS
# ============================================================================

def print_header(title):
    """Print a formatted section header."""
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def print_subheader(title):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def report_snapshot(data):
    """Current state snapshot from WandB summary."""
    print_header("CURRENT SNAPSHOT")
    s = data["summary"]

    steps = s.get("Cumulative Timesteps", 0)
    ent = s.get("Policy Entropy", 0)
    ent_pct = (ent / MAX_ENTROPY) * 100

    print(f"  Steps:         {steps:>14,} ({steps/1e9:.3f}B)")
    print(f"  Entropy:       {ent:>14.4f} ({ent_pct:.1f}% of max {MAX_ENTROPY:.3f})")
    print(f"  Clip Fraction: {s.get('SB3 Clip Fraction', 0):>14.5f}")
    print()

    # Game metrics
    metric_labels = {
        "vyrex/airborne_frac": ("Airborne",      "{:.1%}"),
        "vyrex/avg_boost":     ("Avg Boost",      "{:.1f}"),
        "vyrex/avg_speed":     ("Avg Speed",      "{:.1f} uu/s"),
        "vyrex/avg_teammate_dist": ("Team Dist",  "{:.0f} uu"),
        "vyrex/goal_diff_blue":("Goal Diff",      "{:+.0f}"),
        "vyrex/touches_per_step":("Touches/Step", "{:.5f}"),
        "vyrex/demos_per_step":("Demos/Step",     "{:.6f}"),
        "vyrex/aerial_touch_rate":("Aerial Touch", "{:.4f}"),
    }
    for key, (label, fmt) in metric_labels.items():
        val = s.get(key)
        if val is not None:
            print(f"  {label:<16} {fmt.format(val)}")


def report_entropy_detail(data, since=None):
    """Detailed entropy trajectory analysis with plateau detection."""
    print_header("ENTROPY ANALYSIS")

    ppo_rows = data["ppo_rows"]
    min_step = since or 0

    # Fine-grained buckets (25M)
    buckets_25m = bucket_data(ppo_rows, 25_000_000, min_step=min_step)
    bucket_ents = {}
    for b, pts in buckets_25m.items():
        vals = extract_metric(pts, "Policy Entropy")
        if vals:
            bucket_ents[b] = compute_stats(vals)

    if not bucket_ents:
        print("  No entropy data in range.")
        return

    # Print trajectory
    print_subheader("Trajectory (25M bins)")
    print(f"  {'Bucket':>8} | {'Mean':>7} | {'Min':>7} | {'Max':>7} | {'Std':>6} | {'%Max':>5} | N")
    print(f"  {'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*5}-+---")
    for b in sorted(bucket_ents.keys()):
        s = bucket_ents[b]
        pct = (s["mean"] / MAX_ENTROPY) * 100
        print(f"  {b/1e9:>7.3f}B | {s['mean']:>7.4f} | {s['min']:>7.4f} | "
              f"{s['max']:>7.4f} | {s['std']:>6.4f} | {pct:>4.1f}% | {s['n']}")

    # Plateau detection
    print_subheader("Plateau Detection")
    sorted_buckets = sorted(bucket_ents.keys())
    means = [bucket_ents[b]["mean"] for b in sorted_buckets]
    steps = [b for b in sorted_buckets]

    is_plateau, duration, level, delta = detect_plateau(means, steps)
    if is_plateau:
        print(f"  STATUS:   PLATEAU DETECTED")
        print(f"  Level:    {level:.4f} ({level/MAX_ENTROPY*100:.1f}%)")
        print(f"  Duration: {duration/1e6:.0f}M steps ({duration/1e9:.2f}B)")
        print(f"  Delta:    {delta:.4f} per bucket")
    else:
        print(f"  STATUS:   DESCENDING")
        print(f"  Current:  {means[-1]:.4f} ({means[-1]/MAX_ENTROPY*100:.1f}%)")
        print(f"  Trend:    {delta:+.4f} per bucket")

    # Rate of descent analysis
    print_subheader("Descent Rate by Phase")
    if len(sorted_buckets) >= 6:
        third = len(sorted_buckets) // 3
        phase1_rate = (means[third] - means[0]) / third if third > 0 else 0
        phase2_rate = (means[2*third] - means[third]) / third if third > 0 else 0
        phase3_rate = (means[-1] - means[2*third]) / (len(means) - 2*third) if (len(means) - 2*third) > 0 else 0
        print(f"  Early:  {phase1_rate:+.4f}/bin  ({sorted_buckets[0]/1e9:.2f}B - {sorted_buckets[third]/1e9:.2f}B)")
        print(f"  Mid:    {phase2_rate:+.4f}/bin  ({sorted_buckets[third]/1e9:.2f}B - {sorted_buckets[2*third]/1e9:.2f}B)")
        print(f"  Late:   {phase3_rate:+.4f}/bin  ({sorted_buckets[2*third]/1e9:.2f}B - {sorted_buckets[-1]/1e9:.2f}B)")


def report_game_metrics(data, since=None, bucket_size=100_000_000):
    """Game metrics trajectory with trend analysis."""
    print_header("GAME METRICS TRAJECTORY")

    game_rows = data["game_rows"]
    min_step = since or 0
    buckets = bucket_data(game_rows, bucket_size, min_step=min_step)

    if not buckets:
        print("  No game data in range.")
        return

    # Compute per-bucket stats for each metric
    metric_trajectories = {}
    for name, key in GAME_METRICS.items():
        trajectory = []
        for b in sorted(buckets.keys()):
            vals = extract_metric(buckets[b], key)
            if vals:
                trajectory.append((b, compute_stats(vals)))
        metric_trajectories[name] = trajectory

    # Print table — Core metrics
    sorted_bs = sorted(buckets.keys())
    print(f"\n  {'Bucket':>7} | {'Air%':>6} | {'Boost':>5} | {'Speed':>6} | "
          f"{'TDist':>5} | {'GDiff':>5} | {'Touch':>7} | {'Demo':>8} | {'AerTch':>6} | N")
    print(f"  {'-'*7}-+-{'-'*6}-+-{'-'*5}-+-{'-'*6}-+-"
          f"{'-'*5}-+-{'-'*5}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}-+---")

    for b in sorted_bs:
        pts = buckets[b]
        n = len(pts)
        def a(k):
            vals = [r.get(k) for r in pts if r.get(k) is not None]
            return sum(vals) / len(vals) if vals else 0

        print(f"  {b/1e9:>6.1f}B | {a(GAME_METRICS['airborne'])*100:>5.1f}% | "
              f"{a(GAME_METRICS['boost']):>5.1f} | {a(GAME_METRICS['speed']):>6.1f} | "
              f"{a(GAME_METRICS['team_dist']):>5.0f} | {a(GAME_METRICS['goal_diff']):>+5.1f} | "
              f"{a(GAME_METRICS['touches']):>.5f} | {a(GAME_METRICS['demos']):>.6f} | "
              f"{a(GAME_METRICS['aerial_touch']):>.4f} | {n}")

    # Print table — Enhanced metrics (only printed if data exists)
    has_enhanced = False
    for b in sorted_bs:
        pts = buckets[b]
        vals = [r.get(GAME_METRICS.get("zero_boost", "")) for r in pts if r.get(GAME_METRICS.get("zero_boost", "")) is not None]
        if vals:
            has_enhanced = True
            break

    if has_enhanced:
        print(f"\n  {'Bucket':>7} | {'0Bst%':>6} | {'SSon%':>6} | {'OwnH%':>6} | "
              f"{'Wall%':>6} | {'DblCm':>5} | {'BallD':>6} | {'BSpd':>6} | {'TchSpd':>6} | {'BColl':>6}")
        print(f"  {'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-"
              f"{'-'*6}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

        for b in sorted_bs:
            pts = buckets[b]
            def a(k):
                vals = [r.get(k) for r in pts if r.get(k) is not None]
                return sum(vals) / len(vals) if vals else 0

            print(f"  {b/1e9:>6.1f}B | {a(GAME_METRICS['zero_boost'])*100:>5.1f}% | "
                  f"{a(GAME_METRICS['supersonic'])*100:>5.1f}% | "
                  f"{a(GAME_METRICS['own_half'])*100:>5.1f}% | "
                  f"{a(GAME_METRICS['near_wall'])*100:>5.1f}% | "
                  f"{a(GAME_METRICS['dbl_commit']):>5.3f} | "
                  f"{a(GAME_METRICS['ball_dist']):>6.0f} | "
                  f"{a(GAME_METRICS['ball_speed']):>6.0f} | "
                  f"{a(GAME_METRICS['touch_speed']):>6.0f} | "
                  f"{a(GAME_METRICS['boost_coll']):>6.3f}")

    # Trend analysis
    print_subheader("Trend Analysis")
    for name, trajectory in metric_trajectories.items():
        if len(trajectory) < 3:
            continue
        means = [s["mean"] for _, s in trajectory]
        direction, slope, r2 = compute_trend(means)
        current = means[-1]
        health = assess_health(name, current, direction)
        status_icon = {"OK": "[OK]", "CONCERNING": "[!!]", "CRITICAL": "[XX]"}.get(health, "[??]")
        print(f"  {status_icon} {name:<14} {direction:<10} (slope={slope:+.4f}, R2={r2:.2f}, current={current:.4f})")


def report_ppo_health(data, since=None):
    """PPO optimization health diagnostics."""
    print_header("PPO HEALTH DIAGNOSTICS")

    ppo_rows = data["ppo_rows"]
    min_step = since or 0
    buckets = bucket_data(ppo_rows, 50_000_000, min_step=min_step)

    if not buckets:
        print("  No PPO data in range.")
        return

    sorted_bs = sorted(buckets.keys())

    print(f"\n  {'Bucket':>7} | {'Entropy':>7} | {'Clip':>7} | {'KL':>9} | {'VFLoss':>8} | N")
    print(f"  {'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*9}-+-{'-'*8}-+---")

    for b in sorted_bs:
        pts = buckets[b]
        n = len(pts)
        def a(k):
            vals = [r.get(k) for r in pts if r.get(k) is not None]
            return sum(vals) / len(vals) if vals else 0

        ent = a(PPO_METRICS["entropy"])
        clip = a(PPO_METRICS["clip"])
        kl = a(PPO_METRICS["kl"])
        vf = a(PPO_METRICS["vf_loss"])
        print(f"  {b/1e9:>6.2f}B | {ent:>7.4f} | {clip:>7.5f} | {kl:>9.6f} | {vf:>8.6f} | {n}")

    # Health assessment
    print_subheader("Health Assessment")
    last_bucket = buckets[sorted_bs[-1]]

    for metric_name, ppo_key in [("clip", "SB3 Clip Fraction"), ("entropy", "Policy Entropy")]:
        vals = extract_metric(last_bucket, ppo_key)
        if vals:
            current = np.mean(vals)
            if metric_name in HEALTHY_RANGES:
                low, high = HEALTHY_RANGES[metric_name]
                in_range = low <= current <= high
                status = "HEALTHY" if in_range else "OUT OF RANGE"
                print(f"  {metric_name:<10}: {current:.5f}  [{low}-{high}]  {status}")


def report_version_comparison(data, versions=None):
    """Compare metrics across version boundaries."""
    print_header("VERSION COMPARISON")

    if versions is None:
        versions = list(VERSIONS.keys())

    game_rows = data["game_rows"]
    ppo_rows = data["ppo_rows"]

    version_stats = {}
    for ver in versions:
        if ver not in VERSIONS:
            continue
        start, end = VERSIONS[ver]
        end = end or 999_999_999_999

        v_game = [r for r in game_rows if start <= r["_es"] < end]
        v_ppo = [r for r in ppo_rows if start <= r["_es"] < end]

        stats = {"n_game": len(v_game), "n_ppo": len(v_ppo)}
        for name, key in GAME_METRICS.items():
            vals = extract_metric(v_game, key)
            if vals:
                stats[name] = compute_stats(vals)
        for name, key in PPO_METRICS.items():
            vals = extract_metric(v_ppo, key)
            if vals:
                stats[name] = compute_stats(vals)

        version_stats[ver] = stats

    # Print comparison table
    metric_display = [
        ("entropy",      "Entropy",      "{:.4f}"),
        ("airborne",     "Airborne%",    "{:.1%}"),
        ("boost",        "Boost",        "{:.1f}"),
        ("speed",        "Speed",        "{:.1f}"),
        ("team_dist",    "TeamDist",     "{:.0f}"),
        ("goal_diff",    "GoalDiff",     "{:+.1f}"),
        ("touches",      "Touches",      "{:.5f}"),
        ("demos",        "Demos",        "{:.6f}"),
        ("aerial_touch", "AerialTouch",  "{:.4f}"),
        ("clip",         "Clip",         "{:.5f}"),
        # Enhanced metrics (will show N/A for pre-v3.10 versions)
        ("zero_boost",   "ZeroBoost%",   "{:.1%}"),
        ("supersonic",   "Supersonic%",  "{:.1%}"),
        ("own_half",     "OwnHalf%",     "{:.1%}"),
        ("dbl_commit",   "DblCommit",    "{:.3f}"),
        ("ball_dist",    "BallDist",     "{:.0f}"),
        ("touch_speed",  "TouchSpeed",   "{:.0f}"),
        ("boost_coll",   "BoostColl",    "{:.3f}"),
    ]

    header = f"  {'Metric':<14}"
    for ver in versions:
        if ver in version_stats:
            header += f" | {ver:>12}"
    print(header)
    print(f"  {'-'*14}" + "".join(f"-+-{'-'*12}" for ver in versions if ver in version_stats))

    for metric_name, label, fmt in metric_display:
        row = f"  {label:<14}"
        for ver in versions:
            if ver not in version_stats:
                continue
            stats = version_stats[ver]
            if metric_name in stats:
                val = stats[metric_name]["mean"]
                row += f" | {fmt.format(val):>12}"
            else:
                row += f" | {'N/A':>12}"
        print(row)

    # Sample counts
    row = f"  {'N (game)':14}"
    for ver in versions:
        if ver in version_stats:
            row += f" | {version_stats[ver]['n_game']:>12}"
    print(row)


def report_match(match_path):
    """Analyze a match JSON file."""
    print_header(f"MATCH ANALYSIS: {os.path.basename(match_path)}")

    with open(match_path) as f:
        data = json.load(f)

    # Match overview
    score = data.get("final_score", {})
    duration = data.get("match_duration_sec", 0)
    ticks = data.get("total_ticks", 0)
    print(f"\n  Score:    Blue {score.get('blue', 0)} - {score.get('orange', 0)} Orange")
    print(f"  Duration: {duration:.0f}s ({duration/60:.1f} min)")
    print(f"  Ticks:    {ticks:,}")

    # Scoreboard
    print_subheader("Scoreboard")
    print(f"  {'Name':<16} {'Team':<6} {'G':>2} {'A':>2} {'Sv':>3} {'Sh':>3} {'Dm':>3} {'Score':>5}")
    for p in data.get("scoreboard", []):
        print(f"  {p['name']:<16} {p['team']:<6} {p['goals']:>2} {p['assists']:>2} "
              f"{p['saves']:>3} {p['shots']:>3} {p['demos']:>3} {p['score']:>5}")

    # Bot analysis
    for bot_key, bot_data in data.get("bot_analysis", {}).items():
        print_subheader(f"Bot: {bot_key}")

        dq = bot_data.get("decision_quality", {})
        conf = dq.get("avg_confidence", 0)
        ent = dq.get("avg_entropy", 0)
        low_conf = dq.get("low_confidence_pct", 0)
        print(f"  Confidence:    {conf:.2%} (low-conf ticks: {low_conf:.1f}%)")
        print(f"  Match Entropy: {ent:.3f} ({ent/MAX_ENTROPY*100:.1f}%)")
        print(f"  Actions Used:  {dq.get('unique_actions_used', 0)}/{MAX_ACTIONS}")

        # Top actions
        top5 = dq.get("top_5_actions", [])
        if top5:
            total_ticks = bot_data.get("total_ticks", 1)
            print(f"  Top 5 Actions: ", end="")
            parts = []
            for action_id, count in top5:
                parts.append(f"#{action_id}({count/total_ticks*100:.1f}%)")
            print(", ".join(parts))

        # Behavioral state
        bs = bot_data.get("behavioral_state", {})
        print(f"  States:        ATK={bs.get('ATK',0):.1f}% ROT={bs.get('ROT',0):.1f}% "
              f"DEF={bs.get('DEF',0):.1f}% CHASE={bs.get('BALL_CHASE',0):.1f}%")

        print(f"  Double Commit: {bot_data.get('double_commit_pct', 0):.1f}%")

        # Touches
        t = bot_data.get("touches", {})
        print(f"  Touches:       {t.get('total',0)} total, {t.get('directed_pct',0):.1f}% directed, "
              f"avg speed {t.get('avg_touch_speed',0):.0f}")

        # Mechanical
        m = bot_data.get("mechanical", {})
        flip_eff = (m.get('flips',0) / max(m.get('jumps',1), 1)) * 100
        print(f"  Mechanical:    {m.get('airborne_pct',0):.1f}% air, {m.get('supersonic_pct',0):.1f}% supersonic, "
              f"{m.get('flips',0)} flips/{m.get('jumps',0)} jumps ({flip_eff:.0f}% efficiency)")

        # Boost
        b = bot_data.get("boost_mgmt", {})
        print(f"  Boost:         avg={b.get('avg_boost',0):.1f}, zero={b.get('zero_boost_pct',0):.1f}%, "
              f"waste={b.get('waste_pct',0):.1f}%")

        # Positioning
        p = bot_data.get("positioning", {})
        print(f"  Positioning:   ball_dist={p.get('avg_dist_to_ball',0):.0f}, "
              f"tm_dist={p.get('avg_dist_to_teammate',0):.0f}, "
              f"own_goal_dist={p.get('avg_dist_to_own_goal',0):.0f}")

        # Defense
        d = bot_data.get("defense", {})
        print(f"  Defense:       {d.get('def_third_pct',0):.1f}% def third, "
              f"{d.get('clears_attempted',0)} clears")

    # Event timeline analysis
    events = data.get("event_timeline", [])
    if events:
        print_subheader("Event Analysis")
        goals = [e for e in events if e["event"] == "GOAL"]
        touches = [e for e in events if e["event"] == "TOUCH"]
        clears = [e for e in events if e["event"] == "CLEAR"]

        toward = [e for e in touches if e.get("direction") == "TOWARD"]
        away = [e for e in touches if e.get("direction") == "AWAY"]

        print(f"  Timeline events: {len(events)} total ({len(goals)} goals, "
              f"{len(touches)} touches, {len(clears)} clears)")
        if touches:
            print(f"  Touch direction: {len(toward)} toward ({len(toward)/len(touches)*100:.1f}%), "
                  f"{len(away)} away ({len(away)/len(touches)*100:.1f}%)")

        # Power shots (>1800 uu/s toward)
        power = [e for e in toward if e.get("speed", 0) > 1800]
        print(f"  Power shots:   {len(power)} touches >1800 uu/s toward goal")
        if power:
            speeds = [e["speed"] for e in power]
            print(f"                 avg={np.mean(speeds):.0f}, max={max(speeds):.0f}")

        # Aerial touches (z > 300)
        aerial = [e for e in touches if e.get("ball_z", 0) > 300]
        print(f"  Aerial touches: {len(aerial)} (ball_z > 300)")

        # Stuck detection (10+ identical consecutive touches)
        stuck_sequences = []
        i = 0
        while i < len(touches) - 1:
            j = i + 1
            while j < len(touches) and touches[j].get("speed") == touches[i].get("speed") and \
                  touches[j].get("ball_z") == touches[i].get("ball_z"):
                j += 1
            if j - i >= 10:
                stuck_sequences.append((i, j, j - i, touches[i].get("speed"), touches[i].get("ball_z")))
            i = j if j > i + 1 else i + 1

        if stuck_sequences:
            print(f"  STUCK SEQUENCES: {len(stuck_sequences)} detected!")
            for idx, (start, end, count, spd, z) in enumerate(stuck_sequences):
                print(f"    #{idx+1}: {count} identical touches (speed={spd}, z={z}, "
                      f"ticks {touches[start]['tick']}-{touches[end-1]['tick']})")
        else:
            print(f"  Stuck sequences: None detected")


def report_recommendations(data, since=None):
    """Generate actionable recommendations based on current analysis."""
    print_header("STRATEGIC RECOMMENDATIONS")

    s = data["summary"]
    ppo_rows = data["ppo_rows"]
    game_rows = data["game_rows"]

    steps = s.get("Cumulative Timesteps", 0)
    ent = s.get("Policy Entropy", 0)
    clip = s.get("SB3 Clip Fraction", 0)
    boost = s.get("vyrex/avg_boost", 0)
    aerial = s.get("vyrex/aerial_touch_rate", 0)

    recommendations = []

    # Entropy plateau check — use bucketed means for robustness
    ent_buckets = bucket_data(ppo_rows, 50_000_000, min_step=max(0, steps - 600_000_000))
    if ent_buckets:
        sorted_ebs = sorted(ent_buckets.keys())
        bucket_means = []
        for b in sorted_ebs:
            vals = extract_metric(ent_buckets[b], "Policy Entropy")
            if vals:
                bucket_means.append(np.mean(vals))
        if len(bucket_means) >= 4:
            # Check last half of range for plateau
            half = len(bucket_means) // 2
            recent_means = bucket_means[half:]
            ent_range = max(recent_means) - min(recent_means)
            ent_level = np.mean(recent_means)
            plateau_duration = (sorted_ebs[-1] - sorted_ebs[half]) if half < len(sorted_ebs) else 0
            if ent_range < 0.04:  # bucket means within 0.04 = plateau
                recommendations.append({
                    "priority": "HIGH",
                    "area": "Entropy",
                    "finding": f"Plateau at {ent_level:.4f} (bucket range {ent_range:.4f}) for ~{plateau_duration/1e6:.0f}M steps",
                    "action": "Reduce ent_coef by 20-25% to break through",
                    "risk": "Too aggressive reduction can cause premature convergence"
                })

    # Boost check
    if boost < 10:
        recommendations.append({
            "priority": "MEDIUM",
            "area": "Boost",
            "finding": f"Avg boost at {boost:.1f} — still below target of 12+",
            "action": "Monitor. Boost has been slowly improving with entropy descent.",
            "risk": "Increasing boost_change_weight further may cause reward hacking"
        })

    # Clip fraction check
    if clip > 0.04:
        recommendations.append({
            "priority": "HIGH",
            "area": "Stability",
            "finding": f"Clip fraction at {clip:.4f} — approaching danger zone (>0.05)",
            "action": "Consider reducing learning rate or increasing batch size",
            "risk": "Policy instability, catastrophic forgetting"
        })
    elif clip < 0.008:
        recommendations.append({
            "priority": "LOW",
            "area": "Learning",
            "finding": f"Clip fraction at {clip:.4f} — very low, updates may be too small",
            "action": "Learning rate may need increase, or batch size decrease",
            "risk": "Slow learning"
        })

    # Print recommendations
    if not recommendations:
        print("\n  No immediate action items. All metrics within acceptable ranges.")
    else:
        for i, rec in enumerate(recommendations, 1):
            priority_marker = {"HIGH": "!!!", "MEDIUM": " !!", "LOW": "  !"}.get(rec["priority"], "   ")
            print(f"\n  [{priority_marker}] #{i}: {rec['area']}")
            print(f"      Finding: {rec['finding']}")
            print(f"      Action:  {rec['action']}")
            print(f"      Risk:    {rec['risk']}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VYREX Training Analyzer — Comprehensive WandB Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py                      Full analysis
  python analyze.py --since 4.5          Analysis from 4.5B
  python analyze.py --version v3.8       Version-specific analysis
  python analyze.py --compare v3.7 v3.8  Compare versions
  python analyze.py --match match.json   Analyze match file
  python analyze.py --all                Everything
        """
    )

    parser.add_argument("--since", type=float, default=None,
                        help="Start analysis from this step count in billions (e.g., 4.5)")
    parser.add_argument("--version", type=str, default=None,
                        help="Analyze specific version (e.g., v3.8)")
    parser.add_argument("--compare", nargs="+", default=None,
                        help="Compare versions (e.g., --compare v3.7 v3.8)")
    parser.add_argument("--health", action="store_true",
                        help="PPO health check only")
    parser.add_argument("--entropy-detail", action="store_true",
                        help="Detailed entropy analysis")
    parser.add_argument("--match", type=str, default=None,
                        help="Path to match JSON file for analysis")
    parser.add_argument("--all", action="store_true",
                        help="Run all analyses")
    parser.add_argument("--bucket-size", type=int, default=100,
                        help="Bucket size in millions for game metrics (default: 100)")
    parser.add_argument("--samples", type=int, default=2000,
                        help="WandB sample count (default: 2000)")

    args = parser.parse_args()

    # Determine analysis scope
    since_steps = None
    if args.since:
        since_steps = int(args.since * 1e9)
    elif args.version and args.version in VERSIONS:
        since_steps = VERSIONS[args.version][0]

    bucket_size = args.bucket_size * 1_000_000

    # Handle match-only analysis (no WandB needed)
    if args.match and not args.all:
        report_match(args.match)
        return

    # Load WandB data
    print("\n" + "=" * 72)
    print("  VYREX TRAINING ANALYZER")
    print("=" * 72)
    data = load_wandb_data(samples_ppo=args.samples)

    # Default: run full analysis if no specific flags
    run_all = args.all or not any([args.health, args.entropy_detail, args.compare, args.match])

    if run_all or not any([args.health, args.entropy_detail, args.compare]):
        report_snapshot(data)

    if run_all or args.entropy_detail:
        report_entropy_detail(data, since=since_steps)

    if run_all:
        report_game_metrics(data, since=since_steps, bucket_size=bucket_size)

    if run_all or args.health:
        report_ppo_health(data, since=since_steps)

    if run_all or args.compare:
        versions_to_compare = args.compare or list(VERSIONS.keys())
        report_version_comparison(data, versions=versions_to_compare)

    if args.match:
        report_match(args.match)

    if run_all:
        report_recommendations(data, since=since_steps)

    print(f"\n{'='*72}")
    print(f"  Analysis complete. Steps: {data['summary'].get('Cumulative Timesteps',0)/1e9:.3f}B")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
