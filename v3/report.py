#!/usr/bin/env python3
"""
VYREX — Diagnostic Report Generator
=====================================
Run this script to generate a comprehensive training report.
Copy the output and paste it to Claude for optimization advice.

Usage:
    python report.py                   # Read latest diagnostics
    python report.py --all             # Show all diagnostic history
    python report.py --export report.txt  # Export to file

The output is specifically formatted for LLM consumption — it includes
all the context Claude needs to give you specific, actionable advice
on how to improve VYREX's training.
"""

import os
import sys
import json
import glob
import argparse
from datetime import datetime


def find_diagnostics_dir():
    """Find the diagnostics output directory."""
    # Try standard location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    standard_path = os.path.join(script_dir, "data", "diagnostics")
    if os.path.exists(standard_path):
        return standard_path

    # Try config
    try:
        from config import DEFAULT_CONFIG
        return DEFAULT_CONFIG.diagnostics.diagnostics_output_dir
    except ImportError:
        pass

    return standard_path


def read_latest_report(diag_dir: str) -> dict:
    """Read the latest diagnostics report."""
    latest_path = os.path.join(diag_dir, "latest.json")
    if not os.path.exists(latest_path):
        return None
    with open(latest_path, "r") as f:
        return json.load(f)


def read_all_reports(diag_dir: str) -> list:
    """Read all diagnostics reports, sorted by step count."""
    reports = []
    for fpath in sorted(glob.glob(os.path.join(diag_dir, "diagnostics_*.json"))):
        with open(fpath, "r") as f:
            reports.append(json.load(f))
    return reports


def format_report_for_llm(report: dict) -> str:
    """Format a single report for LLM consumption."""
    lines = []
    lines.append("=" * 72)
    lines.append("VYREX TRAINING DIAGNOSTICS REPORT")
    lines.append(f"Generated at step: {report['total_steps']:,}")
    lines.append(f"Training time: {report['elapsed_hours']:.1f} hours")
    lines.append(f"Steps/second: {report['steps_per_second']:.1f}")
    lines.append("=" * 72)

    lines.append("\n--- PERFORMANCE ---")
    perf = report["performance"]
    lines.append(f"  Reward Mean:  {perf['reward_mean']:.4f}")
    lines.append(f"  Reward Std:   {perf['reward_std']:.4f}")
    lines.append(f"  Reward Range: [{perf['reward_min']:.4f}, {perf['reward_max']:.4f}]")

    lines.append("\n--- GAME METRICS ---")
    gm = report["game_metrics"]
    lines.append(f"  Goals/ep (Blue):   {gm['goals_per_ep_blue']:.3f}")
    lines.append(f"  Goals/ep (Orange): {gm['goals_per_ep_orange']:.3f}")
    lines.append(f"  Goal Differential: {gm['goal_differential']:.3f}")
    lines.append(f"  Touches/ep:        {gm['touches_per_ep']:.1f}")
    lines.append(f"  Demos/ep:          {gm['demos_per_ep']:.2f}")
    lines.append(f"  Avg Episode Length: {gm['ep_length_mean']:.0f} steps")

    lines.append("\n--- BEHAVIOR PROFILE ---")
    bp = report["behavior_profile"]
    lines.append(f"  Aerial Touch %:     {bp['aerial_touch_fraction']*100:.2f}%")
    lines.append(f"  Avg Boost Level:    {bp['avg_boost_level']*100:.1f}%")
    lines.append(f"  Avg Speed:          {bp['avg_speed']:.1f} uu/s")
    lines.append(f"  Time Airborne:      {bp['time_airborne_fraction']*100:.2f}%")

    lines.append("\n--- TRAINING SCHEDULE ---")
    ts = report["training_schedule"]
    lines.append(f"  Team Spirit:  {ts['current_team_spirit']:.4f} → {ts['target_team_spirit']}")
    lines.append(f"  Policy LR:    {ts['learning_rate_policy']}")
    lines.append(f"  Critic LR:    {ts['learning_rate_critic']}")
    lines.append(f"  Entropy Coef: {ts['entropy_coef']}")

    lines.append("\n--- CURRENT CONFIG ---")
    cc = report["current_config_summary"]
    lines.append(f"  Policy Network:  {cc['network_policy']}")
    lines.append(f"  Critic Network:  {cc['network_critic']}")
    lines.append(f"  Batch Size:      {cc['batch_size']:,}")
    lines.append(f"  Parallel Envs:   {cc['n_proc']}")
    lines.append(f"  PPO Epochs:      {cc['ppo_epochs']}")
    lines.append(f"  Team Size:       {cc['team_size']}v{cc['team_size']}")

    lines.append("\n--- INSTRUCTIONS FOR OPTIMIZER ---")
    lines.append(report.get("instructions_for_optimizer", ""))

    lines.append("\n" + "=" * 72)
    lines.append("END OF REPORT — Feed this to Claude for optimization advice.")
    lines.append("=" * 72)

    return "\n".join(lines)


def format_trend_report(reports: list) -> str:
    """Format a trend report showing progress over multiple checkpoints."""
    if not reports:
        return "No reports found."

    lines = []
    lines.append("=" * 72)
    lines.append("VYREX TRAINING TREND REPORT")
    lines.append(f"Reports: {len(reports)} snapshots")
    lines.append("=" * 72)

    lines.append(f"\n{'Steps':>14} | {'Reward':>10} | {'GoalDiff':>10} | "
                 f"{'Touches':>9} | {'Aerial%':>8} | {'TeamSprt':>8} | {'SPS':>8}")
    lines.append("-" * 85)

    for r in reports:
        steps = r["total_steps"]
        reward = r["performance"]["reward_mean"]
        goal_diff = r["game_metrics"]["goal_differential"]
        touches = r["game_metrics"]["touches_per_ep"]
        aerial = r["behavior_profile"]["aerial_touch_fraction"] * 100
        ts = r["training_schedule"]["current_team_spirit"]
        sps = r["steps_per_second"]

        lines.append(f"{steps:>14,} | {reward:>10.4f} | {goal_diff:>10.3f} | "
                     f"{touches:>9.1f} | {aerial:>7.2f}% | {ts:>8.4f} | {sps:>8.1f}")

    lines.append("\n" + "=" * 72)
    lines.append("Feed this trend data to Claude for trajectory analysis.")
    lines.append("=" * 72)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="VYREX Diagnostic Report Reader")
    parser.add_argument("--all", action="store_true", help="Show trend across all reports")
    parser.add_argument("--export", type=str, default=None, help="Export report to file")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    diag_dir = find_diagnostics_dir()

    if args.all:
        reports = read_all_reports(diag_dir)
        if not reports:
            print(f"No diagnostic reports found in: {diag_dir}")
            print("Start training first: python train.py")
            sys.exit(1)
        output = format_trend_report(reports)
        # Also append the latest full report
        output += "\n\n" + format_report_for_llm(reports[-1])
    else:
        report = read_latest_report(diag_dir)
        if report is None:
            print(f"No diagnostic reports found in: {diag_dir}")
            print("Start training first: python train.py")
            sys.exit(1)

        if args.json:
            output = json.dumps(report, indent=2)
        else:
            output = format_report_for_llm(report)

    if args.export:
        with open(args.export, "w") as f:
            f.write(output)
        print(f"Report exported to: {args.export}")
    else:
        print(output)


if __name__ == "__main__":
    main()
