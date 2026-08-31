"""
VYREX v4 — Export trained policy for RLBot deployment.

Copies the latest actor.pt checkpoint to POLICY.pt in an RLBot bot directory.

Usage:
  python export_policy.py                          # Export to default test deploy
  python export_policy.py --dest path/to/bot/src   # Export to custom destination
  python export_policy.py --checkpoint <timestamp>  # Export a specific checkpoint
"""

import argparse
import json
import os
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_ROOT = os.path.join(PROJECT_ROOT, "agent_controllers_checkpoints", "PPO", "vyrex-v4-2v2")
DEFAULT_DEST = os.path.join(PROJECT_ROOT, "..", "bots", "rlbot_test_deploy", "src")


def find_latest_checkpoint():
    """Find the checkpoint with the highest cumulative_timesteps."""
    if not os.path.isdir(CHECKPOINT_ROOT):
        print(f"ERROR: No checkpoints found at {CHECKPOINT_ROOT}")
        sys.exit(1)

    best_ts = -1
    best_dir = None

    for entry in os.listdir(CHECKPOINT_ROOT):
        ckpt_dir = os.path.join(CHECKPOINT_ROOT, entry)
        agent_json = os.path.join(ckpt_dir, "ppo_agent.json")
        actor_pt = os.path.join(ckpt_dir, "ppo_learner", "actor.pt")

        if not os.path.isfile(agent_json) or not os.path.isfile(actor_pt):
            continue

        with open(agent_json, "r") as f:
            data = json.load(f)
        cum_ts = data.get("cumulative_timesteps", 0)

        if cum_ts > best_ts:
            best_ts = cum_ts
            best_dir = ckpt_dir

    if best_dir is None:
        print("ERROR: No valid checkpoints found (need ppo_agent.json + ppo_learner/actor.pt)")
        sys.exit(1)

    return best_dir, best_ts


def export(checkpoint_dir, dest_dir):
    """Copy actor.pt → POLICY.pt to the destination."""
    actor_path = os.path.join(checkpoint_dir, "ppo_learner", "actor.pt")
    dest_path = os.path.join(dest_dir, "POLICY.pt")

    if not os.path.isfile(actor_path):
        print(f"ERROR: actor.pt not found at {actor_path}")
        sys.exit(1)

    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(actor_path, dest_path)

    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"Exported: {dest_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export VYREX v4 policy for RLBot")
    parser.add_argument("--dest", type=str, default=DEFAULT_DEST,
                        help=f"Destination directory (default: {DEFAULT_DEST})")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Specific checkpoint timestamp folder to export")
    args = parser.parse_args()

    if args.checkpoint:
        ckpt_dir = os.path.join(CHECKPOINT_ROOT, args.checkpoint)
        if not os.path.isdir(ckpt_dir):
            print(f"ERROR: Checkpoint not found: {ckpt_dir}")
            sys.exit(1)
        agent_json = os.path.join(ckpt_dir, "ppo_agent.json")
        if os.path.isfile(agent_json):
            with open(agent_json) as f:
                cum_ts = json.load(f).get("cumulative_timesteps", 0)
        else:
            cum_ts = 0
    else:
        ckpt_dir, cum_ts = find_latest_checkpoint()

    print(f"[VYREX v4] Exporting policy...")
    print(f"  Checkpoint:  {ckpt_dir}")
    print(f"  Timesteps:   {cum_ts:,}")
    print(f"  Destination: {os.path.abspath(args.dest)}")
    print()

    export(ckpt_dir, args.dest)
    print("\nDone! Launch with: cd ../bots/rlbot_test_deploy && python launch_match.py")


if __name__ == "__main__":
    main()
