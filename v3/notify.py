"""
VYREX - Discord Step Notification Monitor
==========================================
Standalone script that watches training checkpoints and sends a
Discord webhook notification when cumulative steps reach your targets.

Setup:
    1. Create a Discord webhook:
       Server Settings → Integrations → Webhooks → New Webhook → Copy URL

    2. Set the webhook URL (pick one):
       a. Environment variable:  set VYREX_DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
       b. Paste it directly into DISCORD_WEBHOOK_URL below

    3. Run alongside training:
       python notify.py 930000000 970000000 1000000000
       python notify.py 930M 970M 1B          # shorthand works too

Options:
    --poll      Seconds between checks (default: 30)
    --webhook   Pass webhook URL directly
    --once      Exit after the first notification fires
"""

import os
import sys
import time
import json
import argparse
import urllib.request
import urllib.error
from datetime import datetime, timedelta


# ============================================================================
# CONFIG — paste your webhook URL here if you don't want to use env var
# ============================================================================
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1472965335186538547/GCJM0z_5wD0jKB20KLL96QbQIByscjSXAACG2sKNr2eak_rdCxrd-0BiPEunahuos2wP"


def get_webhook_url(cli_url: str = None) -> str:
    """Resolve webhook URL from CLI arg, env var, or hardcoded constant."""
    url = cli_url or os.environ.get("VYREX_DISCORD_WEBHOOK") or DISCORD_WEBHOOK_URL
    if not url:
        print("ERROR: No Discord webhook URL configured.")
        print("  Option 1: set VYREX_DISCORD_WEBHOOK=https://discord.com/api/webhooks/...")
        print("  Option 2: python notify.py --webhook https://discord.com/api/webhooks/... 930M")
        print("  Option 3: Edit DISCORD_WEBHOOK_URL in notify.py")
        sys.exit(1)
    return url


def parse_step_value(s: str) -> int:
    """Parse human-friendly step strings: 930M, 1B, 1.5B, 500K, 900000000."""
    s = s.strip().upper().replace(",", "").replace("_", "")
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[:-1]) * mult)
    return int(s)


def fmt_steps(n: int) -> str:
    """Format step count for display: 930000000 → 930.0M"""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}K"
    return str(n)


# ============================================================================
# CHECKPOINT SCANNER
# ============================================================================

def get_latest_step(checkpoint_dir: str) -> int:
    """
    Find the highest step number across all checkpoint directories.
    Mirrors train.py's find_latest_checkpoint logic.
    """
    best = 0

    search_roots = []
    if os.path.exists(checkpoint_dir):
        search_roots.append(checkpoint_dir)

    # Also search sibling dirs matching checkpoints-* pattern
    parent = os.path.dirname(checkpoint_dir)
    base = os.path.basename(checkpoint_dir)
    if os.path.exists(parent):
        for entry in os.listdir(parent):
            if entry.startswith(base + "-") and os.path.isdir(os.path.join(parent, entry)):
                search_roots.append(os.path.join(parent, entry))

    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for entry in os.listdir(root):
            full_path = os.path.join(root, entry)
            if not os.path.isdir(full_path):
                continue
            # Direct step directory
            if os.path.isfile(os.path.join(full_path, "PPO_POLICY.pt")):
                try:
                    best = max(best, int(entry))
                except ValueError:
                    pass
            else:
                # Nested directory
                for sub in os.listdir(full_path):
                    sub_path = os.path.join(full_path, sub)
                    if os.path.isdir(sub_path) and \
                       os.path.isfile(os.path.join(sub_path, "PPO_POLICY.pt")):
                        try:
                            best = max(best, int(sub))
                        except ValueError:
                            pass
    return best


# ============================================================================
# DISCORD NOTIFICATION
# ============================================================================

def send_discord(webhook_url: str, title: str, message: str, color: int = 0x00FF88):
    """Send a rich embed to a Discord webhook. No dependencies needed."""
    payload = json.dumps({
        "embeds": [{
            "title": title,
            "description": message,
            "color": color,
            "footer": {"text": "VYREX Training Monitor"},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }]
    }).encode("utf-8")

    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "VYREX-Notify/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status in (200, 204):
                return True
    except urllib.error.HTTPError as e:
        print(f"  [!] Discord HTTP error: {e.code} {e.reason}")
    except Exception as e:
        print(f"  [!] Discord send failed: {e}")
    return False


# ============================================================================
# MONITOR LOOP
# ============================================================================

def monitor(checkpoint_dir: str, targets: list, webhook_url: str,
            poll_interval: float = 30.0, exit_on_first: bool = False):
    """
    Poll checkpoint directory and fire Discord notifications when
    step targets are reached.
    """
    # Sort targets ascending
    remaining = sorted(targets)
    start_step = get_latest_step(checkpoint_dir)

    print("=" * 55)
    print("  VYREX Step Notification Monitor")
    print("=" * 55)
    print(f"  Checkpoint dir : {checkpoint_dir}")
    print(f"  Current steps  : {fmt_steps(start_step)}")
    print(f"  Poll interval  : {poll_interval}s")
    print(f"  Targets        : {', '.join(fmt_steps(t) for t in remaining)}")
    print("=" * 55)
    print()

    # Remove targets already passed
    already_passed = [t for t in remaining if t <= start_step]
    if already_passed:
        print(f"  Skipping already-reached targets: {', '.join(fmt_steps(t) for t in already_passed)}")
        remaining = [t for t in remaining if t > start_step]

    if not remaining:
        print("  All targets already reached. Nothing to monitor.")
        return

    print(f"  Waiting for: {', '.join(fmt_steps(t) for t in remaining)}")
    print(f"  Press Ctrl+C to stop.\n")

    monitor_start = time.time()
    prev_step = start_step

    try:
        while remaining:
            time.sleep(poll_interval)
            current = get_latest_step(checkpoint_dir)

            if current > prev_step:
                elapsed = timedelta(seconds=int(time.time() - monitor_start))
                print(f"  [{elapsed}] Steps: {fmt_steps(current)}  "
                      f"(+{fmt_steps(current - prev_step)} since last check)  "
                      f"Next target: {fmt_steps(remaining[0])}")
                prev_step = current

            # Check targets
            fired = []
            for target in remaining:
                if current >= target:
                    # TARGET HIT
                    delta = current - start_step
                    wall_time = timedelta(seconds=int(time.time() - monitor_start))

                    title = f"🎯 VYREX reached {fmt_steps(target)} steps!"
                    msg = (
                        f"**Current steps:** {current:,}\n"
                        f"**Target:** {target:,} ({fmt_steps(target)})\n"
                        f"**Steps since monitor start:** {delta:,}\n"
                        f"**Wall time monitoring:** {wall_time}\n"
                        f"**Time:** {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}"
                    )

                    print(f"\n  >>> TARGET HIT: {fmt_steps(target)} "
                          f"(actual: {fmt_steps(current)}) <<<")

                    if send_discord(webhook_url, title, msg):
                        print(f"  >>> Discord notification sent!\n")
                    else:
                        print(f"  >>> Discord notification FAILED\n")

                    fired.append(target)

                    if exit_on_first:
                        remaining = []
                        break

            for t in fired:
                remaining.remove(t)

            if remaining:
                # Show countdown
                steps_to_go = remaining[0] - current
                if steps_to_go > 0 and current > start_step:
                    steps_per_sec = (current - start_step) / (time.time() - monitor_start)
                    if steps_per_sec > 0:
                        eta_sec = steps_to_go / steps_per_sec
                        eta = timedelta(seconds=int(eta_sec))
                    else:
                        eta = "?"
                else:
                    eta = "calculating..."

    except KeyboardInterrupt:
        print("\n  Monitor stopped by user.")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VYREX — Discord notification when training hits step targets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python notify.py 930M 970M 1B
  python notify.py --poll 60 930M         # Check every 60 seconds
  python notify.py --webhook https://discord.com/api/webhooks/... 1B
  python notify.py --once 930M            # Exit after first notification
        """,
    )
    parser.add_argument("targets", nargs="+",
                        help="Step targets (e.g. 930M, 1B, 500000000)")
    parser.add_argument("--poll", type=float, default=30,
                        help="Seconds between checkpoint checks (default: 30)")
    parser.add_argument("--webhook", type=str, default=None,
                        help="Discord webhook URL (or use VYREX_DISCORD_WEBHOOK env var)")
    parser.add_argument("--once", action="store_true",
                        help="Exit after the first target is hit")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Override checkpoint directory")

    args = parser.parse_args()

    webhook_url = get_webhook_url(args.webhook)
    targets = [parse_step_value(t) for t in args.targets]

    if args.checkpoint_dir:
        cp_dir = args.checkpoint_dir
    else:
        cp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "data", "checkpoints")

    monitor(cp_dir, targets, webhook_url,
            poll_interval=args.poll, exit_on_first=args.once)


if __name__ == "__main__":
    main()
