"""
VYREX Checkpoint Cleanup Tool
===============================
Intelligently thins out checkpoint directories to reclaim disk space.

Features:
  - Auto-detects checkpoint directories across all training versions
  - Reports size, count, step range, and density
  - Protects the latest checkpoint and version boundary checkpoints
  - Supports multiple thinning strategies:
      * Uniform: keep every Nth checkpoint
      * Tiered: denser near recent, sparser for old
      * Age-based: thin checkpoints older than X billion steps
  - Dry-run by default — requires explicit --execute to delete
  - Interactive confirmation before any deletion

Usage:
    python checkpoint_cleanup.py                          # Analyze & recommend
    python checkpoint_cleanup.py --strategy uniform --keep-every 3  # Preview keeping every 3rd
    python checkpoint_cleanup.py --strategy tiered        # Tiered thinning (recommended)
    python checkpoint_cleanup.py --strategy tiered --execute  # Actually delete
    python checkpoint_cleanup.py --dir data/v3/checkpoints_permanent  # Specific directory
"""

import os
import sys
import re
import shutil
import argparse
from dataclasses import dataclass, field
from typing import Optional

# ============================================================================
# Version boundaries — imported from analyze.py
# These step counts mark version transitions and MUST be protected.
# ============================================================================

VERSION_BOUNDARIES = {
    "v1.0":  (0, 251_000_000),
    "v2.0":  (251_000_000, 401_000_000),
    "v3.0":  (401_000_000, 433_000_000),
    "v3.1":  (433_000_000, 461_000_000),
    "v3.2":  (461_000_000, 521_000_000),
    "v3.3":  (521_000_000, 1_770_000_000),
    "v3.4":  (1_770_000_000, 2_375_000_000),
    "v3.5":  (2_375_000_000, 2_860_000_000),
    "v3.6":  (2_860_000_000, 3_130_000_000),
    "v3.7":  (3_130_000_000, 4_030_000_000),
    "v3.8":  (4_030_000_000, 5_005_000_000),
    "v3.9":  (5_005_000_000, 5_720_000_000),
    "v3.10": (5_720_000_000, 6_500_000_000),
    "v3.11": (6_500_000_000, 7_200_000_000),
    "v3.12": (7_200_000_000, 7_845_000_000),
    "v3.13": (7_845_000_000, 8_565_000_000),
    "v3.14": (8_565_000_000, 11_000_000_000),
    "v3.15": (11_000_000_000, 11_200_000_000),
    "v3.16": (11_200_000_000, 11_400_000_000),
    "v3.17": (11_400_000_000, 11_960_000_000),
    "v3.18": (11_960_000_000, 12_500_000_000),
    "v3.19": (12_500_000_000, None),
}


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class Checkpoint:
    """Represents a single checkpoint directory."""
    path: str               # Full filesystem path
    name: str               # Directory name (e.g., "3414400112" or "414m")
    step: Optional[int]     # Parsed step count (None if unparseable)
    size_bytes: int         # Total size of all files in the checkpoint
    is_protected: bool = False
    protection_reason: str = ""

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 * 1024 * 1024)

    @property
    def step_display(self) -> str:
        if self.step is None:
            return self.name
        if self.step >= 1_000_000_000:
            return f"{self.step / 1_000_000_000:.2f}B"
        elif self.step >= 1_000_000:
            return f"{self.step / 1_000_000:.0f}M"
        else:
            return f"{self.step:,}"


@dataclass
class CheckpointDir:
    """Represents a checkpoint directory root (e.g., data/v3/checkpoints_permanent)."""
    path: str
    checkpoints: list = field(default_factory=list)  # List[Checkpoint]

    @property
    def total_size_bytes(self) -> int:
        return sum(c.size_bytes for c in self.checkpoints)

    @property
    def total_size_gb(self) -> float:
        return self.total_size_bytes / (1024 * 1024 * 1024)

    @property
    def count(self) -> int:
        return len(self.checkpoints)

    @property
    def sorted_checkpoints(self) -> list:
        """Return checkpoints sorted by step count (unparseable ones first)."""
        none_steps = [c for c in self.checkpoints if c.step is None]
        valid_steps = sorted([c for c in self.checkpoints if c.step is not None],
                             key=lambda c: c.step)
        return none_steps + valid_steps


# ============================================================================
# Parsing & Discovery
# ============================================================================

def parse_step_from_name(name: str) -> Optional[int]:
    """
    Parse a step count from a checkpoint directory name.

    Handles:
      - Pure numeric: "3414400112" → 3414400112
      - Millions suffix: "414m" → 414000000
      - Billions suffix: "1.5b" → 1500000000
    """
    name = name.strip()

    # Pure numeric
    if name.isdigit():
        return int(name)

    # Millions: "414m", "414M"
    m = re.match(r'^(\d+)m$', name, re.IGNORECASE)
    if m:
        return int(m.group(1)) * 1_000_000

    # Billions: "1.5b", "2b"
    m = re.match(r'^(\d+(?:\.\d+)?)b$', name, re.IGNORECASE)
    if m:
        return int(float(m.group(1)) * 1_000_000_000)

    return None


def get_dir_size(path: str) -> int:
    """Calculate total size of all files in a directory (non-recursive into subdirs)."""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
    except OSError:
        pass
    return total


def discover_checkpoints(root_path: str) -> CheckpointDir:
    """Scan a checkpoint root directory and build a CheckpointDir."""
    ckpt_dir = CheckpointDir(path=root_path)

    if not os.path.exists(root_path):
        return ckpt_dir

    for entry in os.scandir(root_path):
        if entry.is_dir():
            step = parse_step_from_name(entry.name)
            size = get_dir_size(entry.path)
            ckpt = Checkpoint(
                path=entry.path,
                name=entry.name,
                step=step,
                size_bytes=size,
            )
            ckpt_dir.checkpoints.append(ckpt)

    return ckpt_dir


def find_all_checkpoint_dirs(project_root: str) -> list:
    """Auto-discover all checkpoint directories in the project."""
    data_dir = os.path.join(project_root, "data")
    dirs = []

    if not os.path.exists(data_dir):
        return dirs

    for version_entry in os.scandir(data_dir):
        if version_entry.is_dir():
            # Look for checkpoints_permanent and checkpoints-* patterns
            for sub_entry in os.scandir(version_entry.path):
                if sub_entry.is_dir() and "checkpoint" in sub_entry.name.lower():
                    dirs.append(sub_entry.path)

    return sorted(dirs)


# ============================================================================
# Protection Logic
# ============================================================================

def get_version_boundary_steps() -> list:
    """Extract all unique step counts that mark version transitions."""
    steps = set()
    for start, end in VERSION_BOUNDARIES.values():
        steps.add(start)
        if end is not None:
            steps.add(end)
    return sorted(steps)


def find_nearest_checkpoint(checkpoints: list, target_step: int) -> Optional[int]:
    """Find the index of the checkpoint nearest to target_step."""
    valid = [(i, c) for i, c in enumerate(checkpoints) if c.step is not None]
    if not valid:
        return None
    return min(valid, key=lambda x: abs(x[1].step - target_step))[0]


def apply_protections(ckpt_dir: CheckpointDir):
    """Mark checkpoints that must never be deleted."""
    sorted_ckpts = ckpt_dir.sorted_checkpoints
    if not sorted_ckpts:
        return

    # 1. Protect the LATEST checkpoint (highest step count)
    valid_step_ckpts = [c for c in sorted_ckpts if c.step is not None]
    if valid_step_ckpts:
        latest = max(valid_step_ckpts, key=lambda c: c.step)
        latest.is_protected = True
        latest.protection_reason = "LATEST checkpoint"

    # 2. Protect the FIRST checkpoint (earliest step — genesis)
    if valid_step_ckpts:
        earliest = min(valid_step_ckpts, key=lambda c: c.step)
        earliest.is_protected = True
        if earliest.protection_reason:
            earliest.protection_reason += " + EARLIEST"
        else:
            earliest.protection_reason = "EARLIEST checkpoint"

    # 3. Protect checkpoints nearest to version boundaries
    boundary_steps = get_version_boundary_steps()
    for boundary in boundary_steps:
        idx = find_nearest_checkpoint(sorted_ckpts, boundary)
        if idx is not None:
            ckpt = sorted_ckpts[idx]
            # Only protect if reasonably close (within 20M steps)
            if ckpt.step is not None and abs(ckpt.step - boundary) < 20_000_000:
                version_name = None
                for v, (s, e) in VERSION_BOUNDARIES.items():
                    if s == boundary or e == boundary:
                        version_name = v
                        break
                ckpt.is_protected = True
                reason = f"Version boundary ({version_name}: {boundary/1e9:.2f}B)"
                if ckpt.protection_reason:
                    ckpt.protection_reason += f" + {reason}"
                else:
                    ckpt.protection_reason = reason

    # 4. Protect non-parseable checkpoints (unusual names = likely important)
    for c in sorted_ckpts:
        if c.step is None:
            c.is_protected = True
            if not c.protection_reason:
                c.protection_reason = f"Non-numeric name '{c.name}' (manually created?)"


# ============================================================================
# Thinning Strategies
# ============================================================================

def strategy_uniform(ckpt_dir: CheckpointDir, keep_every: int) -> list:
    """
    Keep every Nth checkpoint (by step order). Delete the rest.
    Protected checkpoints are always kept regardless.

    Returns list of Checkpoint objects to DELETE.
    """
    sorted_ckpts = ckpt_dir.sorted_checkpoints
    to_delete = []

    # Work only with non-protected, step-sortable checkpoints
    deletable = [c for c in sorted_ckpts if not c.is_protected and c.step is not None]
    # Sort by step
    deletable.sort(key=lambda c: c.step)

    for i, ckpt in enumerate(deletable):
        # Keep indices 0, keep_every, 2*keep_every, ...
        if (i % keep_every) != 0:
            to_delete.append(ckpt)

    return to_delete


def strategy_tiered(ckpt_dir: CheckpointDir, tiers: list = None) -> list:
    """
    Tiered thinning: different keep-rates for different step ranges.

    Default tiers (designed for VYREX at 12.5B steps):
      - 0-2B:     keep every 5th (ancient history, rarely needed)
      - 2B-5B:    keep every 4th (early experimental, some value)
      - 5B-8B:    keep every 3rd (mid-training)
      - 8B-11B:   keep every 2nd (recent-ish)
      - 11B+:     keep ALL (recent, actively evaluated)

    Returns list of Checkpoint objects to DELETE.
    """
    if tiers is None:
        # (start_step, end_step, keep_every_N)
        tiers = [
            (0,              2_000_000_000,  5),   # Ancient
            (2_000_000_000,  5_000_000_000,  4),   # Early experimental
            (5_000_000_000,  8_000_000_000,  3),   # Mid-training
            (8_000_000_000,  11_000_000_000, 2),   # Recent-ish
            (11_000_000_000, float('inf'),   1),   # Keep all recent
        ]

    sorted_ckpts = ckpt_dir.sorted_checkpoints
    to_delete = []

    # Bucket deletable checkpoints into tiers
    deletable = [c for c in sorted_ckpts if not c.is_protected and c.step is not None]
    deletable.sort(key=lambda c: c.step)

    # Group by tier
    tier_buckets = {i: [] for i in range(len(tiers))}
    for ckpt in deletable:
        for i, (start, end, keep_n) in enumerate(tiers):
            if start <= ckpt.step < end:
                tier_buckets[i].append(ckpt)
                break

    # Apply keep_every_N within each tier
    for i, (start, end, keep_n) in enumerate(tiers):
        bucket = tier_buckets[i]
        for j, ckpt in enumerate(bucket):
            if keep_n <= 0:
                # keep_n = 0 means delete ALL (except protected)
                to_delete.append(ckpt)
            elif (j % keep_n) != 0:
                to_delete.append(ckpt)

    return to_delete


# ============================================================================
# Reporting
# ============================================================================

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def format_size(size_bytes: int) -> str:
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.2f} GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.1f} MB"
    else:
        return f"{size_bytes / 1024:.0f} KB"


def format_steps(steps: int) -> str:
    if steps >= 1_000_000_000:
        return f"{steps / 1_000_000_000:.2f}B"
    elif steps >= 1_000_000:
        return f"{steps / 1_000_000:.0f}M"
    else:
        return f"{steps:,}"


def print_report(ckpt_dir: CheckpointDir):
    """Print a comprehensive analysis report for a checkpoint directory."""
    sorted_ckpts = ckpt_dir.sorted_checkpoints
    valid_step_ckpts = [c for c in sorted_ckpts if c.step is not None]

    print(f"\n{BOLD}{'='*72}{RESET}")
    print(f"{BOLD}  CHECKPOINT ANALYSIS: {os.path.basename(ckpt_dir.path)}{RESET}")
    print(f"  {DIM}{ckpt_dir.path}{RESET}")
    print(f"{BOLD}{'='*72}{RESET}")

    # Summary stats
    print(f"\n  {CYAN}Directory Count:{RESET}    {ckpt_dir.count:,}")
    print(f"  {CYAN}Total Size:{RESET}         {format_size(ckpt_dir.total_size_bytes)}")

    if valid_step_ckpts:
        min_step = min(c.step for c in valid_step_ckpts)
        max_step = max(c.step for c in valid_step_ckpts)
        print(f"  {CYAN}Step Range:{RESET}         {format_steps(min_step)} → {format_steps(max_step)}")

        step_span = max_step - min_step
        if step_span > 0 and len(valid_step_ckpts) > 1:
            avg_interval = step_span / (len(valid_step_ckpts) - 1)
            density = len(valid_step_ckpts) / (step_span / 1_000_000_000)
            print(f"  {CYAN}Avg Interval:{RESET}      {format_steps(int(avg_interval))} steps")
            print(f"  {CYAN}Density:{RESET}            {density:.1f} checkpoints/billion steps")

    avg_size = ckpt_dir.total_size_bytes / max(ckpt_dir.count, 1)
    print(f"  {CYAN}Avg Size/Ckpt:{RESET}     {format_size(int(avg_size))}")

    # Non-numeric checkpoints
    non_numeric = [c for c in sorted_ckpts if c.step is None]
    if non_numeric:
        print(f"\n  {YELLOW}Non-numeric checkpoints ({len(non_numeric)}):{RESET}")
        for c in non_numeric:
            print(f"    {DIM}•{RESET} {c.name} ({format_size(c.size_bytes)})")

    # Protected checkpoints
    protected = [c for c in sorted_ckpts if c.is_protected]
    print(f"\n  {GREEN}Protected checkpoints ({len(protected)}):{RESET}")
    for c in protected:
        print(f"    {GREEN}🛡{RESET} {c.step_display:>10s}  {c.protection_reason}")

    # Version coverage
    if valid_step_ckpts:
        print(f"\n  {BLUE}Version coverage:{RESET}")
        for version, (start, end) in sorted(VERSION_BOUNDARIES.items(),
                                              key=lambda x: x[1][0]):
            end_val = end if end is not None else float('inf')
            in_range = [c for c in valid_step_ckpts
                       if start <= c.step < end_val]
            if in_range:
                print(f"    {version:>6s}: {len(in_range):>4d} checkpoints "
                      f"({format_steps(start)}–{format_steps(end) if end else 'now'})")


def print_strategy_preview(ckpt_dir: CheckpointDir, to_delete: list, strategy_name: str):
    """Show what a strategy would delete."""
    total_freed = sum(c.size_bytes for c in to_delete)
    remaining = ckpt_dir.count - len(to_delete)

    print(f"\n{BOLD}{'─'*72}{RESET}")
    print(f"  {BOLD}Strategy: {strategy_name}{RESET}")
    print(f"{'─'*72}")
    print(f"  {RED}Would delete:{RESET}      {len(to_delete):,} checkpoints")
    print(f"  {RED}Space freed:{RESET}       {format_size(total_freed)}")
    print(f"  {GREEN}Would keep:{RESET}        {remaining:,} checkpoints")
    print(f"  {GREEN}Remaining size:{RESET}    {format_size(ckpt_dir.total_size_bytes - total_freed)}")

    # Show deletion density by version
    valid_delete = [c for c in to_delete if c.step is not None]
    if valid_delete:
        print(f"\n  {YELLOW}Deletion breakdown by version:{RESET}")
        for version, (start, end) in sorted(VERSION_BOUNDARIES.items(),
                                              key=lambda x: x[1][0]):
            end_val = end if end is not None else float('inf')
            total_in_range = len([c for c in ckpt_dir.sorted_checkpoints
                                  if c.step is not None and start <= c.step < end_val])
            deleted_in_range = len([c for c in valid_delete
                                    if start <= c.step < end_val])
            if total_in_range > 0:
                pct = (deleted_in_range / total_in_range) * 100
                kept = total_in_range - deleted_in_range
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(f"    {version:>6s}: {bar} {deleted_in_range:>4d}/{total_in_range:<4d} deleted "
                      f"({pct:.0f}%), {kept} kept")


def print_recommendations(ckpt_dir: CheckpointDir):
    """Generate intelligent recommendations based on the data."""
    print(f"\n{BOLD}{'='*72}{RESET}")
    print(f"  {BOLD}RECOMMENDATIONS{RESET}")
    print(f"{'='*72}")

    if ckpt_dir.count < 10:
        print(f"\n  {GREEN}✓ Only {ckpt_dir.count} checkpoints — no cleanup needed.{RESET}")
        return

    avg_size = ckpt_dir.total_size_bytes / max(ckpt_dir.count, 1)
    protected_count = len([c for c in ckpt_dir.checkpoints if c.is_protected])
    deletable_count = ckpt_dir.count - protected_count

    # Build recommendations for different strategies
    strategies = []

    for keep_n in [2, 3, 4, 5, 10]:
        to_del = strategy_uniform(ckpt_dir, keep_n)
        freed = sum(c.size_bytes for c in to_del)
        strategies.append((
            f"Uniform (keep every {keep_n})",
            len(to_del),
            freed,
            f"--strategy uniform --keep-every {keep_n}",
        ))

    to_del_tiered = strategy_tiered(ckpt_dir)
    freed_tiered = sum(c.size_bytes for c in to_del_tiered)
    strategies.append((
        "Tiered (recommended)",
        len(to_del_tiered),
        freed_tiered,
        "--strategy tiered",
    ))

    print(f"\n  {CYAN}Available strategies (from {deletable_count} deletable checkpoints):{RESET}\n")
    print(f"  {'Strategy':<28s} {'Delete':>8s} {'Free':>10s} {'Keep':>8s}  Command")
    print(f"  {'─'*28} {'─'*8} {'─'*10} {'─'*8}  {'─'*35}")

    for name, del_count, freed, cmd in strategies:
        kept = ckpt_dir.count - del_count
        marker = f" {GREEN}◀ RECOMMENDED{RESET}" if "recommended" in name.lower() else ""
        print(f"  {name:<28s} {del_count:>8,} {format_size(freed):>10s} {kept:>8,}  {DIM}{cmd}{RESET}{marker}")

    # Disk space context
    try:
        drive = os.path.splitdrive(ckpt_dir.path)[0] or "/"
        stat = shutil.disk_usage(drive + os.sep if drive else "/")
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        used_pct = ((stat.total - stat.free) / stat.total) * 100

        print(f"\n  {CYAN}Disk space ({drive or '/'}):{RESET}")
        print(f"    Free: {free_gb:.2f} GB / {total_gb:.0f} GB ({used_pct:.1f}% used)")

        if free_gb < 5:
            print(f"    {RED}⚠ CRITICALLY LOW DISK SPACE! Cleanup strongly recommended.{RESET}")
        elif free_gb < 50:
            print(f"    {YELLOW}⚠ Low disk space. Consider cleanup.{RESET}")
    except Exception:
        pass

    print(f"\n  {DIM}To preview a strategy:  python checkpoint_cleanup.py <command>{RESET}")
    print(f"  {DIM}To execute deletion:    python checkpoint_cleanup.py <command> --execute{RESET}")
    print(f"  {DIM}To target specific dir: python checkpoint_cleanup.py --dir <path> <command>{RESET}")


# ============================================================================
# Execution
# ============================================================================

def execute_deletion(to_delete: list, dry_run: bool = True) -> tuple:
    """
    Delete checkpoint directories.

    Returns (deleted_count, freed_bytes, errors).
    """
    deleted = 0
    freed = 0
    errors = []

    for ckpt in to_delete:
        if ckpt.is_protected:
            errors.append(f"SKIPPED (protected): {ckpt.path}")
            continue

        if dry_run:
            deleted += 1
            freed += ckpt.size_bytes
        else:
            try:
                shutil.rmtree(ckpt.path)
                deleted += 1
                freed += ckpt.size_bytes
            except Exception as e:
                errors.append(f"FAILED: {ckpt.path} — {e}")

    return deleted, freed, errors


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VYREX Checkpoint Cleanup Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python checkpoint_cleanup.py                                    # Full analysis
  python checkpoint_cleanup.py --strategy uniform --keep-every 3  # Preview uniform thinning
  python checkpoint_cleanup.py --strategy tiered                  # Preview tiered (recommended)
  python checkpoint_cleanup.py --strategy tiered --execute        # Execute tiered deletion
  python checkpoint_cleanup.py --dir data/v3/checkpoints_permanent --strategy uniform --keep-every 5
        """,
    )
    parser.add_argument(
        "--dir", type=str, default=None,
        help="Specific checkpoint directory to clean (default: auto-discover all)",
    )
    parser.add_argument(
        "--strategy", type=str, choices=["uniform", "tiered"],
        default=None,
        help="Thinning strategy to apply",
    )
    parser.add_argument(
        "--keep-every", type=int, default=3,
        help="For uniform strategy: keep every Nth checkpoint (default: 3)",
    )
    parser.add_argument(
        "--tiers", type=str, default=None,
        help="Custom tiers as JSON: [[start, end, keep_every], ...]. "
             "Example: '[[0, 5e9, 5], [5e9, 10e9, 3], [10e9, 1e18, 1]]'",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually delete files (default: dry-run preview only)",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip confirmation prompt (use with --execute)",
    )

    args = parser.parse_args()

    # Resolve project root
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Discover checkpoint directories
    if args.dir:
        dir_path = args.dir
        if not os.path.isabs(dir_path):
            dir_path = os.path.join(project_root, dir_path)
        target_dirs = [dir_path]
    else:
        target_dirs = find_all_checkpoint_dirs(project_root)
        if not target_dirs:
            print(f"{RED}No checkpoint directories found in {project_root}/data/{RESET}")
            sys.exit(1)

    # Process each directory
    all_ckpt_dirs = []
    for d in target_dirs:
        if not os.path.exists(d):
            print(f"{YELLOW}Warning: {d} does not exist, skipping.{RESET}")
            continue
        ckpt_dir = discover_checkpoints(d)
        if ckpt_dir.count == 0:
            continue
        apply_protections(ckpt_dir)
        all_ckpt_dirs.append(ckpt_dir)

    if not all_ckpt_dirs:
        print(f"{YELLOW}No checkpoints found.{RESET}")
        sys.exit(0)

    # Print reports
    grand_total_size = 0
    grand_total_count = 0
    for ckpt_dir in all_ckpt_dirs:
        print_report(ckpt_dir)
        grand_total_size += ckpt_dir.total_size_bytes
        grand_total_count += ckpt_dir.count

    if len(all_ckpt_dirs) > 1:
        print(f"\n{BOLD}{'━'*72}{RESET}")
        print(f"  {BOLD}GRAND TOTAL: {grand_total_count:,} checkpoints, "
              f"{format_size(grand_total_size)}{RESET}")
        print(f"{'━'*72}")

    # Strategy mode
    if args.strategy:
        for ckpt_dir in all_ckpt_dirs:
            if args.strategy == "uniform":
                to_delete = strategy_uniform(ckpt_dir, args.keep_every)
                strategy_name = f"Uniform (keep every {args.keep_every})"
            elif args.strategy == "tiered":
                custom_tiers = None
                if args.tiers:
                    import json
                    custom_tiers = [(int(s), float(e), int(k))
                                    for s, e, k in json.loads(args.tiers)]
                to_delete = strategy_tiered(ckpt_dir, custom_tiers)
                strategy_name = "Tiered"

            print_strategy_preview(ckpt_dir, to_delete, strategy_name)

            if args.execute:
                if not to_delete:
                    print(f"\n  {GREEN}Nothing to delete.{RESET}")
                    continue

                # Confirmation
                if not args.yes:
                    print(f"\n  {RED}{BOLD}⚠ WARNING: This will PERMANENTLY delete "
                          f"{len(to_delete):,} checkpoint directories!{RESET}")
                    print(f"  {RED}Total space freed: {format_size(sum(c.size_bytes for c in to_delete))}{RESET}")
                    response = input(f"\n  Type 'DELETE' to confirm: ")
                    if response.strip() != "DELETE":
                        print(f"  {YELLOW}Aborted.{RESET}")
                        continue

                print(f"\n  {YELLOW}Deleting checkpoints...{RESET}")
                deleted, freed, errors = execute_deletion(to_delete, dry_run=False)
                print(f"  {GREEN}✓ Deleted {deleted:,} checkpoints, freed {format_size(freed)}{RESET}")
                if errors:
                    for e in errors:
                        print(f"  {RED}  {e}{RESET}")
            else:
                print(f"\n  {DIM}This is a DRY RUN. Add --execute to actually delete.{RESET}")
    else:
        # Analysis-only mode: show recommendations
        for ckpt_dir in all_ckpt_dirs:
            print_recommendations(ckpt_dir)

    print()


if __name__ == "__main__":
    main()
