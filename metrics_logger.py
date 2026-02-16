"""
VYREX - Metrics Logger & Diagnostics System
=============================================
Subclasses rlgym-ppo's MetricsLogger ABC to collect per-step game
metrics from the RLGym v2 GameState inside worker processes, then
aggregates and reports them on the main process.

Collects per step:
  - Ball touches, aerial touches, demos
  - Goal events (blue / orange)
  - Boost levels, car speeds, airborne status
  - Teammate spacing

Reports:
  - Aggregated WandB metrics every iteration
  - JSON diagnostic reports at configurable intervals
  - Progress bar with ETA
"""

import os
import sys
import json
import time
import shutil
from typing import Dict, Any, List, Optional

import numpy as np
from tqdm import tqdm

from rlgym_ppo.util import MetricsLogger

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ============================================================================
# Metric indices — layout of the flat array returned by _collect_metrics
# ============================================================================
N_METRICS = 10

IDX_TOUCHES         = 0
IDX_AERIAL_TOUCHES  = 1
IDX_DEMOS           = 2
IDX_GOAL_BLUE       = 3
IDX_GOAL_ORANGE     = 4
IDX_AVG_BOOST       = 5
IDX_AVG_SPEED       = 6
IDX_N_AIRBORNE      = 7
IDX_N_AGENTS        = 8
IDX_AVG_TEAMMATE_DIST = 9


# ============================================================================
# Human-readable step formatter
# ============================================================================

def _fmt_steps(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    elif n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    elif n >= 1_000:
        return f"{n / 1e3:.0f}K"
    return str(n)


# ============================================================================
# VYREX METRICS LOGGER — rlgym-ppo MetricsLogger ABC
# ============================================================================

class VyrexMetricsLogger(MetricsLogger):
    """
    Collects per-step game metrics from RLGym v2 GameState in workers
    and reports aggregated statistics + diagnostics on the main process.

    Lifecycle:
      Worker processes:  _collect_metrics(game_state) -> [np.ndarray]
      Main process:      _report_metrics(collected, wandb_run, steps)
    """

    def __init__(self, config=None):
        if config is None:
            from config import DEFAULT_CONFIG
            config = DEFAULT_CONFIG

        self.config = config
        self.diag_config = config.diagnostics

        # Diagnostics tracking (main process only)
        self._last_diagnostics_step: int = 0
        self._start_time: float = time.time()
        self._last_report_time: float = time.time()
        self._last_report_steps: int = -1  # -1 = not yet initialized
        self._rolling_sps: float = 0.0

        # tqdm progress bar (created on first _report_metrics call)
        self._tqdm_bar: Optional[tqdm] = None
        self._prev_steps: int = 0

        # Permanent checkpoint tracking
        self._last_permanent_checkpoint_step: int = 0

        # Create directories
        if self.diag_config.enable_diagnostics:
            os.makedirs(self.diag_config.diagnostics_output_dir, exist_ok=True)
        os.makedirs(config.paths.permanent_checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # MetricsLogger ABC: Worker-side collection
    # ------------------------------------------------------------------

    def _collect_metrics(self, game_state) -> list:
        """
        Called once per env step inside each worker process.
        Receives the RLGym v2 GameState and returns [np.ndarray].
        """
        m = np.zeros(N_METRICS, dtype=np.float32)

        cars = game_state.cars
        agents = list(cars.keys())
        n_agents = len(agents)
        m[IDX_N_AGENTS] = n_agents

        if n_agents == 0:
            return [m]

        total_touches = 0
        aerial_touches = 0
        total_demos = 0
        boost_sum = 0.0
        speed_sum = 0.0
        n_airborne = 0

        # Team grouping for spacing
        team_positions: Dict[int, list] = {}

        for agent in agents:
            car = cars[agent]

            # Ball touches
            touches = car.ball_touches
            total_touches += touches
            if touches > 0 and game_state.ball.position[2] > 400.0:
                aerial_touches += touches

            # Demos
            if car.bump_victim_id is not None and car.bump_victim_id in cars:
                victim = cars[car.bump_victim_id]
                if victim.demo_respawn_timer > 0:
                    total_demos += 1

            # Boost (0-1 range in rlgym v2)
            boost_sum += car.boost_amount  # Already [0, 100] in rlgym v2

            # Speed
            vel = car.physics.linear_velocity
            speed_sum += float(np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2))

            # Airborne
            if not car.on_ground:
                n_airborne += 1

            # Team grouping
            team = 1 if car.is_orange else 0
            if team not in team_positions:
                team_positions[team] = []
            team_positions[team].append(car.physics.position)

        m[IDX_TOUCHES] = total_touches
        m[IDX_AERIAL_TOUCHES] = aerial_touches
        m[IDX_DEMOS] = total_demos
        m[IDX_AVG_BOOST] = boost_sum / n_agents
        m[IDX_AVG_SPEED] = speed_sum / n_agents
        m[IDX_N_AIRBORNE] = n_airborne

        # Goal detection
        if game_state.goal_scored:
            if game_state.scoring_team == 0:
                m[IDX_GOAL_BLUE] = 1.0
            else:
                m[IDX_GOAL_ORANGE] = 1.0

        # Average teammate distance
        dists = []
        for team_id, positions in team_positions.items():
            if len(positions) >= 2:
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        d = float(np.linalg.norm(
                            np.array(positions[i]) - np.array(positions[j])
                        ))
                        dists.append(d)
        m[IDX_AVG_TEAMMATE_DIST] = np.mean(dists) if dists else 0.0

        return [m]

    # ------------------------------------------------------------------
    # MetricsLogger ABC: Main-process reporting
    # ------------------------------------------------------------------

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        """
        Called once per iteration on the main process.
        collected_metrics: list of (list of np.ndarray) from all workers.
        """
        if not collected_metrics:
            return

        # Flatten collected arrays
        all_steps = []
        for step_arrays in collected_metrics:
            if step_arrays and len(step_arrays) > 0:
                arr = step_arrays[0]
                if hasattr(arr, '__len__') and len(arr) == N_METRICS:
                    all_steps.append(np.asarray(arr, dtype=np.float32))

        if not all_steps:
            return

        data = np.stack(all_steps)  # (n_steps, N_METRICS)

        # Aggregate
        total_touches = float(np.sum(data[:, IDX_TOUCHES]))
        total_aerial = float(np.sum(data[:, IDX_AERIAL_TOUCHES]))
        total_goals_blue = float(np.sum(data[:, IDX_GOAL_BLUE]))
        total_goals_orange = float(np.sum(data[:, IDX_GOAL_ORANGE]))

        metrics = {
            "vyrex/touches_per_step": float(np.mean(data[:, IDX_TOUCHES])),
            "vyrex/aerial_touch_rate": total_aerial / max(total_touches, 1),
            "vyrex/demos_per_step": float(np.mean(data[:, IDX_DEMOS])),
            "vyrex/goals_blue": total_goals_blue,
            "vyrex/goals_orange": total_goals_orange,
            "vyrex/goal_diff_blue": total_goals_blue - total_goals_orange,
            "vyrex/avg_boost": float(np.mean(data[:, IDX_AVG_BOOST])),
            "vyrex/avg_speed": float(np.mean(data[:, IDX_AVG_SPEED])),
            "vyrex/airborne_frac": float(
                np.mean(data[:, IDX_N_AIRBORNE] / np.maximum(data[:, IDX_N_AGENTS], 1))
            ),
            "vyrex/avg_teammate_dist": float(np.mean(data[:, IDX_AVG_TEAMMATE_DIST])),
        }

        # Team spirit
        ts = self._compute_team_spirit(cumulative_timesteps)
        metrics["vyrex/team_spirit"] = ts

        # SPS calculation — skip the very first call on resume to avoid
        # a massive delta (cumulative goes from 0 → checkpoint_steps).
        now = time.time()
        if self._last_report_steps < 0:
            # First call: initialise without computing SPS
            self._last_report_steps = cumulative_timesteps
            self._last_report_time = now
        else:
            dt = now - self._last_report_time
            ds = cumulative_timesteps - self._last_report_steps
            if dt > 0 and ds > 0:
                instant_sps = ds / dt
                if self._rolling_sps == 0:
                    self._rolling_sps = instant_sps
                else:
                    self._rolling_sps = 0.8 * self._rolling_sps + 0.2 * instant_sps
            self._last_report_time = now
            self._last_report_steps = cumulative_timesteps

        elapsed = now - self._start_time
        metrics["vyrex/sps_rolling"] = self._rolling_sps
        metrics["vyrex/elapsed_hours"] = round(elapsed / 3600, 3)

        # Log to WandB
        if wandb_run is not None:
            try:
                wandb_run.log(metrics)
            except Exception:
                pass

        # ── tqdm progress bar ─────────────────────────────────────────
        total = self.config.ppo.timestep_limit
        if self._tqdm_bar is None:
            self._tqdm_bar = tqdm(
                total=total,
                initial=cumulative_timesteps,
                desc="\033[36m▶ VYREX\033[0m",
                unit="step",
                unit_scale=True,
                bar_format=(
                    "{desc} {bar:30} {percentage:5.1f}% "
                    "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
                ),
                smoothing=0.1,
                dynamic_ncols=True,
                mininterval=1.0,
            )
            self._prev_steps = cumulative_timesteps

        delta = cumulative_timesteps - self._prev_steps
        if delta > 0:
            self._tqdm_bar.update(delta)
        self._prev_steps = cumulative_timesteps

        sps_str = f"{self._rolling_sps / 1000:.1f}K" if self._rolling_sps >= 1000 else f"{self._rolling_sps:.0f}"
        self._tqdm_bar.set_postfix_str(
            f"SPS:{sps_str} TS:{ts:.2f}",
            refresh=True,
        )

        # ── Diagnostics report ────────────────────────────────────────
        if self.diag_config.enable_diagnostics:
            if cumulative_timesteps - self._last_diagnostics_step >= self.diag_config.diagnostics_interval_steps:
                self._generate_diagnostics_report(cumulative_timesteps, metrics)
                self._last_diagnostics_step = cumulative_timesteps

        # ── Permanent checkpoint archive ──────────────────────────────
        self._maybe_archive_permanent_checkpoint(cumulative_timesteps)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_team_spirit(self, total_steps: int) -> float:
        rc = self.config.rewards
        progress = min(total_steps / max(rc.team_spirit_ramp_steps, 1), 1.0)
        return rc.team_spirit_start + progress * (rc.team_spirit_end - rc.team_spirit_start)

    def _maybe_archive_permanent_checkpoint(self, cumulative_timesteps: int):
        """
        Archive the latest checkpoint to the permanent directory every N steps.

        Unlike the rolling checkpoint dir (which keeps only the last 5),
        the permanent directory retains every archived checkpoint forever.
        This runs inside _report_metrics on the main process.
        """
        interval = self.config.ppo.permanent_save_every_ts
        if interval <= 0:
            return

        # Determine which 7M-aligned milestone we've crossed
        current_milestone = (cumulative_timesteps // interval) * interval
        if current_milestone <= self._last_permanent_checkpoint_step:
            return  # Haven't crossed a new boundary yet

        # Find the latest checkpoint subdirectory in the rolling dir
        ckpt_dir = self.config.paths.checkpoint_dir
        if not os.path.isdir(ckpt_dir):
            return

        best_step = -1
        best_path = None
        for name in os.listdir(ckpt_dir):
            sub = os.path.join(ckpt_dir, name)
            if not os.path.isdir(sub):
                continue
            try:
                step_num = int(name)
            except ValueError:
                continue
            if step_num > best_step:
                best_step = step_num
                best_path = sub

        if best_path is None:
            return

        # Check if we already have this exact checkpoint archived
        perm_dir = self.config.paths.permanent_checkpoint_dir
        dest = os.path.join(perm_dir, os.path.basename(best_path))
        if os.path.exists(dest):
            # Already archived this step — just update the milestone tracker
            self._last_permanent_checkpoint_step = current_milestone
            return

        # Copy the entire checkpoint directory
        try:
            shutil.copytree(best_path, dest)
            self._last_permanent_checkpoint_step = current_milestone
            msg = (
                f"\n[VYREX] Permanent checkpoint archived: "
                f"{os.path.basename(best_path)} → {perm_dir}\n"
            )
            if self._tqdm_bar is not None:
                tqdm.write(msg)
            else:
                print(msg)
        except Exception as e:
            err_msg = f"\n[VYREX] WARNING: Failed to archive permanent checkpoint: {e}\n"
            if self._tqdm_bar is not None:
                tqdm.write(err_msg)
            else:
                print(err_msg)

    def _generate_diagnostics_report(self, total_steps: int, latest_metrics: dict):
        """Generate a structured JSON diagnostics report for LLM optimization."""
        elapsed = time.time() - self._start_time

        report = {
            "bot_name": "VYREX",
            "report_type": "training_diagnostics",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_steps": total_steps,
            "elapsed_hours": round(elapsed / 3600, 2),
            "steps_per_second": round(self._rolling_sps, 1),

            "hardware": {
                "gpu": "RTX 4070 Ti (12GB VRAM, Ada Lovelace)",
                "cpu": "i7-14700K (20 cores / 28 threads)",
                "ram": "48 GB DDR5",
                "tf32_enabled": True,
            },

            "game_metrics": {
                "touches_per_step": round(latest_metrics.get("vyrex/touches_per_step", 0), 5),
                "aerial_touch_rate": round(latest_metrics.get("vyrex/aerial_touch_rate", 0), 5),
                "demos_per_step": round(latest_metrics.get("vyrex/demos_per_step", 0), 5),
                "goals_blue": latest_metrics.get("vyrex/goals_blue", 0),
                "goals_orange": latest_metrics.get("vyrex/goals_orange", 0),
                "goal_diff_blue": latest_metrics.get("vyrex/goal_diff_blue", 0),
                "avg_boost": round(latest_metrics.get("vyrex/avg_boost", 0), 2),
                "avg_speed": round(latest_metrics.get("vyrex/avg_speed", 0), 1),
                "airborne_frac": round(latest_metrics.get("vyrex/airborne_frac", 0), 4),
                "avg_teammate_dist": round(latest_metrics.get("vyrex/avg_teammate_dist", 0), 1),
            },

            "training_schedule": {
                "current_team_spirit": round(self._compute_team_spirit(total_steps), 4),
                "target_team_spirit": self.config.rewards.team_spirit_end,
                "learning_rate_policy": self.config.ppo.policy_lr,
                "learning_rate_critic": self.config.ppo.critic_lr,
                "entropy_coef": self.config.ppo.ppo_ent_coef,
            },

            "current_config_summary": {
                "network_policy": self.config.network.policy_layer_sizes,
                "network_critic": self.config.network.critic_layer_sizes,
                "batch_size": self.config.ppo.ppo_batch_size,
                "n_proc": self.config.ppo.n_proc,
                "ppo_epochs": self.config.ppo.ppo_epochs,
                "team_size": self.config.env.team_size,
            },

            "instructions_for_optimizer": (
                "This is a VYREX training diagnostics report. Hardware: RTX 4070 Ti (12GB), "
                "i7-14700K (20C/28T), 48GB RAM. Analyze the metrics above and provide "
                "specific, actionable advice to improve the bot. Consider: "
                "1) Are rewards trending up? If flat, suggest reward weight changes. "
                "2) Is goal_diff_blue positive? If not, increase offensive rewards. "
                "3) Is aerial_touch_rate > 0.05? If not, increase InAirReward weight. "
                "4) Is avg_boost > 30? If too low, increase BoostConservationReward. "
                "5) Is airborne_frac reasonable? Too high = bot is stuck in air. "
                "6) Is team_spirit appropriate for current step count? "
                "7) Are touches_per_step increasing? If stalled, check dense reward weights. "
                "8) Should we transition to next curriculum phase? "
                "9) Is SPS optimal for this hardware? If <15000, suggest n_proc/batch changes. "
                "10) Should ppo_minibatch_size be increased given 12GB VRAM headroom? "
                "Return your advice as specific config.py changes with exact values."
            ),
        }

        # Save to file
        filename = f"diagnostics_{total_steps:012d}.json"
        filepath = os.path.join(self.diag_config.diagnostics_output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        # Latest copy
        latest_path = os.path.join(self.diag_config.diagnostics_output_dir, "latest.json")
        with open(latest_path, "w") as f:
            json.dump(report, f, indent=2)

        diag_msg = (
            f"\n{'='*60}\n"
            f"  VYREX Diagnostics Report @ {total_steps:,} steps\n"
            f"  Saved to: {filepath}\n"
            f"  Team Spirit: {report['training_schedule']['current_team_spirit']:.4f}\n"
            f"  SPS:         {report['steps_per_second']:.1f}\n"
            f"{'='*60}\n"
        )
        if self._tqdm_bar is not None:
            tqdm.write(diag_msg)
        else:
            print(diag_msg)

        # Log summary to WandB
        if HAS_WANDB and wandb.run is not None:
            try:
                wandb.log({
                    "diagnostics/goal_diff": report["game_metrics"]["goal_diff_blue"],
                    "diagnostics/touches_per_step": report["game_metrics"]["touches_per_step"],
                    "diagnostics/aerial_rate": report["game_metrics"]["aerial_touch_rate"],
                    "diagnostics/avg_boost": report["game_metrics"]["avg_boost"],
                    "diagnostics/sps": report["steps_per_second"],
                })
            except Exception:
                pass

        return report


# ============================================================================
# STANDALONE DIAGNOSTIC READER
# ============================================================================

def read_latest_diagnostics(diagnostics_dir: str = None) -> Optional[Dict]:
    """
    Read the latest diagnostics report and return it as a dict.

    Usage:
        from metrics_logger import read_latest_diagnostics
        report = read_latest_diagnostics()
        print(json.dumps(report, indent=2))
    """
    if diagnostics_dir is None:
        from config import DEFAULT_CONFIG
        diagnostics_dir = DEFAULT_CONFIG.diagnostics.diagnostics_output_dir

    latest_path = os.path.join(diagnostics_dir, "latest.json")
    if not os.path.exists(latest_path):
        print(f"No diagnostics found at {latest_path}")
        return None

    with open(latest_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    report = read_latest_diagnostics()
    if report:
        print(json.dumps(report, indent=2))
    else:
        print("No diagnostics report found yet. Start training first!")
