"""
VYREX v4 - Metrics Logger & Diagnostics System
=================================================
Extends rlgym-learn-algos PPOMetricsLogger to provide:
  - PPO training metrics (loss, entropy, etc.) via base class
  - TrueSkill rating tracking metrics
  - Team spirit progression
  - SPS tracking with rolling average
  - alive-progress bar (pinned to bottom, logs scroll above)
  - Permanent checkpoint archiving
  - JSON diagnostic reports

Architecture difference from v3:
  v3 (rlgym-ppo): _collect_metrics(game_state) in workers + _report_metrics() on main
  v4 (rlgym-learn): get_metrics() called by WandbMetricsLogger after each learn step.
    Game-level metrics (touches, aerials, etc.) from SharedInfoProvider are
    accumulated via the custom AgentController's process_timestep_data().
"""

import os
import sys
import json
import time
import shutil
from typing import Dict, Any, List, Optional

import numpy as np
from alive_progress import alive_bar


try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


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


def _fmt_eta(seconds: float) -> str:
    if seconds <= 0:
        return "?"
    d = int(seconds // 86400)
    h = int((seconds % 86400) // 3600)
    m = int((seconds % 3600) // 60)
    if d > 0:
        return f"{d}d{h:02d}h"
    elif h > 0:
        return f"{h}h{m:02d}m"
    else:
        return f"{m}m"


# ============================================================================
# Game Metrics Indices — layout of accumulated metrics from SharedInfoProvider
# ============================================================================
N_METRICS = 23

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
IDX_N_ZERO_BOOST    = 10
IDX_N_SUPERSONIC    = 11
IDX_AVG_BALL_DIST   = 12
IDX_BALL_SPEED      = 13
IDX_N_OWN_HALF      = 14
IDX_N_DOUBLE_COMMIT = 15
IDX_TOUCH_BALL_SPEED = 16
IDX_N_TOUCH_AGENTS  = 17
IDX_BOOST_COLLECTED = 18
IDX_N_NEAR_WALL     = 19
IDX_N_HAS_FLIP      = 20
IDX_N_IS_BOOSTING   = 21
IDX_AERIAL_SEQ_TOUCHES = 22


# ============================================================================
# VYREX METRICS LOGGER — extends PPOMetricsLogger
# ============================================================================

class VyrexMetricsLogger:
    """
    Custom metrics logger for VYREX v4.

    Extends the base PPOMetricsLogger to add:
    - Team spirit tracking
    - SPS calculation
    - alive-progress bar (pinned to bottom, logs scroll above)
    - Permanent checkpoint archiving
    - JSON diagnostics reports
    - TrueSkill rating metrics (when tracker is provided)
    - Accumulated game metrics from SharedInfoProvider
    """

    def __init__(self, config=None, trueskill_tracker=None):
        from rlgym_learn_algos.ppo import PPOMetricsLogger
        self._base_logger = PPOMetricsLogger()

        if config is None:
            from config import DEFAULT_CONFIG
            config = DEFAULT_CONFIG

        self.config = config
        self.diag_config = config.diagnostics
        self.trueskill_tracker = trueskill_tracker

        # Timing
        self._start_time: float = time.time()
        self._last_report_time: float = time.time()
        self._last_report_steps: int = -1
        self._rolling_sps: float = 0.0

        # alive-progress bar (manual context manager lifecycle)
        self._bar = None          # bar callable from alive_bar context
        self._bar_ctx = None      # context manager reference for cleanup
        self._prev_steps: int = 0

        # Diagnostics
        self._last_diagnostics_step: int = 0

        # Permanent checkpoint tracking
        self._last_permanent_checkpoint_step: int = 0

        # Accumulated game metrics (populated by custom AgentController)
        self._game_metrics_buffer: List[np.ndarray] = []
        self._game_metrics_lock = None  # Set if threading needed

        # Create directories
        if self.diag_config.enable_diagnostics:
            os.makedirs(self.diag_config.diagnostics_output_dir, exist_ok=True)
        os.makedirs(config.paths.permanent_checkpoint_dir, exist_ok=True)

    def close(self):
        """Clean up alive-progress bar context."""
        if self._bar_ctx is not None:
            self._bar_ctx.__exit__(None, None, None)
            self._bar = None
            self._bar_ctx = None

    def accumulate_game_metrics(self, metrics_array: np.ndarray):
        """Called by custom AgentController to add game metrics from shared_info."""
        self._game_metrics_buffer.append(metrics_array)

    def get_metrics(self, cumulative_timesteps: int = 0) -> Dict[str, Any]:
        """
        Called after each learning iteration to get metrics for wandb logging.
        Returns a dict of metric_name -> value.
        """
        metrics = {}

        # Team spirit
        ts = self._compute_team_spirit(cumulative_timesteps)
        metrics["vyrex/team_spirit"] = ts

        # SPS calculation
        now = time.time()
        if self._last_report_steps < 0:
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

        # Process accumulated game metrics if any
        if self._game_metrics_buffer:
            game_metrics = self._aggregate_game_metrics()
            metrics.update(game_metrics)
            self._game_metrics_buffer.clear()

        # TrueSkill metrics
        if self.trueskill_tracker is not None:
            ts_metrics = self.trueskill_tracker.get_metrics()
            metrics.update(ts_metrics)

        # alive-progress bar
        self._update_progress_bar(cumulative_timesteps, ts, metrics)

        # Diagnostics
        if self.diag_config.enable_diagnostics:
            if cumulative_timesteps - self._last_diagnostics_step >= self.diag_config.diagnostics_interval_steps:
                self._generate_diagnostics_report(cumulative_timesteps, metrics)
                self._last_diagnostics_step = cumulative_timesteps

        # Permanent checkpoint archive
        self._maybe_archive_permanent_checkpoint(cumulative_timesteps)

        return metrics

    def _aggregate_game_metrics(self) -> Dict[str, float]:
        """Aggregate accumulated game metrics from SharedInfoProvider."""
        if not self._game_metrics_buffer:
            return {}

        data = np.stack(self._game_metrics_buffer)  # (n_steps, N_METRICS)

        total_touches = float(np.sum(data[:, IDX_TOUCHES]))
        total_aerial = float(np.sum(data[:, IDX_AERIAL_TOUCHES]))
        total_goals_blue = float(np.sum(data[:, IDX_GOAL_BLUE]))
        total_goals_orange = float(np.sum(data[:, IDX_GOAL_ORANGE]))

        agent_counts = np.maximum(data[:, IDX_N_AGENTS], 1)

        return {
            "vyrex/touches_per_step": float(np.mean(data[:, IDX_TOUCHES])),
            "vyrex/aerial_touch_rate": total_aerial / max(total_touches, 1),
            "vyrex/demos_per_step": float(np.mean(data[:, IDX_DEMOS])),
            "vyrex/goals_blue": total_goals_blue,
            "vyrex/goals_orange": total_goals_orange,
            "vyrex/goal_diff_blue": total_goals_blue - total_goals_orange,
            "vyrex/avg_boost": float(np.mean(data[:, IDX_AVG_BOOST])),
            "vyrex/avg_speed": float(np.mean(data[:, IDX_AVG_SPEED])),
            "vyrex/airborne_frac": float(np.mean(data[:, IDX_N_AIRBORNE] / agent_counts)),
            "vyrex/avg_teammate_dist": float(np.mean(data[:, IDX_AVG_TEAMMATE_DIST])),
            "vyrex/zero_boost_frac": float(np.mean(data[:, IDX_N_ZERO_BOOST] / agent_counts)),
            "vyrex/supersonic_frac": float(np.mean(data[:, IDX_N_SUPERSONIC] / agent_counts)),
            "vyrex/own_half_frac": float(np.mean(data[:, IDX_N_OWN_HALF] / agent_counts)),
            "vyrex/near_wall_frac": float(np.mean(data[:, IDX_N_NEAR_WALL] / agent_counts)),
            "vyrex/has_flip_frac": float(np.mean(data[:, IDX_N_HAS_FLIP] / agent_counts)),
            "vyrex/boosting_frac": float(np.mean(data[:, IDX_N_IS_BOOSTING] / agent_counts)),
            "vyrex/avg_ball_dist": float(np.mean(data[:, IDX_AVG_BALL_DIST])),
            "vyrex/ball_speed": float(np.mean(data[:, IDX_BALL_SPEED])),
            "vyrex/double_commit_frac": float(np.mean(data[:, IDX_N_DOUBLE_COMMIT])),
            "vyrex/boost_collected_per_step": float(np.mean(data[:, IDX_BOOST_COLLECTED])),
            "vyrex/touch_ball_speed": float(
                np.sum(data[:, IDX_TOUCH_BALL_SPEED]) /
                max(np.sum(data[:, IDX_N_TOUCH_AGENTS]), 1)
            ),
            "vyrex/aerial_seq_touches": float(np.mean(data[:, IDX_AERIAL_SEQ_TOUCHES])),
        }

    def _compute_team_spirit(self, total_steps: int) -> float:
        rc = self.config.rewards
        progress = min(total_steps / max(rc.team_spirit_ramp_steps, 1), 1.0)
        return rc.team_spirit_start + progress * (rc.team_spirit_end - rc.team_spirit_start)

    def _update_progress_bar(self, cumulative_timesteps: int, team_spirit: float,
                             metrics: Dict[str, Any]):
        """Update alive-progress bar (pinned to bottom, logs scroll above)."""
        total = self.config.ppo.timestep_limit
        if self._bar is None:
            self._bar_ctx = alive_bar(
                manual=True,
                title="\033[36m VYREX v4\033[0m",
                force_tty=True,
                dual_line=True,
                length=30,
                spinner="dots_waves"
            )
            self._bar = self._bar_ctx.__enter__()
            self._prev_steps = cumulative_timesteps

        fraction = cumulative_timesteps / total if total > 0 else 0.0
        self._bar(min(fraction, 1.0))
        self._prev_steps = cumulative_timesteps

        sps_str = f"{self._rolling_sps / 1000:.1f}K" if self._rolling_sps >= 1000 else f"{self._rolling_sps:.0f}"
        boost_str = f"{metrics.get('vyrex/avg_boost', 0):.0f}"

        # ETA from rolling SPS
        remaining = total - cumulative_timesteps
        eta_str = _fmt_eta(remaining / self._rolling_sps) if self._rolling_sps > 0 else "?"

        # TrueSkill rating (2v2 deterministic)
        ts_rating = metrics.get("Rating 2v2/Deterministic")
        rating_str = f" | R:{ts_rating:.1f}" if ts_rating is not None else ""

        self._bar.text(
            f"  {_fmt_steps(cumulative_timesteps)}/{_fmt_steps(total)}"
            f" | SPS:{sps_str} | ETA:{eta_str} | TS:{team_spirit:.2f}"
            f" | B:{boost_str}{rating_str}"
        )

    def _maybe_archive_permanent_checkpoint(self, cumulative_timesteps: int):
        """Archive the latest checkpoint to permanent directory every N steps."""
        interval = self.config.ppo.permanent_save_every_ts
        if interval <= 0:
            return

        current_milestone = (cumulative_timesteps // interval) * interval
        if current_milestone <= self._last_permanent_checkpoint_step:
            return

        # rlgym-learn checkpoint structure:
        #   agent_controllers_checkpoints/PPO/<run_name>/<timestamp_ns>/
        # Checkpoint folders are named with time.time_ns(), not cumulative_timesteps.
        run_dir = os.path.join(
            self.config.paths.project_root,
            "agent_controllers_checkpoints", "PPO", "vyrex-v4-2v2",
        )
        if not os.path.isdir(run_dir):
            return

        # Find latest checkpoint (highest nanosecond timestamp)
        best_ts = -1
        best_path = None
        for name in os.listdir(run_dir):
            sub = os.path.join(run_dir, name)
            if not os.path.isdir(sub):
                continue
            try:
                ts = int(name)
            except ValueError:
                continue
            if ts > best_ts:
                best_ts = ts
                best_path = sub

        if best_path is None:
            return

        perm_dir = self.config.paths.permanent_checkpoint_dir
        # Name the permanent copy by cumulative_timesteps for readability
        agent_json = os.path.join(best_path, "ppo_agent.json")
        if os.path.isfile(agent_json):
            with open(agent_json, "r") as f:
                cts = json.load(f).get("cumulative_timesteps", best_ts)
            dest_name = str(cts)
        else:
            dest_name = os.path.basename(best_path)
        dest = os.path.join(perm_dir, dest_name)
        if os.path.exists(dest):
            self._last_permanent_checkpoint_step = current_milestone
            return

        try:
            shutil.copytree(best_path, dest)
            self._last_permanent_checkpoint_step = current_milestone
            msg = (
                f"\n[VYREX] Permanent checkpoint archived: "
                f"{dest_name} → {perm_dir}\n"
            )
            print(msg)
        except Exception as e:
            print(f"\n[VYREX] WARNING: Failed to archive permanent checkpoint: {e}\n")

    def _generate_diagnostics_report(self, total_steps: int, latest_metrics: dict):
        """Generate a structured JSON diagnostics report."""
        elapsed = time.time() - self._start_time

        report = {
            "bot_name": "VYREX",
            "version": "v4",
            "framework": "rlgym-learn",
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
                k.replace("vyrex/", ""): round(v, 5) if isinstance(v, float) else v
                for k, v in latest_metrics.items()
                if k.startswith("vyrex/") and k not in (
                    "vyrex/team_spirit", "vyrex/sps_rolling", "vyrex/elapsed_hours"
                )
            },

            "training_schedule": {
                "current_team_spirit": round(self._compute_team_spirit(total_steps), 4),
                "target_team_spirit": self.config.rewards.team_spirit_end,
                "learning_rate_actor": self.config.ppo.actor_lr,
                "learning_rate_critic": self.config.ppo.critic_lr,
                "entropy_coef": self.config.ppo.ent_coef,
            },

            "current_config_summary": {
                "network_policy": self.config.network.policy_layer_sizes,
                "network_critic": self.config.network.critic_layer_sizes,
                "batch_size": self.config.ppo.batch_size,
                "n_proc": self.config.ppo.n_proc,
                "n_epochs": self.config.ppo.n_epochs,
                "team_size": self.config.env.team_size,
            },
        }

        # Save to file
        filename = f"diagnostics_{total_steps:012d}.json"
        filepath = os.path.join(self.diag_config.diagnostics_output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        latest_path = os.path.join(self.diag_config.diagnostics_output_dir, "latest.json")
        with open(latest_path, "w") as f:
            json.dump(report, f, indent=2)

        diag_msg = (
            f"\n{'='*60}\n"
            f"  VYREX v4 Diagnostics Report @ {total_steps:,} steps\n"
            f"  Saved to: {filepath}\n"
            f"  Team Spirit: {report['training_schedule']['current_team_spirit']:.4f}\n"
            f"  SPS:         {report['steps_per_second']:.1f}\n"
            f"{'='*60}\n"
        )
        print(diag_msg)

        # Log summary to WandB
        if HAS_WANDB and wandb.run is not None:
            try:
                wandb.log({
                    "diagnostics/sps": report["steps_per_second"],
                    "diagnostics/team_spirit": report["training_schedule"]["current_team_spirit"],
                })
            except Exception:
                pass

        return report


# ============================================================================
# STANDALONE DIAGNOSTIC READER
# ============================================================================

def read_latest_diagnostics(diagnostics_dir: str = None) -> Optional[Dict]:
    """Read the latest diagnostics report and return it as a dict."""
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
