"""
VYREX — WandB API Client with TTL Caching
==========================================
Handles all WandB data fetching with an in-memory cache
to avoid hammering the API on every request.
"""

import os
import time
import logging
import math
from typing import Dict, List, Optional, Any

logger = logging.getLogger("vyrex.wandb")

# ============================================================================
# Key Mapping: WandB raw keys → Frontend-friendly keys
# ============================================================================
# WandB logs use verbose names like "Policy Reward" and "vyrex/avg_speed".
# The frontend expects short, consistent keys like "reward" and "game/avg_speed".

WANDB_TO_FRONTEND: Dict[str, str] = {
    # PPO training metrics
    "Policy Reward": "reward",
    "Policy Entropy": "entropy",
    "Mean KL Divergence": "kl_divergence",
    "Value Function Loss": "vf_loss",
    "SB3 Clip Fraction": "clip_fraction",
    "Overall Steps per Second": "overall_sps",
    "Policy Update Magnitude": "policy_update_magnitude",
    "Value Function Update Magnitude": "vf_update_magnitude",
    "Cumulative Timesteps": "cumulative_timesteps",
    # Game metrics (vyrex/ namespace → game/ namespace)
    "vyrex/touches_per_step": "game/touches_per_step",
    "vyrex/aerial_touch_rate": "game/aerial_touches_per_step",
    "vyrex/avg_speed": "game/avg_speed",
    "vyrex/airborne_frac": "game/airborne_fraction",
    "vyrex/avg_boost": "game/avg_boost",
    "vyrex/goals_blue": "game/goals_blue",
    "vyrex/goals_orange": "game/goals_orange",
    "vyrex/avg_teammate_dist": "game/avg_teammate_dist",
    "vyrex/demos_per_step": "game/demos_per_step",
    "vyrex/goal_diff_blue": "game/goal_diff_blue",
    "vyrex/team_spirit": "game/team_spirit",
    "vyrex/sps_rolling": "game/sps_rolling",
    # Diagnostics namespace (pass through with prefix)
    "diagnostics/touches_per_step": "diag/touches_per_step",
    "diagnostics/aerial_rate": "diag/aerial_rate",
    "diagnostics/avg_boost": "diag/avg_boost",
    "diagnostics/sps": "diag/sps",
    "diagnostics/goal_diff": "diag/goal_diff",
}

# Reverse mapping: frontend key → WandB key (for history queries)
FRONTEND_TO_WANDB: Dict[str, str] = {v: k for k, v in WANDB_TO_FRONTEND.items()}


class WandBClient:
    """
    Thread-safe WandB API client with TTL-based caching.

    Usage:
        client = WandBClient()
        if client.is_configured:
            info = client.get_run_info()
            latest = client.get_latest_metrics()
            history = client.get_metric_history(["reward", "entropy"], samples=300)
    """

    def __init__(self):
        self._api = None
        self._run = None
        self._cache: Dict[str, tuple] = {}  # key -> (data, timestamp)

        self.api_key = os.getenv("WANDB_API_KEY", "")
        self.entity = os.getenv("WANDB_ENTITY", "")
        self.project = os.getenv("WANDB_PROJECT", "vyrex-rl")
        self.run_id = os.getenv("WANDB_RUN_ID", "")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.run_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_connection(self):
        """Lazily initialize the wandb API and run object."""
        if self._api is None:
            import wandb
            os.environ["WANDB_API_KEY"] = self.api_key
            self._api = wandb.Api(timeout=30)
            logger.info("WandB API initialized")

        if self._run is None and self.run_id:
            path = (
                f"{self.entity}/{self.project}/{self.run_id}"
                if self.entity
                else f"{self.project}/{self.run_id}"
            )
            self._run = self._api.run(path)
            logger.info(f"Connected to run: {path}")

    def _cache_get(self, key: str, ttl: int = 30) -> Optional[Any]:
        if key in self._cache:
            data, ts = self._cache[key]
            if time.time() - ts < ttl:
                return data
        return None

    def _cache_set(self, key: str, data: Any):
        self._cache[key] = (data, time.time())

    @staticmethod
    def _sanitize_value(v: Any) -> Any:
        """Convert NaN/Inf to None for JSON serialization."""
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_run_info(self) -> Dict:
        cached = self._cache_get("run_info", ttl=60)
        if cached is not None:
            return cached

        try:
            self._ensure_connection()
            # Use Cumulative Timesteps for real step count (WandB _step is
            # the logging step, not the environment step count)
            summary = self._run.summary
            total_steps = int(
                summary.get("Cumulative Timesteps", 0)
                or summary.get("_step", 0)
            )
            info = {
                "id": self._run.id,
                "name": self._run.name or "",
                "state": self._run.state,
                "createdAt": str(getattr(self._run, "created_at", "")),
                "heartbeatAt": str(getattr(self._run, "heartbeat_at", "")),
                "tags": list(self._run.tags) if self._run.tags else [],
                "totalSteps": total_steps,
                "config": {
                    k: self._sanitize_value(v)
                    for k, v in dict(self._run.config).items()
                    if not k.startswith("_")
                },
            }
            self._cache_set("run_info", info)
            return info
        except Exception as e:
            logger.error(f"get_run_info failed: {e}")
            return {"error": str(e)}

    def get_latest_metrics(self) -> Dict[str, Any]:
        cached = self._cache_get("latest_metrics", ttl=15)
        if cached is not None:
            return cached

        try:
            self._ensure_connection()
            # Re-fetch run to get latest summary
            path = (
                f"{self.entity}/{self.project}/{self.run_id}"
                if self.entity
                else f"{self.project}/{self.run_id}"
            )
            self._run = self._api.run(path)
            summary = self._run.summary

            metrics: Dict[str, Any] = {}
            for key in summary.keys():
                if key.startswith("_"):
                    continue
                val = summary[key]
                if not isinstance(val, (int, float)):
                    continue
                sanitized = self._sanitize_value(val)
                # Map to frontend key if mapping exists, otherwise keep raw
                frontend_key = WANDB_TO_FRONTEND.get(key, key)
                metrics[frontend_key] = sanitized

            # Use Cumulative Timesteps as _step (real env step count)
            cum_steps = summary.get("Cumulative Timesteps")
            if cum_steps is not None:
                metrics["_step"] = int(cum_steps)
            else:
                metrics["_step"] = self._sanitize_value(summary.get("_step", 0))

            self._cache_set("latest_metrics", metrics)
            return metrics
        except Exception as e:
            logger.error(f"get_latest_metrics failed: {e}")
            return {"error": str(e)}

    def _build_step_interpolator(self, n_points: int = 200) -> callable:
        """
        Build a function that maps WandB logging _step → Cumulative Timesteps.
        Uses linear interpolation from PPO metric rows that have both.
        Returns a callable: step_interpolator(_step) -> env_step
        """
        cache_key = "_step_interpolator"
        cached = self._cache_get(cache_key, ttl=300)
        if cached is not None:
            return cached

        rows = self._run.history(
            keys=["Cumulative Timesteps", "_step"],
            samples=n_points,
            pandas=False,
        )
        if not isinstance(rows, list):
            rows = list(rows)

        # Build sorted list of (_step, cumulative) pairs
        pairs = []
        for row in rows:
            s = row.get("_step")
            c = row.get("Cumulative Timesteps")
            if s is not None and c is not None:
                pairs.append((int(s), int(c)))
        pairs.sort()

        if not pairs:
            # Fallback: identity
            fn = lambda step: step
            self._cache_set(cache_key, fn)
            return fn

        def interpolate(step: int) -> int:
            if step <= pairs[0][0]:
                return pairs[0][1]
            if step >= pairs[-1][0]:
                return pairs[-1][1]
            # Binary search for bracket
            lo, hi = 0, len(pairs) - 1
            while lo < hi - 1:
                mid = (lo + hi) // 2
                if pairs[mid][0] <= step:
                    lo = mid
                else:
                    hi = mid
            s0, c0 = pairs[lo]
            s1, c1 = pairs[hi]
            if s1 == s0:
                return c0
            frac = (step - s0) / (s1 - s0)
            return int(c0 + frac * (c1 - c0))

        self._cache_set(cache_key, interpolate)
        return interpolate

    def _fetch_single_key_history(
        self, wandb_key: str, samples: int, include_cumulative: bool = True
    ) -> List[Dict]:
        """Fetch history for a single WandB key + step columns."""
        fetch_keys = [wandb_key, "_step"]
        if include_cumulative:
            fetch_keys.append("Cumulative Timesteps")
        rows = self._run.history(
            keys=fetch_keys,
            samples=samples,
            pandas=False,
        )
        if not isinstance(rows, list):
            rows = list(rows)
        return rows

    def get_metric_history(
        self,
        keys: List[str],
        samples: int = 500,
    ) -> Dict[str, Any]:
        """
        Fetch historical metric data, downsampled to `samples` points.

        WandB logs PPO metrics and game metrics on separate history rows
        (different wandb.log() calls). PPO rows contain `Cumulative Timesteps`
        (real env step count), while game metric rows only have `_step`
        (logging step). We handle this by:
        1. Fetching each key independently
        2. For PPO metrics: using Cumulative Timesteps directly
        3. For game metrics: using a step interpolation table to convert
           _step → Cumulative Timesteps

        Returns { "metric_name": {"_step": [...], "value": [...]}, ... }
        """
        cache_key = f"history_{'_'.join(sorted(keys))}_{samples}"
        cached = self._cache_get(cache_key, ttl=120)
        if cached is not None:
            return cached

        try:
            self._ensure_connection()

            # Identify which keys are PPO (have Cumulative Timesteps) vs game
            # PPO keys are those whose WandB name does NOT start with "vyrex/" or "diagnostics/"
            ppo_keys = []
            game_keys = []
            for fk in keys:
                wk = FRONTEND_TO_WANDB.get(fk, fk)
                if wk.startswith("vyrex/") or wk.startswith("diagnostics/"):
                    game_keys.append(fk)
                else:
                    ppo_keys.append(fk)

            # Build step interpolator if we have game keys
            interpolator = None
            if game_keys:
                interpolator = self._build_step_interpolator()

            result: Dict[str, Any] = {}

            # Fetch PPO metrics (have Cumulative Timesteps)
            for fk in ppo_keys:
                wk = FRONTEND_TO_WANDB.get(fk, fk)
                rows = self._fetch_single_key_history(wk, samples, include_cumulative=True)

                steps: List = []
                values: List = []
                for row in rows:
                    val = row.get(wk)
                    if val is None:
                        continue
                    cum = row.get("Cumulative Timesteps")
                    step_val = int(cum) if cum is not None else row.get("_step", 0)
                    steps.append(self._sanitize_value(step_val))
                    values.append(self._sanitize_value(val))

                result[fk] = {"_step": steps, "value": values}

            # Fetch game metrics (no Cumulative Timesteps, use interpolation)
            for fk in game_keys:
                wk = FRONTEND_TO_WANDB.get(fk, fk)
                rows = self._fetch_single_key_history(wk, samples, include_cumulative=False)

                steps: List = []
                values: List = []
                for row in rows:
                    val = row.get(wk)
                    if val is None:
                        continue
                    raw_step = row.get("_step", 0)
                    # Convert logging step → env step
                    env_step = interpolator(raw_step) if interpolator else raw_step
                    steps.append(self._sanitize_value(env_step))
                    values.append(self._sanitize_value(val))

                result[fk] = {"_step": steps, "value": values}

            self._cache_set(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"get_metric_history failed: {e}")
            return {"error": str(e)}

    def invalidate_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        logger.info("Cache invalidated")
