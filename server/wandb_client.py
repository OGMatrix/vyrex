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
            info = {
                "id": self._run.id,
                "name": self._run.name or "",
                "state": self._run.state,
                "createdAt": str(getattr(self._run, "created_at", "")),
                "heartbeatAt": str(getattr(self._run, "heartbeat_at", "")),
                "tags": list(self._run.tags) if self._run.tags else [],
                "totalSteps": int(self._run.summary.get("_step", 0)),
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
                    # Keep _step for reference
                    if key == "_step":
                        metrics[key] = self._sanitize_value(summary[key])
                    continue
                val = summary[key]
                if isinstance(val, (int, float)):
                    metrics[key] = self._sanitize_value(val)

            self._cache_set("latest_metrics", metrics)
            return metrics
        except Exception as e:
            logger.error(f"get_latest_metrics failed: {e}")
            return {"error": str(e)}

    def get_metric_history(
        self,
        keys: List[str],
        samples: int = 500,
    ) -> Dict[str, List]:
        """
        Fetch historical metric data, downsampled to `samples` points.
        Returns { "_step": [...], "metric_name": [...], ... }
        """
        cache_key = f"history_{'_'.join(sorted(keys))}_{samples}"
        cached = self._cache_get(cache_key, ttl=120)
        if cached is not None:
            return cached

        try:
            self._ensure_connection()

            # Use pandas=False for lighter weight when possible
            try:
                df = self._run.history(
                    keys=keys + ["_step"],
                    samples=samples,
                    pandas=True,
                )
                result: Dict[str, List] = {
                    "_step": [self._sanitize_value(v) for v in df["_step"].tolist()]
                }
                for key in keys:
                    if key in df.columns:
                        result[key] = [
                            self._sanitize_value(v) for v in df[key].tolist()
                        ]
                    else:
                        result[key] = []
            except Exception:
                # Fallback: use scan_history (slower but more reliable)
                rows = list(self._run.scan_history(keys=keys + ["_step"]))
                # Downsample if needed
                if len(rows) > samples:
                    step = max(1, len(rows) // samples)
                    rows = rows[::step]

                result = {"_step": []}
                for key in keys:
                    result[key] = []

                for row in rows:
                    result["_step"].append(self._sanitize_value(row.get("_step", 0)))
                    for key in keys:
                        result[key].append(self._sanitize_value(row.get(key)))

            self._cache_set(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"get_metric_history failed: {e}")
            return {"error": str(e)}

    def invalidate_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        logger.info("Cache invalidated")
