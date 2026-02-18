"""
VYREX — Backend API Server
============================
FastAPI server that proxies WandB data and serves version history.

Usage:
    cd server
    cp .env.example .env   # Fill in your WandB credentials
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000
"""

import os
import json
import asyncio
import logging
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

load_dotenv()

from wandb_client import WandBClient

# ============================================================================
# App Setup
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vyrex.server")

app = FastAPI(
    title="VYREX API",
    description="Backend for the VYREX Rocket League AI dashboard",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:4173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

wandb_client = WandBClient()

# ============================================================================
# Version History — Manually Curated from Training Reports
# ============================================================================

VERSIONS = [
    {
        "id": "v1.0",
        "name": "Genesis",
        "stepRange": [0, 251_000_000],
        "date": "2026-02-15",
        "status": "completed",
        "summary": "Initial training run. Bot learned basic ball approach and driving, but developed a severe pathological jumping behavior — airborne 85% of the time due to unconditional InAirReward giving free reward for any jump.",
        "changes": [],
        "impact": [
            {
                "metric": "Airborne",
                "before": None,
                "after": 85.5,
                "unit": "%",
                "trend": "up",
                "isGood": False,
            },
            {
                "metric": "Touches/step",
                "before": 0,
                "after": 0.013,
                "unit": "",
                "trend": "up",
                "isGood": True,
            },
            {
                "metric": "Avg Speed",
                "before": 0,
                "after": 502,
                "unit": "uu/s",
                "trend": "up",
                "isGood": True,
            },
            {
                "metric": "Entropy",
                "before": None,
                "after": 4.452,
                "unit": "",
                "trend": "stable",
                "isGood": True,
            },
        ],
        "assessment": {
            "grade": "C",
            "summary": "Bot learned ball approach but jumping pathology destroyed all gameplay. 250M steps of bad reward signal deeply entrenched in the policy.",
        },
    },
    {
        "id": "v2.0",
        "name": "The Fix",
        "stepRange": [251_000_000, 401_000_000],
        "date": "2026-02-15",
        "status": "completed",
        "summary": "Applied three critical fixes: conditional InAirReward (ball height > 300uu), added GroundedReward, and wired in curriculum state mutators. Airborne dropped from 85% to 70% — directionally correct but painfully slow at 0.075pp per million steps.",
        "changes": [
            {
                "parameter": "InAirReward",
                "oldValue": "Unconditional",
                "newValue": "Conditional (ball > 300uu)",
                "rationale": "Stop rewarding pointless jumping when the ball is on the ground",
            },
            {
                "parameter": "grounded_weight",
                "oldValue": "0 (none)",
                "newValue": "0.005",
                "rationale": "Dense positive signal for staying grounded",
            },
            {
                "parameter": "Curriculum Phase",
                "oldValue": "Phase 1 (kickoffs only)",
                "newValue": "Phase 2 (30% random states)",
                "rationale": "Diversify starting positions for more robust learning",
            },
        ],
        "impact": [
            {
                "metric": "Airborne",
                "before": 85.5,
                "after": 70.7,
                "unit": "%",
                "trend": "down",
                "isGood": True,
            },
            {
                "metric": "Touches/step",
                "before": 0.013,
                "after": 0.026,
                "unit": "",
                "trend": "up",
                "isGood": True,
            },
            {
                "metric": "Avg Speed",
                "before": 502,
                "after": 669,
                "unit": "uu/s",
                "trend": "up",
                "isGood": True,
            },
            {
                "metric": "Touch Rate",
                "before": 0.013,
                "after": 0.025,
                "unit": "",
                "trend": "up",
                "isGood": True,
            },
        ],
        "assessment": {
            "grade": "B-",
            "summary": "Right direction but too slow. GroundedReward at 0.005 was a tiny signal competing against 250M steps of deeply-entrenched jumping behavior.",
        },
    },
    {
        "id": "v3.0",
        "name": "Shock Therapy",
        "stepRange": [401_000_000, 433_000_000],
        "date": "2026-02-16",
        "status": "completed",
        "summary": "10× increase in grounded_weight (0.005→0.05) combined with halved entropy coefficient (0.01→0.005) and Phase 3 curriculum. The jumping pathology was eliminated within 5M steps — but overcorrected to just 3.2% airborne. Touch rate doubled. Goals more than doubled.",
        "changes": [
            {
                "parameter": "grounded_weight",
                "oldValue": "0.005",
                "newValue": "0.05",
                "rationale": "10× increase to force behavioral reset from 70% airborne",
            },
            {
                "parameter": "ppo_ent_coef",
                "oldValue": "0.01",
                "newValue": "0.005",
                "rationale": "Reduce exploration, encourage policy specialization",
            },
            {
                "parameter": "Curriculum Phase",
                "oldValue": "Phase 2 (30% random)",
                "newValue": "Phase 3 (70% random)",
                "rationale": "More diverse game states with established ground play",
            },
        ],
        "impact": [
            {
                "metric": "Airborne",
                "before": 70.7,
                "after": 3.2,
                "unit": "%",
                "trend": "down",
                "isGood": False,
            },
            {
                "metric": "Touches/step",
                "before": 0.026,
                "after": 0.042,
                "unit": "",
                "trend": "up",
                "isGood": True,
            },
            {
                "metric": "Avg Speed",
                "before": 669,
                "after": 848,
                "unit": "uu/s",
                "trend": "up",
                "isGood": True,
            },
            {
                "metric": "Goals/window",
                "before": 16,
                "after": 41,
                "unit": "",
                "trend": "up",
                "isGood": True,
            },
        ],
        "assessment": {
            "grade": "B+",
            "summary": "Effective behavioral reset — jumping pathology completely eliminated. But overcorrected: bot now refuses to leave the ground, even for essential dodges and low aerials.",
        },
    },
    {
        "id": "v3.1",
        "name": "Gentle Recovery",
        "stepRange": [433_000_000, 461_000_000],
        "date": "2026-02-16",
        "status": "completed",
        "summary": "Reduced grounded_weight from 0.05 to 0.015 to coax the bot back into the air. Too conservative — policy got trapped at 4.3% airborne equilibrium. Still too strong a ground signal for meaningful recovery.",
        "changes": [
            {
                "parameter": "grounded_weight",
                "oldValue": "0.05",
                "newValue": "0.015",
                "rationale": "Relax ground constraint to allow jumping recovery",
            },
        ],
        "impact": [
            {
                "metric": "Airborne",
                "before": 3.2,
                "after": 4.3,
                "unit": "%",
                "trend": "up",
                "isGood": True,
            },
            {
                "metric": "Touches/step",
                "before": 0.042,
                "after": 0.054,
                "unit": "",
                "trend": "up",
                "isGood": True,
            },
            {
                "metric": "Avg Speed",
                "before": 848,
                "after": 955,
                "unit": "uu/s",
                "trend": "up",
                "isGood": True,
            },
        ],
        "assessment": {
            "grade": "C+",
            "summary": "Equilibrium trap. The dense grounded signal at 0.015 still dominated, holding airborne at ~4%. Speed and touches improved but aerial recovery stalled.",
        },
    },
    {
        "id": "v3.2",
        "name": "Goldilocks",
        "stepRange": [461_000_000, 521_000_000],
        "date": "2026-02-16",
        "status": "completed",
        "summary": "Found the sweet spot at grounded_weight=0.003. Airborne recovered from 4.4% to 34.6% over 60M steps. Most remarkable: entropy INCREASED for the first time ever — the policy actively rediscovered exploration. Most stable PPO session to date.",
        "changes": [
            {
                "parameter": "grounded_weight",
                "oldValue": "0.015",
                "newValue": "0.003",
                "rationale": "Break equilibrium trap — make jump cost negligible",
            },
        ],
        "impact": [
            {
                "metric": "Airborne",
                "before": 4.4,
                "after": 34.6,
                "unit": "%",
                "trend": "up",
                "isGood": True,
            },
            {
                "metric": "Entropy",
                "before": 4.111,
                "after": 4.226,
                "unit": "",
                "trend": "up",
                "isGood": True,
            },
            {
                "metric": "KL Divergence",
                "before": 0.002,
                "after": 0.0015,
                "unit": "",
                "trend": "down",
                "isGood": True,
            },
            {
                "metric": "VF Loss",
                "before": 0.018,
                "after": 0.017,
                "unit": "",
                "trend": "down",
                "isGood": True,
            },
        ],
        "assessment": {
            "grade": "A",
            "summary": "Textbook recovery. Entropy rising for the first time = healthy re-exploration. PPO training rock-solid: KL flat, VF loss declining, clip fraction low.",
        },
    },
    {
        "id": "v3.3",
        "name": "The Marathon",
        "stepRange": [521_000_000, 897_000_000],
        "date": "2026-02-16",
        "status": "completed",
        "summary": "376M steps overnight with zero config changes — the longest uninterrupted session ever. Touch rate exploded +53% to all-time high. Speed hit all-time record. Most stable PPO run in history. Airborne settled at 47% (higher than target but coupled with real skill gains).",
        "changes": [],
        "impact": [
            {
                "metric": "Touches/step",
                "before": 0.049,
                "after": 0.075,
                "unit": "",
                "trend": "up",
                "isGood": True,
            },
            {
                "metric": "Avg Speed",
                "before": 889,
                "after": 976,
                "unit": "uu/s",
                "trend": "up",
                "isGood": True,
            },
            {
                "metric": "Airborne",
                "before": 34.6,
                "after": 47.8,
                "unit": "%",
                "trend": "up",
                "isGood": False,
            },
            {
                "metric": "VF Loss",
                "before": 0.017,
                "after": 0.013,
                "unit": "",
                "trend": "down",
                "isGood": True,
            },
            {
                "metric": "Goals/window",
                "before": 64,
                "after": 51,
                "unit": "",
                "trend": "down",
                "isGood": False,
            },
            {
                "metric": "Entropy",
                "before": 4.226,
                "after": 4.229,
                "unit": "",
                "trend": "stable",
                "isGood": True,
            },
        ],
        "assessment": {
            "grade": "A-",
            "summary": "Strongest skill gains in training history. Touch rate +53%, speed ATH. Aerial efficiency declined 61% — the extra air time is mostly dodge-jumping for speed, not productive aerials.",
        },
    },
]


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "wandb_configured": wandb_client.is_configured,
    }


@app.get("/api/run")
async def get_run():
    if not wandb_client.is_configured:
        return {"error": "WandB not configured. Set WANDB_API_KEY and WANDB_RUN_ID in .env"}
    return wandb_client.get_run_info()


@app.get("/api/metrics/latest")
async def get_latest_metrics():
    if not wandb_client.is_configured:
        return {"error": "WandB not configured"}
    return wandb_client.get_latest_metrics()


@app.get("/api/metrics/history")
async def get_metric_history(
    keys: str = Query(
        default="reward,entropy",
        description="Comma-separated metric keys",
    ),
    samples: int = Query(default=500, ge=10, le=5000),
):
    if not wandb_client.is_configured:
        return {"error": "WandB not configured"}
    key_list = [k.strip() for k in keys.split(",") if k.strip()]
    return wandb_client.get_metric_history(key_list, samples=samples)


@app.get("/api/versions")
async def get_versions():
    return VERSIONS


@app.get("/api/metrics/stream")
async def metrics_stream():
    """
    Server-Sent Events endpoint for live metric updates.
    Polls WandB every 30 seconds and pushes new data to connected clients.
    """

    async def event_generator():
        while True:
            try:
                if wandb_client.is_configured:
                    wandb_client.invalidate_cache()
                    metrics = wandb_client.get_latest_metrics()
                    if "error" not in metrics:
                        yield f"data: {json.dumps(metrics)}\n\n"
                    else:
                        yield f"data: {json.dumps({'_error': metrics['error']})}\n\n"
                else:
                    yield f"data: {json.dumps({'_error': 'not_configured'})}\n\n"
            except Exception as e:
                logger.error(f"SSE error: {e}")
                yield f"data: {json.dumps({'_error': str(e)})}\n\n"

            await asyncio.sleep(30)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/cache/invalidate")
async def invalidate_cache():
    wandb_client.invalidate_cache()
    return {"status": "cache_invalidated"}
