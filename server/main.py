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
        "stepRange": [521_000_000, 1_770_000_000],
        "date": "2026-02-16",
        "status": "completed",
        "summary": "1.25B steps with zero config changes — the longest uninterrupted stretch of training. Began with an overnight session (521M-897M) where touch rate exploded +53% and speed hit ATH. Then the policy matured (897M-1.1B, speed peaked 1059 uu/s), crashed catastrophically (1.1B-1.2B, speed plummeted to 776, touches spiked 2×), and slowly recovered over a 560M-step plateau (1.2B-1.77B). By the end, speed recovered to 960 and touches normalized — but the policy was treading water. The case for config intervention was clear.",
        "changes": [],
        "impact": [
            {
                "metric": "Touches/step",
                "before": 0.049,
                "after": 0.052,
                "unit": "",
                "trend": "up",
                "isGood": True,
            },
            {
                "metric": "Avg Speed",
                "before": 889,
                "after": 960,
                "unit": "uu/s",
                "trend": "up",
                "isGood": True,
            },
            {
                "metric": "Airborne",
                "before": 34.6,
                "after": 51.4,
                "unit": "%",
                "trend": "up",
                "isGood": False,
            },
            {
                "metric": "Avg Boost",
                "before": 7.5,
                "after": 7.5,
                "unit": "",
                "trend": "stable",
                "isGood": False,
            },
            {
                "metric": "Entropy",
                "before": 4.226,
                "after": 4.19,
                "unit": "",
                "trend": "stable",
                "isGood": True,
            },
        ],
        "assessment": {
            "grade": "B-",
            "summary": "Massive training runway but uneven results. The overnight session showed the policy's potential (touch rate +53%, speed ATH). But the subsequent crash and 560M-step plateau exposed limits of the reward structure — the bot needed new signals to break through.",
        },
    },
    {
        "id": "v3.4",
        "name": "The Overhaul",
        "stepRange": [1_770_000_000, 2_375_000_000],
        "date": "2026-02-17",
        "status": "completed",
        "summary": "Three rapid-fire config iterations applied over 600M steps. First: BoostChangeReward and WavedashReward taught boost economy and ground technique (airborne instantly dropped from 51% to 36%). Then boost_conservation tripled to lock in the gains (speed jumped 975→1045). Finally: asymmetric GoalReward (2× concede penalty), WallPlayReward, doubled demo_weight, and increased save_weight rounded out the defensive toolkit. By the end, boost averaged 13.5 (from 7.5) and speed hit a new ATH of 1090.",
        "changes": [
            {
                "parameter": "BoostChangeReward",
                "oldValue": "None",
                "newValue": "0.04",
                "rationale": "Reward boost pad collection — bot was at 7% avg boost, catastrophically low",
            },
            {
                "parameter": "WavedashReward",
                "oldValue": "None",
                "newValue": "0.02",
                "rationale": "Reward controlled flip technique for ground-level speed",
            },
            {
                "parameter": "boost_conservation",
                "oldValue": "0.01",
                "newValue": "0.03",
                "rationale": "Triple the boost economy signal to lock in pad-collection behavior",
            },
            {
                "parameter": "boost_change_lose",
                "oldValue": "None",
                "newValue": "0.3",
                "rationale": "Mild penalty for wasteful boost spending",
            },
            {
                "parameter": "WallPlayReward",
                "oldValue": "None",
                "newValue": "0.06",
                "rationale": "Encourage side wall and corner plays — bot avoided walls entirely",
            },
            {
                "parameter": "GoalReward",
                "oldValue": "Symmetric",
                "newValue": "Asymmetric (concede_multiplier=2.0)",
                "rationale": "Punish conceding 2× harder than scoring — defend more seriously",
            },
            {
                "parameter": "save_weight",
                "oldValue": "0.15",
                "newValue": "0.25",
                "rationale": "Stronger incentive for defensive saves",
            },
            {
                "parameter": "demo_weight",
                "oldValue": "0.4",
                "newValue": "0.8",
                "rationale": "Zero demos in 400M+ steps — needs much stronger signal",
            },
        ],
        "impact": [
            {
                "metric": "Avg Boost",
                "before": 7.5,
                "after": 13.5,
                "unit": "",
                "trend": "up",
                "isGood": True,
            },
            {
                "metric": "Airborne",
                "before": 51.4,
                "after": 38.2,
                "unit": "%",
                "trend": "down",
                "isGood": True,
            },
            {
                "metric": "Avg Speed",
                "before": 960,
                "after": 1090,
                "unit": "uu/s",
                "trend": "up",
                "isGood": True,
            },
            {
                "metric": "Touches/step",
                "before": 0.052,
                "after": 0.059,
                "unit": "",
                "trend": "up",
                "isGood": True,
            },
        ],
        "assessment": {
            "grade": "A",
            "summary": "The plateau breaker. Three targeted interventions transformed boost economy (7.5→13.5), grounded the bot (51%→38% airborne), and pushed speed to ATH 1090. The most productive config period in training history.",
        },
    },
    {
        "id": "v3.5",
        "name": "Unlock Techniques",
        "stepRange": [2_375_000_000, 2_860_000_000],
        "date": "2026-02-17",
        "status": "completed",
        "summary": "Philosophical shift: REMOVED FaceBallReward and GroundedReward entirely. Halved speed_toward_ball. Added AerialPlayReward (0.15). Raised goal_weight to 25, directed_touch to 0.40, velocity_toward_goal to 0.20. The intent was to reduce the dense reward comfort floor and let techniques emerge. Over 275M steps, airborne rose from 38% to 49% but aerial touch rate did NOT improve — the positioning component rewarded floating without contact. In-game test revealed one bot playing correctly while the teammate drove on walls or stood still.",
        "changes": [
            {
                "parameter": "FaceBallReward",
                "oldValue": "0.002",
                "newValue": "REMOVED",
                "rationale": "Stop rewarding passive ball-watching — internalized at 2.38B",
            },
            {
                "parameter": "GroundedReward",
                "oldValue": "0.008",
                "newValue": "REMOVED",
                "rationale": "Stop penalizing airborne play — airborne stable at 38%, safe to remove",
            },
            {
                "parameter": "AerialPlayReward",
                "oldValue": "None",
                "newValue": "0.15",
                "rationale": "Reward aerial positioning and aerial ball contacts",
            },
            {
                "parameter": "goal_weight",
                "oldValue": "15",
                "newValue": "25",
                "rationale": "Dense reward floor was 3.2× goal reward — goals MUST dominate",
            },
            {
                "parameter": "directed_touch",
                "oldValue": "0.25",
                "newValue": "0.40",
                "rationale": "53% directed at 2.38B — barely above coin flip. Must aim.",
            },
            {
                "parameter": "velocity_toward_goal",
                "oldValue": "0.12",
                "newValue": "0.20",
                "rationale": "Stronger directional shooting incentive",
            },
            {
                "parameter": "speed_toward_ball",
                "oldValue": "0.012",
                "newValue": "0.005",
                "rationale": "Halved — was #4 comfort signal, internalized at 2.38B",
            },
        ],
        "impact": [
            {
                "metric": "Airborne",
                "before": 38.2,
                "after": 49.0,
                "unit": "%",
                "trend": "up",
                "isGood": False,
            },
            {
                "metric": "Avg Boost",
                "before": 13.5,
                "after": 11.2,
                "unit": "",
                "trend": "down",
                "isGood": False,
            },
            {
                "metric": "Avg Speed",
                "before": 1090,
                "after": 1075,
                "unit": "uu/s",
                "trend": "down",
                "isGood": False,
            },
            {
                "metric": "Aerial Touches",
                "before": 0.021,
                "after": 0.021,
                "unit": "",
                "trend": "stable",
                "isGood": False,
            },
        ],
        "assessment": {
            "grade": "C+",
            "summary": "Brave philosophical pivot but mixed results. Removing ground/face constraints unleashed airborne play, yet aerial touch rate flatlined. The AerialPlayReward's positioning component rewarded floating without contact. In-game test exposed teammate passivity.",
        },
    },
    {
        "id": "v3.6",
        "name": "Fix the Teammate",
        "stepRange": [2_860_000_000, 3_130_000_000],
        "date": "2026-02-18",
        "status": "completed",
        "summary": "Three surgical fixes targeting the teammate passivity observed in v3.5 testing. AerialPlayReward positioning component removed (weight 0.15→0.10) to stop rewarding pointless floating. RotationReward gained X-axis centering and speed awareness. TeammateSpacingReward overshoot decay steepened (0.3→0.6). Over 270M steps, airborne stayed at 51-53% (unchanged), teammate distance didn't converge, and goal diff was volatile. The X-centering in RotationReward caused boost starvation by pulling bot away from wall boost pads. Match test at 3.13B: 8-0 win vs weak bots but entropy at 93%, 58.6% zero boost, 48.8% directed touches (coin flip), and constant left-right wiggling from near-random action selection.",
        "changes": [
            {
                "parameter": "AerialPlayReward",
                "oldValue": "0.15 (positioning + touches)",
                "newValue": "0.10 (touches only)",
                "rationale": "52% airborne was too high — remove free reward for floating",
            },
            {
                "parameter": "RotationReward",
                "oldValue": "Context-aware (near/far)",
                "newValue": "+X-centering +speed awareness",
                "rationale": "Reward staying near field center with speed, not wall-camping",
            },
            {
                "parameter": "TeammateSpacingReward",
                "oldValue": "overshoot_decay=0.3",
                "newValue": "overshoot_decay=0.6",
                "rationale": "3462 avg spacing was barely penalized — steeper decay needed",
            },
        ],
        "impact": [
            {
                "metric": "Airborne %",
                "before": 49.2,
                "after": 51.9,
                "unit": "%",
                "trend": "up",
                "isGood": False,
            },
            {
                "metric": "Teammate Distance",
                "before": 3111,
                "after": 3383,
                "unit": "uu",
                "trend": "up",
                "isGood": False,
            },
            {
                "metric": "Goal Diff",
                "before": 1,
                "after": 1,
                "unit": "",
                "trend": "stable",
                "isGood": False,
            },
            {
                "metric": "Avg Boost (match)",
                "before": 0,
                "after": 19.7,
                "unit": "",
                "trend": "down",
                "isGood": False,
            },
        ],
        "assessment": {
            "grade": "C",
            "summary": "1-for-3 version. Teammate spacing decay slightly improved but airborne unchanged, and RotationReward's X-centering caused severe boost starvation (58.6% zero boost in match). Match test revealed 93% entropy — near-random action selection causing visible wiggling. Policy cannot commit to actions. Config changes were too indirect to move the needle.",
        },
    },
    {
        "id": "v3.7",
        "name": "The Sharpening",
        "stepRange": [3_130_000_000, None],
        "date": "2026-02-18",
        "status": "active",
        "summary": "Revolution through subtraction. Match analysis at 3.13B revealed entropy at 93% (4.165/4.500), 58.6% zero boost, 48.8% directed touches, and constant wiggling from near-random action selection. Root causes: 15 reward components creating gradient collision, team_spirit at 0.8 diluting 40% of individual gradient with teammate noise, and ent_coef too high. Fix: Remove 3 noise components (SpeedTowardBall, BoostConservation, RotationReward X-centering/speed), reduce team_spirit 0.8→0.3 for 5.7x cleaner individual gradient, drop ent_coef 0.003→0.001 for action commitment, increase batch 150K→200K, boost collection 2.5x stronger (0.04→0.10), wavedash 2.5x (0.02→0.05), concede 2.0→1.5 for aggression. 15→12 components.",
        "changes": [
            {
                "parameter": "ppo_ent_coef",
                "oldValue": "0.003",
                "newValue": "0.001",
                "rationale": "Entropy at 93% (4.165/4.50) at 3.13B — policy near-random. Must commit to actions to stop wiggling.",
            },
            {
                "parameter": "team_spirit_end",
                "oldValue": "0.8",
                "newValue": "0.3",
                "rationale": "40% teammate noise gradient. Individual mechanics not solid — 5.7x cleaner signal at 0.3.",
            },
            {
                "parameter": "ppo_batch_size",
                "oldValue": "150,000",
                "newValue": "200,000",
                "rationale": "Larger batches for cleaner gradient estimates. Zealan: increase once scoring.",
            },
            {
                "parameter": "SpeedTowardBallReward",
                "oldValue": "0.005",
                "newValue": "0.0 (REMOVED)",
                "rationale": "Fully internalized — 0.001/tick is pure gradient noise at 3.13B.",
            },
            {
                "parameter": "BoostConservationReward",
                "oldValue": "0.03",
                "newValue": "0.0 (REMOVED)",
                "rationale": "Redundant with BoostChangeReward. Rewards HAVING boost, not GETTING it. 80 rew/ep noise.",
            },
            {
                "parameter": "RotationReward",
                "oldValue": "Y-pos + X-centering + speed",
                "newValue": "Y-position only",
                "rationale": "X-centering pulled bot away from wall boost pads → 58.6% zero boost. Removed 2 competing gradients.",
            },
            {
                "parameter": "boost_change_weight",
                "oldValue": "0.04",
                "newValue": "0.10",
                "rationale": "58.6% zero boost is catastrophic. #1 mechanical prerequisite for speed, aerials, power shots.",
            },
            {
                "parameter": "wavedash_weight",
                "oldValue": "0.02",
                "newValue": "0.05",
                "rationale": "Only 39 flips in 7-min match. Flipping is THE primary speed mechanic.",
            },
            {
                "parameter": "concede_multiplier",
                "oldValue": "2.0",
                "newValue": "1.5",
                "rationale": "Zealan's aggression_bias — reduce fear of conceding to promote risky, aggressive play.",
            },
        ],
        "impact": [
            "Entropy: 4.07→3.63 (dropped 0.44 in first 200M, then plateaued for 500M steps)",
            "Airborne: 53%→25-26% in match (grounded play restored)",
            "Supersonic: 1.5%→3.3-4.6% (2-3x improvement)",
            "Flips: 39→37-59 per match (flip commitment increased)",
            "DEF state: 1.5%→7.4-22.9% (defensive awareness emerged)",
            "Confidence: 4.46%→9.35-10.92% (2x improvement)",
            "PROBLEM: Entropy wall at 3.63 for 500M+ steps — policy frozen",
            "PROBLEM: Zero boost 54.9%→71-73% — boost starvation WORSENED",
            "PROBLEM: Directed touches 50%→36-55% — flip overshoot causing wrong-direction hits",
            "PROBLEM: Double commits 8.7%→10.9% (Match 1) — spacing signal too weak",
        ],
        "assessment": {
            "grade": "B",
            "summary": "Transformative first half, frozen second half. Entropy dropped 4.07→3.63 in first 200M steps — wiggling eliminated, confidence doubled, defense emerged, airborne halved. But entropy hit a wall at 3.63 and ALL metrics plateaued for 500M steps. Boost starvation worsened to 73% zero-boost in match. Bot learned to flip aggressively (wavedash_weight worked) but overshoots — flips are ballistic and the bot can't course-correct without boost. 4-1 and 6-2 wins show raw effectiveness but precision is missing. v3.7's subtraction philosophy was correct; the policy just needs more room to sharpen.",
        },
    },
    {
        "id": "v3.8",
        "name": "The Coordinator",
        "stepRange": [4_030_000_000, 5_005_000_000],
        "date": "2026-02-19",
        "status": "closed",
        "summary": "Two surgical changes to break the entropy wall and fix coordination. Entropy stuck at 3.63 for 500M+ steps — policy frozen, can't learn pad routes or flip accuracy. Match analysis at 4.03B revealed: 73% zero boost (up from 55%), 36-55% directed touches (flip overshoot), 10.9% double commits (up from 8.7%). Root cause chain: entropy wall → can't learn pad collection → boost starvation → forced flip-approaches → ballistic whiffs. Fix: ent_coef 0.001→0.0008 (break the wall), teammate_spacing 0.012→0.025 (fix coordination drift). No new rewards — preserve v3.7's clean 12-component architecture.",
        "changes": [
            {
                "parameter": "ppo_ent_coef",
                "oldValue": "0.001",
                "newValue": "0.0008",
                "rationale": "Entropy wall at 3.63 for 500M+ steps. Policy frozen — can't develop pad routes, flip timing, or touch precision. 20% reduction to push toward 3.45-3.50.",
            },
            {
                "parameter": "teammate_spacing_weight",
                "oldValue": "0.012",
                "newValue": "0.025",
                "rationale": "Double commits 8.7%→10.9%, teammate distance drifted 2085→2801 between 3.73B and 4.03B matches. 2x increase to restore coordination.",
            },
        ],
        "impact": [],
        "assessment": {
            "grade": "B+",
            "summary": "Successfully broke entropy wall #1 (3.63→3.49) and restored team coordination. Aerial touch rate surged +65% (0.023→0.039). VF loss fell steadily (0.019→0.015) showing critic improvement. But entropy hit wall #2 at 3.51 for 825M steps. Boost starvation persisted at 72.4% zero-boost. Movement quality improved visibly — bot now does small aerials and has solid ground movement — but can't sustain high aerials or saves without boost. 8-1 win at 5.0B shows raw effectiveness but 0 saves, 2% supersonic, 40.6% directed touches reveal ceiling imposed by boost drought.",
        },
    },
    {
        "id": "v3.9",
        "name": "The Fuel Line",
        "stepRange": [5_005_000_000, 5_720_000_000],
        "date": "2026-02-20",
        "status": "closed",
        "summary": "Break entropy wall #2 and address the boost starvation bottleneck that blocks all advanced mechanics. Match at 5.0B: 8-1 win but 72.4% zero boost, 0 saves, 2.0% supersonic, 40.6% directed touches. Root cause chain: boost starvation → can't sustain aerials above double-jump → can't reach supersonic → can't rotate back for saves → can't generate power shots. Entropy plateau at 3.51 for 825M steps (descent rate literally zero). Three fixes: ent_coef 0.0008→0.0006 (break plateau), boost_change 0.10→0.15 (make pad collection worth a detour), save_weight 0.25→0.35 with defense_zone 3500→4500 (earlier defensive commitment). Still 12 reward components.",
        "changes": [
            {
                "parameter": "ppo_ent_coef",
                "oldValue": "0.0008",
                "newValue": "0.0006",
                "rationale": "Entropy plateau at 3.51 for 825M+ steps. Descent rate +0.0004/bin = flat. VF loss still falling (critic improving) but policy frozen. 25% reduction — same strategy that broke wall #1.",
            },
            {
                "parameter": "boost_change_weight",
                "oldValue": "0.10",
                "newValue": "0.15",
                "rationale": "72.4% zero boost at 5.0B. Full 100-pad at 0.10 = 0.01 reward = same as 1 tick of vel_ball_to_goal. Zero incentive to detour through pad lanes. At 0.15: meaningful detour value to unlock sustained aerials, supersonic, defense rotation.",
            },
            {
                "parameter": "save_weight",
                "oldValue": "0.25",
                "newValue": "0.35",
                "rationale": "0+1 saves in 8-1 match. DEF state only 8.9% despite 48.1% ROT. Bot rotates but doesn't commit to defensive touches. User-flagged priority.",
            },
            {
                "parameter": "defense_zone_y (in SaveReward)",
                "oldValue": "3500",
                "newValue": "4500",
                "rationale": "Bot only received save signal when ball was dangerously close to goal. Wider zone gives earlier defensive incentive so bot commits to saves before it's too late.",
            },
        ],
        "impact": [],
        "assessment": {
            "grade": "A",
            "summary": "Every target hit. Entropy wall #2 demolished: 3.51→3.25 (-0.26, fastest descent in training history). Boost reversal confirmed for the first time in 4 versions (9.0→9.8). Aerial touch +41% (0.039→0.055). Speed new high (1248). Goal diff flipped positive (-0.3→+0.7). Double commits halved (9.9%→4.5-5.6%). First save recorded. Kickoff rotation learned (one stays, one goes). Only limitation: entropy wall #3 forming at 3.25 after 375M steps, and boost starvation still chronic (70% zero-boost in match).",
        },
    },
    {
        "id": "v3.10",
        "name": "The Clean Hit",
        "stepRange": [5_720_000_000, None],
        "date": "2026-02-21",
        "status": "active",
        "summary": "Break entropy wall #3 and surgically fix the lingering contact pattern that causes boost starvation. Three matches at 5.72B reveal the root cause: DirectedTouchReward gives full reward for EVERY consecutive step of ball contact, making prolonged ball-pushing (20-40+ tick sequences) the most consistent reward source. Bot never disengages to collect boost. Proof: when VYREX is dominated by Necto and forced to rotate, avg boost is 15.6 (vs 9.9 when in control). The bot CAN collect boost — it just won't when ball contact is more rewarding. Fix: Add sustain_decay (0.5) to DirectedTouchReward — geometric decay on consecutive contacts caps total reward at 2x first touch. Combined with boost_change +33% and lose_weight +67%, creates a crossover point where boost collection beats lingering after ~3 steps. ent_coef -20% breaks wall #3. Still 12 components.",
        "changes": [
            {
                "parameter": "ppo_ent_coef",
                "oldValue": "0.0006",
                "newValue": "0.00048",
                "rationale": "Entropy wall #3 at 3.25 for 375M steps. Descent rate collapsed from -0.0131/bin to -0.0039/bin. 20% reduction — proven pattern that broke walls #1 and #2.",
            },
            {
                "parameter": "directed_touch_sustain_decay (NEW)",
                "oldValue": "None (full reward every step)",
                "newValue": "0.5",
                "rationale": "ROOT CAUSE FIX. Lingering contact gives full DirectedTouchReward every step — 40 consecutive steps = 40x first touch reward. Decay: step 0 = 100%, step 1 = 50%, step 2 = 25%... Total capped at 2.0x first touch. Eliminates the perverse incentive to push ball instead of hitting and moving to boost.",
            },
            {
                "parameter": "boost_change_weight",
                "oldValue": "0.15",
                "newValue": "0.20",
                "rationale": "Boost starvation persistent at 5.72B (70% zero-boost). Combined with touch decay, creates crossover: after 3 steps of contact, boost collection reward exceeds lingering contact reward.",
            },
            {
                "parameter": "boost_change_lose_weight",
                "oldValue": "0.3",
                "newValue": "0.5",
                "rationale": "Bot burns all boost during lingering contact sequences then has 0 for next play. 67% increase makes wasteful boosting more costly, teaching conservation.",
            },
        ],
        "impact": [],
        "assessment": {
            "grade": "Pending",
            "summary": "Targets the structural reward flaw enabling lingering contact. Key metrics to watch: avg_boost (target >14), zero_boost% (target <55%), lingering contact sequences (target <5 consecutive steps), entropy (target <3.10). If touch decay works as designed, expect cascading improvement: shorter contacts → boost collection → better attacks → more goals → less boost waste.",
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
