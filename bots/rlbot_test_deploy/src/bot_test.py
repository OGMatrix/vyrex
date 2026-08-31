"""
VYREX — TEST Bot with Deep Behavioral Analysis
=================================================
Same inference as production bot, but tracks every decision, touch,
positioning choice, and team coordination event. Produces:
  1. Rich in-game HUD showing behavioral state + live reward signals
  2. JSON match report on exit for post-match analysis

Usage: Copy POLICY.pt from rlbot_deploy/src/ into this directory.
       Then launch via match.toml with RLBot v5.

Output: data/test_logs/match_YYYYMMDD_HHMMSS.json
"""

import os
import sys
import time
import json
import traceback
import math
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    sys.path.insert(0, "../../torch-archive")
    import torch

import rlbot_flatbuffers as flat
from rlbot.flat import ControllerState, GamePacket
from rlbot.managers import Bot

from rlgym_compat import GameState
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.action_parsers import LookupTableAction
from rlgym.rocket_league import common_values

torch.set_num_threads(1)  # Avoid PyTorch using too many threads in RLBot environment


# ============================================================================
# CONSTANTS
# ============================================================================

FIELD_HALF_X = common_values.SIDE_WALL_X       # 4096
FIELD_HALF_Y = common_values.BACK_WALL_Y       # 5120
GOAL_Y = common_values.BACK_NET_Y              # 6000
CEILING_Z = common_values.CEILING_Z             # 2044
MAX_SPEED = common_values.CAR_MAX_SPEED         # 2300
BALL_RADIUS = common_values.BALL_RADIUS          # 91.25
BALL_MAX_SPEED = common_values.BALL_MAX_SPEED    # 6000
SUPERSONIC = common_values.SUPERSONIC_THRESHOLD  # 2200
BLUE_GOAL = np.array(common_values.BLUE_GOAL_CENTER)   # (0, -5120, 321)
ORANGE_GOAL = np.array(common_values.ORANGE_GOAL_CENTER) # (0, 5120, 321)

# Field zones for positional analysis (5 vertical strips x 3 horizontal strips = 15 zones)
ZONE_NAMES = [
    "DEF_LEFT", "DEF_MID", "DEF_RIGHT",
    "DTHIRD_LEFT", "DTHIRD_MID", "DTHIRD_RIGHT",
    "MID_LEFT", "MID_MID", "MID_RIGHT",
    "ATHIRD_LEFT", "ATHIRD_MID", "ATHIRD_RIGHT",
    "ATK_LEFT", "ATK_MID", "ATK_RIGHT",
]

LOG_INTERVAL = 120  # Ticks between log prints (~1 second at 120 tick)


# ============================================================================
# NEURAL NETWORK — Same as production
# ============================================================================

class DiscreteFF(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int = 90,
                 layer_sizes=None):
        super().__init__()
        if layer_sizes is None:
            layer_sizes = [2048, 2048, 1024, 1024]
        layers = []
        prev = input_size
        for sz in layer_sizes:
            layers.append(torch.nn.Linear(prev, sz))
            layers.append(torch.nn.LeakyReLU())
            prev = sz
        layers.append(torch.nn.Linear(prev, output_size))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ============================================================================
# BEHAVIORAL ANALYZER — Per-bot tracking
# ============================================================================

class BotAnalyzer:
    """Tracks all behavioral metrics for a single VYREX bot instance."""

    def __init__(self, index: int, team: int):
        self.index = index
        self.team = team
        self.ticks = 0

        # --- Decision quality ---
        self.confidence_sum = 0.0
        self.entropy_sum = 0.0
        self.low_confidence_ticks = 0     # Below 5% (essentially guessing)
        self.action_counts = defaultdict(int)

        # --- Behavioral state tracking ---
        self.state_ticks = defaultdict(int)  # ATK/DEF/ROT/BALL_CHASE ticks
        self.double_commit_ticks = 0

        # --- Touch analysis ---
        self.touches_total = 0
        self.touches_toward_goal = 0
        self.touches_away_from_goal = 0
        self.touch_speeds = []             # Ball speed after each touch
        self.touch_directions = []          # Toward-goal component per touch
        self.last_touch_tick = -999
        self.touch_outcomes = []            # (tick, toward_goal_speed, ball_height)

        # --- Positioning ---
        self.zone_ticks = defaultdict(int)
        self.dist_to_ball_sum = 0.0
        self.dist_to_teammate_sum = 0.0
        self.dist_to_own_goal_sum = 0.0
        self.time_in_air = 0
        self.time_on_ground = 0
        self.time_supersonic = 0

        # --- Boost ---
        self.boost_sum = 0.0
        self.boost_waste_ticks = 0         # Boosting while supersonic
        self.boost_zero_ticks = 0           # Time spent at 0 boost
        self.boost_full_pickups = 0         # Count transitions to 100

        # --- Defensive ---
        self.in_defensive_third_ticks = 0
        self.save_positions = 0             # Between ball and own goal when ball is in def third
        self.clears_attempted = 0           # Touches in defensive third

        # --- Mechanical ---
        self.aerial_frames = 0             # Airborne + ball height > 300
        self.flip_count = 0
        self.jump_count = 0
        self.prev_has_jumped = False
        self.prev_has_flipped = False

        # --- Live reward signals (computed each tick) ---
        self.reward_signals = defaultdict(float)  # Running sum for averaging

        # --- Rolling windows for HUD ---
        self.recent_confidence = deque(maxlen=120)
        self.recent_entropy = deque(maxlen=120)
        self.recent_states = deque(maxlen=360)  # 3 seconds

        # --- Prev frame state for delta detection ---
        self.prev_boost = 0.0

    def get_field_zone(self, pos: np.ndarray, is_orange: bool) -> str:
        """Classify position into one of 15 field zones (team-relative)."""
        x, y = pos[0], pos[1]
        # Normalize Y so positive = attacking direction for both teams
        norm_y = y if not is_orange else -y

        # Y bands: def(-5120 to -3072), dthird(-3072 to -1024), mid(-1024 to 1024),
        #          athird(1024 to 3072), atk(3072 to 5120)
        if norm_y < -3072:
            row = 0  # DEF
        elif norm_y < -1024:
            row = 1  # DTHIRD
        elif norm_y < 1024:
            row = 2  # MID
        elif norm_y < 3072:
            row = 3  # ATHIRD
        else:
            row = 4  # ATK

        # X bands: left, mid, right
        if x < -1365:
            col = 0
        elif x < 1365:
            col = 1
        else:
            col = 2

        return ZONE_NAMES[row * 3 + col]

    def classify_state(self, car_pos: np.ndarray, ball_pos: np.ndarray,
                       teammate_pos: Optional[np.ndarray],
                       is_orange: bool) -> str:
        """Classify current behavioral state: ATK, DEF, ROT, or BALL_CHASE."""
        dist_to_ball = np.linalg.norm(car_pos[:2] - ball_pos[:2])

        # Own goal direction
        own_goal_y = -FIELD_HALF_Y if not is_orange else FIELD_HALF_Y
        ball_to_own_goal = abs(ball_pos[1] - own_goal_y)
        car_to_own_goal = abs(car_pos[1] - own_goal_y)

        # Ball in defensive third?
        ball_in_def = ball_to_own_goal < FIELD_HALF_Y * 0.6

        # Am I between ball and own goal?
        if not is_orange:
            between = car_pos[1] < ball_pos[1]
        else:
            between = car_pos[1] > ball_pos[1]

        # Check for double-commit (both close to ball)
        if teammate_pos is not None:
            tm_dist_to_ball = np.linalg.norm(teammate_pos[:2] - ball_pos[:2])
            both_close = dist_to_ball < 1500 and tm_dist_to_ball < 1500
            tm_closer = tm_dist_to_ball < dist_to_ball
        else:
            both_close = False
            tm_closer = False

        if both_close and tm_closer:
            return "BALL_CHASE"  # Teammate is closer but we're also close = chasing
        elif ball_in_def and between:
            return "DEF"
        elif dist_to_ball < 2000 and not tm_closer:
            return "ATK"
        elif between and car_to_own_goal < ball_to_own_goal:
            return "ROT"  # Behind ball, heading back
        else:
            return "ATK"

    def compute_reward_signals(self, car, ball_pos: np.ndarray,
                               ball_vel: np.ndarray, is_orange: bool,
                               teammate_pos: Optional[np.ndarray]) -> Dict[str, float]:
        """Compute the same reward signals used in training (for live display)."""
        signals = {}
        car_pos = car.physics.position
        car_vel = car.physics.linear_velocity

        # 1. Speed toward ball
        diff = ball_pos - car_pos
        dist = np.linalg.norm(diff)
        if dist > 1e-5:
            dir_to_ball = diff / dist
            stb = np.dot(car_vel, dir_to_ball) / MAX_SPEED
            signals["spd_to_ball"] = max(stb, 0.0)
        else:
            signals["spd_to_ball"] = 0.0

        # 2. Ball velocity toward opponent goal
        opp_goal = ORANGE_GOAL if not is_orange else BLUE_GOAL
        bg = opp_goal - ball_pos
        bg_dist = np.linalg.norm(bg)
        if bg_dist > 1e-5:
            bg_dir = bg / bg_dist
            vbg = np.dot(ball_vel, bg_dir) / BALL_MAX_SPEED
            signals["ball_to_goal"] = max(vbg, 0.0)
        else:
            signals["ball_to_goal"] = 0.0

        # 3. Face ball alignment
        forward = car.physics.forward
        if dist > 1e-5:
            signals["face_ball"] = max(0.0, np.dot(forward, diff / dist))
        else:
            signals["face_ball"] = 1.0

        # 4. Save signal
        own_goal_y = -GOAL_Y if not is_orange else GOAL_Y
        ball_dist_to_goal = abs(ball_pos[1] - own_goal_y)
        between = (car_pos[1] < ball_pos[1]) if not is_orange else (car_pos[1] > ball_pos[1])
        if ball_dist_to_goal < 3500 and between:
            signals["save"] = 1.0 - (ball_dist_to_goal / 3500)
        else:
            signals["save"] = 0.0

        # 5. Teammate spacing
        if teammate_pos is not None:
            tm_dist = np.linalg.norm(car_pos - teammate_pos)
            if tm_dist <= 3000:
                signals["spacing"] = tm_dist / 3000
            else:
                overshoot = (tm_dist - 3000) / 5000
                signals["spacing"] = max(0.0, 1.0 - 0.3 * overshoot)
        else:
            signals["spacing"] = 0.5

        # 6. Boost conservation
        signals["boost"] = math.sqrt(max(0, car.boost_amount / 100.0))

        return signals

    def to_summary(self) -> dict:
        """Generate the final analysis summary."""
        t = max(self.ticks, 1)
        return {
            "bot_index": self.index,
            "total_ticks": self.ticks,
            "decision_quality": {
                "avg_confidence": round(self.confidence_sum / t, 4),
                "avg_entropy": round(self.entropy_sum / t, 3),
                "low_confidence_pct": round(self.low_confidence_ticks / t * 100, 1),
                "top_5_actions": sorted(self.action_counts.items(),
                                        key=lambda x: -x[1])[:5],
                "unique_actions_used": len(self.action_counts),
            },
            "behavioral_state": {
                k: round(v / t * 100, 1)
                for k, v in sorted(self.state_ticks.items())
            },
            "double_commit_pct": round(self.double_commit_ticks / t * 100, 1),
            "touches": {
                "total": self.touches_total,
                "toward_goal": self.touches_toward_goal,
                "away_from_goal": self.touches_away_from_goal,
                "directed_pct": round(
                    self.touches_toward_goal / max(self.touches_total, 1) * 100, 1
                ),
                "avg_touch_speed": round(
                    np.mean(self.touch_speeds) if self.touch_speeds else 0, 1
                ),
            },
            "positioning": {
                "zone_pct": {
                    k: round(v / t * 100, 1)
                    for k, v in sorted(self.zone_ticks.items())
                    if v > 0
                },
                "avg_dist_to_ball": round(self.dist_to_ball_sum / t, 0),
                "avg_dist_to_teammate": round(self.dist_to_teammate_sum / t, 0),
                "avg_dist_to_own_goal": round(self.dist_to_own_goal_sum / t, 0),
            },
            "mechanical": {
                "airborne_pct": round(self.time_in_air / t * 100, 1),
                "supersonic_pct": round(self.time_supersonic / t * 100, 1),
                "aerial_frames_pct": round(self.aerial_frames / t * 100, 1),
                "flips": self.flip_count,
                "jumps": self.jump_count,
            },
            "boost_mgmt": {
                "avg_boost": round(self.boost_sum / t, 1),
                "zero_boost_pct": round(self.boost_zero_ticks / t * 100, 1),
                "waste_pct": round(self.boost_waste_ticks / t * 100, 1),
            },
            "defense": {
                "def_third_pct": round(self.in_defensive_third_ticks / t * 100, 1),
                "save_positions": self.save_positions,
                "clears_attempted": self.clears_attempted,
            },
            "avg_reward_signals": {
                k: round(v / t, 4)
                for k, v in sorted(self.reward_signals.items())
            },
        }


# ============================================================================
# MATCH TRACKER — Global match state
# ============================================================================

class MatchTracker:
    """Tracks match-level events and produces the final report."""

    def __init__(self):
        self.start_time = time.time()
        self.events: List[dict] = []        # Timeline of major events
        self.prev_blue_goals = 0
        self.prev_orange_goals = 0
        self.ticks = 0

    def check_goals(self, packet: GamePacket) -> Optional[dict]:
        """Detect goal events from the packet."""
        if len(packet.teams) < 2:
            return None
        blue_g = packet.teams[0].score
        orange_g = packet.teams[1].score
        event = None
        if blue_g > self.prev_blue_goals:
            event = {"tick": self.ticks, "event": "GOAL", "team": "blue",
                     "score": f"{blue_g}-{orange_g}"}
        elif orange_g > self.prev_orange_goals:
            event = {"tick": self.ticks, "event": "GOAL", "team": "orange",
                     "score": f"{blue_g}-{orange_g}"}
        self.prev_blue_goals = blue_g
        self.prev_orange_goals = orange_g
        if event:
            self.events.append(event)
        return event

    def build_report(self, analyzers: Dict[int, "BotAnalyzer"],
                     packet: GamePacket) -> dict:
        """Build the complete match analysis JSON."""
        elapsed = time.time() - self.start_time
        blue_g = packet.teams[0].score if len(packet.teams) >= 2 else 0
        orange_g = packet.teams[1].score if len(packet.teams) >= 2 else 0

        # Per-player scoreboard from packet
        scoreboard = []
        for p in packet.players:
            si = p.score_info
            scoreboard.append({
                "name": p.name, "team": "blue" if p.team == 0 else "orange",
                "goals": si.goals, "assists": si.assists, "saves": si.saves,
                "shots": si.shots, "demos": si.demolitions, "score": si.score,
            })

        report = {
            "generated": datetime.now().isoformat(),
            "match_duration_sec": round(elapsed, 1),
            "total_ticks": self.ticks,
            "final_score": {"blue": blue_g, "orange": orange_g},
            "scoreboard": scoreboard,
            "bot_analysis": {
                f"VYREX_{a.index}": a.to_summary()
                for a in analyzers.values()
            },
            "event_timeline": self.events[-100:],  # Last 100 events max
        }
        return report


# ============================================================================
# TEST BOT — Main class
# ============================================================================

class VyrexTestBot(Bot):
    """
    VYREX Test Bot — same policy inference as production, with deep
    behavioral tracking and match analysis logging.
    """

    POLICY_FILE = "POLICY.pt"
    DETERMINISTIC = True

    # Shared across all bot instances in the match
    _match_tracker: Optional[MatchTracker] = None
    _analyzers: Dict[int, BotAnalyzer] = {}
    _report_saved = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.game_state = None
        self.obs_builder = None
        self.action_table = None
        self._obs_initialized = False
        self._tick_count = 0

        # Per-tick inference results (for HUD)
        self._last_action_idx = 0
        self._last_confidence = 0.0
        self._last_entropy = 0.0
        self._last_top3 = []
        self._last_signals = {}
        self._last_state = "?"
        self._max_entropy = np.log(90)

    def initialize(self):
        """Called once per bot when RLBot is ready."""
        # Initialize shared match tracker (first bot creates it)
        if VyrexTestBot._match_tracker is None:
            VyrexTestBot._match_tracker = MatchTracker()
            VyrexTestBot._analyzers = {}
            VyrexTestBot._report_saved = False

        # Create analyzer for this bot instance
        self.analyzer = BotAnalyzer(self.index, self.team)
        VyrexTestBot._analyzers[self.index] = self.analyzer

        # Load policy (same as production)
        self._load_policy()

        # Setup rlgym compat
        self.game_state = GameState.create_compat_game_state(
            field_info=self.field_info,
            match_configuration=self.match_config,
        )
        self.obs_builder = DefaultObs(
            zero_padding=None,
            pos_coef=np.asarray([
                1.0 / common_values.SIDE_WALL_X,
                1.0 / common_values.BACK_NET_Y,
                1.0 / common_values.CEILING_Z,
            ]),
            ang_coef=1.0 / np.pi,
            lin_vel_coef=1.0 / common_values.CAR_MAX_SPEED,
            ang_vel_coef=1.0 / common_values.CAR_MAX_ANG_VEL,
            boost_coef=1.0 / 100.0,
        )
        self.action_table = LookupTableAction.make_lookup_table()

        # Prepare log directory
        log_dir = os.path.join(os.path.dirname(__file__), "..", "data", "test_logs")
        os.makedirs(log_dir, exist_ok=True)

        print(f"[VYREX-TEST #{self.index}] Initialized. Logging to {log_dir}")

    def _load_policy(self):
        """Load trained policy network."""
        policy_path = os.path.join(os.path.dirname(__file__), self.POLICY_FILE)
        if not os.path.exists(policy_path):
            print(f"[VYREX-TEST] ERROR: POLICY.pt not found at {policy_path}")
            print("[VYREX-TEST] Copy POLICY.pt from rlbot_deploy/src/ into this directory.")
            return
        try:
            sd = torch.load(policy_path, map_location=self.device, weights_only=True)
            first_key = next(iter(sd))
            in_sz = sd[first_key].shape[1]
            keys = list(sd.keys())
            out_sz = sd[[k for k in keys if 'weight' in k][-1]].shape[0]
            self.policy = DiscreteFF(in_sz, out_sz, [2048, 2048, 1024, 1024])
            self.policy.load_state_dict(sd)
            self.policy.to(self.device)
            self.policy.eval()
            print(f"[VYREX-TEST #{self.index}] Policy loaded: {in_sz} -> {out_sz}")
        except Exception as e:
            print(f"[VYREX-TEST] Policy load failed: {e}")
            traceback.print_exc()

    def get_output(self, packet: GamePacket) -> ControllerState:
        """Main tick: inference + deep analysis."""
        controller = ControllerState()
        if self.policy is None or self.game_state is None:
            return controller

        try:
            self.game_state.update(packet)
            agent_id = self.player_id
            if agent_id not in self.game_state.cars:
                return controller

            if not self._obs_initialized:
                agents = list(self.game_state.cars.keys())
                self.obs_builder.reset(agents, self.game_state, {})
                self._obs_initialized = True

            # --- Inference (same as production) ---
            obs = self.obs_builder.build_obs([agent_id], self.game_state, {})[agent_id]
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.policy(obs_t)
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                if self.DETERMINISTIC:
                    action_idx = torch.argmax(probs).item()
                else:
                    action_idx = torch.multinomial(probs.unsqueeze(0), 1).item()

                self._last_action_idx = action_idx
                self._last_confidence = probs[action_idx].item()
                log_probs = torch.log(probs + 1e-10)
                self._last_entropy = -(probs * log_probs).sum().item()
                top3 = torch.topk(probs, min(3, len(probs)))
                self._last_top3 = list(zip(
                    top3.indices.cpu().tolist(), top3.values.cpu().tolist()))

            # Apply action
            if action_idx < len(self.action_table):
                a = self.action_table[action_idx]
                controller.throttle = float(a[0])
                controller.steer = float(a[1])
                controller.pitch = float(a[2])
                controller.yaw = float(a[3])
                controller.roll = float(a[4])
                controller.jump = bool(a[5])
                controller.boost = bool(a[6])
                controller.handbrake = bool(a[7])

            # --- Deep Analysis ---
            self._analyze_tick(packet, controller)

            # # --- HUD ---
            # self._draw_analysis_hud(packet, controller)

            # --- Periodic console log ---
            self._tick_count += 1
            if self._tick_count % (LOG_INTERVAL * 5) == 1:
                self._print_status()

            # --- Check match end and save report ---
            self._check_save_report(packet)

        except Exception as e:
            if self._tick_count % 500 == 0:
                print(f"[VYREX-TEST #{self.index}] ERROR: {e}")
                traceback.print_exc()

        return controller

    # ------------------------------------------------------------------
    # ANALYSIS CORE — Called every tick
    # ------------------------------------------------------------------

    def _analyze_tick(self, packet: GamePacket, ctrl: ControllerState):
        """The heart of the analysis — track everything meaningful."""
        a = self.analyzer
        a.ticks += 1
        mt = VyrexTestBot._match_tracker
        mt.ticks = max(mt.ticks, a.ticks)

        car_rlgym = self.game_state.cars[self.player_id]
        car_pos = car_rlgym.physics.position
        car_vel = car_rlgym.physics.linear_velocity
        ball_pos = self.game_state.ball.position
        ball_vel = self.game_state.ball.linear_velocity
        is_orange = car_rlgym.is_orange

        # --- Find teammate ---
        teammate_pos = None
        for aid, other_car in self.game_state.cars.items():
            if aid != self.player_id and other_car.is_orange == is_orange:
                teammate_pos = other_car.physics.position
                break

        # --- Decision quality ---
        a.confidence_sum += self._last_confidence
        a.entropy_sum += self._last_entropy
        a.recent_confidence.append(self._last_confidence)
        a.recent_entropy.append(self._last_entropy)
        if self._last_confidence < 0.05:
            a.low_confidence_ticks += 1
        a.action_counts[self._last_action_idx] += 1

        # --- Behavioral state ---
        state = a.classify_state(car_pos, ball_pos, teammate_pos, is_orange)
        self._last_state = state
        a.state_ticks[state] += 1
        a.recent_states.append(state)

        # Double commit detection
        if teammate_pos is not None:
            my_d = np.linalg.norm(car_pos[:2] - ball_pos[:2])
            tm_d = np.linalg.norm(teammate_pos[:2] - ball_pos[:2])
            if my_d < 1200 and tm_d < 1200:
                a.double_commit_ticks += 1

        # --- Touch analysis ---
        if car_rlgym.ball_touches > 0 and (a.ticks - a.last_touch_tick) > 10:
            a.touches_total += 1
            a.last_touch_tick = a.ticks

            # Direction of ball relative to opponent goal
            opp_goal = ORANGE_GOAL if not is_orange else BLUE_GOAL
            to_goal = opp_goal - ball_pos
            d = np.linalg.norm(to_goal)
            if d > 1e-5:
                toward_goal_speed = np.dot(ball_vel, to_goal / d)
                a.touch_speeds.append(np.linalg.norm(ball_vel))
                a.touch_directions.append(toward_goal_speed)

                if toward_goal_speed > 0:
                    a.touches_toward_goal += 1
                else:
                    a.touches_away_from_goal += 1

                a.touch_outcomes.append({
                    "tick": a.ticks, "ball_speed": float(np.linalg.norm(ball_vel)),
                    "toward_goal": float(toward_goal_speed),
                    "ball_height": float(ball_pos[2]),
                })

                # Log significant touches
                if abs(toward_goal_speed) > 500:
                    direction = "TOWARD" if toward_goal_speed > 0 else "AWAY"
                    mt.events.append({
                        "tick": a.ticks, "event": "TOUCH",
                        "bot": f"VYREX_{self.index}",
                        "direction": direction,
                        "speed": round(toward_goal_speed, 0),
                        "ball_z": round(ball_pos[2], 0),
                    })

            # Defensive clear — only count actual TOUCHES that CLEAR in defensive position
            # v3 FIX: Was counting every touch in defensive third as a "clear",
            # even when the touch sent ball toward own goal. Now only counts
            # touches that send the ball AWAY from own goal (actual clears).
            own_goal_y = -GOAL_Y if not is_orange else GOAL_Y
            if abs(ball_pos[1] - own_goal_y) < 3500:
                # Check if the touch sent the ball away from own goal (an actual clear)
                ball_vel_y = ball_vel[1] if ball_vel is not None else 0
                if not is_orange:
                    clearing_away = ball_vel_y > 0  # Away from blue goal (-Y)
                else:
                    clearing_away = ball_vel_y < 0  # Away from orange goal (+Y)
                if clearing_away:
                    a.clears_attempted += 1
                    mt.events.append({
                        "tick": a.ticks, "event": "CLEAR",
                        "bot": f"VYREX_{self.index}",
                        "speed": round(abs(ball_vel_y), 0),
                    })

        # --- Positioning ---
        zone = a.get_field_zone(car_pos, is_orange)
        a.zone_ticks[zone] += 1
        a.dist_to_ball_sum += np.linalg.norm(car_pos - ball_pos)
        if teammate_pos is not None:
            a.dist_to_teammate_sum += np.linalg.norm(car_pos - teammate_pos)
        own_goal_y = -GOAL_Y if not is_orange else GOAL_Y
        a.dist_to_own_goal_sum += abs(car_pos[1] - own_goal_y)

        # Defensive third time
        if abs(car_pos[1] - own_goal_y) < FIELD_HALF_Y * 0.6:
            a.in_defensive_third_ticks += 1

        # Save positioning check
        ball_dist_to_own = abs(ball_pos[1] - own_goal_y)
        between = (car_pos[1] < ball_pos[1]) if not is_orange else (car_pos[1] > ball_pos[1])
        if ball_dist_to_own < 3500 and between:
            a.save_positions += 1

        # --- Mechanical ---
        speed = np.linalg.norm(car_vel)
        if not car_rlgym.on_ground:
            a.time_in_air += 1
            if ball_pos[2] > 300:
                a.aerial_frames += 1
        else:
            a.time_on_ground += 1

        if speed >= SUPERSONIC:
            a.time_supersonic += 1

        # Jump/flip detection from packet
        p_car = packet.players[self.index]
        if p_car.has_jumped and not a.prev_has_jumped:
            a.jump_count += 1
        if p_car.has_dodged and not a.prev_has_flipped:
            a.flip_count += 1
        a.prev_has_jumped = p_car.has_jumped
        a.prev_has_flipped = p_car.has_dodged

        # --- Boost ---
        boost = car_rlgym.boost_amount
        a.boost_sum += boost
        if boost < 1:
            a.boost_zero_ticks += 1
        if ctrl.boost and speed >= SUPERSONIC:
            a.boost_waste_ticks += 1
        if boost >= 99 and a.prev_boost < 90:
            a.boost_full_pickups += 1
        a.prev_boost = boost

        # --- Live reward signals ---
        signals = a.compute_reward_signals(
            car_rlgym, ball_pos, ball_vel, is_orange, teammate_pos)
        self._last_signals = signals
        for k, v in signals.items():
            a.reward_signals[k] += v

        # --- Goal detection (match tracker) ---
        mt.check_goals(packet)

    # ------------------------------------------------------------------
    # HUD — In-game analysis display
    # ------------------------------------------------------------------

    def _draw_analysis_hud(self, packet: GamePacket, ctrl: ControllerState):
        """Draw compact but information-dense analysis HUD."""
        try:
            a = self.analyzer
            group = f"vyrex_test_{self.index}"
            self.renderer.begin_rendering(group)

            # Colors
            WHITE = flat.Color(r=255, g=255, b=255, a=255)
            CYAN = flat.Color(r=0, g=255, b=255, a=220)
            GREEN = flat.Color(r=0, g=255, b=0, a=220)
            YELLOW = flat.Color(r=255, g=255, b=0, a=220)
            RED = flat.Color(r=255, g=60, b=60, a=220)
            ORANGE = flat.Color(r=255, g=165, b=0, a=220)
            DIM = flat.Color(r=140, g=140, b=140, a=160)
            BG = flat.Color(r=0, g=0, b=0, a=180)
            BAR_BG = flat.Color(r=30, g=30, b=30, a=200)

            hud_x = 0.005 if self.index == 0 else 0.74
            y = 0.02
            s = 1

            # --- Header: State + Confidence ---
            state_colors = {"ATK": GREEN, "DEF": CYAN, "ROT": YELLOW, "BALL_CHASE": RED}
            sc = state_colors.get(self._last_state, DIM)
            conf_pct = self._last_confidence * 100
            ent_ratio = self._last_entropy / self._max_entropy * 100

            self.renderer.draw_string_2d(
                f"VYREX #{self.index} [{self._last_state}]",
                hud_x, y, s + 0.3, sc, BG)
            y += 0.03

            self.renderer.draw_string_2d(
                f"Conf:{conf_pct:4.1f}%  Ent:{ent_ratio:4.1f}%  Act:{self._last_action_idx}",
                hud_x, y, s * 0.85, WHITE, BG)
            y += 0.022

            # --- Live Reward Signals (bar chart) ---
            self.renderer.draw_string_2d(
                "Reward Signals:", hud_x, y, s * 0.8, DIM, BG)
            y += 0.018
            sig_order = ["spd_to_ball", "ball_to_goal", "face_ball",
                         "save", "spacing", "boost"]
            sig_labels = ["Ball", "Goal", "Face", "Save", "Team", "Bst"]
            sig_colors = [CYAN, GREEN, YELLOW, RED, ORANGE, DIM]
            for label, key, col in zip(sig_labels, sig_order, sig_colors):
                val = self._last_signals.get(key, 0)
                bar = "|" * int(val * 20)
                self.renderer.draw_string_2d(
                    f" {label}: {val:.2f} {bar}",
                    hud_x, y, s * 0.75, col, BG)
                y += 0.015

            y += 0.005

            # --- Match Stats (live) ---
            minutes = a.ticks / 120 / 60
            self.renderer.draw_string_2d(
                f"Touches:{a.touches_total} "
                f"Dir:{a.touches_toward_goal}/{max(a.touches_total,1)}"
                f"({a.touches_toward_goal/max(a.touches_total,1)*100:.0f}%) "
                f"DC:{a.double_commit_ticks/max(a.ticks,1)*100:.0f}%",
                hud_x, y, s * 0.75, WHITE, BG)
            y += 0.018

            boost = self.game_state.cars[self.player_id].boost_amount
            self.renderer.draw_string_2d(
                f"Boost:{boost:.0f} "
                f"Air:{a.time_in_air/max(a.ticks,1)*100:.0f}% "
                f"Flips:{a.flip_count} Jumps:{a.jump_count} "
                f"SSonic:{a.time_supersonic/max(a.ticks,1)*100:.0f}%",
                hud_x, y, s * 0.75, ORANGE, BG)
            y += 0.018

            # State distribution (live)
            state_str = " ".join(
                f"{k}:{v/max(a.ticks,1)*100:.0f}%"
                for k, v in sorted(a.state_ticks.items())
            )
            self.renderer.draw_string_2d(
                state_str, hud_x, y, s * 0.7, DIM, BG)
            y += 0.018

            # Controls
            self.renderer.draw_string_2d(
                f"T:{ctrl.throttle:+.0f} S:{ctrl.steer:+.0f} "
                f"P:{ctrl.pitch:+.0f} Y:{ctrl.yaw:+.0f} R:{ctrl.roll:+.0f} "
                f"B:{'Y' if ctrl.boost else '-'} J:{'Y' if ctrl.jump else '-'} "
                f"H:{'Y' if ctrl.handbrake else '-'}",
                hud_x, y, s * 0.7, DIM, BG)

            # --- 3D: State label above car ---
            car_anchor = flat.CarAnchor(
                index=self.index,
                local=flat.Vector3(x=0, y=0, z=130)
            )
            self.renderer.draw_string_3d(
                f"{self._last_state} {conf_pct:.0f}%",
                car_anchor, 1.2, sc, BG)

            # 3D line to ball (color = state)
            if packet.balls:
                ball_anchor = flat.BallAnchor(local=flat.Vector3(x=0, y=0, z=0))
                car_line = flat.CarAnchor(
                    index=self.index,
                    local=flat.Vector3(x=0, y=0, z=50))
                self.renderer.draw_line_3d(car_line, ball_anchor, sc)

            self.renderer.end_rendering()

        except Exception:
            pass  # Never crash on render errors

    # ------------------------------------------------------------------
    # CONSOLE OUTPUT
    # ------------------------------------------------------------------

    def _print_status(self):
        """Periodic console status dump."""
        a = self.analyzer
        t = max(a.ticks, 1)
        secs = t / 120

        state_pcts = {k: f"{v/t*100:.0f}%" for k, v in a.state_ticks.items()}
        dir_pct = a.touches_toward_goal / max(a.touches_total, 1) * 100

        print(f"\n[VYREX-TEST #{self.index}] === {secs:.0f}s ===")
        print(f"  State: {state_pcts}")
        print(f"  Touches: {a.touches_total} (directed: {dir_pct:.0f}%)")
        print(f"  Confidence: {a.confidence_sum/t:.3f}  "
              f"Entropy: {a.entropy_sum/t:.2f}/{self._max_entropy:.2f}")
        print(f"  Airborne: {a.time_in_air/t*100:.1f}%  "
              f"Supersonic: {a.time_supersonic/t*100:.1f}%  "
              f"Boost avg: {a.boost_sum/t:.0f}")
        print(f"  Double-commits: {a.double_commit_ticks/t*100:.1f}%  "
              f"Flips: {a.flip_count}  Clears: {a.clears_attempted}")

    # ------------------------------------------------------------------
    # REPORT SAVING
    # ------------------------------------------------------------------

    def _check_save_report(self, packet: GamePacket):
        """Save report when match ends or periodically."""
        if VyrexTestBot._report_saved:
            return

        # Save every 60 seconds of game time, and on match end
        game_time_elapsed = self.analyzer.ticks / 120
        should_save = (self.analyzer.ticks % (120 * 60) == 0 and
                       self.analyzer.ticks > 0)

        # Check for match end (MatchPhase.Ended)
        match_ended = False
        if hasattr(packet, 'match_info') and packet.match_info:
            try:
                from rlbot.flat import MatchPhase
                match_ended = packet.match_info.match_phase == MatchPhase.Ended
            except Exception:
                pass

        if should_save or match_ended:
            self._save_report(packet, final=match_ended)

    def _save_report(self, packet: GamePacket, final: bool = False):
        """Write the analysis JSON to disk."""
        try:
            mt = VyrexTestBot._match_tracker
            report = mt.build_report(VyrexTestBot._analyzers, packet)
            report["is_final"] = final

            log_dir = os.path.join(os.path.dirname(__file__),
                                   "..", "data", "test_logs")
            os.makedirs(log_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "_FINAL" if final else ""
            filename = f"match_{timestamp}{suffix}.json"
            filepath = os.path.join(log_dir, filename)

            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            print(f"\n[VYREX-TEST] Report saved: {filepath}")

            if final:
                VyrexTestBot._report_saved = True
                # Print summary to console
                print("\n" + "=" * 60)
                print("  VYREX TEST MATCH — FINAL REPORT")
                print("=" * 60)
                score = report["final_score"]
                print(f"  Score: Blue {score['blue']} - {score['orange']} Orange")
                print(f"  Duration: {report['match_duration_sec']:.0f}s "
                      f"({report['total_ticks']} ticks)")
                print()
                for name, analysis in report["bot_analysis"].items():
                    print(f"  --- {name} ---")
                    dq = analysis["decision_quality"]
                    print(f"    Confidence: {dq['avg_confidence']:.3f}  "
                          f"Entropy: {dq['avg_entropy']:.2f}")
                    bs = analysis["behavioral_state"]
                    print(f"    States: {bs}")
                    t = analysis["touches"]
                    print(f"    Touches: {t['total']} "
                          f"(directed: {t['directed_pct']}%)")
                    d = analysis["defense"]
                    print(f"    Defense: {d['def_third_pct']}% in def third, "
                          f"{d['clears_attempted']} clears")
                    m = analysis["mechanical"]
                    print(f"    Airborne: {m['airborne_pct']}%, "
                          f"Supersonic: {m['supersonic_pct']}%, "
                          f"Flips: {m['flips']}")
                    print(f"    Double-commit: {analysis['double_commit_pct']}%")
                    print(f"    Reward signals: {analysis['avg_reward_signals']}")
                    print()
                print("=" * 60)

        except Exception as e:
            print(f"[VYREX-TEST] Report save error: {e}")
            traceback.print_exc()

    def retire(self):
        """Called when bot is removed. Save final report."""
        print(f"[VYREX-TEST #{self.index}] Retiring — saving final report...")
        # Try to get packet from last state for a final save
        if not VyrexTestBot._report_saved and VyrexTestBot._match_tracker:
            try:
                # Build a minimal report without packet
                mt = VyrexTestBot._match_tracker
                report = {
                    "generated": datetime.now().isoformat(),
                    "match_duration_sec": round(time.time() - mt.start_time, 1),
                    "total_ticks": mt.ticks,
                    "final_score": "unknown (retired)",
                    "bot_analysis": {
                        f"VYREX_{a.index}": a.to_summary()
                        for a in VyrexTestBot._analyzers.values()
                    },
                    "event_timeline": mt.events[-100:],
                    "is_final": True,
                }
                log_dir = os.path.join(os.path.dirname(__file__),
                                       "..", "data", "test_logs")
                os.makedirs(log_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fp = os.path.join(log_dir, f"match_{ts}_FINAL.json")
                with open(fp, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                print(f"[VYREX-TEST] Final report saved: {fp}")
                VyrexTestBot._report_saved = True
            except Exception as e:
                print(f"[VYREX-TEST] Final report error: {e}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    bot = VyrexTestBot("vyrex/vyrex-test")
    bot.run()
