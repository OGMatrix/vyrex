"""
VYREX - Custom Reward Functions for 2v2 Domination
====================================================
Each reward outputs in [-1, 1] (or [0, 1]) range before weighting.

Reward Design Principles (from Nexto, Lucy-SKG, Zealan's Guide, Seer thesis):
  1. Dense rewards guide early learning (velocity toward ball, face ball)
  2. Sparse rewards drive late-game skill (goals, saves, demos)
  3. Team rewards prevent ball-chasing (teammate spacing, rotation)
  4. All continuous rewards are normalized to consistent ranges
  5. Negative rewards used sparingly to avoid reward hacking

Reference: rlgym.rocket_league.common_values for game constants
"""

from typing import List, Dict, Any
import numpy as np

from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values


# ============================================================================
# DENSE CONTINUOUS REWARDS
# ============================================================================

class SpeedTowardBallReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards the agent for having velocity directed toward the ball.
    Output: [0, 1] — 0 when moving away, 1 at max car speed toward ball.

    This is the single most important early-training reward. It teaches the
    bot to drive toward the ball, which is a prerequisite for everything else.
    """

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            # Use inverted physics for orange team so rewards are team-agnostic
            car_phys = car.physics if not car.is_orange else car.inverted_physics
            ball_phys = state.ball if not car.is_orange else state.inverted_ball

            pos_diff = ball_phys.position - car_phys.position
            dist = np.linalg.norm(pos_diff)

            if dist < 1e-5:
                rewards[agent] = 0.0
                continue

            dir_to_ball = pos_diff / dist
            speed_toward = np.dot(car_phys.linear_velocity, dir_to_ball)
            rewards[agent] = max(speed_toward / common_values.CAR_MAX_SPEED, 0.0)

        return rewards


class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards when the ball has velocity toward the opponent's goal.
    Output: [0, 1] — scales with ball speed toward goal.

    This reward teaches the bot that hitting the ball toward the goal is good,
    regardless of whether it scores. Crucial for offensive play development.
    """

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            ball = state.ball

            # Orange team attacks the blue goal (negative Y), blue attacks orange (positive Y)
            goal_y = common_values.BACK_NET_Y if not car.is_orange else -common_values.BACK_NET_Y
            goal_pos = np.array([0.0, goal_y, 0.0])

            pos_diff = goal_pos - ball.position
            dist = np.linalg.norm(pos_diff)

            if dist < 1e-5:
                rewards[agent] = 0.0
                continue

            dir_to_goal = pos_diff / dist
            vel_toward = np.dot(ball.linear_velocity, dir_to_goal)
            rewards[agent] = max(vel_toward / common_values.BALL_MAX_SPEED, 0.0)

        return rewards


class FaceBallReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards the agent for facing toward the ball.
    Output: [0, 1] — 1 when perfectly facing ball, 0 when facing opposite.

    Teaches spatial awareness. Agents that face the ball can react faster
    and are more likely to make good hits.
    """

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            car_phys = car.physics if not car.is_orange else car.inverted_physics
            ball_phys = state.ball if not car.is_orange else state.inverted_ball

            pos_diff = ball_phys.position - car_phys.position
            dist = np.linalg.norm(pos_diff)

            if dist < 1e-5:
                rewards[agent] = 1.0
                continue

            dir_to_ball = pos_diff / dist
            # Forward direction is the first column of the rotation matrix
            # In RLGym v2, forward is along the x-axis of the car's local frame
            forward = car_phys.forward
            alignment = np.dot(forward, dir_to_ball)
            rewards[agent] = max(0.0, alignment)  # [0, 1]

        return rewards


class AerialPlayReward(RewardFunction[AgentID, GameState, float]):
    """
    Unified aerial skill development reward.

    v3.5: Replaces BOTH InAirReward AND GroundedReward.

    PROBLEM: The old setup was anti-aerial:
      - InAirReward (weight 0.01) gave tiny reward for being airborne
      - GroundedReward (weight 0.008) actively PUNISHED being airborne
      - Dense comfort rewards (spacing, face_ball) meant going airborne
        had massive OPPORTUNITY COST — bot lost ~0.02/step of comfort
        rewards every tick it was in the air.
      Result: Bot waits on ground when ball is in the air.

    SOLUTION: Strong, focused aerial technique reward:
      1. AERIAL TOUCH (big): Touching ball while both player and ball are
         elevated. This is the core technique learning signal. Scales with
         ball height — higher touches earn more.
      2. APPROACH VELOCITY (small, v3.20): Speed toward ball while airborne
         without touching. Provides gradient for aerial approach PRECISION.
         Speed-based, not position-based — avoids v3.6's "hover near ball"
         problem. A car flying directly at the ball gets reward; a car
         hovering nearby does not.

    v3.6: Removed positioning component (proximity-based hover farming).
    v3.20: Added approach velocity component (speed-based, no hover abuse).
        The bot goes for aerials (16.1% aerial_frames, rising) but MISSES
        because there's NO gradient between "close miss" and "way off."
        Both get 0.0 reward. Speed-toward-ball rewards the APPROACH, not
        the position, teaching trajectory control.

    Output: [0, 1]
    """

    def __init__(self, min_ball_height: float = 150.0, approach_weight: float = 0.15):
        """
        Args:
            min_ball_height: Minimum ball Z for aerial play to activate.
                150 = even moderately elevated balls trigger aerial reward.
                Was 300 in old InAirReward — too high, bot ignored mid-height balls.
            approach_weight: Weight of the approach velocity component relative
                to the touch component. 0.15 = approach gives up to 15% of
                max reward per step. Touch gives 0.5-1.0 per step. This ensures
                approach guides the trajectory but doesn't dominate over touches.
        """
        self.min_ball_height = min_ball_height
        self.approach_weight = approach_weight

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        ball_z = state.ball.position[2]
        ball_high = ball_z > self.min_ball_height

        # Height factor: how elevated is the ball? [0, 1]
        if ball_high:
            height_factor = min(1.0, (ball_z - self.min_ball_height) /
                                (common_values.CEILING_Z - self.min_ball_height))
        else:
            height_factor = 0.0

        rewards = {}
        for agent in agents:
            if not ball_high:
                rewards[agent] = 0.0
                continue

            car = state.cars[agent]
            airborne = not car.on_ground

            if airborne and car.ball_touches > 0:
                # === AERIAL TOUCH — the technique discovery signal ===
                # This is the big payoff. Scale with ball height.
                # Height 150 → 0.5, Height 1100 (mid-ceiling) → 0.75, Ceiling → 1.0
                rewards[agent] = 0.5 + 0.5 * height_factor
            elif airborne:
                # === AERIAL APPROACH — v3.20: trajectory guidance signal ===
                # Speed toward ball while airborne but not touching.
                # Provides gradient for approach precision: "aim AT the ball."
                # Speed-based (not proximity-based) — won't reward hovering.
                # A car flying at 1500 uu/s directly at ball gets max reward.
                # A car hovering at 0 uu/s near ball gets 0.
                pos_diff = state.ball.position - car.physics.position
                dist = np.linalg.norm(pos_diff)
                if dist < 1e-5:
                    rewards[agent] = self.approach_weight
                else:
                    dir_to_ball = pos_diff / dist
                    speed_toward = np.dot(car.physics.linear_velocity, dir_to_ball)
                    approach_reward = max(0.0, speed_toward / common_values.CAR_MAX_SPEED)
                    rewards[agent] = approach_reward * self.approach_weight * height_factor
            else:
                rewards[agent] = 0.0

        return rewards


# === LEGACY CLASSES (kept for checkpoint compatibility, no longer used in build) ===

class InAirReward(RewardFunction[AgentID, GameState, float]):
    """LEGACY — replaced by AerialPlayReward in v3.5."""
    def __init__(self, min_ball_height: float = 300.0):
        self.min_ball_height = min_ball_height
    def reset(self, agents, initial_state, shared_info): pass
    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        ball_high = state.ball.position[2] > self.min_ball_height
        return {a: float(ball_high and not state.cars[a].on_ground) for a in agents}


class GroundedReward(RewardFunction[AgentID, GameState, float]):
    """LEGACY — removed in v3.5. Was actively preventing aerial development."""
    def reset(self, agents, initial_state, shared_info): pass
    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        return {a: float(state.cars[a].on_ground) for a in agents}


class BoostConservationReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards having boost available. Uses sqrt scaling so low boost
    is penalized more than high boost (going from 0→50 is more
    valuable than 50→100 in Rocket League).
    Output: [0, 1]

    v3.7: REMOVED — deemed redundant with BoostChangeReward.
    v3.12: RE-ADDED — Research shows it serves a DIFFERENT function.
        BoostChangeReward rewards GETTING boost (action, sparse pickup event).
        BoostConservation rewards HAVING boost (state, continuous pressure).
        Now paired with BoostApproachReward providing that gradient,
        conservation becomes meaningful: "approach gets you there,
        change rewards pickup, conservation rewards keeping it."

    v3.19: BLANKET airborne exemption — returns 0 when car is airborne.
    v3.20: CONDITIONAL airborne exemption — only exempt when ball is elevated
        (z > min_ball_height). Same fix as AirborneAwareBoostChangeReward.
        The blanket exemption taught the bot "jump = free boost zone" —
        avg_boost crashed from 26.8 to 13.5 as the bot exploited this.
        With the conditional exemption, conservation pressure is only
        removed during legitimate aerial play (ball actually elevated).
        Ground-level jumps, kickoff hops, random jumps still face
        conservation pressure.

    Note: rlgym v2 boost_amount is in [0, 100] (not [0, 1]).
    """

    def __init__(self, min_ball_height: float = 300.0):
        self.min_ball_height = min_ball_height

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        ball_elevated = state.ball.position[2] > self.min_ball_height
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            # v3.20: Only exempt when airborne AND ball is elevated
            # (legitimate aerial play, not random jumping)
            if not car.on_ground and ball_elevated:
                rewards[agent] = 0.0
            else:
                boost_amount = car.boost_amount / 100.0  # [0,100] -> [0,1]
                rewards[agent] = np.sqrt(max(0.0, boost_amount))  # sqrt scaling
        return rewards


class AirborneAwareBoostChangeReward(RewardFunction[AgentID, GameState, float]):
    """
    Wraps rlgym_tools BoostChangeReward with CONDITIONAL airborne exemption.

    v3.19: Replaces raw BoostChangeReward. Airborne boost loss exempt.
    v3.20: CONDITIONAL exemption — ball must be elevated (z > min_ball_height)
    for the airborne exemption to apply. Fixes blanket exemption exploit.

    v3.19 PROBLEM: Blanket `not car.on_ground` exemption was too permissive.
    The bot learned: jump = free boost zone. ANY hop disables boost cost.
    Match evidence (14B): avg_boost crashed 26.8→13.5, zero_boost 30%→45%.
    Bot dumps boost freely during ANY jump, lands empty, can't challenge.
    Double-commits rose from 5.7% to 8.5-13.8% because boost cost was the
    implicit coordination mechanism — without it both bots go for every ball.

    v3.20 FIX: Only exempt boost loss when ball_z > min_ball_height AND car
    is airborne. This confirms the car is engaged in legitimate aerial play
    (ball IS elevated, worth going for). Low-altitude jumps, kickoff hops,
    random jumps still incur boost cost. Preserves aerial incentive while
    closing the exploit.

    No abuse vector: AerialPlayReward/AerialDistanceReward require actual
    ball contact at height. Going airborne near ground-level ball earns
    nothing positive AND boost cost is restored.
    """

    def __init__(self, gain_weight: float = 1.0, lose_weight: float = 0.5,
                 min_ball_height: float = 300.0):
        import math
        self.gain_weight = gain_weight
        self.lose_weight = lose_weight
        self.min_ball_height = min_ball_height
        self.activation_fn = lambda x: math.sqrt(0.01 * x)
        self.prev_values = None

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        self.prev_values = {
            agent: self.activation_fn(initial_state.cars[agent].boost_amount)
            for agent in agents
        }

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        ball_elevated = state.ball.position[2] > self.min_ball_height
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            current_value = self.activation_fn(car.boost_amount)
            delta = current_value - self.prev_values[agent]

            if delta > 0:
                # Boost gained — always reward (ground pad pickups, aerial pad catches)
                rewards[agent] = delta * self.gain_weight
            elif delta < 0:
                # Boost lost:
                # - On ground → always penalize (wasteful ground boosting)
                # - Airborne + ball elevated → exempt (legitimate aerial play)
                # - Airborne + ball LOW → penalize (not real aerial, just jumping)
                if car.on_ground or not ball_elevated:
                    rewards[agent] = delta * self.lose_weight
                else:
                    # Legitimate aerial: airborne AND ball is elevated
                    rewards[agent] = 0.0
            else:
                rewards[agent] = 0.0

            self.prev_values[agent] = current_value

        return rewards


class BoostApproachReward(RewardFunction[AgentID, GameState, float]):
    """
    Dense per-tick reward for velocity toward the nearest available boost pad
    when the agent's boost is below a threshold.

    v3.12 NEW — THE critical fix for boost starvation.

    ROOT CAUSE (from Claude Research at 7.18B):
        BoostChangeReward fires ONLY at the pad pickup instant, but the DECISION
        to drive toward a pad occurs 5-20 action steps earlier. Ball rewards
        (VelocityBallToGoalReward, GoalViewReward) fire EVERY tick, creating
        a ~100x denser gradient signal. Increasing BoostChangeReward weight
        7.5x (0.04→0.30) across 5 versions produced ZERO improvement because
        the temporal credit assignment gap remained.

    SOLUTION:
        Convert boost collection from a sparse delayed event into a continuous
        dense signal — same signal density as ball rewards. This provides the
        PRE-PICKUP gradient: "you're heading toward a pad = good" on every tick,
        bridging the 5-20 step credit assignment gap.

    DESIGN:
        - Rewards velocity component toward nearest available pad (like
          SpeedTowardBallReward but for pads)
        - Only active when boost < threshold (30%) — doesn't interfere with
          ball play when boost is adequate
        - Urgency scaling: lower boost → stronger signal (0 boost = max urgency)
        - Small pads weighted 25% "closer" to prevent the universal small-pad
          neglect problem (ZealanL: "make small pad pickup much stronger")
        - Uses world-coordinate pad positions (BOOST_LOCATIONS) + pad timers
          to identify available pads

    THEORETICAL BASIS:
        Inspired by Potential-Based Reward Shaping (Ng et al., 1999). The boost-
        conditional activation breaks the formal PBRS guarantee but provides
        practical benefit: dense gradient signal matching ball rewards' density
        while avoiding pad-chasing when boost is sufficient.

    No public Rocket League RL codebase has implemented an approach-based
    boost reward. This is novel in the domain.

    Output: [0, 1]
    """

    # Pre-computed pad positions and big pad indices
    PAD_LOCATIONS = np.array(common_values.BOOST_LOCATIONS)  # (34, 3)
    BIG_PAD_INDICES = frozenset({3, 4, 15, 18, 29, 30})

    def __init__(self, boost_threshold: float = 0.30, small_pad_distance_mult: float = 0.8):
        """
        Args:
            boost_threshold: Boost level (0-1 scale) above which the reward
                deactivates. 0.30 = reward only fires when boost < 30%.
                At 30%, the agent has enough for a few actions — don't distract
                from ball play.
            small_pad_distance_mult: Multiplier on distance to small pads.
                < 1.0 makes small pads appear "closer" than they are, biasing
                the agent toward the more abundant small pads. 0.8 = small pads
                are effectively 25% closer. This addresses ZealanL's explicit
                recommendation: "Bots have a tendency to ignore small pads."
        """
        self.boost_threshold = boost_threshold
        self.small_pad_distance_mult = small_pad_distance_mult

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}

        # Pre-compute which pads are available (timer == 0 means active)
        pad_timers = state.boost_pad_timers  # shape (34,)
        available_mask = pad_timers == 0.0  # bool array (34,)

        if not np.any(available_mask):
            # No pads available at all (extremely rare)
            return {agent: 0.0 for agent in agents}

        # Get positions of available pads only
        available_positions = self.PAD_LOCATIONS[available_mask]  # (N, 3)
        available_indices = np.where(available_mask)[0]  # original indices

        for agent in agents:
            car = state.cars[agent]
            boost = car.boost_amount / 100.0  # [0, 100] → [0, 1]

            if boost >= self.boost_threshold:
                rewards[agent] = 0.0
                continue

            # Urgency: stronger signal at lower boost levels
            # boost=0 → urgency=1.0, boost=0.15 → urgency=0.5, boost=0.29 → urgency=0.03
            urgency = 1.0 - (boost / self.boost_threshold)

            car_pos = car.physics.position  # (3,)

            # Compute distances to all available pads (vectorized)
            deltas = available_positions - car_pos  # (N, 3)
            distances = np.linalg.norm(deltas, axis=1)  # (N,)

            # Apply small pad distance multiplier:
            # Small pads get their distance reduced, making them appear "closer"
            effective_distances = distances.copy()
            for idx_in_available, original_idx in enumerate(available_indices):
                if original_idx not in self.BIG_PAD_INDICES:
                    effective_distances[idx_in_available] *= self.small_pad_distance_mult

            # Find nearest pad by effective distance
            nearest_idx = np.argmin(effective_distances)
            nearest_delta = deltas[nearest_idx]  # (3,) vector to nearest pad
            nearest_dist = distances[nearest_idx]  # actual distance (not effective)

            if nearest_dist < 1e-5:
                # Already on top of the pad
                rewards[agent] = urgency
                continue

            # Direction to nearest available pad (unit vector)
            dir_to_pad = nearest_delta / nearest_dist

            # Velocity component toward the nearest pad
            speed_toward = np.dot(car.physics.linear_velocity, dir_to_pad)

            # Normalize by max car speed, clip to [0, 1] — no penalty for moving away
            normalized = max(0.0, speed_toward / common_values.CAR_MAX_SPEED)

            rewards[agent] = normalized * urgency

        return rewards


# ============================================================================
# 2v2 TEAM-SPECIFIC REWARDS
# ============================================================================

class TeammateSpacingReward(RewardFunction[AgentID, GameState, float]):
    """
    Penalizes being too close to teammates (ball-chasing prevention).
    Rewards maintaining good spacing from teammates.

    In 2v2, ideal spacing is roughly 2000-4000 units between teammates.
    Output: [0, 1] — 0 when on top of teammate, 1 at ideal distance.

    This is CRITICAL for 2v2. Without this, both bots will chase the ball
    and leave the goal wide open.
    """

    IDEAL_SPACING = 3000.0  # Unreal units
    MAX_SPACING = 8000.0    # Beyond this, no additional reward

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}

        # Group agents by team
        teams: Dict[bool, List[AgentID]] = {False: [], True: []}
        for agent in agents:
            teams[state.cars[agent].is_orange].append(agent)

        for agent in agents:
            car = state.cars[agent]
            team = teams[car.is_orange]

            if len(team) < 2:
                rewards[agent] = 0.5  # Solo, neutral reward
                continue

            # Calculate distance to all teammates
            my_pos = car.physics.position
            teammate_dists = []
            for other in team:
                if other == agent:
                    continue
                other_pos = state.cars[other].physics.position
                dist = np.linalg.norm(my_pos - other_pos)
                teammate_dists.append(dist)

            if not teammate_dists:
                rewards[agent] = 0.5
                continue

            avg_dist = np.mean(teammate_dists)
            # Smooth reward: ramps up to 1 at ideal distance, decays slowly after
            if avg_dist <= self.IDEAL_SPACING:
                rewards[agent] = avg_dist / self.IDEAL_SPACING
            else:
                # v3.6: STEEPER decay past ideal — was 0.3, now 0.6.
                # At 3462 avg distance overnight, old formula gave 0.97 reward
                # (barely penalized). New formula: 0.94 at 3462, 0.76 at 4500.
                # This discourages parking far away from teammate.
                overshoot = (avg_dist - self.IDEAL_SPACING) / (self.MAX_SPACING - self.IDEAL_SPACING)
                rewards[agent] = max(0.0, 1.0 - 0.6 * overshoot)

        return rewards


class RotationReward(RewardFunction[AgentID, GameState, float]):
    """
    Rotation-behind-ball reward with distance scaling.

    v4.2 REWRITE: Replaced the complex role-differentiated rotation system
    (v3.3→v3.26) with a clean, universal signal. The old system had separate
    logic for closest-to-ball vs support, attacking vs defensive half,
    ideal_y positions with retreat distances — too many competing gradients
    that caused parking and confusion.

    New design (based on RocketSim community pattern):
      - "behind" = how far the player is on the defensive side of the ball,
        normalized by 2200uu (roughly half-field). Clamped to [0, 1].
      - "dist_scale" = proximity to ball, inverse of distance normalized
        by 3500uu. Closer = higher scale.
      - reward = behind * (0.4 + 0.6 * dist_scale)

    This creates a single clean gradient: be behind the ball AND close to it.
    At max distance (3500+uu), being behind still gives 0.4 * behind — the bot
    doesn't lose all signal for rotating back from far away. But being close
    AND behind gives up to 1.0 — rewarding active support positioning.

    Applied to ALL players equally — no role detection needed. The closest
    player naturally gets high dist_scale while engaging, and the support
    player gets rewarded for staying behind the play.

    Output: [0, 1]
    """

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        ball_pos = state.ball.position
        rewards = {}

        for agent in agents:
            car = state.cars[agent]
            car_pos = car.physics.position

            # How far behind the ball the player is (defensive side)
            # Blue attacks +Y → behind = ball_y - car_y (positive when car is closer to own goal)
            # Orange attacks -Y → behind = car_y - ball_y
            if not car.is_orange:
                rel = ball_pos[1] - car_pos[1]
            else:
                rel = car_pos[1] - ball_pos[1]

            behind = max(0.0, min(rel / 2200.0, 1.0))

            # Distance scaling: closer to ball = stronger signal
            to_ball = ball_pos - car_pos
            dist = np.linalg.norm(to_ball)
            dist_scale = 1.0 - max(0.0, min(dist / 3500.0, 1.0))

            rewards[agent] = behind * (0.4 + 0.6 * dist_scale)

        return rewards


# ============================================================================
# SPARSE EVENT REWARDS
# ============================================================================

class GoalReward(RewardFunction[AgentID, GameState, float]):
    """
    Large positive reward for scoring, large negative for conceding.
    Output: -concede_multiplier to +1.

    v3.4: Made ASYMMETRIC. Conceding is punished harder than scoring is rewarded.
    With concede_multiplier=2.0 and goal_weight=15:
      Score a goal:   +1.0 * 15.0 = +15.0
      Get scored on:  -2.0 * 15.0 = -30.0
    This creates STRONG defensive urgency without needing dense defense rewards.
    The bot learns: conceding is catastrophic → must prevent goals.

    Uses GameState.goal_scored (bool) and GameState.scoring_team (0=blue, 1=orange).
    """

    def __init__(self, concede_multiplier: float = 1.0):
        """
        Args:
            concede_multiplier: How much harder to punish conceding vs scoring.
                1.0 = symmetric (old behavior), 2.0 = conceding hurts 2x more.
        """
        self.concede_multiplier = concede_multiplier

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}

        if state.goal_scored:
            # scoring_team: 0 = blue scored, 1 = orange scored
            blue_scored = state.scoring_team == 0
            for agent in agents:
                car = state.cars[agent]
                if blue_scored:
                    # Blue scored: blue gets +1, orange gets -concede_multiplier
                    rewards[agent] = -self.concede_multiplier if car.is_orange else 1.0
                else:
                    # Orange scored: orange gets +1, blue gets -concede_multiplier
                    rewards[agent] = 1.0 if car.is_orange else -self.concede_multiplier
        else:
            for agent in agents:
                rewards[agent] = 0.0

        return rewards


class TouchBallReward(RewardFunction[AgentID, GameState, float]):
    """
    Reward for touching the ball, scaled by the height of the touch.
    Higher touches (aerials) get more reward to encourage aerial play.
    Output: [0, 1]
    """

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            if car.ball_touches > 0:
                # Scale by ball height for aerial encouragement
                height_bonus = min(state.ball.position[2] / common_values.CEILING_Z, 1.0)
                rewards[agent] = 0.5 + 0.5 * height_bonus  # [0.5, 1.0]
            else:
                rewards[agent] = 0.0
        return rewards


class DemoReward(RewardFunction[AgentID, GameState, float]):
    """
    Reward for demolishing opponents.
    Output: 0 or 1.

    Uses bump_victim_id to detect if this car bumped someone,
    and checks if the victim is demoed (demo_respawn_timer > 0).
    """

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            victim_id = car.bump_victim_id
            if victim_id is not None and victim_id in state.cars:
                victim = state.cars[victim_id]
                # Confirm it was a demo, not just a bump
                if victim.demo_respawn_timer > 0:
                    rewards[agent] = 1.0
                else:
                    rewards[agent] = 0.0
            else:
                rewards[agent] = 0.0
        return rewards


class SaveReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards actual defensive touches (clears) near own goal.

    v3 FIX at 1.2B: Was dense (fired every tick for positioning alone) which
    caused the bot to CAMP in front of own goal farming save reward. The dense
    positional component was the single highest reward signal (0.30-0.40 avg)
    and overwhelmed all offensive signals combined.

    Now primarily EVENT-BASED:
      - BIG reward for actually touching/clearing the ball in defensive position
      - BONUS for clearing ball away from own goal (proper defensive direction)
      - TINY positional guidance (25x smaller than before) — just enough to teach
        that being between ball and goal is good, NOT enough to farm

    Output: [0, 1]
    """

    def __init__(self, defense_zone_y: float = 3500.0, proximity_bonus_dist: float = 1500.0):
        """
        Args:
            defense_zone_y: Y distance from own goal where defensive reward activates
            proximity_bonus_dist: Distance to ball within which proximity bonus applies
        """
        self.defense_zone_y = defense_zone_y
        self.proximity_bonus_dist = proximity_bonus_dist

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        ball_pos = state.ball.position

        for agent in agents:
            car = state.cars[agent]
            car_pos = car.physics.position

            # Own goal Y: blue = -BACK_NET_Y, orange = +BACK_NET_Y
            own_goal_y = -common_values.BACK_NET_Y if not car.is_orange else common_values.BACK_NET_Y

            # Is ball in our defensive zone?
            ball_dist_to_goal = abs(ball_pos[1] - own_goal_y)
            if ball_dist_to_goal > self.defense_zone_y:
                rewards[agent] = 0.0
                continue

            # Is the agent between the ball and own goal?
            if not car.is_orange:
                between_ball_and_goal = car_pos[1] < ball_pos[1]
            else:
                between_ball_and_goal = car_pos[1] > ball_pos[1]

            if not between_ball_and_goal:
                rewards[agent] = 0.0
                continue

            # Danger level: how close is the ball to our goal? [0, 1]
            danger = 1.0 - (ball_dist_to_goal / self.defense_zone_y)

            if car.ball_touches > 0:
                # === PRIMARY: Actual defensive touch (clearing the ball) ===
                # Check if the clear went in the right direction (away from own goal)
                ball_vel_y = state.ball.linear_velocity[1]
                if not car.is_orange:
                    clearing_away = ball_vel_y > 0  # Ball moving away from blue goal (-Y)
                else:
                    clearing_away = ball_vel_y < 0  # Ball moving away from orange goal (+Y)

                if clearing_away:
                    # Great clear — full reward scaled by danger
                    rewards[agent] = danger
                else:
                    # Touched ball but bad direction — small reward for effort
                    rewards[agent] = danger * 0.2
            else:
                # v3.25: Positional guide REMOVED entirely.
                # Was: danger * proximity * 0.02 per tick for being between ball
                # and goal. Even at 0.002/step, the user correctly identified this
                # as encouraging "staying behind." Clearing-only: reward touches
                # that redirect ball, never reward just standing near goal.
                rewards[agent] = 0.0

        return rewards


class DirectedTouchReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards touching the ball in a useful direction (toward opponent goal).
    Unlike TouchBallReward which rewards ANY touch equally, this only
    rewards touches where the ball ends up moving toward the opponent's goal.

    This teaches the bot to aim its hits, not just whack the ball randomly.

    v3.10: Added sustained contact decay. The first touch in a contact sequence
    gets full reward. Each consecutive step of continued contact decays by
    sustain_decay multiplier. This prevents the bot from farming reward by
    lingering on the ball (20-40+ consecutive tick sequences in every match)
    and teaches "hit and move" behavior. Total reward per contact sequence is
    capped at 1/(1-decay) * first_touch_reward regardless of duration.

    Output: [0, 1] — 1 when ball is moving fast toward opponent goal after touch.
    """

    def __init__(self, sustain_decay: float = 0.5):
        """
        Args:
            sustain_decay: Geometric decay factor for consecutive-step contacts.
                0.5 = each consecutive step gets half the previous reward.
                Step 0: 100%, Step 1: 50%, Step 2: 25%, Step 3: 12.5%...
                Infinite series converges to 2x first-touch reward.
        """
        self.sustain_decay = sustain_decay
        self._touch_streak: Dict[AgentID, int] = {}

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        self._touch_streak = {a: 0 for a in agents}

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            if car.ball_touches == 0:
                self._touch_streak[agent] = 0
                rewards[agent] = 0.0
                continue

            # Calculate ball velocity direction relative to opponent goal
            goal_y = common_values.BACK_NET_Y if not car.is_orange else -common_values.BACK_NET_Y
            goal_pos = np.array([0.0, goal_y, 0.0])

            ball_to_goal = goal_pos - state.ball.position
            dist = np.linalg.norm(ball_to_goal)
            if dist < 1e-5:
                base_reward = 1.0
            else:
                direction = ball_to_goal / dist
                ball_speed_toward_goal = np.dot(state.ball.linear_velocity, direction)

                if ball_speed_toward_goal <= 0:
                    # Ball moving away from opponent goal after our touch — no reward
                    base_reward = 0.0
                else:
                    # Scale by how fast ball is going toward goal
                    base_reward = min(ball_speed_toward_goal / common_values.BALL_MAX_SPEED, 1.0)

            # Apply sustained contact decay: consecutive-step contacts get
            # geometrically diminishing reward. First touch = full, then decay.
            decay = self.sustain_decay ** self._touch_streak[agent]
            rewards[agent] = base_reward * decay
            self._touch_streak[agent] += 1

        return rewards


class WallPlayReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards the agent for making plays near/off walls and corners.

    v3.4 NEW: The bot avoids walls entirely, always playing through center field.
    Wall plays are crucial for Rocket League development:
      - Cross-field passes from side walls
      - Corner clears and pinches
      - Setting up aerial opportunities (ball rolling up wall → aerial redirect)
      - More varied, unpredictable offense

    Fires on ball touches when the ball is near a side wall (|X| close to
    SIDE_WALL_X) or in a corner zone. Scales with touch quality — directed
    touches toward goal earn more than random whacks.

    Event-based (only on touches) so it won't create farming behavior.
    Output: [0, 1]
    """

    def __init__(self, wall_threshold: float = 1200.0):
        """
        Args:
            wall_threshold: Distance from wall within which the ball counts
                as "near wall". Side walls are at ±4096 (SIDE_WALL_X).
                1200 means ball at |X| > 2896 counts as wall play zone.
        """
        self.wall_threshold = wall_threshold

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        ball_pos = state.ball.position

        # Check if ball is near side walls or in corner areas
        near_side_wall = abs(ball_pos[0]) > (common_values.SIDE_WALL_X - self.wall_threshold)
        # Corner zone: ball near both side wall AND back wall area
        near_corner = (abs(ball_pos[0]) > (common_values.SIDE_WALL_X - self.wall_threshold * 1.5) and
                       abs(ball_pos[1]) > (common_values.BACK_NET_Y - self.wall_threshold * 1.5))
        ball_in_wall_zone = near_side_wall or near_corner

        for agent in agents:
            car = state.cars[agent]
            if not ball_in_wall_zone or car.ball_touches == 0:
                rewards[agent] = 0.0
                continue

            # Ball is in wall zone AND we touched it — reward!
            # Scale by how well we directed the touch toward opponent goal
            goal_y = common_values.BACK_NET_Y if not car.is_orange else -common_values.BACK_NET_Y
            goal_pos = np.array([0.0, goal_y, 0.0])

            ball_to_goal = goal_pos - state.ball.position
            dist = np.linalg.norm(ball_to_goal)
            if dist < 1e-5:
                rewards[agent] = 1.0
                continue

            direction = ball_to_goal / dist
            ball_vel_toward_goal = np.dot(state.ball.linear_velocity, direction)

            # Base reward for touching ball in wall zone (0.4)
            # Directed bonus for hitting it toward goal (up to 0.6 more)
            # Corner touches get extra bonus (+0.2 on base)
            base = 0.6 if near_corner else 0.4
            directed_bonus = max(0.0, ball_vel_toward_goal / common_values.BALL_MAX_SPEED) * (1.0 - base)
            rewards[agent] = base + directed_bonus

        return rewards


class WallBallChallengeReward(RewardFunction[AgentID, GameState, float]):
    """
    Dense per-tick reward for velocity toward the ball when ball is on/near a wall.

    v3.18 NEW: Addresses the SAME temporal credit assignment gap that
    BoostApproachReward solved for boost starvation. WallPlayReward fires
    ONLY on touches near walls (event-based), but the DECISION to drive toward
    the wall ball occurs 5-20 steps earlier with zero positive feedback. The bot
    rationally waits for the ball to come to ground because:
      1. No dense approach signal for wall ball (unlike ball on ground where
         internalized SpeedTowardBall provides general approach gradient)
      2. WallPlayReward payoff (0.06) is 3-8x weaker than DirectedTouch ground
         hits (0.20+), making wall play an irrational choice

    This reward provides the PRE-TOUCH gradient: "heading toward ball on wall =
    good" on every tick, bridging the temporal credit assignment gap. Same
    architectural pattern as BoostApproachReward.

    Activates when ball is:
      - On side wall surface (|x| > SIDE_WALL_X - threshold) AND elevated (z > min_height)
      - OR in corner area (near both side wall and back wall)
      - OR on back wall (|y| > BACK_NET_Y - threshold) AND elevated

    Output: [0, 1] — speed toward ball / CAR_MAX_SPEED when wall ball detected.
    """

    def __init__(self, wall_threshold: float = 1200.0, min_ball_height: float = 200.0):
        """
        Args:
            wall_threshold: Distance from wall within which the ball counts
                as "near wall". Same threshold as WallPlayReward. Side walls
                at ±4096 (SIDE_WALL_X). 1200 means ball at |X| > 2896.
            min_ball_height: Minimum ball Z for wall ball detection.
                200uu ensures ball is actually ON the wall surface (rolling up,
                bouncing off), not just on the ground near a wall edge.
                Ball on ground = 93uu. Ball rolling on side wall = 200+uu.
                Exception: corner areas don't need height check (ground play
                in corners IS wall play).
        """
        self.wall_threshold = wall_threshold
        self.min_ball_height = min_ball_height

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        ball_pos = state.ball.position
        ball_z = ball_pos[2]

        # Check if ball is on/near a wall surface
        near_side_wall = abs(ball_pos[0]) > (common_values.SIDE_WALL_X - self.wall_threshold)
        near_back_wall = abs(ball_pos[1]) > (common_values.BACK_NET_Y - self.wall_threshold)

        # Corner zone: near both side wall AND back wall (any height — corners are wall play)
        in_corner = (abs(ball_pos[0]) > (common_values.SIDE_WALL_X - self.wall_threshold * 1.5) and
                     abs(ball_pos[1]) > (common_values.BACK_NET_Y - self.wall_threshold * 1.5))

        # Wall ball: on wall surface (elevated) OR in corner (any height)
        ball_on_wall = (
            in_corner or
            (near_side_wall and ball_z > self.min_ball_height) or
            (near_back_wall and ball_z > self.min_ball_height)
        )

        if not ball_on_wall:
            return {agent: 0.0 for agent in agents}

        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            car_pos = car.physics.position

            # Direction to ball
            pos_diff = ball_pos - car_pos
            dist = np.linalg.norm(pos_diff)

            if dist < 1e-5:
                rewards[agent] = 1.0
                continue

            dir_to_ball = pos_diff / dist

            # Speed toward ball
            speed_toward = np.dot(car.physics.linear_velocity, dir_to_ball)

            # Normalize and clip to [0, 1] — no penalty for moving away
            rewards[agent] = max(0.0, speed_toward / common_values.CAR_MAX_SPEED)

        return rewards


class SpeedGainReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards gaining speed over a lookback window.

    v3.22 NEW: User request — "rewarded relatively to the speed it gained
    in the last x seconds / ticks so it learns to gain speed."

    Complements WavedashReward: wavedash rewards the FLIP TECHNIQUE (mechanic),
    speed gain rewards the RESULT (actual velocity increase). Teaches:
      - Boosting in forward direction (converts boost → speed efficiently)
      - Speed flips and wavedashes (technique → speed gain → reward)
      - Not braking unnecessarily (sharp turns kill speed with no penalty,
        but you miss the reward for maintaining/gaining speed)
      - Recovery from low speed (after landing, demos, wall bounces)

    Uses a sliding window: tracks speed N ticks ago, rewards the delta.
    Only rewards POSITIVE speed gains (gain_only=True by default). Braking
    can be tactically correct (approaching ball, aligning for shot), so
    slowing down is NOT penalized.

    With action_repeat=8 at 120Hz physics:
      lookback_ticks=15 → 15 × 8/120 = 1.0 second window

    Output: [0, 1] — 1 when going from 0 to CAR_MAX_SPEED in one window.
    """

    def __init__(self, lookback_ticks: int = 15):
        """
        Args:
            lookback_ticks: Number of reward steps to look back for speed
                comparison. At action_repeat=8, 120Hz: 15 ticks ≈ 1 second.
                Shorter = rewards quick bursts (flips). Longer = rewards
                sustained acceleration (boosting across field).
        """
        self.lookback_ticks = lookback_ticks
        self.speed_buffers = {}

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        self.speed_buffers = {}
        for agent in agents:
            speed = np.linalg.norm(initial_state.cars[agent].physics.linear_velocity)
            self.speed_buffers[agent] = [speed]

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            current_speed = np.linalg.norm(car.physics.linear_velocity)

            buf = self.speed_buffers.get(agent)
            if buf is None:
                buf = [current_speed]
                self.speed_buffers[agent] = buf
                rewards[agent] = 0.0
                continue

            # Compare with oldest entry in buffer
            if len(buf) >= self.lookback_ticks:
                old_speed = buf[0]
                speed_gain = current_speed - old_speed
                rewards[agent] = max(0.0, speed_gain / common_values.CAR_MAX_SPEED)
            else:
                rewards[agent] = 0.0

            # Maintain buffer: append current, trim to lookback_ticks
            buf.append(current_speed)
            if len(buf) > self.lookback_ticks:
                # Remove oldest entries to keep buffer at lookback_ticks
                del buf[0:len(buf) - self.lookback_ticks]

        return rewards


class GroundDribbleReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards carrying the ball on top of the car while moving toward goal.

    v3.17 NEW: The bot has demonstrated ground ball control — 59 consecutive
    touches in a single sequence (5.4 seconds sustained contact). The mechanical
    skill EXISTS but there's no explicit reward for the ground dribble position
    (ball on car roof, car driving toward goal). DirectedTouchReward rewards
    touches-toward-goal but DECAYS with sustain_decay=0.85, teaching "hit and
    carry briefly." Ground dribbling is the opposite: SUSTAINED carry IS the skill.

    This reward provides a continuous, non-decaying signal for the ground dribble:
      - Ball within carry_radius horizontally of car center
      - Ball directly above car (80-300uu above car center — physics: ball resting
        on octane roof = ball_center at car_z + 128uu)
      - Car on or near ground
      - Reward = car speed toward opponent goal / CAR_MAX_SPEED

    Unlike DirectedTouchReward:
      1. Does NOT require ball_touches > 0 — fires on POSITION not contact frame
      2. Does NOT decay — sustained carry IS the desired behavior
      3. Measures CAR velocity toward goal, not ball velocity — in a dribble
         the ball follows the car, and car velocity IS the control signal

    Output: [0, 1] — 1 at max speed toward goal while carrying ball.
    """

    def __init__(self, carry_radius: float = 200.0, min_height_above: float = 80.0,
                 max_height_above: float = 300.0):
        """
        Args:
            carry_radius: Max horizontal distance from car center to ball center
                for dribble detection. 200uu ≈ 1.7 car lengths. Generous but
                excludes "ball just happens to be nearby" situations.
            min_height_above: Minimum ball center height above car center.
                80uu ensures ball is actually above car (not beside/in-front).
                Ball resting on roof ≈ 128uu above car center.
            max_height_above: Maximum ball center height above car center.
                300uu catches pop-up dribbles but excludes aerial plays (those
                are handled by AerialPlayReward + AerialDistanceReward).
        """
        self.carry_radius = carry_radius
        self.min_height_above = min_height_above
        self.max_height_above = max_height_above

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        ball_pos = state.ball.position
        rewards = {}

        for agent in agents:
            car = state.cars[agent]
            car_pos = car.physics.position

            # Check horizontal distance (XY plane)
            dx = ball_pos[0] - car_pos[0]
            dy = ball_pos[1] - car_pos[1]
            horiz_dist = np.sqrt(dx * dx + dy * dy)

            # Check vertical: ball above car within dribble range
            height_above = ball_pos[2] - car_pos[2]

            is_carrying = (
                horiz_dist < self.carry_radius and
                height_above > self.min_height_above and
                height_above < self.max_height_above and
                (car.on_ground or car_pos[2] < 100.0)  # Car near ground (inc. wall)
            )

            if not is_carrying:
                rewards[agent] = 0.0
                continue

            # Reward speed toward CENTER of opponent goal
            goal_y = common_values.BACK_NET_Y if not car.is_orange else -common_values.BACK_NET_Y
            goal_center = np.array([0.0, goal_y, 0.0])
            to_goal = goal_center - car_pos
            dist_to_goal = np.linalg.norm(to_goal)
            if dist_to_goal < 1.0:
                rewards[agent] = 0.0
                continue
            goal_dir = to_goal / dist_to_goal
            speed_toward_goal_center = np.dot(car.physics.linear_velocity, goal_dir)

            # Normalize and clip to [0, 1] — no reward for backward carries
            rewards[agent] = max(0.0, speed_toward_goal_center / common_values.CAR_MAX_SPEED)

        return rewards


class KickoffReward(RewardFunction[AgentID, GameState, float]):
    """
    Role-differentiated kickoff reward for 2v2 with speed urgency and ball control.

    v3.16: Rewards speed toward ball during kickoff.
    v3.20: ROLE DIFFERENTIATION — only the closest bot goes for ball.
    v4.0:  ENHANCED — three-component go-er reward:
      1. SPEED TOWARD BALL (40%): Faster approach = higher reward.
         Directly rewards high velocity toward the ball.
      2. PROXIMITY BONUS (30%): Closer you are = higher reward.
         Creates urgency: being fastest to arrive matters. Uses inverse
         distance scaled to spawning distance (~3500uu diagonal kickoff).
      3. BALL CONTROL (30%): After first contact, reward for keeping
         the ball near the car and directing it toward the opponent goal.
         Teaches the bot to win the kickoff by gaining possession, not
         just slamming into the ball blindly.

    Support bot still rewards defensive positioning (stay back, cover).

    Output: [0, 1]
    """

    def __init__(self, ball_speed_threshold: float = 100.0,
                 ball_center_threshold: float = 200.0,
                 post_touch_window: float = 1500.0,
                 control_distance: float = 400.0):
        """
        Args:
            ball_speed_threshold: Ball speed below which pre-touch kickoff is active.
            ball_center_threshold: Max ball offset from center for pre-touch kickoff.
            post_touch_window: Distance from center the ball can be for post-touch
                control phase to still apply (ball moves away after touch).
            control_distance: Ball distance from car considered "controlled" (~4 car lengths).
        """
        self.ball_speed_threshold = ball_speed_threshold
        self.ball_center_threshold = ball_center_threshold
        self.post_touch_window = post_touch_window
        self.control_distance = control_distance
        self._kickoff_toucher: Dict[str, Optional[AgentID]] = {}  # team key -> who touched first
        self._kickoff_active: bool = False

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        self._kickoff_toucher = {}
        self._kickoff_active = True

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        ball = state.ball
        ball_pos = ball.position
        ball_speed = np.linalg.norm(ball.linear_velocity)
        ball_near_center = (abs(ball_pos[0]) < self.ball_center_threshold and
                            abs(ball_pos[1]) < self.ball_center_threshold and
                            ball_pos[2] < 200.0)

        pre_touch = ball_speed < self.ball_speed_threshold and ball_near_center

        # Post-touch phase: ball has been hit but is still near-ish center
        ball_from_center = np.linalg.norm(ball_pos[:2])
        post_touch = (not pre_touch and ball_from_center < self.post_touch_window
                      and self._kickoff_active)

        if not pre_touch and not post_touch:
            self._kickoff_active = False
            return {a: 0.0 for a in agents}

        # Track who touched the ball first per team
        for agent in agents:
            car = state.cars[agent]
            team_key = "orange" if car.is_orange else "blue"
            if car.ball_touches > 0 and team_key not in self._kickoff_toucher:
                self._kickoff_toucher[team_key] = agent

        # Group agents by team and determine roles
        teams: Dict[bool, List[AgentID]] = {False: [], True: []}
        for agent in agents:
            teams[state.cars[agent].is_orange].append(agent)

        # For each team, find who is closest to ball → they go for kickoff
        closest_to_ball = set()
        for team_agents in teams.values():
            if len(team_agents) == 0:
                continue
            min_dist = float('inf')
            closest = team_agents[0]
            for a in team_agents:
                d = np.linalg.norm(ball_pos - state.cars[a].physics.position)
                if d < min_dist:
                    min_dist = d
                    closest = a
            closest_to_ball.add(closest)

        # Max spawning distance for normalization (~diagonal kickoff spawn)
        max_spawn_dist = 4100.0

        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            team_key = "orange" if car.is_orange else "blue"

            if agent in closest_to_ball:
                pos_diff = ball_pos - car.physics.position
                dist = np.linalg.norm(pos_diff)

                if pre_touch:
                    # === PRE-TOUCH: Speed + Proximity ===
                    # Component 1: Speed toward ball [0, 1]
                    if dist < 1e-5:
                        speed_component = 1.0
                    else:
                        dir_to_ball = pos_diff / dist
                        speed_toward = np.dot(car.physics.linear_velocity, dir_to_ball)
                        speed_component = max(0.0, speed_toward / common_values.CAR_MAX_SPEED)

                    # Component 2: Proximity bonus [0, 1] — closer = better
                    proximity_component = max(0.0, 1.0 - dist / max_spawn_dist)

                    # Combined: 55% speed, 45% proximity
                    rewards[agent] = 0.55 * speed_component + 0.45 * proximity_component

                else:
                    # === POST-TOUCH: Ball control ===
                    # Reward for keeping ball close AND directing it at opponent goal
                    opp_goal_y = common_values.BACK_NET_Y if not car.is_orange else -common_values.BACK_NET_Y

                    # Ball proximity to car [0, 1]
                    ball_closeness = max(0.0, 1.0 - dist / self.control_distance)

                    # Ball moving toward opponent goal [0, 1]
                    ball_vel_y = ball.linear_velocity[1]
                    if not car.is_orange:
                        goal_direction = max(0.0, ball_vel_y) / max(ball_speed, 1.0)
                    else:
                        goal_direction = max(0.0, -ball_vel_y) / max(ball_speed, 1.0)

                    # 50% closeness + 50% direction
                    rewards[agent] = 0.5 * ball_closeness + 0.5 * goal_direction

            else:
                # === SUPPORT: Stay behind, cover goal ===
                own_goal_y = -common_values.BACK_NET_Y if not car.is_orange else common_values.BACK_NET_Y
                car_y = car.physics.position[1]

                field_half = abs(own_goal_y)
                if not car.is_orange:
                    depth = max(0.0, -car_y) / field_half
                else:
                    depth = max(0.0, car_y) / field_half

                rewards[agent] = min(1.0, depth * 1.5) * 0.6

        return rewards


class FakingPenaltyReward(RewardFunction[AgentID, GameState, float]):
    """
    Penalizes flipping near the ball without making contact ("faking").

    v3.16 NEW: Self-play artifact — bot approaches ball, flips to fake a
    challenge, then retreats. Works against identical policy (opponent also
    faked) but catastrophic against real opponents who simply take the ball.

    Detection: Car starts a flip (is_flipping transitions from False to True)
    while within proximity_threshold of the ball, AND has zero ball_touches
    on that step. This is a reliable signal for:
      - Intentional fakes near ball (self-play artifact)
      - Whiffs at close range (also undesirable at 11B steps — should be
        mechanically accurate enough to not whiff from 500uu)

    Does NOT fire when:
      - Flipping far from ball (rotation/speed flips for movement)
      - Near ball without flipping (positioning, shadowing)
      - Flip makes contact (ball_touches > 0 → successful challenge)

    Event-based (fires once at flip START), not continuous.
    Output: [-1, 0] — -1 on detected fake/whiff, 0 otherwise.
    """

    def __init__(self, proximity_threshold: float = 500.0):
        """
        Args:
            proximity_threshold: Maximum distance from ball center for a flip
                to count as "near ball". Ball radius is ~93uu, car length ~118uu.
                500uu ≈ 2 car lengths from ball surface = very close range.
        """
        self.proximity_threshold = proximity_threshold
        self._prev_flipping: Dict[AgentID, bool] = {}

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        self._prev_flipping = {a: False for a in agents}

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            # Detect flip INITIATION (transition from not-flipping to flipping)
            just_started_flip = car.is_flipping and not self._prev_flipping.get(agent, False)

            if just_started_flip:
                ball_dist = np.linalg.norm(
                    state.ball.position - car.physics.position
                )
                if ball_dist < self.proximity_threshold and car.ball_touches == 0:
                    # Flipped near ball without contact → fake/whiff penalty
                    rewards[agent] = -1.0
                else:
                    rewards[agent] = 0.0
            else:
                rewards[agent] = 0.0

            self._prev_flipping[agent] = car.is_flipping

        return rewards


# ============================================================================
# COMBINED REWARD BUILDER
# ============================================================================

def build_vyrex_reward(config=None):
    """
    Build the complete VYREX reward function from config weights.

    Returns a CombinedReward with all reward components properly weighted.
    Import this in train.py and pass it to the RLGym environment.

    v3 FIX (1.2B) — Fixed reward hacking from v2 overhaul:
      PROBLEM: v2 overhaul overcorrected. Bot scored 0 goals against Psyonix Rookie.
        Dense defensive+touch rewards (1.0 combined) overwhelmed offensive signals (0.155)
        by 6.5:1. Bot learned to camp + farm touches instead of scoring.
      EVIDENCE: 0 goals, 0 shots, 67-80% of touches toward OWN goal, 0.3% supersonic.
      FIX:
        - REMOVED AdvancedTouchReward (direction-agnostic → rewarded own-goals)
        - SaveReward rewritten: event-based (only fires on touches), not dense every tick
        - save_weight: 0.5→0.15, directed_touch: 0.3→0.15
        - velocity_ball_to_goal: 0.05→0.12 (need goal direction)
        - speed_toward_ball: 0.005→0.012 (bot was too slow)
        - GoalViewReward: 0.1→0.15 (more goal-directedness)
        - demo: 0.3→0.4 (more aggression)
      NEW BALANCE: Offensive ~0.29 vs Defensive ~0.15 (was 0.155 vs 1.0)
    """
    from rlgym.rocket_league.reward_functions import CombinedReward
    from rlgym_tools.rocket_league.reward_functions.goal_prob_reward import GoalViewReward
    # v3.19: BoostChangeReward replaced by AirborneAwareBoostChangeReward (defined above)
    from rlgym_tools.rocket_league.reward_functions.wavedash_reward import WavedashReward
    from rlgym_tools.rocket_league.reward_functions.aerial_distance_reward import AerialDistanceReward
    from rlgym_tools.rocket_league.reward_functions.distribute_rewards_wrapper import DistributeRewardsWrapper

    if config is None:
        from config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG

    rc = config.rewards

    # ====================================================================
    # v3.7 — "THE SHARPENING"
    # ====================================================================
    # PROBLEM: Entropy at 4.165/4.500 (93% random) at 3.13B steps.
    #   Match data: avg_confidence=4.46%, 75% low-confidence ticks.
    #   Bot wiggles (steer oscillates -1/0/+1 randomly at 15Hz).
    #   58.6% zero boost. 1.5% supersonic. 39 flips in 7 min.
    #   48.8% directed touches (coin flip). 15 reward components
    #   create gradient collision — too many weak signals = no signal.
    #
    # FIX: Subtract noise, sharpen existing signals.
    #   - REMOVED: SpeedTowardBallReward (internalized, contributes 0.001/tick noise)
    #   - REMOVED: BoostConservationReward (redundant with BoostChangeReward, 80 rew/ep noise)
    #   - SIMPLIFIED: RotationReward (removed X-centering + speed_factor — 2 competing
    #     gradients that pulled bot away from wall boost pads)
    #   - BOOSTED: boost_change (0.04→0.10), wavedash (0.02→0.05)
    #   - PPO: ent_coef 0.003→0.001, batch 150K→200K, team_spirit 0.8→0.3
    #   - concede_multiplier 2.0→1.5 (Zealan's aggression_bias)
    #
    # 15 → 12 components. Fewer competing gradients = sharper policy updates.
    # Goal ratio: ~250 goal vs ~60 dense = 4.2:1 (was 1.6:1 in v3.5).
    # ====================================================================

    # ====================================================================
    # v3.10 — "THE CLEAN HIT"
    # ====================================================================
    # PROBLEM: Lingering contact is THE root cause of boost starvation.
    #   At 5.72B: 70% zero-boost, avg_boost 9.9 in matches. Bot pushes ball
    #   for 20-40+ consecutive ticks, getting full DirectedTouchReward every
    #   step. A 40-step push = 40x first-touch reward. Meanwhile, collecting
    #   a small boost pad gives 0.052 reward total. The bot NEVER disengages.
    #   PROOF: vs Necto (where bot is forced to rotate), avg_boost is 15.6.
    #   The bot CAN collect boost — the reward structure prevents it.
    #
    # FIX: Break the lingering contact incentive.
    #   - DirectedTouchReward: sustain_decay=0.5 — geometric decay on
    #     consecutive contacts. Step 0=100%, 1=50%, 2=25%, 3=12.5%...
    #     Total capped at 2x first touch. 40-step push: 2.0x (was 40x).
    #   - boost_change 0.15→0.20 — fill the vacuum left by touch decay
    #   - boost_change_lose 0.3→0.5 — costlier to waste boost
    #   - ent_coef 0.0006→0.00048 — break entropy wall #3 at 3.25
    #
    # Still 12 components. Same reward set, just fixing the exploitation.
    # ====================================================================

    # ====================================================================
    # v3.11 — "THE BOOST HUNTER"
    # ====================================================================
    # PROBLEM: Boost starvation persists as #1 competitive weakness at 6.5B.
    #   Match data (vs Element 2-10, vs Necto 0-15):
    #     - 63-65% zero_boost, avg_boost 10-12
    #     - 1.4% supersonic (needs boost OR flips to reach speed)
    #     - 0 power shots (>1800 uu/s) — can't finish on open nets
    #     - Bot rotates but doesn't collect pads during rotation
    #     - Bot waits on defense with 0 boost instead of pathing to pickup
    #     - Drives straight instead of speed flipping when low boost
    #   Entropy wall #4 at 3.0752 (68.3%) for 725M+ steps.
    #   Bot's wall play and aerial touch are IMPROVING — learned techniques
    #   from v3.10 but can't deploy them without boost.
    #
    # ROOT CAUSE #1: BoostChangeReward at 0.20 still loses to ball-contact
    #   signals. Even with sustain_decay, a directed touch (0.40 * direction)
    #   dominates over small pad pickup (sqrt(0.12) * 0.20 = 0.069). Bot
    #   always prefers staying near ball over detouring for pads.
    #
    # ROOT CAUSE #2 (from Kaiyotech / Nexto creator): SaveReward at 0.35
    #   creates PERVERSE INCENTIVE. "Save rewards encourage allowing the
    #   ball to get into save territory instead of just, not doing that."
    #   Match evidence: 43.7% def_third but only 11% DEF state — bot
    #   positions near goal WAITING for ball to enter danger zone (the
    #   prerequisite for save reward) instead of preventing it from getting
    #   there. Correct defense: concede penalty (-32.5) + VelBallToGoal.
    #
    # FIX: Boost as dominant dense signal + remove save trap.
    #   - SaveReward 0.35→0.0 — REMOVED. Defense from concede penalty
    #     (-32.5) + VelocityBallToGoalReward (0.20). Per Kaiyotech.
    #   - boost_change 0.20→0.30 — small pad at 0 boost: 0.104 reward
    #     (EXCEEDS mediocre touch). Big pad at 0 boost: 0.300 reward.
    #     Creates genuine incentive to path through pads during rotation.
    #   - boost_lose 0.5→0.7 — asymmetric pressure: collect=1.0×0.30,
    #     burn=0.7×0.30. Only boost when it creates value.
    #   - wavedash 0.05→0.08 — speed flip alternative when boost-starved.
    #     1800 uu/s flip → 0.063 reward. Competitive with pad pickup.
    #   - concede 1.5→1.3 — unlock aggression. Bot won't take risks with
    #     -37.5 penalty. At -32.5 it can commit to saves and attacks.
    #   - ent_coef 0.00048→0.00038 — break entropy wall #4. 21% cut.
    #
    # 12→11 components. Save trap removed, boost signal strengthened.
    # ====================================================================

    # ====================================================================
    # v3.12 — "THE BOOST PATHFINDER"
    # ====================================================================
    # PROBLEM: Boost starvation WORSENING despite v3.11's 7.5x weight increase.
    #   At 7.2B: zero_boost=75.2% (RISING), boost_collected FALLING,
    #   supersonic FALLING to 4.5%, touch_speed FALLING.
    #   Meanwhile: entropy wall #5 BROKE without intervention (2.854→2.773),
    #   aerial_touch at ATH 0.0845, demos at ATH 0.00096, speed rising.
    #   The bot is learning EVERYTHING except boost collection.
    #
    # ROOT CAUSE (Claude Research — Solving VYREX's boost starvation at 7.18B):
    #   *"The core problem is not reward weight — it's reward timing."*
    #   BoostChangeReward fires ONLY at the pad pickup instant, but the
    #   DECISION to drive toward a pad occurs 5-20 action steps earlier.
    #   VelocityBallToGoalReward shapes EVERY tick — ~100x denser signal.
    #   Multiplying a temporally-displaced sparse signal by 7.5x makes the
    #   spike taller but doesn't help PPO bridge the credit assignment gap.
    #   RUDDER (NeurIPS 2019) formally proves TD methods require exponentially
    #   many updates for delayed rewards.
    #
    #   Community evidence (sly at 50B, Foxe at 3B, Fredrik): Even at 50B
    #   with SaveBoostReward dominating 60%+ of total reward, avg boost
    #   only reaches ~25%. Zero-sum on nearly everything from the start
    #   helps (Fredrik trains at "insane necto in 1b speeds").
    #
    #   No public RL bot has an approach-based boost reward. Novel in domain.
    #
    # FIX: Three-part boost reward ARCHITECTURE (not just weight).
    #
    #   APPROACH (NEW): BoostApproachReward at 0.02
    #     Dense per-tick velocity-toward-nearest-available-pad when boost < 30%.
    #     Converts 5-20 step delayed pickup into immediate continuous feedback.
    #     Same signal density as VelocityBallToGoalReward (~0.006/step when
    #     active). Small pads weighted 25% "closer" (ZealanL recommendation).
    #     The missing link: PPO now gets gradient signal for the DECISION
    #     to drive toward a pad, not just the PICKUP moment.
    #
    #   CONSERVATION (RE-ADDED): BoostConservationReward at 0.02
    #     sqrt(boost) continuous state pressure. Removed in v3.7 ("redundant
    #     with BoostChange, 80 rew/ep noise"). Research proves it serves
    #     DIFFERENT function: with approach gradient now providing the "how
    #     to get boost" signal, conservation provides "why having boost
    #     matters" pressure. Three roles:
    #       Approach → guides TOWARD pads (pre-pickup gradient)
    #       Change   → rewards the PICKUP (event payoff)
    #       Conservation → rewards KEEPING it (retention pressure)
    #
    #   BoostChangeReward UNCHANGED at 0.30 — still provides pickup payoff.
    #
    # 11→13 active components. Two new signals are COMPLEMENTARY to existing
    # BoostChangeReward — gradient reinforcement, not gradient collision.
    # Analogous to ball play having two complementary signals
    # (VelocityBallToGoal + GoalView).
    # ====================================================================

    # ====================================================================
    # v3.13 — "THE EQUALIZER"
    # ====================================================================
    # CONTEXT: v3.12 is a DEFINITIVE SUCCESS after 645M steps (7.2B→7.845B).
    #   Zero-boost: 74.7%→62.5% (SOLVED). Avg boost: 11.2→19.0 (+70%).
    #   Touch speed: 2036→2117 (NEW ATH). Supersonic: 4.7%→5.5% (recovered).
    #   Boost collected: 1.118→1.167 (+4.4%). Aerial touch ratio: rising (R²=0.91).
    #   Demos: rising (R²=0.86). Goal diff: +17. PPO health: ALL HEALTHY.
    #   The 7.5B bucket regression was a false alarm — fully self-corrected.
    #
    # PROBLEMS REMAINING (match evidence at 7.84B):
    #   1. Passive defending: bot retreats instead of challenging when opponent
    #      has ball. Match vs Element: clears at 8, 24, 31 speed (tentative
    #      contact). Gets outdribbled by Element consistently. ROOT CAUSE:
    #      concede_multiplier=1.3 creates 30% defensive bias — failing to
    #      defend is 30% more costly than failing to score is rewarding. Bot
    #      learns "don't challenge = don't risk conceding."
    #   2. "Faking" from self-play: bot approaches ball then retreats. Works
    #      against identical policy (self-play echo chamber) but fails against
    #      Element. This is strategy overfitting — self-corrects with continued
    #      training. Same root cause: asymmetric concede penalty rewards caution.
    #   3. No ball control (dribbles): Expected at 7.8B. Community warns against
    #      explicit dribble rewards ("abuse dribbling"). Air dribble ATTEMPTS
    #      visible in self-play match (ball_z 465→802→149 with sustained contact).
    #      Dribbling emerges from: proximity + touch quality + boost. All improving.
    #   4. Basic techniques: Touch speed at ATH and rising. Techniques evolve
    #      slowly through 90-action discrete space. Kaiyotech: "Flipping for
    #      speed often comes back with time."
    #
    # FIX: Two surgical changes. No new components. Conservative for 6-hour
    #   overnight session (~378M steps unattended).
    #
    #   concede_multiplier: 1.3 → 1.0 (SYMMETRIC goal reward)
    #     Score=+25, Concede=-25 (was -32.5). Removes the 30% defensive bias
    #     that causes passive retreating and tentative touches. Bot now weighs
    #     offensive opportunity and defensive risk equally → committed challenges
    #     instead of backing off. ZealanL: "Decrease concede penalty to make
    #     the bot more aggressive." History: 2.0→1.5→1.3→1.0.
    #
    #   boost_change_weight: 0.30 → 0.20 (boost SOLVED, rebalance)
    #     Total boost weight: 0.34→0.24. Ball-play/boost ratio: 2.2:1→3.1:1.
    #     645M steps of three-part boost system (approach+change+conserve) have
    #     deeply internalized pad-pathing. Reducing event-based pickup reward
    #     shifts relative emphasis toward ball play signals (DirectedTouch 0.40,
    #     VelocityBallToGoal 0.20, GoalView 0.15). Approach(0.02) + Conservation
    #     (0.02) maintain continuous boost guidance. Pickup from 0→12% still
    #     gives 0.069 reward — meaningful but no longer dominates touches.
    #
    # 12 active components UNCHANGED. Only weights adjusted.
    # ====================================================================

    # ====================================================================
    # v3.14 — "THE AERIAL PREDATOR"
    # ====================================================================
    # CONTEXT: v3.13 ran 720M steps (7.845B→8.565B). Training metrics
    #   improving across the board: zero_boost 65.1%→57.4%, touch_speed
    #   2083→2120, aerial_touch 0.0881→0.1065 (+21%), demos rising,
    #   speed rising. PPO ALL HEALTHY. Entropy plateauing at 2.58 (57.3%).
    #   Goal diff dipped to -0.7 (expected from symmetric reward exploration).
    #
    # MATCH EVIDENCE (8.56B, 3 matches):
    #   vs Element (4-7): directed 37.7%, boost 12.1, zero_boost 58.7%
    #   vs Necto (1-9):   directed 37.4%, boost 33.3, zero_boost 29.8%
    #   vs Samara (5-1):  directed 51.6%, boost 26.3, zero_boost 36.0%
    #   KEY PATTERN: Bot plays well vs weaker opponents (51.6% directed
    #   vs Samara!) but collapses under pressure from strong opponents.
    #   Boost starvation RETURNS in competitive matches (12.1 avg vs Element).
    #   Bot has NO incentive to challenge opponent ball-carries — VBG gives
    #   0.0 when ball moves toward own goal (clamped). 63.7% of touches
    #   go AWAY from opponent goal with zero penalty.
    #
    # PROBLEMS (from user observation):
    #   1. "Bot often gets outdribbled and then scored on"
    #   2. "Time the bot learns to do proper aerials and aerial techniques"
    #   3. "Saving techniques still very undeveloped"
    #   4. "Attacks, gets defended, no boost to rotate back"
    #
    # FIX: Two changes addressing all four problems.
    #
    #   AerialDistanceReward: NEW at 0.15 (from rlgym-tools)
    #     Our AerialPlayReward gives flat 0.5-1.0 per aerial touch — teaches
    #     "touch ball while airborne" but NOT aerial technique sequences.
    #     AerialDistanceReward adds:
    #       - First touch: rewarded by HEIGHT (like AerialPlayReward)
    #       - Consecutive touches: rewarded by DISTANCE TRAVELED (car+ball)
    #         → teaches sustained air carries, air dribbles, wall-to-air
    #       - Resets on landing or opponent touch → teaches maintaining control
    #     Combined aerial signal: AerialPlay(0.10) + AerialDistance(0.15) = 0.25
    #     WARNING: At weight 10, community reported "abuse dribbling." At 0.15
    #     (67x lower), competitive with existing aerial but won't dominate.
    #     Addresses problem #2. New metric tracks abuse.
    #
    #   ZeroSumReward on VelocityBallToGoalReward: WRAPPED
    #     VBG currently clamped at 0 for ball moving toward own goal. With
    #     zero-sum: player_reward = own_vbg - opponent_avg_vbg. When opponent
    #     advances ball toward our goal, we get NEGATIVE signal. Addresses:
    #       Problem #1: Bot now incentivized to CHALLENGE opponent ball-carries
    #         instead of watching them dribble past (negative reward for inaction)
    #       Problem #3: Implicit save pressure WITHOUT Kaiyotech's "save trap."
    #         Ball moving toward our goal = negative, redirecting it = positive.
    #         No perverse incentive to let ball reach danger zone.
    #       Problem #4: When opponent has ball control and advances, negative
    #         VBG signal encourages disengaging from lost challenges and
    #         rotating. Bot won't camp near ball hoping for touch when opponent
    #         is pushing ball goalward — it'll seek defensive position.
    #     Fredrik: "i zerosum almost all my rewards from the start while still
    #       improving at insane necto in 1b speeds"
    #     Weight unchanged at 0.20. Just wrapped in ZeroSumReward.
    #
    # 13→15 components (14 active). +1 AerialDistanceReward, VBG wrapped.
    # ====================================================================

    # ====================================================================
    # v3.15 — "THE DRIBBLER"
    # ====================================================================
    # CONTEXT: v3.14 ran from ~8.5B to 11B (user tweaked aerial_distance
    #   to 0.45 and ent_coef). Plateau for 750M+ steps:
    #   Touch speed FLAT (2113→2133), supersonic FLAT (5.5%), speed FLAT (1310).
    #   Aerial touch% rising slowly (13.1%→14.5%) but plateauing.
    #   Boost improving slowly (20→23 training, 13-20 in matches).
    #   Goal diff improving in self-play (+6) but loses to Element 5-7, 5-8.
    #
    # MATCH EVIDENCE (11.0B, 2 matches vs Element):
    #   Match 1: 5-8 loss. avg_boost=13.0(!), directed=36.4%, supersonic=2.0%
    #   Match 2: 5-7 loss. avg_boost=20.5, directed=38.7%, supersonic=2.9%
    #   Event logs: REPEATED "TOUCH AWAY speed:-1533 ball_z:97" every 11 ticks
    #   — bot PUSHING ball toward own goal during sustained ground contact.
    #   36-39% directed touches means 61-64% go in useless/harmful direction.
    #
    # USER REQUEST: "more boost and to (air)dribble (air and ground)"
    # Zealan community discussion: implicit penalties (like upside-down penalty)
    # block technique learning. Our sustain_decay=0.5 IS that implicit penalty.
    #
    # ROOT CAUSES (4):
    #   1. sustain_decay=0.5 is ANTI-DRIBBLE. Dribbling = sustained directed
    #      contact. Decay penalizes exactly this. 5-step carry = only 1.94x
    #      first-touch; teaches "hit and disengage" over "control the ball."
    #      Zealan's pattern: implicit penalty blocking technique learning.
    #   2. Boost starvation collapses in matches (13-20 avg vs 23 training).
    #      Air dribbles need 40-60 boost. threshold=0.30 too late, weight too low.
    #   3. No ground dribble pathway — AerialDistance (0.45) teaches aerial
    #      sequences but ground carry is the prerequisite for air dribbles.
    #   4. Touch direction 36-39% (worse than random in meaningful metric).
    #
    # FIX: 4 surgical parameter changes. No new components. No gradient collision.
    #
    #   directed_touch_sustain_decay: 0.5 → 0.85
    #     Removes the implicit anti-dribble penalty. 10-step carry now gives
    #     4.46x (was 1.9x). Cap at 6.67x (was 2x). SAFE because decay only
    #     affects directed touches (ball toward opponent goal). Lingering in
    #     wrong direction gets base_reward=0 regardless of decay.
    #
    #   boost_conservation_weight: 0.02 → 0.06
    #     3x stronger "keep your boost" pressure. sqrt(0.5)×0.06=0.042/step
    #     matches GoalView density. Creates pressure to RETAIN boost for
    #     techniques instead of burning on contested approaches.
    #
    #   boost_approach_threshold: 0.30 → 0.50
    #     Seek pads at 50% instead of 30%. Air dribbles need 40-60 boost;
    #     at 30% it's too late. ~67% more active time for approach gradient.
    #
    #   boost_approach_weight: 0.02 → 0.04
    #     2x approach gradient. ~0.012/step active, competitive with VBG.
    #
    # 14 active components UNCHANGED. Only parameters adjusted.
    # ====================================================================

    # ====================================================================
    # v3.16 — "THE CHALLENGER"
    # ====================================================================
    # CONTEXT: v3.15 ran 200M steps (11.0B→11.2B).
    #   BOOST SUCCESS: avg_boost DOUBLED 21→39, zero_boost HALVED 53%→25%.
    #   BOOST PARADOX: boosting_frac DOWN 12.5%, boost_collected DOWN 3.3%.
    #   Bot is HOARDING boost — "boost miser" from conservation=0.06 being
    #   4x stronger than VBG. Has boost but won't spend it.
    #   Speed FLAT (1310-1326). Supersonic FLAT (4.8-5.0%). Despite 2x boost.
    #   Touch speed UP +5.3% (2080→2191). Ball speed UP +5.6%.
    #   Aerial touch UP +19% (12.4→14.8%). Bot IS better at aerials.
    #   Touches DOWN 11% (0.066→0.059). Ball distance UP (2282→2483).
    #
    # MATCH (11.2B vs Element): 1-14 LOSS (worse than 5-7/5-8!)
    #   Boost IMPROVED (25.6 avg, 31.4% zero — best match boost ever)
    #   BUT directed_pct DROPPED to 29.2% (from 36-39%)
    #   avg_touch_speed 1272 (training: 2191 — massive gap)
    #   Still doing sustained AWAY touches (29 consecutive)
    #   Most goals from kickoff losses — bot doesn't boost/flip to ball
    #
    # USER OBSERVATIONS:
    #   1. "Boost went up greatly but speed didn't" → miser behavior
    #   2. "It often lost kickoffs — doesn't flip or boost to ball" → no kickoff reward
    #   3. "Doesn't wave dash or speed flip to rotate" → wavedash too weak
    #   4. "Address faking — give penalty when it fakes" → self-play artifact
    #   5. "Happy with air hitting improvement" → aerial changes working
    #
    # ROOT CAUSES (4):
    #   1. boost_conservation=0.06 is 4x VBG → bot values HAVING boost over
    #      USING it. Boosting_frac DOWN despite 2x more boost. Miser trap.
    #   2. boost_change_lose=0.7 reinforces miser: every boost use penalized.
    #   3. No kickoff signal. SpeedTowardBall removed in v3.7. At kickoff,
    #      conservation penalty PUNISHES burning 33% initial boost.
    #   4. Self-play faking: flipping near ball without contact. No penalty.
    #   5. wavedash=0.08: speed flip (0.063 one-time) can't compete with
    #      conservation (0.042/step). Rational choice: sit with boost.
    #
    # FIX: 3 weight changes + 2 new components.
    #
    #   boost_conservation_weight: 0.06 → 0.04
    #     Reduce miser 33%. Still 2x initial (0.02). sqrt(0.5)×0.04=0.028/step
    #     competitive with VBG instead of dominating 4x.
    #
    #   boost_change_lose_weight: 0.7 → 0.5
    #     Reduce boost USE penalty. 10% boost use costs 0.010 (was 0.014).
    #     Bot can afford to boost for speed, aerials, kickoff.
    #
    #   wavedash_weight: 0.08 → 0.12
    #     Speed flip: 0.78×0.12=0.094 (was 0.063). Now exceeds ~3.4 steps
    #     of conservation reward. Speed flips become genuinely attractive.
    #
    #   KickoffReward: NEW at 0.10
    #     Speed toward ball when ball at center with zero velocity. Fires
    #     ~2-3 sec per kickoff. Addresses "doesn't go for ball at kickoff."
    #
    #   FakingPenaltyReward: NEW at 0.15
    #     Flip START within 500uu of ball without contact = -0.15 penalty.
    #     Each fake ≈ 2× small pad cost. Event-based (once per fake).
    #
    # 14→16 active components. +KickoffReward, +FakingPenaltyReward.
    # ====================================================================

    # ====================================================================
    # v3.17 — "THE CONTROLLER"
    # ====================================================================
    # CONTEXT: v3.16 ran 200M steps (11.2B→11.4B).
    #   FIRST WIN VS ELEMENT: 6-4! No more faking, committed kickoffs, good
    #   boost management. directed_pct 43.7% (up from 29.2%). Confidence 48.1%.
    #   User: "very happy, doesn't fake anymore, much more controlled, kickoffs
    #   much better, good boost management."
    #
    # TRAINING TRENDS (v3.16 window):
    #   avg_boost: 38.12→32.46 (normalized from miser peak — FIX WORKING)
    #   boosting_frac: 9.58%→10.69% (using boost more — FIX WORKING)
    #   supersonic: 4.59%→5.32% (+16%)
    #   goal_diff: -9→+27 (strong positive swing)
    #   touches: 0.0596→0.0658 (+10%)
    #   aerial_touch: stable 15.2%
    #
    # MATCH (6-4 WIN vs Element):
    #   POSITIVES: First win. 2 demos. ATK 47.4%. 90 unique actions.
    #   KEY FINDING: 59 consecutive ground AWAY touches (ball_z=93, speed=-1988,
    #   5.4 seconds of sustained contact) + 37 aerial AWAY touches carrying ball
    #   from z=97→1945→191. These occur in LAST 35 seconds of match (winning 6-4).
    #   CRITICAL INSIGHT: Bot HAS ground ball control mechanics (59 consecutive
    #   touches!) AND aerial carry skill (37-touch aerial sequence to ceiling!).
    #   The MECHANICS exist. The DIRECTION needs fixing.
    #
    # USER REQUEST: "Ball control, ground and air dribbles, control in the air.
    #   Still has to learn wavedashes, speed flips. Gets outdribbled sometimes."
    #
    # ROOT CAUSES (3):
    #   1. No explicit ground dribble reward. DirectedTouchReward decays with
    #      sustain_decay=0.85 — teaches "hit and carry briefly." But ground
    #      dribbling IS sustained carry. The decay penalizes the exact skill.
    #   2. AerialDistanceReward (0.45) is direction-agnostic. It's the strongest
    #      continuous active reward and incentivizes aerial carries in ANY
    #      direction. Match evidence: 37-touch AWAY aerial carry rewarded.
    #      VBG zero-sum (0.20) provides opposing pressure but 0.45 > 0.20.
    #   3. wavedash=0.12: supersonic 3.7% in match. Speed flip (0.094) barely
    #      exceeds conservation (0.028/step × 3.4 = 0.095). Needs more headroom.
    #
    # FIX: 2 weight changes + 1 new reward.
    #
    #   GroundDribbleReward: NEW at 0.08
    #     Dense per-tick reward for ball-on-car-roof while moving toward goal.
    #     Ball within 200uu horizontal, 80-300uu above car, car on ground.
    #     Reward: car speed toward opponent goal / CAR_MAX_SPEED.
    #     Unlike DirectedTouchReward: no touch requirement, no decay. Sustained
    #     carry IS the desired behavior. At 0.08: 3-second carry ≈ 2.16 total
    #     reward ≈ 6 small pad pickups. Meaningful but not farmable.
    #
    #   aerial_distance_weight: 0.45 → 0.30
    #     Direction-agnostic aerial signal reduced 33%. Combined aerial:
    #     0.10 (play) + 0.30 (distance) = 0.40 (was 0.55). Still strong but
    #     no longer overwhelms directional signals (DirectedTouch 0.40, VBG 0.20).
    #     Aerials are learned (15.3%, stable). SHIFTS emphasis from "maintain
    #     any aerial contact" to "maintain DIRECTED aerial contact."
    #
    #   wavedash_weight: 0.12 → 0.15
    #     Speed flip: 0.78×0.15=0.117 (exceeds 4+ steps of conservation).
    #     User explicitly wants wavedashes/speed flips. 3.7% supersonic in match.
    #
    # 16→17 active components. +GroundDribbleReward.
    # ====================================================================

    # ====================================================================
    # v3.18 "THE WALL RIDER" — 11.96B steps
    # ====================================================================
    #
    # v3.17 RESULT: 2-7 LOSS vs Element (regression from v3.16's 6-4 WIN)
    #
    # TRAINING TRENDS (v3.17 window: 11.4B→11.96B, 560M steps):
    #   FLAT metrics. near_wall_frac 15-17% (no increase). aerial_touch 15%
    #   (stable). goal_diff oscillating wildly (-23→+18→-7). No convergence.
    #   GroundDribbleReward IS producing ground carries (26 consecutive TOWARD
    #   touches in match event timeline, z=93-141, speed=502-2020). Positive.
    #
    # MATCH (2-7 LOSS vs Element):
    #   VYREX_0: 2g/2sv/4sh. VYREX_1: 0g/0sv/1sh. Element: 7 goals, 2 demos.
    #   directed_pct: 34.8% (DOWN from 43.7% in v3.16 — CRITICAL REGRESSION)
    #   avg_boost: 19.8 (training: 31-33). zero_boost: 37.8%. supersonic: 3.1%.
    #   ATK: 48.4%. ROT: 41.6%. DEF: 5.4% (DOWN from 11.1% — less defending).
    #
    #   EVENT TIMELINE DIAGNOSIS:
    #   72 consecutive AWAY touches (ball_z=302, speed=-788) — BOTH bots double-
    #   committing on a backward carry. Neither can redirect. Mechanical failure
    #   under pressure. Then 26 TOWARD ground dribble touches (speed 2020→502)
    #   — GroundDribbleReward IS working but too rare.
    #
    # USER REPORT: "Doesn't go for wall plays, always waits for ball to come
    #   down. Kickoffs worse again. Still gets outdribbled. Aerial improvements.
    #   Does focus on speed more."
    #
    # ROOT CAUSES (2 critical, 1 secondary):
    #   1. WALL PLAY AVOIDANCE — identical temporal credit assignment gap as
    #      boost starvation (v3.11-3.14). WallPlayReward fires ONLY on wall
    #      touches (event-based), but the DECISION to drive toward wall ball
    #      occurs 5-20 steps earlier with ZERO positive feedback. Bot rationally
    #      waits for ball to come to ground because:
    #        a) No dense approach signal for wall ball situations
    #        b) WallPlayReward payoff (0.06) is 3-8x weaker than DirectedTouch
    #           ground hits (0.20+), making wall play an irrational choice
    #        c) Waiting → easier ground touch → DirectedTouch(0.40×0.5=0.20)
    #      Same problem, same solution: BoostApproachReward solved boost
    #      starvation by providing dense pre-event gradient. Apply to walls.
    #
    #   2. KICKOFF REGRESSION — KickoffReward at 0.10 insufficient. May be
    #      drowned out by competing signals (GroundDribbleReward could fire
    #      as ball lifts after first contact, pulling attention from follow-up).
    #      Strengthen to 0.15 to ensure committed kickoff remains dominant.
    #
    #   3. BALL CONTROL (ongoing) — directed_pct 34.8% is dire. 65% of touches
    #      go wrong direction. VBG zero-sum already provides negative signal for
    #      bad touches. GroundDribbleReward IS working (26-touch carry). Needs
    #      more training time, not weight changes. No action this version.
    #
    # FIX: 2 weight changes + 1 new reward.
    #
    #   wall_play_weight: 0.06 → 0.15
    #     Directed wall touch: 0.15×1.0 = 0.15 (was 0.06). Now competitive
    #     with ground DirectedTouch (0.40×0.5=0.20). Payoff ratio: 0.75×
    #     ground (was 0.30×). Bot no longer punished for choosing wall play.
    #
    #   WallBallChallengeReward: NEW at 0.06
    #     Dense per-tick speed-toward-ball when ball is on wall surface. Same
    #     architectural pattern as BoostApproachReward. Fires when ball is:
    #     elevated on side/back wall OR in corner zone (any height). Output:
    #     max(0, speed_toward_ball / CAR_MAX_SPEED). Bridges the temporal
    #     credit gap between "decide to go to wall" and "touch ball on wall."
    #     At 0.06: 2 seconds of max-speed approach ≈ 1.2 total reward.
    #
    #   kickoff_weight: 0.10 → 0.15
    #     User reports regression. Stronger signal for committed kickoffs.
    #     Perfect kickoff: 0.15/tick × 27 ticks ≈ 4.05 total (was 2.7).
    #
    # 17→18 active components. +WallBallChallengeReward.
    # ====================================================================

    # ====================================================================
    # v3.19 "THE LIBERATOR" — 12.50B steps
    # ====================================================================
    #
    # v3.18 RESULT: 5-8 LOSS vs Element (improved from v3.17's 2-7 LOSS)
    #
    # TRAINING TRENDS (v3.18 window: 11.96B→12.50B, 540M steps):
    #   aerial_touch_rate: 15.3%→16.0% (stable, slight up)
    #   avg_boost: 33→29 (DOWN — more boost consumption from wall/aerial)
    #   zero_boost: 31%→32% (slight UP — expected given more aerial usage)
    #   supersonic: 5.3%→6.0% (UP — good, speed focus working)
    #   goal_diff: -7→-1 (IMPROVED — less negative)
    #   touch_ball_speed: 2117→2188 (+3.3% — harder hits)
    #   near_wall: 16.3%→15.4% (FLAT — wall play changes need more time)
    #
    # MATCH (5-8 LOSS vs Element):
    #   Score improved: 5-8 (was 2-7). VYREX_0: 3g/1sv/4sh. VYREX_1: 2g/2sv/5sh.
    #   Confidence: 0.5642 (UP from 0.4942). Entropy: 1.48 (DOWN from 1.729).
    #   ATK: 52.1% (UP from 48.4% — more aggressive). NOT FAKING ANYMORE.
    #   directed_pct: 31.5% (STILL DECLINING: 43.7%→34.8%→31.5%)
    #   avg_boost: 17.0 (DOWN from 19.8). zero_boost: 45.6% (UP from 37.8%).
    #   supersonic: 2.3% (DOWN from 3.1%). Boost starvation WORSENING in match.
    #
    #   EVENT TIMELINE BREAKTHROUGHS:
    #   60 consecutive TOWARD ground dribble touches (tick 60852-61512,
    #   ball_z=171, speed=1250, ~5.5 seconds). PERFECT ground carry.
    #   8-touch ascending aerial carry (z=334→1205, all TOWARD goal).
    #   18-touch descending aerial carry (z=1940→359 — near-ceiling to ground,
    #   all 18 touches TOWARD goal). ADVANCED aerial control demonstrated.
    #
    # USER INSIGHT: "Boost rewards penalize aerial play. Either aerial reward
    #   should overcome boost cost, or deactivate boost rewards when in air!"
    #
    # ROOT CAUSE — BOOST PENALTY KILLS AERIALS (exact math):
    #   During aerial APPROACH (boosting toward ball, no touch yet):
    #     AerialPlayReward: 0.0/step (requires touch — doesn't fire)
    #     AerialDistanceReward: 0.0/step (requires touch — doesn't fire)
    #     BoostChangeReward: -0.002/step (penalty for consuming boost)
    #     BoostConservation: -0.001/step (lower boost state = lower reward)
    #     NET: -0.003/step of PURE PENALTY before any aerial reward fires
    #
    #   Going aerial burns ~30 boost over 1 second (15 steps):
    #     BoostChangeReward: sqrt(0.20)-sqrt(0.50) = -0.26 total
    #       × 0.5 (lose_weight) × 0.20 (weight) = -0.026 total penalty
    #     BoostConservation: ~-0.010/step reduction for rest of episode
    #
    #   vs GROUND PLAY: zero boost cost, DirectedTouch(0.40×0.5=0.20) +
    #   GroundDribble(0.08). Bot RATIONALLY waits for ball to descend.
    #   Same temporal credit gap pattern as boost starvation and wall play.
    #
    # FIX: 2 behavioral corrections + 1 weight increase.
    #
    #   AirborneAwareBoostChangeReward: REPLACES BoostChangeReward
    #     Boost loss penalty = 0 when car is airborne. Gain always rewarded.
    #     Ground behavior unchanged. Using boost for aerials is not punished.
    #     No abuse vector: AerialPlayReward requires actual ball contact.
    #     Going airborne without purpose earns zero positive reward.
    #
    #   BoostConservationReward: MODIFIED with airborne exemption
    #     Returns 0 when car is airborne. Having low boost during aerial is
    #     EXPECTED, not a failure state. Combined with AirborneAware above,
    #     eliminates ALL boost-related disincentives for aerial play.
    #
    #   aerial_play_weight: 0.10 → 0.15
    #     With boost penalty removed, aerial touches become clearly net-positive.
    #     At mid-height: 0.15×0.75=0.1125/step (was 0.075). No longer competing
    #     against ~0.003/step boost penalty. Combined aerial after fix:
    #     0.15 (play) + 0.30 (distance) = 0.45 total (was 0.40 with penalty).
    #
    # 18 active components (same count — 2 modifications, 1 weight change).
    # ====================================================================

    # ====================================================================
    # v3.20 "THE COORDINATOR" — 14.0B steps
    # ====================================================================
    #
    # v3.19 RESULT: REGRESSION TRAJECTORY
    #   13.0B: 5-4 WIN (initial success)
    #   13.5B: 5-6 LOSS (starting to regress)
    #   14.0B: 4-9 HEAVY LOSS (worst performance ever)
    #
    # CRITICAL METRICS (14B vs Element):
    #   avg_boost: 13.5 (WORST EVER — crashed from 26.8 at 13.5B)
    #   zero_boost: 44.5% (back to pre-v3.15 levels)
    #   double_commit: 8.5% (v3.18: 5.7% — DOUBLED by v3.19)
    #   directed_pct: 32.1% (REGRESSED from 42.1% at 13B)
    #   aerial_frames: 16.1% (RISING — bot goes for aerials BUT MISSES)
    #   DEF state: 6.7% (very low — poor save/defensive awareness)
    #   saves: 3 scoreboard total (consistently low)
    #
    # ROOT CAUSE #1: BLANKET AIRBORNE BOOST EXEMPTION (exploited)
    #   v3.19 disabled ALL boost disincentives when `not car.on_ground`.
    #   Bot learned: "jump = free boost zone." ANY jump/hop disables cost.
    #   avg_boost crashed 26.8→13.5 as bot dumps boost during ANY airborne
    #   moment. Additionally, boost cost was an IMPLICIT COORDINATOR —
    #   one bot deferred on aerials due to boost cost. Without cost, BOTH
    #   bots go for every ball → double_commit 5.7%→8.5-13.8%.
    #
    # ROOT CAUSE #2: KICKOFF DOUBLE-COMMIT (no role differentiation)
    #   KickoffReward rewarded ALL agents for speed-toward-ball at kickoff.
    #   User: "double-committing on kickoff and then getting scored on
    #   because both of them are in front."
    #
    # ROOT CAUSE #3: AERIAL MISS (no approach gradient)
    #   Bot goes for aerials (16.1% aerial_frames, highest ever) but MISSES.
    #   AerialPlayReward only fires on TOUCH — binary signal. A 10uu miss
    #   and 1000uu miss both get 0.0. No gradient for approach precision.
    #   v3.6 removed positioning component (hover farming) but left no
    #   approach signal at all.
    #
    # ROOT CAUSE #4: DEFENSIVE PASSIVITY
    #   DEF state 6.7%, saves consistently low. Spacing (0.025) and rotation
    #   (0.03) too weak relative to 0.45 aerial signal. Bot commits forward
    #   with teammate (double-commit) leaving goal exposed.
    #
    # FIX: 5 changes — close exploits, differentiate roles, improve precision.
    #
    #   1. AirborneAwareBoostChangeReward: CONDITIONAL exemption
    #      Only exempt when ball_z > 300 (legitimate aerial play, not random
    #      jumping). Low-altitude hops, kickoff boosts, random jumps still
    #      incur boost loss penalty. Closes "jump = free boost" exploit.
    #
    #   2. BoostConservationReward: CONDITIONAL exemption (matching)
    #      Same ball_z > 300 condition. Conservation pressure maintained
    #      during ground-level airborne time. Only released for true aerials.
    #
    #   3. KickoffReward: ROLE DIFFERENTIATION
    #      Closest bot: speed toward ball (go-er). Second bot: reward for
    #      staying back in own half (support/defense). Directly fixes
    #      kickoff double-commit. Natural role separation.
    #
    #   4. AerialPlayReward: APPROACH VELOCITY gradient (NEW)
    #      When airborne near elevated ball without touching: small reward
    #      for SPEED toward ball (0.15 × speed_frac × height_factor).
    #      Speed-based, not proximity-based — avoids v3.6 hover farming.
    #      Flying AT ball = reward. Hovering near ball = 0. Teaches
    #      trajectory control for aerial approaches.
    #
    #   5. teammate_spacing_weight: 0.025 → 0.04
    #      rotation_reward_weight: 0.03 → 0.04
    #      Double-commit prevention + better defensive positioning.
    #      Combined team signal: 0.08 (was 0.055). Still 5.6x weaker
    #      than aerial (0.45) but competitive with individual dense signals.
    #
    # 18 active components (same count — 4 modifications, 2 weight changes).
    # ====================================================================

    # ====================================================================
    # v3.21 "THE DEFENDER" — 14.7B steps
    # ====================================================================
    #
    # v3.20 RESULT: TWO MATCHES, BOTH 2-8 LOSS vs Element
    #   v3.20 fixed the boost exploit (avg_boost recovered 13.5→20-21.5)
    #   but scoring COLLAPSED (4→2 goals/match) and defense still terrible.
    #
    # CRITICAL METRICS (v3.20, 2 matches):
    #   Match 1: ATK=50.1%, DEF=6.2%, ROT=36.7%, directed=36.0%
    #   Match 2: ATK=55.7%, DEF=6.9%, ROT=31.8%, directed=33.5%
    #   avg_dist_to_own_goal: 5042-5201 (too far from goal!)
    #   avg_dist_to_teammate: 2753-2998 (support bot too far for passes)
    #   saves: 4 and 2 (vs 8 goals conceded each match)
    #   Event timeline: 60+ consecutive AWAY touches (ball toward own goal)
    #
    # USER OBSERVATIONS:
    #   1. "Passes good but other bot too far behind — gets intercepted"
    #   2. "Doesn't save, defense is big problem"
    #   3. "Own-goals when trying to save"
    #   4. "Still not dribbling"
    #   5. "Sometimes could play more aggressive"
    #   6. "Often misses the ball" (need more training time for v3.20 approach gradient)
    #   7. "Too far rotating"
    #
    # ROOT CAUSE #1: SUPPORT BOT AT MIDFIELD DURING ATTACKS
    #   RotationReward ideal_y = (ball_y + own_goal_y) * 0.5 → always clamps
    #   to midfield (y≈0). When attacker is at y=4000, support is at y=0 — 
    #   4000uu away. Passes intercepted by Element before support arrives.
    #   avg_dist_to_teammate: 2753-2998 confirms support is too distant.
    #
    # ROOT CAUSE #2: ZERO DEFENSIVE CLEARING REWARD (save_weight=0.0)
    #   DEF state stuck at 6-7% for THREE versions. Concede penalty (-35)
    #   fires once per goal — too sparse for clearing mechanics. VBG zero-sum
    #   provides implicit defense but not clearing DIRECTION. Match 1: 60+
    #   AWAY touches at ball_z=263 (bot pushing ball toward own goal).
    #   No reward teaches "redirect ball AWAY from own goal."
    #
    # ROOT CAUSE #3: GROUND DRIBBLE SIGNAL TOO WEAK (0.08)
    #   At 0.08: 3-second carry = 2.16 total reward. DirectedTouch with
    #   decay gives 1.33 for same period. Ratio only 1.6:1 — not compelling
    #   enough to prefer sustained carries over hit-and-disengage.
    #
    # ROOT CAUSE #4: OFFENSIVE/DEFENSIVE SIGNAL TOO WEAK (VBG 0.20)
    #   ball_to_goal avg reward: 0.04/step. Bot scores only 2 goals/match
    #   (was 4-5). Zero-sum at 0.25 makes both directions more urgent.
    #
    # FIX: 4 changes — defend, position, dribble, intensify.
    #
    #   1. RotationReward: FORWARD SUPPORT POSITIONING
    #      When ball in attacking half, support bot now positions at
    #      ball_y - 2500uu (trails ball by 2500uu) instead of midfield.
    #      Blue example: ball at y=4000 → ideal_y=1500 (attacking half,
    #      close for pass!). Ball at y=2000 → ideal_y=-500 (just behind
    #      midfield, still supportive). Clamped at -2000 to prevent
    #      over-retreat. Directly fixes "passes but teammate too far."
    #
    #   2. save_weight: 0.0 → 0.10 (RE-ENABLED)
    #      At 0.10: event-based clearing reward (danger × 0.10 per touch)
    #      teaches proper save direction (AWAY from own goal = full reward,
    #      bad direction = 0.2x). Camping incentive negligible: positional
    #      guide = 0.10×danger×proximity×0.02 ≈ 0.002/step (the 0.35 that
    #      caused camping was 35x stronger). Kaiyotech's concern doesn't
    #      apply at this weight — event-based clears dominate.
    #
    #   3. ground_dribble_weight: 0.08 → 0.15
    #      3-second max-speed carry: 5.4 total (vs DirectedTouch 1.33).
    #      Ratio 4.1:1 makes sustained carries genuinely dominant.
    #      User requested dribbling in 3+ analyses.
    #
    #   4. velocity_ball_to_goal_weight: 0.20 → 0.25
    #      Zero-sum amplifies both attack and defense. Ball toward opponent
    #      goal = +0.25/step, opponent advancing = -0.25/step. Addresses
    #      "more aggressive" + "more defense" simultaneously. #2 continuous
    #      signal after DirectedTouch (0.40).
    #
    # 19 active components (SaveReward re-enabled from 0.0 to 0.10).
    # ====================================================================

    # ====================================================================
    # v3.22 "THE PRECISION" — ~15B+ steps
    # ====================================================================
    #
    # v3.21 RESULT: MILESTONE — 8-5 WIN vs Element (FIRST WIN EVER!)
    #   Also: 5-12 loss vs Necto (competitive — scored 5)
    #
    # CRITICAL METRICS (v3.21, 2 matches):
    #   vs Element: 8-5 WIN! VYREX_0: 5g/2a. VYREX_1: 3g/1a.
    #   vs Necto: 5-12 LOSS. VYREX_0: 2g/2a. VYREX_1: 3g/1a.
    #   double_commit: 5.6% (BEST EVER — was 8.9-13.8%)
    #   directed_pct: 41.0% (recovered from 33-36%)
    #   avg_boost: 21.9-24.0 (stable, healthy)
    #   zero_boost: 28.5-31.3% (BEST EVER range)
    #   aerial_frames: 11-14% (stable — going for aerials)
    #   supersonic: 2.8-4.5% (still low — needs speed work)
    #   saves: 0 in both (re-enabled at 0.10 — needs training time)
    #
    # USER OBSERVATIONS:
    #   1. "Boosts into the air but flies under the ball or doesn't hit it"
    #      → aerial trajectory precision needs stronger gradient
    #   2. "I want it to gain speed — rewarded relative to speed gained"
    #      → new SpeedGainReward requested
    #   3. Very happy about beating Element: "FIRST TIME OMG!"
    #   4. Necto match "very strong" — competitive at 5-12
    #
    # ROOT CAUSE #1: AERIAL APPROACH SIGNAL TOO WEAK
    #   aerial_play_approach_weight=0.15 × height_factor(~0.8) ×
    #   aerial_play_weight(0.15) = 0.018/step effective. Among 19 competing
    #   signals, this is noise-level. The approach gradient IS architecturally
    #   correct (speed toward ball = correct trajectory), but needs 2.7x
    #   amplification to produce meaningful trajectory learning pressure.
    #   A car flying slightly under ball gets nearly the same reward as one
    #   flying directly at it — the gradient is too flat to distinguish.
    #
    # ROOT CAUSE #2: NO SPEED BUILDING REWARD
    #   WavedashReward (0.15) rewards the flip TECHNIQUE but not the
    #   RESULT (speed gain). Bot can gain speed via boosting forward,
    #   wavedashes, or just not braking — no signal rewards this.
    #   supersonic: 2.8-4.5% in competitive matches (should be 8-12%).
    #
    # FIX: 1 weight change + 1 new reward.
    #
    #   1. aerial_play_approach_weight: 0.15 → 0.40
    #      Effective signal: 0.40×0.8×0.15 = 0.048/step (2.7x stronger).
    #      Over 15-step aerial approach: 0.72 total (was 0.27). Still
    #      bounded by aerial_play_weight=0.15. Touch component (0.075-0.15)
    #      remains bigger payoff → approach guides but doesn't dominate.
    #      Directly fixes "flies under the ball."
    #
    #   2. SpeedGainReward: NEW at 0.08
    #      Dense per-tick reward for speed gained over a 1-second sliding
    #      window (15 lookback ticks). Reward = max(0, speed_delta / max_speed).
    #      Only rewards gains (braking not penalized — can be tactical).
    #      Teaches: boost forward, speed flip, don't brake unnecessarily.
    #      At 0.08: moderate signal. Complements WavedashReward (technique)
    #      with result-based signal (actual velocity increase).
    #
    # 19→20 active components. +SpeedGainReward.
    # ====================================================================

    # ====================================================================
    # v3.24 "THE AGGRESSOR" — ~16B+ steps
    # ====================================================================
    #
    # v3.23 RESULT: 1-6 LOSS vs Element (WORST REGRESSION — from 8-5 WIN)
    #   Scoring COLLAPSED: 8-5 WIN (v3.21) → 5-10 (v3.22) → 1-6 (v3.23).
    #   Bot has ELITE mechanics (ceiling air dribble, ground carries, 15.7%
    #   aerial_frames ATH) but plays TOO PASSIVELY — only 1 goal scored.
    #   avg_touch_speed: 1408 (LOW — gentle touches easily defended).
    #
    # CRITICAL METRICS (v3.23 vs Element):
    #   Score: 1-6 L. VYREX: 1g total, 3 saves.
    #   avg_boost: 28.3 (RECOVERED from 16.9 — SpeedGain removal worked)
    #   zero_boost: 28.5% (RECOVERED from 47.0%)
    #   DEF: 9.1% (improved but bot TOO defensive)
    #   aerial_frames: 15.7% (HIGHEST EVER — aerials elite)
    #   avg_touch_speed: 1408 (LOW — needs harder hits)
    #   supersonic: 3.5% (still slow)
    #   directed_pct: 35.9% (stable)
    #   double_commit: 7.1% (acceptable)
    #
    # EVENT TIMELINE:
    #   CEILING AIR DRIBBLE: 26 consecutive TOWARD touches z=717→1950 (ELITE!)
    #   MASSIVE AWAY CARRY: 43 consecutive AWAY touches (-539→-2055, own-goal)
    #   HIGH-SPEED AERIAL: 33 TOWARD touches z=728→748 at 1400-1650 speed
    #   Bot HAS the mechanics — it's the PLAY STYLE that's wrong.
    #
    # USER FEEDBACK (explicit requests for v3.24):
    #   1. "ZERO save_reward — encourages bot to stay behind!" × 3rd time
    #   2. "Increase both goal and concede reward — more impactful"
    #   3. "Needs to be more aggressive"
    #   4. "More ball control — currently doing simple dribbles" (positive!)
    #   5. "Faster plays, faster movement — slow touches easily defended"
    #   6. "Drives toward ball without boosting or flipping for speed"
    #   7. "Gets outdribbled but when it learns to dribble it could defend"
    #
    # DIAGNOSIS — why 8-5 WIN became 1-6 LOSS:
    #   save_weight 0.10→0.15 shifted bot toward DEFENSIVE camping.
    #   Combined with SpeedGainReward disruption (added v3.22, removed v3.23),
    #   ~1B steps of suboptimal gradient. Bot learned patience over aggression.
    #   Offense/Defense ratio was 2.4:1 — now shifting to 3.8:1.
    #
    # FIX: 5 weight changes — FUNDAMENTAL AGGRESSION SHIFT.
    #
    #   1. save_weight: 0.15 → 0.0 (REMOVED — third time zeroing)
    #      User explicit: "encourages staying behind." Defense via VBG
    #      zero-sum (0.30) + concede penalty (-40). Aggressive bot defends
    #      by DOMINATION, not camping.
    #
    #   2. goal_weight: 35.0 → 40.0 (14% higher stakes)
    #      User: "increase both goal and concede." Score=+40, Concede=-40
    #      (symmetric at 1.0× multiplier). Both events more impactful.
    #      concede_mult stays 1.0 — history shows >1.0 causes passivity.
    #
    #   3. directed_touch_weight: 0.40 → 0.50 (25% stronger)
    #      "Slow ball touches easily defended." Rewards HARDER directional
    #      hits. 0.50×0.5=0.25/step for decent hit — clear #1 touch signal.
    #      Incentivizes boosting INTO the ball, not gentle approach.
    #
    #   4. ground_dribble_weight: 0.15 → 0.20 (33% stronger)
    #      User: "more ball control." Simple dribbles happening (confirmed!)
    #      — amplify the working signal. 3-second carry: 7.2 total reward
    #      (was 5.4). 4.3:1 ratio over DirectedTouch decay sum.
    #
    #   5. velocity_ball_to_goal_weight: 0.25 → 0.30 (20% stronger)
    #      User: "more aggressive." #1 continuous zero-sum signal. Push ball
    #      forward = +0.30, opponent advances = -0.30. With save=0, VBG
    #      zero-sum becomes PRIMARY defense — penalizes inaction when
    #      opponent has ball without rewarding camping.
    #
    # Signal balance: Offense 1.15 (was 0.95) vs Defense 0.30 (was 0.40)
    # Offense/Defense ratio: 3.8:1 (was 2.4:1). Clear aggression mandate.
    # 19→18 active components. SaveReward disabled.
    # ====================================================================

    # ====================================================================
    # v3.25 "THE CHALLENGER" — ~16.5B+ steps
    # ====================================================================
    #
    # v3.24 RESULT: 4-9 LOSS vs Element
    #   OFFENSE IMPROVED: goals scored TRIPLED (1→4). Aggression shift worked.
    #   DEFENSE COLLAPSED: 9 goals conceded. Only 2 saves (was 3).
    #   Ceiling air dribble: 27 TOWARD touches z=280→1718 (ELITE mechanics).
    #   BUT: 25-tick AWAY carry at -1538 speed before redirect — CORE FAILURE.
    #
    # CRITICAL METRICS (v3.24 vs Element):
    #   Score: 4-9 L. VYREX: 4g/2a/2sv/7sh/1demo (offense UP!).
    #   avg_boost: 23.6 (REGRESSED from 28.3)
    #   zero_boost: 37.6% (REGRESSED from 28.5%)
    #   ROT: 33.1% (DOWN from 39.4% — CRITICAL DROP)
    #   avg_touch_speed: 1310 (DOWN from 1408 — paradoxical)
    #   double_commit: 8.0% (up from 7.1%)
    #   def_third_pct: 37.7% (spending MORE time defending but doing LESS)
    #   aerial_frames: 14.7% (stable)
    #   directed_pct: 36.7% (stable)
    #
    # USER FEEDBACK:
    #   1. "Cannot get the ball away from own goal fast enough" — CLEARING
    #   2. "When element has ball, should try challenging" — ACTIVE DEFENSE
    #   3. "Currently often just waits what element does" — PASSIVITY
    #   4. "The goal is open and element scores" — NO ROTATION BACK
    #   5. "Often does unnecessary moves on defending" — INEFFICIENCY
    #
    # ROOT CAUSE #1: ZERO CLEARING INCENTIVE (save_weight=0)
    #   During 25-tick AWAY carry, DirectedTouch gives 0.0, VBG zero-sum
    #   gives negative (correct but no ACTION guidance), AerialDistanceReward
    #   gives POSITIVE (direction-agnostic — rewards AWAY aerial carries!).
    #   SaveReward was the ONLY component providing clearing direction gradient.
    #   Without it: VBG says "state bad" but not "redirect THIS WAY."
    #   Same temporal credit assignment gap pattern (v3.12, v3.18, v3.20).
    #
    # ROOT CAUSE #2: AerialDistanceReward COMPETING IN DEFENSE
    #   At 0.30 (direction-agnostic), aerial AWAY carries at z=462-570 get
    #   POSITIVE reward that partially OFFSETS VBG's negative signal. Bot
    #   maintains wrong-direction aerial contact because it's rewarded for it.
    #
    # ROOT CAUSE #3: ROTATION COLLAPSED (39.4%→33.1%)
    #   Offensive signals (DT 0.50 + VBG 0.30 + Dribble 0.20 = 1.0 total)
    #   drown out rotation at 0.04 by 25:1. Both bots chase → open net.
    #
    # ROOT CAUSE #4: avg_touch_speed DROP (1408→1310)
    #   GroundDribble (0.20) + high sustain_decay (0.85) makes 10-tick gentle
    #   carries 3.6x more rewarding than single hard hits. Bot rationally
    #   prefers gentle dribbles over hard clears — wrong in defense.
    #
    # FIX: 1 CODE CHANGE + 3 weight adjustments.
    #
    #   1. SaveReward: CODE MODIFIED — positional guide REMOVED entirely
    #      The user's objection was "encourages staying behind" — caused by
    #      the per-tick positional guide (danger×proximity×0.02). This has
    #      been STRIPPED from the code. SaveReward now fires ONLY on actual
    #      touches in the defensive zone. Zero camping incentive possible.
    #      Re-enabled at 0.08: good clear = 0.08×danger (event-based).
    #      Bad direction clear = 0.08×danger×0.2. No touch = 0.0.
    #
    #   2. aerial_distance_weight: 0.30 → 0.20
    #      Direction-agnostic signal reduced 33%. Stops rewarding AWAY aerial
    #      carries. Combined aerial: 0.15+0.20=0.35 (still strong, aerials
    #      mature at 14.7%). DirectedTouch(0.50) now clearly dominates for
    #      directional aerial plays.
    #
    #   3. rotation_reward_weight: 0.04 → 0.06
    #      50% stronger rotation signal. Support bot positioning competitive
    #      with spacing(0.04). Ensures one bot rotates back on counter-attacks.
    #      Addresses "goal is open and element scores."
    #
    # 18→19 active components. SaveReward re-enabled (clearing-only).
    # ====================================================================
    #      33% stronger commitment signal.
    #
    #   2. save_weight: 0.10 → 0.15
    #      Full clear: 0.15 (0.75× offense ratio, was 0.50×). Defense
    #      becomes competitive with offense. Saves improved 0→5 at 0.10
    #      — amplifying a working signal. Camping: 0.003/step max (23x
    #      below Kaiyotech's danger threshold of 0.35).
    #
    #   3. speed_gain_weight: 0.08 → 0.0 (REMOVED)
    #      User request. Removes kickoff competing gradient and boost
    #      drain pressure. WavedashReward (0.15) still covers technique.
    #
    # 20→19 active components. SpeedGainReward disabled.
    # ====================================================================

    reward_components = [
        # === SPARSE EVENT REWARDS (the dominant signal — technique payoff) ===
        (GoalReward(concede_multiplier=rc.concede_multiplier), rc.goal_weight),  # 40.0 (v3.24: 35→40). Score=+40, Concede=-40 (1.0× symmetric)
        (DemoReward(), rc.demo_weight),                              # 0.8

        # === DIRECTED TOUCH (v3.24: 0.40→0.50 — reward HARDER directional hits) ===
        # v3.10: sustain_decay stops reward farming during lingering contact
        (DirectedTouchReward(
            sustain_decay=rc.directed_touch_sustain_decay,            # 0.85 — enable dribble sequences
        ), rc.directed_touch_weight),                                # 0.50 (v3.24: "slow touches easily defended")

        # === DEFENSE ===
        # v3.25: SaveReward CODE MODIFIED — positional guide REMOVED.
        # Now PURELY clearing-only: fires only on actual touches in defensive
        # zone. No per-tick reward for just being near goal. Addresses user's
        # camping concern while providing clearing direction gradient.
        # Good clear: 0.08×danger. Bad direction: 0.08×danger×0.2. No touch: 0.0.
        (SaveReward(
            defense_zone_y=4500.0,
            proximity_bonus_dist=1500.0,
        ), rc.save_weight),                                          # 0.08 (v3.25: 0.0→0.08 — clearing-only, positional guide removed)

        # === GOAL-DIRECTED SHAPING ===
        (GoalViewReward(gamma=0.99), rc.goal_view_weight),           # 0.15
        # v3.14: VelocityBallToGoalReward WRAPPED in DistributeRewardsWrapper (zero-sum).
        # Before: clamped at 0 when ball moves toward own goal (no defensive signal).
        # After: player_reward = own_vbg - opponent_avg_vbg. When opponent pushes ball
        # toward our goal, we get NEGATIVE reward. Creates defensive pressure without
        # Kaiyotech's "save trap" (SaveReward encourages letting ball reach danger zone).
        # Fredrik: "i zerosum almost all my rewards from the start"
        (DistributeRewardsWrapper(
            VelocityBallToGoalReward(),
            selflessness=0.0,   # Individual: each agent gets own reward (no team sharing)
            team_coef=1.0,      # Full own reward
            opp_coef=1.0,       # Full opponent subtraction: own_vbg - avg_opponent_vbg
        ) if rc.zero_sum_vbg else VelocityBallToGoalReward(),
         rc.velocity_ball_to_goal_weight),                           # 0.30 (v3.24: 0.25→0.30, #1 aggression driver + defense via zero-sum)

        # === AERIAL PLAY (touch + approach since v3.20, approach amplified v3.22) ===
        (AerialPlayReward(
            min_ball_height=rc.aerial_play_min_height,  # 150
            approach_weight=rc.aerial_play_approach_weight,  # 0.40 — v3.22: 0.15→0.40, 2.7x stronger trajectory gradient
        ), rc.aerial_play_weight),                                   # 0.15 (v3.22: approach_weight 0.15→0.40)

        # === AERIAL DISTANCE (v3.14: NEW — aerial technique sequences) ===
        # From rlgym-tools. Rewards aerial SEQUENCES not just single touches:
        #   - First touch: rewarded by height (similar to AerialPlayReward)
        #   - Consecutive touches by same agent: rewarded by distance traveled
        #     (car+ball movement since last touch) — teaches air carries/dribbles
        #   - Resets on landing or opponent touch — teaches maintaining aerial control
        # Combined aerial signal: AerialPlay(0.10) + AerialDistance(0.15) = 0.25 total
        # WARNING: Community reported "abuse dribbling" at weight 10 — monitoring via
        # new vyrex/aerial_seq_touches metric.
        (AerialDistanceReward(
            touch_height_weight=1.0,      # First touch scales with height
            car_distance_weight=1.0,      # Track car movement between touches
            ball_distance_weight=1.0,     # Track ball movement between touches
        ), rc.aerial_distance_weight),                               # 0.20 (v3.25: 0.30→0.20, reduce direction-agnostic AWAY incentive)

        # === BOOST COLLECTION (v3.7: 2.5x stronger — #1 mechanical priority) ===
        # SpeedTowardBallReward REMOVED in v3.7: Fully internalized at 3.13B.
        #   avg_dist_to_ball=2908, 1634 touches/match. 0.001/tick = gradient noise.
        # BoostConservationReward RE-ADDED in v3.12: see v3.12 block above.
        # v3.10: weight 0.15→0.20, lose_weight 0.3→0.5. Combined with DirectedTouchReward
        #   decay, this creates a crossover point: after 3 steps of ball contact, boost
        #   collection becomes more rewarding than continued lingering.
        # v3.11: weight 0.20→0.30, lose_weight 0.5→0.7. Small pad at 0 boost = 0.104
        #   reward — EXCEEDS mediocre directed touch. Creates genuine pad-pathing incentive.
        # v3.20: CONDITIONAL airborne exemption — only exempt when ball_z > 300.\n        # v3.19's blanket exemption caused boost dumping (avg_boost 26.8→13.5)\n        # and double commits (both bots go for every ball without boost cost).\n        # Now: low-altitude jumps/kickoff hops still cost boost. Only real\n        # aerial play (ball elevated) gets the exemption.
        (AirborneAwareBoostChangeReward(
            gain_weight=1.0,
            lose_weight=rc.boost_change_lose_weight,
            min_ball_height=200.0,  # v3.26: 300→200 — exempt wall aerials (ball on wall at z=200-300)
        ), rc.boost_change_weight),                                  # 0.20 (v3.26: min_ball_height 300→200)

        # === BOOST APPROACH (v3.12: THE missing pre-pickup gradient) ===
        # Dense per-tick velocity toward nearest available pad when boost < 30%.
        # Converts the 5-20 step delayed pickup signal into immediate continuous
        # feedback. ~0.006/step when active, competitive with VelocityBallToGoal.
        # Novel in domain — no public RL bot has this.
        (BoostApproachReward(
            boost_threshold=rc.boost_approach_threshold,            # 0.30
            small_pad_distance_mult=rc.boost_approach_small_pad_mult,  # 0.8
        ), rc.boost_approach_weight),                                # 0.02 (v3.12: NEW)

        # === BOOST CONSERVATION (v3.12: RE-ADDED, v3.20: CONDITIONAL AIRBORNE EXEMPT) ===
        # sqrt(boost) continuous state pressure. v3.20: only exempt when airborne
        # AND ball is elevated (z > 300). v3.19's blanket exemption taught
        # "jump = free boost zone" → avg_boost crashed 26.8→13.5. With conditional
        # exemption: conservation pressure maintained during ground-level jumps,
        # only removed during legitimate aerial play (ball actually elevated).
        (BoostConservationReward(min_ball_height=200.0), rc.boost_conservation_weight),   # 0.03 (v3.26: min_ball_height 300→200, weight 0.04→0.03)

        # === MOVEMENT TECHNIQUE (v3.11: 60% stronger — 1.4% supersonic is disastrous) ===
        (WavedashReward(scale_by_acceleration=True), rc.wavedash_weight),  # 0.15 (v3.17: 0.12→0.15)

        # === WALL PLAY (v3.18: 0.06→0.15 + NEW WallBallChallengeReward) ===
        # Wall touch payoff was 3-8x weaker than ground touches → bot rationally
        # waited for ball to come down. Now competitive with DirectedTouch ground hits.
        (WallPlayReward(wall_threshold=1200.0), rc.wall_play_weight),  # 0.15 (v3.18: 0.06→0.15)

        # NEW v3.18: Dense wall ball approach gradient. Same architecture as
        # BoostApproachReward — provides pre-touch signal for the DECISION to
        # drive toward ball on wall, bridging temporal credit assignment gap.
        (WallBallChallengeReward(
            wall_threshold=rc.wall_ball_challenge_wall_threshold,     # 1200.0 — same zone as WallPlayReward
            min_ball_height=rc.wall_ball_challenge_min_height,       # 200.0 — ball on wall surface, not ground
        ), rc.wall_ball_challenge_weight),                           # 0.06 (v3.18: NEW)

        # === TEAM 2v2 (v3.25: rotation strengthened for defensive recovery) ===
        # v3.24: ROT dropped 39.4%→33.1%. Offensive signals (1.0 total) drowned out
        # rotation (0.04). Both bots chase → open goal. Rotation at 0.06 ensures one
        # bot rotates back. Addresses "goal is open and element scores."
        (TeammateSpacingReward(), rc.teammate_spacing_weight),       # 0.04
        (RotationReward(), rc.rotation_reward_weight),               # 0.06 (v3.25: 0.04→0.06 — fix rotation deficit)

        # === KICKOFF (v3.20: ROLE-DIFFERENTIATED — fix kickoff double-commit) ===
        # v3.19: BOTH bots rewarded for speed toward ball → both commit →
        # ball goes past both → open net → easy goal against.
        # v3.20: Closest bot goes for ball. Second bot stays back/defends.
        # Natural role separation without hardcoding positions.
        (KickoffReward(), rc.kickoff_weight),                        # 0.20 (v3.23: 0.15→0.20 — fix kickoff regression)

        # === FAKING PENALTY (v3.16: NEW — anti-fake/whiff) ===
        # Self-play artifact: bot approaches ball, flips to fake challenge, retreats.
        # Detection: flip START within proximity of ball without ball contact.
        # Fires once per fake attempt (event-based). At 0.15: each fake ≈ 2× small
        # pad cost. Teaches "if you flip near ball, HIT it."
        (FakingPenaltyReward(
            proximity_threshold=rc.faking_proximity_threshold,       # 500 uu
        ), rc.faking_penalty_weight),                                # 0.15 (v3.16: NEW)

        # === GROUND DRIBBLE (v3.24: 0.15→0.20 — "more ball control") ===
        # User: "more ball control, currently doing simple dribbles" (positive!).
        # Amplify working signal. At 0.20: 3-second carry = 7.2 total reward,
        # clearly dominant over decaying DirectedTouch (1.66) for same period.
        (GroundDribbleReward(
            carry_radius=200.0,       # Max horizontal dist for dribble detect
            min_height_above=80.0,    # Ball above car (on roof ≈ 128uu)
            max_height_above=300.0,   # Catches pop-ups, excludes aerials
        ), rc.ground_dribble_weight),                                # 0.20 (v3.24: 0.15→0.20)

        # === SPEED GAIN (v3.22: NEW, v3.23: DISABLED — competing kickoff gradient) ===
        # Created competing gradient during kickoffs (support bot pulled forward)
        # and contributed to boost starvation (16.9 avg, 47.0% zero).
        (SpeedGainReward(
            lookback_ticks=rc.speed_gain_lookback_ticks,             # 15 steps ≈ 1.0 second
        ), rc.speed_gain_weight),                                    # 0.0 (v3.23: 0.08→0.0 DISABLED)
    ]
    # v3.25: 20 components total (19 active — SpeedGainReward disabled at 0.0)
    # Changes from v3.24:
    #   - SaveReward CODE MODIFIED: positional guide removed, clearing-only
    #   - save_weight: 0.0 → 0.08 (re-enabled — clearing direction for defense)
    #   - aerial_distance_weight: 0.30 → 0.20 (reduce AWAY aerial carry incentive)
    #   - rotation_reward_weight: 0.04 → 0.06 (fix rotation deficit 39.4%→33.1%)
    # Active components: 19. SaveReward re-enabled (clearing-only, no camping).

    return CombinedReward(*reward_components)
