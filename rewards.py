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


class InAirReward(RewardFunction[AgentID, GameState, float]):
    """
    Conditional aerial reward — only rewards being airborne when the ball
    is also elevated (Z > min_ball_height). This incentivizes actual aerial
    plays rather than mindless hopping.

    The old unconditional version (reward for any airborne frame) taught the
    bot to jump 85% of the time, destroying ground play and burning boost.

    Output: 0 or 1 (only 1 when both airborne AND ball is high).
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
        ball_high = state.ball.position[2] > self.min_ball_height
        rewards = {}
        for agent in agents:
            if ball_high and not state.cars[agent].on_ground:
                rewards[agent] = 1.0
            else:
                rewards[agent] = 0.0
        return rewards


class GroundedReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards the agent for being on the ground. Counteracts the deeply-learned
    pathological jumping behavior caused by the old unconditioned InAirReward.

    At 250M+ steps the policy has learned that jumping = free reward, and the
    bot is airborne 85% of the time, unable to drive properly. This reward
    provides a constant dense signal that being on the ground is baseline-good,
    teaching the bot to stay grounded unless there's a reason to go airborne.

    Should be used temporarily (until ~400-500M steps) then phased out once
    ground play is re-established.

    Output: 0 or 1. (1 = on ground, 0 = airborne)
    """

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: float(state.cars[agent].on_ground) for agent in agents}


class BoostConservationReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards having boost available. Uses sqrt scaling so low boost
    is penalized more than high boost (going from 0→50 is more
    valuable than 50→100 in Rocket League).
    Output: [0, 1]

    Note: rlgym v2 boost_amount is in [0, 100] (not [0, 1]).
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
            boost_amount = state.cars[agent].boost_amount / 100.0  # [0,100] -> [0,1]
            rewards[agent] = np.sqrt(max(0.0, boost_amount))  # sqrt scaling
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
                # Gentle decay past ideal — being too far is slightly worse than ideal
                overshoot = (avg_dist - self.IDEAL_SPACING) / (self.MAX_SPACING - self.IDEAL_SPACING)
                rewards[agent] = max(0.0, 1.0 - 0.3 * overshoot)

        return rewards


class RotationReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards proper 2v2 rotation: one player near ball, one player behind.

    The "behind" player should be between the ball and their own goal.
    This teaches the fundamental 2v2 rotation concept.
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

        # Group by team
        teams: Dict[bool, List[AgentID]] = {False: [], True: []}
        for agent in agents:
            teams[state.cars[agent].is_orange].append(agent)

        for agent in agents:
            car = state.cars[agent]
            team = teams[car.is_orange]

            if len(team) < 2:
                rewards[agent] = 0.0
                continue

            ball_pos = state.ball.position
            own_goal_y = -common_values.BACK_NET_Y if not car.is_orange else common_values.BACK_NET_Y

            # Determine which teammate is closest to ball
            my_pos = car.physics.position
            my_ball_dist = np.linalg.norm(my_pos[:2] - ball_pos[:2])

            teammates_closer = 0
            for other in team:
                if other == agent:
                    continue
                other_pos = state.cars[other].physics.position
                other_ball_dist = np.linalg.norm(other_pos[:2] - ball_pos[:2])
                if other_ball_dist < my_ball_dist:
                    teammates_closer += 1

            if teammates_closer == 0:
                # I'm closest to ball — I should be attacking. Small reward for being close.
                rewards[agent] = max(0.0, 1.0 - (my_ball_dist / 6000.0))
            else:
                # I'm behind — reward for being between ball and own goal
                # Check if we're on the correct side of the ball (toward own goal)
                ball_to_goal_dir = own_goal_y - ball_pos[1]
                my_to_ball_dir = my_pos[1] - ball_pos[1]

                # Same sign means we're on the right side (between ball and goal)
                if (ball_to_goal_dir > 0 and my_to_ball_dir > 0) or \
                   (ball_to_goal_dir < 0 and my_to_ball_dir < 0):
                    rewards[agent] = 0.8
                else:
                    rewards[agent] = 0.2  # On wrong side, small reward

        return rewards


# ============================================================================
# SPARSE EVENT REWARDS
# ============================================================================

class GoalReward(RewardFunction[AgentID, GameState, float]):
    """
    Large positive reward for scoring, large negative for conceding.
    Output: -1, 0, or 1.

    This is the fundamental game objective. The high weight in CombinedReward
    makes this the dominant signal over long horizons.

    Uses GameState.goal_scored (bool) and GameState.scoring_team (0=blue, 1=orange).
    """

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
                    rewards[agent] = -1.0 if car.is_orange else 1.0
                else:
                    rewards[agent] = 1.0 if car.is_orange else -1.0
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


# ============================================================================
# COMBINED REWARD BUILDER
# ============================================================================

def build_vyrex_reward(config=None):
    """
    Build the complete VYREX reward function from config weights.

    Returns a CombinedReward with all reward components properly weighted.
    Import this in train.py and pass it to the RLGym environment.
    """
    from rlgym.rocket_league.reward_functions import CombinedReward

    if config is None:
        from config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG

    rc = config.rewards

    # Build the reward tuple list: (RewardFunction, weight)
    reward_components = [
        # Sparse
        (GoalReward(), rc.goal_weight),
        (TouchBallReward(), rc.shot_weight),
        (DemoReward(), rc.demo_weight),
        # Dense
        (SpeedTowardBallReward(), rc.speed_toward_ball_weight),
        (VelocityBallToGoalReward(), rc.velocity_ball_to_goal_weight),
        (FaceBallReward(), rc.face_ball_weight),
        (InAirReward(min_ball_height=rc.in_air_min_ball_height), rc.in_air_weight),
        (BoostConservationReward(), rc.boost_conservation_weight),
        # Grounded — counteracts learned jumping pathology
        (GroundedReward(), rc.grounded_weight),
        # Team (2v2)
        (TeammateSpacingReward(), rc.teammate_spacing_weight),
        (RotationReward(), rc.rotation_reward_weight),
    ]

    return CombinedReward(*reward_components)
