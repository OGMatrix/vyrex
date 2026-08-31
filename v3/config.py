"""
VYREX - Central Training Configuration
=======================================
All hyperparameters, reward weights, and architecture settings in one place.
Modify this file to tune your training — never hardcode values elsewhere.

Design Philosophy:
    - 2v2 focused from the ground up
    - Curriculum-based team_spirit ramp
    - Rich reward shaping with KRC-inspired normalization
    - High throughput via RocketSim parallelism
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


# ============================================================================
# PATHS
# ============================================================================
@dataclass
class PathConfig:
    """All filesystem paths used during training and deployment."""
    project_root: str = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir: str = os.path.join(project_root, "data", "checkpoints")
    permanent_checkpoint_dir: str = os.path.join(project_root, "data", "checkpoints_permanent")
    log_dir: str = os.path.join(project_root, "data", "logs")
    model_export_dir: str = os.path.join(project_root, "rlbot_deploy", "src")
    wandb_project: str = "vyrex-rl"
    wandb_entity: str = ""  # Set to your wandb username/org, or leave empty


# ============================================================================
# ENVIRONMENT
# ============================================================================
@dataclass
class EnvConfig:
    """RLGym v2 environment configuration for 2v2 training."""

    # --- Match Setup ---
    team_size: int = 2                       # 2v2 is our target mode
    spawn_opponents: bool = True             # Always train with opponents
    action_repeat: int = 8                   # Physics ticks per decision step (standard)

    # --- Episode Termination ---
    no_touch_timeout_seconds: float = 30.0   # End episode if no one touches ball
    game_timeout_seconds: float = 300.0      # Max episode length (5 min match)

    # --- Observation Normalization ---
    # These denominators normalize raw game values into roughly [-1, 1]
    # Values sourced from rlgym.rocket_league.common_values
    # SIDE_WALL_X=4096, BACK_NET_Y=6000, CEILING_Z=2044
    # CAR_MAX_SPEED=2300, CAR_MAX_ANG_VEL=5.5, BALL_MAX_SPEED=6000

    # Zero-padding for variable player counts (set to max_players_per_team * 2)
    # None = no padding (fixed 2v2). Set to 6 if you want 1v1/2v2/3v3 flexibility
    obs_zero_padding: int = None


# ============================================================================
# REWARDS — The Heart of VYREX
# ============================================================================
@dataclass
class RewardConfig:
    """
    Reward function weights for CombinedReward.

    Design principles (learned from Nexto, Lucy-SKG, Seer, Zealan's guide):
      1. Sparse rewards (goals) need high weight to cut through noise
      2. Dense rewards (velocity, positioning) need low weight to guide without dominating
      3. Negative rewards for conceding create defensive awareness
      4. Team-oriented rewards (passing, rotation) require team_spirit > 0 to matter
      5. Event rewards for discrete game events (demos, saves, shots)
      6. All continuous rewards should output in [-1, 1] range before weighting

    v2 OVERHAUL at 1.1B steps:
      Problem: Bot lost 4-5 to Psyonix Beginner. 0 saves, 0 aerials, double-commits.
      Changes: +SaveReward, +DirectedTouchReward, +AdvancedTouchReward, +GoalViewReward,
               -GroundedReward, reduced ball-chase rewards, 4x team rewards.

    v3 FIX at 1.2B steps:
      Problem: v2 OVERCORRECTED. Bot scored 0 goals vs Psyonix Rookie.
        Dense defense+touch rewards (1.0 total) overwhelmed offense (0.155) by 6.5:1.
        Reward hacking: bot camped near own goal farming SaveReward (dense every tick)
        + farmed AdvancedTouchReward (rewards ANY touch regardless of direction).
        67-80% of touches went toward OWN goal. 0.3% supersonic. 0 shots.
      Evidence: Test match data — VYREX scored 0 goals, 0 shots, 0 demos vs Rookie.
      Changes: -AdvancedTouchReward (direction-agnostic → rewarded own-goals),
               SaveReward rewritten (event-based, not dense), save 0.5→0.15,
               directed_touch 0.3→0.15, velocity_ball_to_goal 0.05→0.12,
               speed_toward_ball 0.005→0.012, goal_view 0.1→0.15, demo 0.3→0.4.
    """

    # --- Sparse Event Rewards (high weight, rare signals) ---
    goal_weight: float = 40.0                    # v3.24: 35→40 — user: "increase both goal and concede reward."
                                                    # Score=+40, Concede=-40 (symmetric at 1.0×). 14% higher stakes
                                                    # for BOTH events. Makes scoring the dominant imperative.
                                                    # History: 15(v3)→25(v3.5)→35(v3.5)→40(v3.24)
    concede_multiplier: float = 1.0              # v3.13: 1.3→1.0 — SYMMETRIC goal reward. Score=+25, Concede=-25.
                                                    # v3.12 match evidence: bot retreats instead of challenging,
                                                    # clears at 8-31 speed (tentative contact), "faking" behavior
                                                    # from self-play. All symptoms of 30% defensive bias.
                                                    # Symmetric reward = bot weighs offense and defense equally.
                                                    # ZealanL: "Decrease concede penalty for more aggression."
                                                    # History: 2.0(v3.4)→1.5(v3.7)→1.3(v3.11)→1.0(v3.13)
    zero_sum_vbg: bool = True                       # v3.14: NEW — zero-sum VelocityBallToGoalReward. When opponent's
                                                    # ball moves toward OUR goal, we get negative reward. Addresses
                                                    # 3 issues: (1) gets outdribbled — now penalized when opponent
                                                    # advances, incentivizes challenging. (2) saving — implicit save
                                                    # pressure without Kaiyotech's "save trap". (3) 63.7% of touches
                                                    # go AWAY from opponent goal at 0 penalty; zero-sum creates
                                                    # negative signal for opponent's ball-toward-goal velocity.
                                                    # Fredrik: "i zerosum almost all my rewards from the start while
                                                    # still improving at insane necto in 1b speeds."
    demo_weight: float = 0.8                     # v3.4: 0.4→0.8 — zero demos across 400M+ steps, need much stronger signal

    # --- Touch Rewards ---
    directed_touch_weight: float = 0.50          # v3.24: 0.40→0.50 — user: "slow ball touches which is easily defensable."
                                                    # 25% stronger touch quality signal. A decent hit gives
                                                    # 0.50×0.5=0.25/step — #1 touch signal by clear margin.
                                                    # Rewards HARDER, DIRECTED hits toward goal. Incentivizes
                                                    # boosting into the ball instead of gentle approach.
                                                    # History: 0.15(v3)→0.25(v3.5)→0.40(v3.5)→0.50(v3.24)
    directed_touch_sustain_decay: float = 0.85   # v3.15: 0.5→0.85 — enable dribble sequences. At 0.5, a 5-step
                                                    # controlled carry toward goal gave only 1.94x first-touch reward
                                                    # (capped at 2x). This taught "hit and disengage" — an IMPLICIT
                                                    # anti-dribble penalty (Zealan's "upside-down penalty" pattern).
                                                    # At 0.85: 5-step carry=3.38x, 10-step=4.46x, cap=6.67x.
                                                    # Safe because decay ONLY affects directed touches (ball moving
                                                    # toward opponent goal). Lingering that pushes ball in wrong
                                                    # direction gets base_reward=0.0 regardless of decay value.
                                                    # v3.10 original: 0.5 (anti-lingering). History: N/A→0.5(v3.10)→0.85(v3.15)
    advanced_touch_weight: float = 0.0           # v3: 0.2→0.0 — REMOVED: direction-agnostic, rewards own-goals!
    # AdvancedTouchReward gives `touch_reward * touches + acceleration_reward * ball_speed_change`
    # for ANY touch regardless of direction. With 67-80% of touches going the WRONG WAY,
    # this was actively rewarding the bot for hitting the ball toward its own goal.

    # --- Defense (event-based in v3 — was dense camping trap in v2) ---
    save_weight: float = 0.08                     # v3.25: 0.0→0.08 — CLEARING-ONLY (positional guide REMOVED in code).
                                                    # v3.24 data: 9 goals conceded, 25-tick AWAY carry with ZERO
                                                    # clearing incentive. SaveReward code MODIFIED: per-tick
                                                    # positional guide stripped — only fires on actual TOUCHES
                                                    # in defensive zone. Camping impossible (0.0/tick without touch).
                                                    # Good clear: 0.08×danger. Bad direction: 0.08×danger×0.2.
                                                    # Addresses "cannot get ball away from own goal" without
                                                    # "encouraging staying behind" (user's original objection).
                                                    # History: 0.5(v2)→0.15(v3)→0.25(v3.4)→0.35(v3.9)→0.0(v3.11)→0.10(v3.21)→0.15(v3.23)→0.0(v3.24)→0.08(v3.25)
    boost_pickup_weight: float = 0.20

    # --- Potential-Based Goal Shaping ---
    goal_view_weight: float = 0.15               # v3: 0.1→0.15 — increase goal-directedness

    # --- Dense Continuous Rewards ---
    # v3.5 PHILOSOPHY: Reduce dense reward floor so techniques can emerge.
    # At v3.4: dense rewards were 3.2x goal reward per game. Bot had zero
    # incentive to develop techniques — comfortable mediocrity from driving around.
    # Reducing comfort rewards, increasing technique payoff signals.
    speed_toward_ball_weight: float = 0.0        # v3.7: 0.005→0.0 — REMOVED. Fully internalized at 3.13B (avg_dist_to_ball=2908,
                                                    # 1634 touches/match). Contributes 0.001/tick = pure gradient noise.
    velocity_ball_to_goal_weight: float = 0.30   # v3.24: 0.25→0.30 — user: "more aggressive." Strongest continuous
                                                    # zero-sum signal. Ball toward opponent goal = +0.30×speed,
                                                    # opponent advancing = -0.30×speed. With save=0, VBG zero-sum
                                                    # is now the PRIMARY defense signal — penalizes inaction when
                                                    # opponent advances. Also the #1 aggression driver: push ball
                                                    # forward HARD. #2 continuous signal after DirectedTouch (0.50).
                                                    # History: 0.05(v2)→0.12(v3)→0.20(v3.5)→0.25(v3.21)→0.30(v3.24)
    face_ball_weight: float = 0.0                # v3.5: 0.002→0.0 — REMOVED. #2 comfort signal (0.575 avg). Internalized at 2.38B.
    boost_conservation_weight: float = 0.04      # v3.16: 0.06→0.04 — v3.15's 3x increase created "boost miser".
                                                    # avg_boost DOUBLED (21→39) but boosting_frac DOWN 12.5%,
                                                    # boost_collected DOWN 3.3%, speed FLAT. Bot hoards boost
                                                    # instead of converting to speed/aerials. sqrt(0.5)×0.04
                                                    # =0.028/step — still competitive with VBG (~0.01/step)
                                                    # without dominating 4x. 2x original (0.02), proven useful.
                                                    # History: 0.03(v3)→0.0(v3.7)→0.02(v3.12)→0.06(v3.15)→0.04(v3.16)

    # --- Aerial Play Reward (v3.5: replaces InAirReward + GroundedReward) ---
    # v3.5 CRITICAL: Bot waits on ground when ball is in the air!
    # Old InAirReward (0.01) was too weak AND GroundedReward (0.008) actively
    # PUNISHED aerial attempts. Net effect: stay on ground = more reward.
    # AerialPlayReward unifies both with strong aerial touch incentive:
    #   - Small reward for being airborne near elevated ball (positioning)
    #   - BIG reward for actually touching ball while airborne (technique)
    # GroundedReward REMOVED: was needed at 250M (85% airborne), harmful at
    # 2.38B (stable 34-38% airborne). It actively prevented aerial development.
    aerial_play_weight: float = 0.15             # v3.19: 0.10→0.15 — with boost penalty removed for airborne cars,
                                                    # aerial touches become clearly net-positive. Strengthens the
                                                    # "go aerial and touch the ball" signal. At mid-height:
                                                    # 0.15×0.75=0.1125/step (was 0.075). No longer competing against
                                                    # ~0.003/step boost penalty. History: 0.15(v3.5)→0.10(v3.6)→0.15(v3.19)
    aerial_play_min_height: float = 150.0        # v3.5: lowered from 300 — bot must attempt aerials at moderate heights
    aerial_play_approach_weight: float = 0.40    # v3.22: 0.15→0.40 — 2.7x stronger aerial trajectory gradient.
                                                    # Bot boosts into air but flies UNDER the ball (v3.21 match
                                                    # evidence: aerial_frames 11-14% but aerial misses persist).
                                                    # Effective signal was 0.15×0.8×0.15=0.018/step — too weak to
                                                    # learn precision trajectories among 19 competing signals.
                                                    # At 0.40: effective 0.40×0.8×0.15=0.048/step (2.7x). Over a
                                                    # 15-step aerial approach: 0.72 total (was 0.27). Still bounded
                                                    # by aerial_play_weight=0.15 in CombinedReward. Touch component
                                                    # (0.075-0.15/step) remains the bigger payoff — approach guides
                                                    # trajectory but doesn't become farmable.
                                                    # History: 0.15(v3.20)→0.40(v3.22)

    # --- Aerial Distance Reward (NEW v3.14 — from rlgym-tools) ---
    # AerialPlayReward gives flat 0.5-1.0 per aerial touch — it teaches
    # "touch ball while airborne" but NOT aerial TECHNIQUE SEQUENCES.
    # AerialDistanceReward adds multi-touch aerial carry tracking:
    #   - First touch: rewarded by HEIGHT (similar to AerialPlayReward)
    #   - Consecutive touches by same agent: rewarded by DISTANCE TRAVELED
    #     (car + ball movement since last touch) — teaches sustained aerial
    #     sequences, air dribbles, wall-to-air carries.
    #   - Resets on landing (below ramp height) or opponent touch.
    # Combined aerial signal: AerialPlay(0.10) + AerialDistance(0.25) = 0.35
    # WARNING: At weight 10 community reported "abuse dribbling" (ground
    # dribble farming). At 0.15 (67x lower) it's competitive with existing
    # aerial reward but won't dominate. Will monitor for abuse via new metric.
    aerial_distance_weight: float = 0.20         # v3.25: 0.30→0.20 — direction-agnostic signal rewards AWAY
                                                    # aerial carries. v3.24 match: 25-tick AWAY aerial carry at
                                                    # z=462-570 — AerialDistanceReward partially OFFSETS the VBG
                                                    # negative signal by rewarding sustained aerial contact in any
                                                    # direction. Aerials are mature (14.7%, stable). Combined aerial:
                                                    # AerialPlay(0.15) + AerialDistance(0.20) = 0.35 (still strong).
                                                    # DirectedTouch(0.50) now clearly dominates over direction-agnostic
                                                    # aerial signal, incentivizing DIRECTED aerial carries.
                                                    # History: 0.15(v3.14)→0.45(user)→0.30(v3.17)→0.20(v3.25)

    # --- Boost Pickup Reward (NEW v3.2) ---
    # Bot is at 7% average boost — catastrophically low. BoostConservationReward
    # rewards HAVING boost but doesn't teach PICKING UP pads. BoostChangeReward
    # rewards the DELTA: gaining boost (driving over pads) = positive reward.
    # gain_only=True: don't penalize using boost, only reward collecting it.
    boost_change_weight: float = 0.20            # v3.13: 0.30→0.20 — boost collection SOLVED after 645M steps of v3.12.
                                                    # Zero-boost 74.7%→62.5%, avg boost 11.2→19.0, boost collected
                                                    # 1.118→1.167. Three-part boost system (approach+change+conserve)
                                                    # deeply internalized. Reducing event-based pickup reward shifts
                                                    # relative emphasis toward ball play signals: ball-play/boost
                                                    # ratio 2.2:1→3.1:1. Approach(0.02) + Conservation(0.02) still
                                                    # provide continuous boost guidance. Small pad from 0 now gives
                                                    # 0.20×0.346=0.069 reward — still meaningful but no longer
                                                    # dominates over directed touches.
                                                    # History: 0.04(v3.2)→0.10(v3.7)→0.15(v3.8)→0.20(v3.10)→0.30(v3.11)→0.20(v3.13)
    boost_change_lose_weight: float = 0.5         # v3.16: 0.7→0.5 — reduce penalty for USING boost. At 0.7 with
                                                    # conservation=0.06, total cost of boost usage was too high: bot
                                                    # preferred sitting with boost over boosting for speed/aerials.
                                                    # At 0.5: using 10 boost costs 0.10×0.5×0.20=0.010 — still
                                                    # noticeable but not prohibitive. Let bot spend boost to create plays.
                                                    # History: 0.3(v3.2)→0.5(v3.10)→0.7(v3.11)→0.5(v3.16)

    # --- Boost Approach Reward (NEW v3.12) ---
    # THE critical fix for boost starvation. Dense per-tick velocity-toward-
    # nearest-available-pad signal when boost < threshold. Converts the 5-20
    # step delayed pickup signal into immediate continuous feedback.
    # No public Rocket League RL codebase has implemented this approach.
    boost_approach_weight: float = 0.04          # v3.15: 0.02→0.04 — 2x stronger approach gradient. Generates
                                                    # ~0.012/step when active, competitive with VBG (0.01/step).
                                                    # Match data: zero_boost 44-52% in competitive play. Stronger
                                                    # pull toward pads during rotation/downtime. History: 0.02(v3.12)→0.04(v3.15)
    boost_approach_threshold: float = 0.50       # v3.15: 0.30→0.50 — seek pads at 50% instead of 30%. At 30%
                                                    # boost, bot can barely do one aerial. At 50%, has options.
                                                    # ~67% more active time for approach gradient. Air dribbles
                                                    # need 40-60+ boost — by 30% it's already too late.
                                                    # History: 0.30(v3.12)→0.50(v3.15)
    boost_approach_small_pad_mult: float = 0.8   # v3.12: Small pads appear 25% closer. ZealanL: "Bots have a tendency
                                                    # to ignore small pads, so I recommend making the small pad pickup
                                                    # reward much stronger than 12% of the big boost pickup reward."

    # --- Wavedash/Flip Technique Reward (NEW v3.2) ---
    # Bot flips for speed but can't control the flips. WavedashReward rewards
    # controlled flips: flip + land on ground = acceleration = reward. Teaches
    # the bot that flips should be intentional and result in speed gains.
    wavedash_weight: float = 0.15                # v3.17: 0.12→0.15 — user: "still has to learn wavedashes, speed
                                                    # flips." 3.7% supersonic in match (training: 5.3%). Perfect speed
                                                    # flip: 0.78×0.15=0.117 — exceeds 4+ steps of conservation reward
                                                    # (0.028/step × 4 = 0.112). With miser behavior now fixed,
                                                    # technique flips become the dominant speed option. Combined with
                                                    # lower conservation, the incentive structure now clearly prefers:
                                                    # speed_flip(0.117) > hold_boost(4_steps × 0.028=0.112).
                                                    # History: 0.02(v3.2)→0.05(v3.7)→0.08(v3.11)→0.12(v3.16)→0.15(v3.17)

    # --- Wall Play Reward (NEW v3.4) ---
    # Bot avoids walls/sides entirely — always plays through center field.
    # Wall plays are crucial for developing aerials (ball rolls up wall → redirect)
    # and creating cross-field opportunities. Event-based (only on touches near walls).
    wall_play_weight: float = 0.15               # v3.18: 0.06→0.15 — bot "doesn't go for wall plays, always
                                                    # waits for ball to come down." At 0.06, a directed wall touch
                                                    # gives 0.06×1.0=0.06 reward while a ground DirectedTouch gives
                                                    # 0.40×0.5=0.20. Wall play is 3-8x UNDERVALUED. At 0.15, directed
                                                    # wall touch=0.15 — same order as ground touches. Plus, intercepting
                                                    # ball on wall prevents opponent possession (VBG zero-sum captures
                                                    # this). History: 0.06(v3.4)→0.15(v3.18)

    # --- Wall Ball Challenge Reward (NEW v3.18) ---
    # Dense per-tick approach signal for when ball is on a wall surface.
    # Same architectural pattern as BoostApproachReward. WallPlayReward fires
    # ONLY on touch (event-based), but the decision to drive toward wall ball
    # occurs 5-20 steps earlier with no gradient signal. This provides the
    # missing PRE-TOUCH gradient: "heading toward ball on wall = good."
    # The same temporal credit assignment gap that crippled boost collection
    # (solved by BoostApproachReward in v3.12) is now crippling wall play.
    # Activates when ball is elevated on side/back wall OR in corner area.
    # Output: speed toward ball / CAR_MAX_SPEED [0, 1].
    wall_ball_challenge_weight: float = 0.06     # v3.18: NEW — wall ball approach gradient
    wall_ball_challenge_wall_threshold: float = 1200.0  # Same zone as WallPlayReward
    wall_ball_challenge_min_height: float = 200.0       # Ball on wall surface (not ground-level)

    # --- Kickoff Reward (NEW v3.16) ---
    # Bot loses kickoffs consistently — doesn't boost or flip toward ball.
    # Self-play artifact: faking kickoffs works vs identical policy but
    # catastrophic vs real opponents. Also, boost_conservation penalty was
    # penalizing burn at kickoff (33% initial boost → conservation likes keeping it).
    # Rewards speed toward ball when ball is at center with zero velocity.
    # Fires ~2-3 seconds per kickoff, sparse but concentrated signal.
    kickoff_weight: float = 0.20                 # v3.23: 0.15→0.20 — kickoff regression in v3.22. SpeedGainReward
                                                    # (0.08) created competing gradient: support bot's stay-back
                                                    # signal (0.15×0.6=0.09) was barely above speed_gain (0.08).
                                                    # At 0.20: go-er at 0.20/tick, support at 0.12/tick — clear
                                                    # dominance over any speed-related noise. Perfect kickoff:
                                                    # 0.20×27 ticks ≈ 5.4 total (was 4.05). 33% stronger.
                                                    # History: 0.10(v3.16)→0.15(v3.18)→0.20(v3.23)

    # --- Faking Penalty (NEW v3.16) ---
    # Self-play artifact: bot approaches ball, flips to fake challenge, retreats.
    # Works vs identical policy (opponent also fakes) but fails vs real opponents.
    # Detects: flip START within proximity_threshold of ball WITHOUT ball contact.
    # Fires once per fake attempt (event-based, not continuous).
    # At 0.15 weight: each fake costs 0.15 ≈ 2× small pad reward (0.069).
    # Significant but not devastating — teaches "if you flip near ball, HIT it."
    faking_penalty_weight: float = 0.15          # v3.16: NEW — penalize faking/whiffing near ball
    faking_proximity_threshold: float = 500.0    # Distance from ball center for fake detection.
                                                    # 500uu ≈ 2 car lengths from ball surface.

    # --- Ground Dribble Reward (NEW v3.17) ---
    # Bot has demonstrated ground ball control — 59 consecutive touches in a
    # single carry sequence (5.4 seconds of sustained contact). The mechanical
    # skill EXISTS but is directionless. DirectedTouchReward rewards touches
    # toward goal but DECAYS with sustain_decay=0.85 — it teaches "hit and
    # carry briefly," the OPPOSITE of ground dribbling. This reward provides
    # a continuous, non-decaying signal for the quintessential RL skill:
    # ball on car roof, car driving toward goal.
    # Detection: ball within 200uu horizontal, 80-300uu above car, car on ground.
    # Physics: ball on octane roof = ball center at car_z + 128uu.
    # Reward: car speed toward opponent goal / CAR_MAX_SPEED [0, 1].
    # At 0.08: max 0.08/tick. 3-second carry ≈ 2.16 total reward ≈ 6 small pads.
    # Meaningful but not farmable. Complements DirectedTouchReward (touch-based
    # decaying) with carry-based non-decaying signal.
    ground_dribble_weight: float = 0.20          # v3.24: 0.15→0.20 — user: "more ball control." Bot is doing
                                                    # simple dribbles (confirmed!) but needs to dribble MORE and
                                                    # FASTER. At 0.20: 3-second max-speed carry yields 7.2 total
                                                    # reward (vs DirectedTouch decay sum of 1.66 for same period).
                                                    # Ratio 4.3:1 makes sustained carries clearly dominant.
                                                    # Combined with DirectedTouch 0.50, rewards aggressive carries
                                                    # toward goal over cautious approach.
                                                    # History: 0.08(v3.17)→0.15(v3.21)→0.20(v3.24)

    # --- Speed Gain Reward (NEW v3.22) ---
    # User request: "rewarded relatively to the speed it gained in the last
    # x seconds / ticks so it learns to gain speed." Complements WavedashReward
    # which rewards the flip TECHNIQUE — this rewards the RESULT (actual speed
    # increase). Teaches boosting in forward direction, not braking unnecessarily,
    # and recovery from low speed. Uses sliding window: compares current speed
    # with speed N ticks ago, rewards positive delta.
    # supersonic still at 2.8-4.5% in matches — needs more speed emphasis.
    # At action_repeat=8, 120Hz: lookback_ticks=15 ≈ 1.0 second window.
    # Output: max(0, (current - old) / CAR_MAX_SPEED) → [0, 1].
    # At 0.08: full 0→supersonic acceleration in window = 0.08 one-time.
    # Moderate signal — teaches "gaining speed good" without dominating ball play.
    speed_gain_weight: float = 0.0               # v3.23: 0.08→0.0 — REMOVED. Created competing gradient during
                                                    # kickoffs (support bot pulled forward by speed_gain vs stay-back
                                                    # signal). Also contributed to boost starvation (16.9 avg,
                                                    # 47.0% zero — rewarded spending boost for speed bursts).
                                                    # User: "not needed anymore." History: 0.08(v3.22)→0.0(v3.23)
    speed_gain_lookback_ticks: int = 15          # v3.22: 15 steps × 8/120 = 1.0 second window

    # --- 2v2-Specific Team Rewards ---
    # v3.5: spacing STILL the #1 dense signal at 0.696 avg in match.
    # Bot has stable spacing (3091). Halving again to reduce comfort floor.
    # Rotation reduced from 40% of game time — bot is too passive.
    teammate_spacing_weight: float = 0.04        # v3.20: 0.025→0.04 — double commits DOUBLED in v3.19 (5.7%→8.5-13.8%).
                                                    # v3.19's airborne boost exemption removed implicit coordination:
                                                    # boost cost made one bot defer, now both go for every ball.
                                                    # Spacing at 0.025 was 18x weaker than aerial signal (0.45).
                                                    # At 0.04: avg 0.028/step — competitive with conservation, provides
                                                    # real pressure against converging on ball. Combined with kickoff
                                                    # role differentiation, should reduce double-commit back below 8%.
                                                    # History: 0.012(v3.5)→0.025(v3.8)→0.04(v3.20)
    rotation_reward_weight: float = 0.06         # v3.25: 0.04→0.06 — ROT dropped 39.4%→33.1% in v3.24. Stronger
                                                    # offensive signals (DT 0.50, VBG 0.30, dribble 0.20 = 1.0 total)
                                                    # drowned out 0.04 rotation signal by 25:1. Both bots chase
                                                    # ball → goal open → Element scores. At 0.06: support bot
                                                    # positioning reward competitive with spacing (0.04). Ensures
                                                    # one bot rotates back on opponent counter-attacks.
                                                    # History: 0.03(v3.20)→0.04(v3.20)→0.06(v3.25)

    # --- Team Spirit Curriculum ---
    # team_spirit interpolates from individual reward to shared team reward
    # 0.0 = fully individual, 1.0 = fully shared among team
    # Ramp: starts low so agents learn individual skills first, then cooperate
    team_spirit_start: float = 0.0
    team_spirit_end: float = 0.3               # v3.7: 0.8→0.3 — at 0.8, 40% of gradient was teammate noise.
                                                    # Individual mechanics (aerials, flips, boost collection) aren't
                                                    # solid yet. At 0.3: 85% self-reward, 15% team-reward = 5.7x
                                                    # cleaner individual gradient. Spacing reward still prevents
                                                    # ball-chasing. Revisit after techniques emerge.
    team_spirit_ramp_steps: int = 500_000_000  # Steps over which to linearly ramp


# ============================================================================
# NETWORK ARCHITECTURE
# ============================================================================
@dataclass
class NetworkConfig:
    """
    Neural network architecture for policy and critic.

    Based on empirical results from the community:
    - [2048, 2048, 1024, 1024] is the gold standard for competitive bots
    - Separate policy and critic networks (no shared backbone)
    - The critic can be slightly larger if you have VRAM to spare
    """
    policy_layer_sizes: List[int] = field(default_factory=lambda: [2048, 2048, 1024, 1024])
    critic_layer_sizes: List[int] = field(default_factory=lambda: [2048, 2048, 1024, 1024])


# ============================================================================
# PPO HYPERPARAMETERS
# ============================================================================
@dataclass
class PPOConfig:
    """
    Proximal Policy Optimization hyperparameters.
    
    ═══════════════════════════════════════════════════════════════════
    HARDWARE-OPTIMIZED FOR:
        GPU:  RTX 4070 Ti — 12GB GDDR6X, Ada Lovelace (AD104)
        CPU:  i7-14700K  — 8 P-cores + 12 E-cores = 20 cores / 28 threads
        RAM:  48 GB DDR5
    ═══════════════════════════════════════════════════════════════════
    
    n_proc = 20:
        One RocketSim instance per CPU core. The i7-14700K has 20 physical 
        cores. We use all 20 to maximize simulation throughput.
        The remaining 8 hyperthreads handle: main process, GPU inference 
        dispatch, OS overhead. If CPU isn't pegged at 100%, try 22-24.
        
    ppo_batch_size = 150,000:
        Larger batches → better gradient estimates → more stable learning.
        The 4070 Ti's 12GB VRAM comfortably handles this. The community
        notes "much higher than 300K doesn't seem to help most people",
        so 150K is a solid middle ground for 2v2 (4 agents per env).
        
    ppo_minibatch_size = 75,000:
        This is what actually goes through the GPU per forward/backward pass.
        RTX 3060 Ti (8GB) handles 50K fine (per Zealan's guide).
        RTX 4070 Ti (12GB, 50% more VRAM + Ada efficiency) → 75K is safe.
        If you get CUDA OOM errors, drop to 50,000.
        If no issues, try pushing to 100,000.
        
    exp_buffer_size = 450,000:
        3x batch size = standard ratio. With 48GB RAM this is negligible.
        Larger buffer gives PPO more diverse experience to sample from.
        
    ppo_epochs = 2:
        More epochs = better learning per iteration but lower SPS.
        2 is the community sweet spot. 3 is okay if SPS is still high.
        1 is fine for very early training where speed matters most.
        
    Learning rates (1e-4):
        Standard starting point. Decay to 5e-5 after 200M+ steps
        if reward curve starts oscillating.
    """
    # --- Parallelism (CPU-bound) ---
    # i7-14700K: 20 physical cores → 20 RocketSim instances
    # Each process is single-threaded, pinned to a core.
    # Monitor CPU usage; if < 90%, increase by 2 until saturated.
    n_proc: int = 20

    # --- Batch Sizes (GPU-bound) ---
    # ts_per_iteration = ppo_batch_size (use all collected data per iteration)
    ts_per_iteration: int = 200_000           # v3.7: 150K→200K — Zealan: "increase once scoring". Larger batches
                                                # give cleaner gradient estimates, critical with 12 reward components.
    ppo_batch_size: int = 200_000             # v3.7: match ts_per_iteration
    exp_buffer_size: int = 600_000            # 3x batch (48GB RAM can easily hold this)
    ppo_minibatch_size: int = 100_000         # v3.7: 75K→100K — 4070 Ti 12GB can handle this. Drop to 75K if OOM.

    # --- PPO Hyperparameters ---
    ppo_epochs: int = 2                       # Optimization passes per iteration
    ppo_ent_coef: float = 0.0005            # v3.11: 0.00048→0.00038 — entropy wall #4 at 3.0752 (68.3%) for 725M+ steps.
                                                # Descent rate collapsed: -0.0176/bin (early) → -0.0018/bin (late).
                                                # 21% reduction to break through. Same pattern as walls #1-3.
                                                # History: 0.01→0.005(400M)→0.003(897M)→0.001(3.13B)→0.0008(4.03B)→0.0006(5.0B)→0.00048(5.72B)→0.00038(6.5B)→0.0006 (v3.15, 9.18B)
    policy_lr: float = 1e-4                   # Policy learning rate (decay to 5e-5 later)
    critic_lr: float = 1e-4                   # Critic learning rate
    ppo_clip_range: float = 0.2              # PPO clipping (standard)
    gamma: float = 0.995                      # Discount factor (high = long-horizon planning)
    gae_lambda: float = 0.95                  # GAE lambda for advantage estimation

    # --- Normalization (DO NOT CHANGE — these are empirically validated) ---
    standardize_returns: bool = True
    standardize_obs: bool = False

    # --- Checkpointing ---
    save_every_ts: int = 2_000_000            # Save every 2M steps (~13 min at ~2500 SPS)
    permanent_save_every_ts: int = 7_000_000  # Permanent archive every 7M steps (never deleted)
    timestep_limit: int = 20_000_000_000      # Total budget: 10 billion steps (raised from 5B at 4.58B)

    # --- WandB ---
    log_to_wandb: bool = True

    # --- Rendering (for development visualization) ---
    render: bool = False
    render_delay: float = 0.05               # Seconds between rendered frames

    @property
    def min_inference_size(self) -> int:
        """Minimum batch size before running inference. ~90% of n_proc."""
        return max(1, int(round(self.n_proc * 0.9)))


# ============================================================================
# DIAGNOSTICS — Output for Iterative Improvement
# ============================================================================
@dataclass
class DiagnosticsConfig:
    """
    Configuration for diagnostic outputs that you feed back to Claude
    for iterative optimization advice.
    """
    enable_diagnostics: bool = True
    diagnostics_interval_steps: int = 5_000_000   # Generate report every 5M steps
    diagnostics_output_dir: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "diagnostics"
    )
    # Number of recent episodes to analyze for behavioral metrics
    episode_buffer_size: int = 200
    # Track individual reward component averages
    track_reward_components: bool = True


# ============================================================================
# CURRICULUM — Training Phases
# ============================================================================
@dataclass
class CurriculumConfig:
    """
    Defines training phases with different state mutators.

    Phase 1 (0 - 100M steps):    Kickoff states only, learn fundamentals
    Phase 2 (100M - 300M steps): Random states mixed in (30%), learn recovery
    Phase 3 (300M+ steps):       Mostly random states (70%), master all situations
    """
    phase_1_end: int = 100_000_000
    phase_2_end: int = 300_000_000
    phase_1_random_ratio: float = 0.0    # 0% random states
    phase_2_random_ratio: float = 0.3    # 30% random states
    phase_3_random_ratio: float = 0.7    # 70% random states


# ============================================================================
# MASTER CONFIG
# ============================================================================
@dataclass
class VyrexConfig:
    """Master configuration object aggregating all sub-configs."""
    paths: PathConfig = field(default_factory=PathConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    def summary(self) -> str:
        """Return a human-readable summary of the full configuration."""
        lines = [
            "=" * 70,
            "  VYREX Configuration Summary",
            "=" * 70,
            "  Hardware Target:",
            "    GPU:  RTX 4070 Ti — 12GB VRAM, Ada Lovelace (TF32 enabled)",
            "    CPU:  i7-14700K  — 20 cores / 28 threads",
            "    RAM:  48 GB",
            "",
            f"  Mode:            2v2 ({'with' if self.env.spawn_opponents else 'without'} opponents)",
            f"  Network:         Policy {self.network.policy_layer_sizes}",
            f"                   Critic {self.network.critic_layer_sizes}",
            f"  PPO:             batch={self.ppo.ppo_batch_size:,}, "
            f"minibatch={self.ppo.ppo_minibatch_size:,}, "
            f"lr={self.ppo.policy_lr}, epochs={self.ppo.ppo_epochs}",
            f"  Parallel Envs:   {self.ppo.n_proc} (1 per CPU core)",
            f"  Training Budget: {self.ppo.timestep_limit:,} steps",
            f"  Team Spirit:     {self.rewards.team_spirit_start} → "
            f"{self.rewards.team_spirit_end} over {self.rewards.team_spirit_ramp_steps:,} steps",
            f"  WandB:           {'ON' if self.ppo.log_to_wandb else 'OFF'} "
            f"(project={self.paths.wandb_project})",
            f"  Diagnostics:     {'ON' if self.diagnostics.enable_diagnostics else 'OFF'}",
            f"  Checkpoints:     every {self.ppo.save_every_ts:,} steps → {self.paths.checkpoint_dir}",
            f"  Perm. Archive:   every {self.ppo.permanent_save_every_ts:,} steps → {self.paths.permanent_checkpoint_dir}",
            "",
            "  Expected SPS:    ~2,000-3,500 steps/sec (with 20 procs + 4070 Ti)",
            "  Est. per 100M:   ~8-14 hours",
            "=" * 70,
        ]
        return "\n".join(lines)


# Singleton default config
DEFAULT_CONFIG = VyrexConfig()

if __name__ == "__main__":
    print(DEFAULT_CONFIG.summary())
