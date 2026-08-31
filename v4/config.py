"""
VYREX v4 - Central Training Configuration
==========================================
All hyperparameters, reward weights, and architecture settings in one place.
Migrated from v3 (rlgym-ppo) to rlgym-learn. All values preserved exactly.

Design Philosophy:
    - 2v2 focused from the ground up
    - Curriculum-based team_spirit ramp
    - Rich reward shaping with KRC-inspired normalization
    - High throughput via RocketSim parallelism + rlgym-learn Rust backend
"""

import os
from dataclasses import dataclass, field
from typing import List


# ============================================================================
# PATHS
# ============================================================================
@dataclass
class PathConfig:
    """All filesystem paths used during training and deployment."""
    project_root: str = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir: str = os.path.join(project_root, "agent_controllers_checkpoints")
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
    Identical to v3.25 "THE CHALLENGER" — all weights preserved exactly.
    """

    # --- Sparse Event Rewards ---
    goal_weight: float = 40.0
    concede_multiplier: float = 1.0
    zero_sum_vbg: bool = True
    demo_weight: float = 1

    # --- Touch Rewards ---
    directed_touch_weight: float = 0.50
    directed_touch_sustain_decay: float = 0.85
    advanced_touch_weight: float = 0.0           # REMOVED

    # --- Defense ---
    save_weight: float = 0.05                     # v4.1: 0.08→0.05 — reduce clearing incentive that encourages defending
    boost_pickup_weight: float = 0.20

    # --- Potential-Based Goal Shaping ---
    goal_view_weight: float = 0.15

    # --- Dense Continuous Rewards ---
    speed_toward_ball_weight: float = 0.0        # REMOVED
    velocity_ball_to_goal_weight: float = 0.30
    face_ball_weight: float = 0.0                # REMOVED
    boost_conservation_weight: float = 0.03        # v4.1: 0.04→0.03 — less "sit with boost" pressure

    # --- Aerial Play ---
    aerial_play_weight: float = 0.15
    aerial_play_min_height: float = 100.0          # v4.1: 150→100 — activate aerial rewards for wall-height balls
    aerial_play_approach_weight: float = 0.40

    # --- Aerial Distance ---
    aerial_distance_weight: float = 0.20

    # --- Boost ---
    boost_change_weight: float = 0.20
    boost_change_lose_weight: float = 0.5
    boost_approach_weight: float = 0.04
    boost_approach_threshold: float = 0.50
    boost_approach_small_pad_mult: float = 0.8

    # --- Wavedash ---
    wavedash_weight: float = 0.15

    # --- Wall Play ---
    wall_play_weight: float = 0.30                 # v4.1: 0.15→0.20 — bigger payoff for wall touches
    wall_ball_challenge_weight: float = 0.15       # v4.1: 0.06→0.10 — stronger wall ball approach gradient
    wall_ball_challenge_wall_threshold: float = 1200.0
    wall_ball_challenge_min_height: float = 200.0

    # --- Kickoff ---
    kickoff_weight: float = 0.30              # v4.1: 0.25→0.30 — encourage more kickoff engagement

    # --- Faking Penalty ---
    faking_penalty_weight: float = 0.10            # v4.1: 0.15→0.10 — less penalty for aggressive challenges
    faking_proximity_threshold: float = 350.0      # v4.1: 500→350 — only penalize very close whiffs

    # --- Ground Dribble ---
    ground_dribble_weight: float = 0.15

    # --- Speed Gain ---
    speed_gain_weight: float = 0.0               # REMOVED
    speed_gain_lookback_ticks: int = 15

    # --- 2v2-Specific Team Rewards ---
    teammate_spacing_weight: float = 0.03          # v4.1: 0.04→0.03 — reduce continuous safe signal
    rotation_reward_weight: float = 0.04            # v4.1: 0.06→0.04 — reduce defensive pull (code fix does more)

    # --- Team Spirit Curriculum ---
    team_spirit_start: float = 0.0
    team_spirit_end: float = 0.3
    team_spirit_ramp_steps: int = 500_000_000


# ============================================================================
# NETWORK ARCHITECTURE
# ============================================================================
@dataclass
class NetworkConfig:
    """Neural network architecture for policy and critic."""
    policy_layer_sizes: List[int] = field(default_factory=lambda: [2048, 2048, 1024, 1024])
    critic_layer_sizes: List[int] = field(default_factory=lambda: [2048, 2048, 1024, 1024])


# ============================================================================
# PPO HYPERPARAMETERS — mapped to rlgym-learn config models
# ============================================================================
@dataclass
class PPOConfig:
    """
    PPO hyperparameters mapped from v3 (rlgym-ppo) to v4 (rlgym-learn).

    rlgym-learn uses different naming conventions:
      v3 policy_lr      → v4 actor_lr
      v3 critic_lr      → v4 critic_lr
      v3 ppo_batch_size  → v4 batch_size
      v3 ppo_minibatch_size → v4 n_minibatches (batch_size / minibatch_size)
      v3 ppo_ent_coef    → v4 ent_coef
      v3 ppo_epochs      → v4 n_epochs
      v3 exp_buffer_size  → v4 experience_buffer max_size
      v3 ts_per_iteration → v4 timesteps_per_iteration

    ═══════════════════════════════════════════════════════════════════
    HARDWARE-OPTIMIZED FOR:
        GPU:  RTX 4070 Ti — 12GB GDDR6X, Ada Lovelace (AD104)
        CPU:  i7-14700K  — 8 P-cores + 12 E-cores = 20 cores / 28 threads
        RAM:  48 GB DDR5
    ═══════════════════════════════════════════════════════════════════
    """
    # --- Parallelism ---
    n_proc: int = 20

    # --- Batch Sizes ---
    timesteps_per_iteration: int = 200_000
    batch_size: int = 200_000
    n_minibatches: int = 2                    # 200K / 100K = 2 minibatches
    experience_buffer_max_size: int = 600_000

    # --- PPO Hyperparameters ---
    n_epochs: int = 2
    ent_coef: float = 0.0005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    clip_range: float = 0.2
    gamma: float = 0.995
    gae_lambda: float = 0.95

    # --- Normalization ---
    standardize_returns: bool = True
    standardize_obs: bool = False

    # --- Checkpointing ---
    save_every_ts: int = 5_000_000
    permanent_save_every_ts: int = 15_000_000
    timestep_limit: int = 25_000_000_000

    # --- WandB ---
    log_to_wandb: bool = True

    # --- Rendering ---
    render: bool = False
    render_delay: float = 0.05

    # --- TrueSkill ---
    trueskill_version_save_interval: int = 5_000_000
    trueskill_max_versions: int = 20
    trueskill_eval_interval: int = 10_000_000
    trueskill_eval_games_per_version: int = 2
    trueskill_versions_to_eval: int = 3
    trueskill_eval_gamemodes: List[str] = field(default_factory=lambda: ["2v2"])

    @property
    def min_inference_size(self) -> int:
        """Minimum batch size before running inference. ~90% of n_proc."""
        return max(1, int(round(self.n_proc * 0.9)))


# ============================================================================
# DIAGNOSTICS
# ============================================================================
@dataclass
class DiagnosticsConfig:
    """Configuration for diagnostic outputs."""
    enable_diagnostics: bool = True
    diagnostics_interval_steps: int = 5_000_000
    diagnostics_output_dir: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "diagnostics"
    )
    episode_buffer_size: int = 200
    track_reward_components: bool = True


# ============================================================================
# CURRICULUM
# ============================================================================
@dataclass
class CurriculumConfig:
    """
    Training phases with different state mutators.
    Phase 1 (0 - 100M):    Kickoff states only
    Phase 2 (100M - 300M): 30% random states mixed in
    Phase 3 (300M+):       70% random states
    """
    phase_1_end: int = 100_000_000
    phase_2_end: int = 300_000_000
    phase_1_random_ratio: float = 0.0
    phase_2_random_ratio: float = 0.3
    phase_3_random_ratio: float = 0.7


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
            "  VYREX v4 Configuration Summary (rlgym-learn)",
            "=" * 70,
            "  Hardware Target:",
            "    GPU:  RTX 4070 Ti — 12GB VRAM, Ada Lovelace (TF32 enabled)",
            "    CPU:  i7-14700K  — 20 cores / 28 threads",
            "    RAM:  48 GB",
            "",
            f"  Framework:       rlgym-learn + rlgym-learn-algos (PPO)",
            f"  Rating:          TrueSkill (rlgym-learn-trueskill)",
            f"  Mode:            2v2 ({'with' if self.env.spawn_opponents else 'without'} opponents)",
            f"  Network:         Policy {self.network.policy_layer_sizes}",
            f"                   Critic {self.network.critic_layer_sizes}",
            f"  PPO:             batch={self.ppo.batch_size:,}, "
            f"minibatches={self.ppo.n_minibatches}, "
            f"lr={self.ppo.actor_lr}, epochs={self.ppo.n_epochs}",
            f"  Parallel Envs:   {self.ppo.n_proc} (1 per CPU core)",
            f"  Training Budget: {self.ppo.timestep_limit:,} steps",
            f"  Team Spirit:     {self.rewards.team_spirit_start} → "
            f"{self.rewards.team_spirit_end} over {self.rewards.team_spirit_ramp_steps:,} steps",
            f"  WandB:           {'ON' if self.ppo.log_to_wandb else 'OFF'} "
            f"(project={self.paths.wandb_project})",
            f"  TrueSkill:       eval every {self.ppo.trueskill_eval_interval:,} steps, "
            f"modes={self.ppo.trueskill_eval_gamemodes}",
            f"  Diagnostics:     {'ON' if self.diagnostics.enable_diagnostics else 'OFF'}",
            f"  Checkpoints:     every {self.ppo.save_every_ts:,} steps",
            f"  Perm. Archive:   every {self.ppo.permanent_save_every_ts:,} steps",
            "",
            "  Expected SPS:    ~3,000-6,000+ steps/sec (rlgym-learn Rust backend)",
            "",
            "  Active Rewards (v3.26 'THE CHALLENGER'):",
        ]

        # List active rewards
        rc = self.rewards
        active = []
        for name, weight in [
            ("GoalReward", rc.goal_weight),
            ("DemoReward", rc.demo_weight),
            ("DirectedTouchReward", rc.directed_touch_weight),
            ("SaveReward", rc.save_weight),
            ("GoalViewReward", rc.goal_view_weight),
            ("VelocityBallToGoal", rc.velocity_ball_to_goal_weight),
            ("AerialPlayReward", rc.aerial_play_weight),
            ("AerialDistanceReward", rc.aerial_distance_weight),
            ("BoostChangeReward", rc.boost_change_weight),
            ("BoostApproachReward", rc.boost_approach_weight),
            ("BoostConservationReward", rc.boost_conservation_weight),
            ("WavedashReward", rc.wavedash_weight),
            ("WallPlayReward", rc.wall_play_weight),
            ("WallBallChallengeReward", rc.wall_ball_challenge_weight),
            ("TeammateSpacingReward", rc.teammate_spacing_weight),
            ("RotationReward", rc.rotation_reward_weight),
            ("KickoffReward", rc.kickoff_weight),
            ("FakingPenaltyReward", rc.faking_penalty_weight),
            ("GroundDribbleReward", rc.ground_dribble_weight),
        ]:
            if weight > 0:
                active.append(f"    {name}: {weight}")
        lines.extend(active)
        lines.append("=" * 70)
        return "\n".join(lines)


DEFAULT_CONFIG = VyrexConfig()
