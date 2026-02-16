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

    The weights below are tuned for early-to-mid training. As the bot improves,
    you should adjust these — particularly reducing dense shaping rewards and
    increasing event/goal weight.
    """

    # --- Sparse Event Rewards (high weight, rare signals) ---
    goal_weight: float = 10.0
    concede_weight: float = -10.0
    shot_weight: float = 0.5
    save_weight: float = 1.0
    demo_weight: float = 0.3
    boost_pickup_weight: float = 0.05

    # --- Dense Continuous Rewards (low weight, constant signal) ---
    speed_toward_ball_weight: float = 0.02
    velocity_ball_to_goal_weight: float = 0.15
    face_ball_weight: float = 0.005
    in_air_weight: float = 0.02            # Doubled at 897M — conditional (ball>200uu), incentivize aerials+wall play
    in_air_min_ball_height: float = 200.0    # Lowered from 300 at 897M — enables wall-play and low-aerial rewards
    boost_conservation_weight: float = 0.01

    # --- Grounded Reward (counteracts learned jumping pathology) ---
    # Dense reward for staying on the ground. Needed because the old
    # unconditioned InAirReward taught the bot to jump 85% of the time.
    # History: 0.005 (251M, too weak) → 0.05 (401M, over-corrected to 3.2%)
    #   → 0.015 (433M, equilibrium trap at 4.3%) → 0.003 (456M, recovered to 47%)
    #   → 0.005 (897M, reduce choppy hops from 47% toward 38-42% target)
    # At 0.005 the per-jump penalty (~0.04) is small enough for purposeful
    # aerials to remain profitable, but large enough to discourage random hops.
    grounded_weight: float = 0.005

    # --- 2v2-Specific Team Rewards ---
    # These only matter when team_spirit > 0
    teammate_spacing_weight: float = 0.01   # Penalize ball-chasing / clustering
    rotation_reward_weight: float = 0.01    # Reward proper positioning relative to ball/goal

    # --- Team Spirit Curriculum ---
    # team_spirit interpolates from individual reward to shared team reward
    # 0.0 = fully individual, 1.0 = fully shared among team
    # Ramp: starts low so agents learn individual skills first, then cooperate
    team_spirit_start: float = 0.0
    team_spirit_end: float = 0.8
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
    ts_per_iteration: int = 150_000
    ppo_batch_size: int = 150_000
    exp_buffer_size: int = 450_000            # 3x batch (48GB RAM can easily hold this)
    ppo_minibatch_size: int = 75_000          # 4070 Ti 12GB → 75K safe; try 100K if no OOM

    # --- PPO Hyperparameters ---
    ppo_epochs: int = 2                       # Optimization passes per iteration
    ppo_ent_coef: float = 0.003              # Reduced at 897M — entropy stable at 94% (4.229/4.500) for 376M steps.
                                                # Movement is choppy due to noisy action selection.
                                                # Lower ent_coef → more committed actions → smoother play,
                                                # enables learning multi-step sequences (drifts, wave dashes).
                                                # History: 0.01 (start) → 0.005 (400M) → 0.003 (897M)
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
    timestep_limit: int = 2_000_000_000       # Total budget: 2 billion steps

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
