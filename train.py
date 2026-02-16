"""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║                        V Y R E X                                         ║
║               Devastating 2v2 Rocket League AI                           ║
║                                                                          ║
║   Framework:   RLGym v2 + RocketSim (headless physics simulation)        ║
║   Algorithm:   Proximal Policy Optimization (PPO)                        ║
║   Monitoring:  Weights & Biases (wandb)                                  ║
║   Deployment:  RLBot v5                                                  ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝

Training Script — Run this to start training VYREX.

Usage:
    conda activate vyrex
    python train.py                    # Train with default config
    python train.py --n_proc 16        # Override parallel environments
    python train.py --render           # Enable visualization
    python train.py --resume           # Resume from latest checkpoint

Architecture Decisions (justified by research):
    - RLGym v2 + RocketSim: Headless simulation runs 100-1000x faster than real-time.
      No Rocket League installation needed. Cross-platform (Linux/Mac/Windows).
    - PPO (rlgym-ppo): Industry standard for game AI (used by Nexto, Necto, Seer,
      Lucy-SKG, GT Sophy). Stable, sample-efficient, well-tested.
    - LookupTableAction: Discrete action space with 90 actions covering all meaningful
      input combinations. Proven better than continuous for RL in this domain.
    - DefaultObs with normalization: All values in [-1, 1] range for stable training.
    - 2v2 from the start: Team_spirit curriculum lets agents learn individual skills
      first, then gradually learn to cooperate.
"""

import os
import sys
import argparse
import shutil
import time
from functools import partial
from typing import Optional

import numpy as np

from config import VyrexConfig, DEFAULT_CONFIG
from rewards import build_vyrex_reward
from metrics_logger import VyrexMetricsLogger


# ============================================================================
# ENVIRONMENT BUILDER
# ============================================================================

def build_vyrex_env(config: VyrexConfig = None, current_step: int = 0):
    """
    Build a 2v2 RLGym v2 environment for VYREX training.

    This function is called once per parallel worker process.
    It must import everything it needs inside the function body
    because each worker is a separate process.

    Args:
        config: VyrexConfig instance (pickled via functools.partial).
        current_step: Approximate cumulative timestep count at env creation.
                      Used to select curriculum phase for state mutator ratios.

    Returns:
        RLGymV2GymWrapper wrapping the configured RLGym environment.
    """
    # All imports inside function — required by rlgym-ppo's multiprocessing
    import random
    import numpy as np
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import (
        GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition
    )
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import CombinedReward
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import (
        MutatorSequence, FixedTeamSizeMutator, KickoffMutator
    )
    from rlgym.rocket_league.api import GameState as GS_Type
    from rlgym.api import StateMutator
    from rlgym.rocket_league import common_values
    from rlgym_ppo.util import RLGymV2GymWrapper

    # Use provided config or default
    if config is None:
        from config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG

    ec = config.env
    cc = config.curriculum

    # --- Team Setup ---
    blue_size = ec.team_size
    orange_size = ec.team_size if ec.spawn_opponents else 0

    # --- Action Space ---
    action_parser = RepeatAction(LookupTableAction(), repeats=ec.action_repeat)

    # --- Episode Termination ---
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=ec.no_touch_timeout_seconds),
        TimeoutCondition(timeout_seconds=ec.game_timeout_seconds),
    )

    # --- Reward Function ---
    reward_fn = build_vyrex_reward(config)

    # --- Observation Builder ---
    obs_builder = DefaultObs(
        zero_padding=ec.obs_zero_padding,
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

    # =====================================================================
    # CURRICULUM STATE MUTATORS
    # =====================================================================

    class RandomStateMutator(StateMutator):
        """
        Randomize ball and car positions/velocities within the field.

        Cars are placed on the ground at random field positions with random
        yaw, random boost (20-100), and small random velocity.
        Ball is placed at random position with random velocity.
        """

        def apply(self, state, shared_info):
            # Random ball position (within field, at or above ground)
            bx = np.random.uniform(-3500, 3500)
            by = np.random.uniform(-4500, 4500)
            bz = np.random.uniform(common_values.BALL_RESTING_HEIGHT,
                                   common_values.CEILING_Z * 0.6)
            state.ball.position = np.array([bx, by, bz], dtype=np.float32)

            # Random ball velocity (moderate, not max)
            bvx = np.random.uniform(-2000, 2000)
            bvy = np.random.uniform(-2000, 2000)
            bvz = np.random.uniform(-500, 1000)
            state.ball.linear_velocity = np.array([bvx, bvy, bvz], dtype=np.float32)
            state.ball.angular_velocity = np.random.uniform(-3, 3, size=3).astype(np.float32)

            # Random car positions
            for car in state.cars.values():
                cx = np.random.uniform(-3800, 3800)
                cy = np.random.uniform(-4800, 4800)
                cz = 17.0  # ground level
                car.physics.position = np.array([cx, cy, cz], dtype=np.float32)

                yaw = np.random.uniform(-np.pi, np.pi)
                car.physics.euler_angles = np.array([0, yaw, 0], dtype=np.float32)

                # Small random velocity
                cvx = np.random.uniform(-500, 500)
                cvy = np.random.uniform(-500, 500)
                car.physics.linear_velocity = np.array([cvx, cvy, 0], dtype=np.float32)
                car.physics.angular_velocity = np.zeros(3, dtype=np.float32)

                car.boost_amount = np.random.uniform(20, 100)

    class WeightedSampleMutator(StateMutator):
        """
        Probabilistically chooses between multiple state mutators.

        Example: WeightedSampleMutator([(KickoffMutator(), 0.7), (RandomMutator(), 0.3)])
        """

        def __init__(self, mutators_with_weights):
            self.mutators = [m for m, _ in mutators_with_weights]
            self.weights = [w for _, w in mutators_with_weights]

        def apply(self, state, shared_info):
            chosen = random.choices(self.mutators, weights=self.weights, k=1)[0]
            chosen.apply(state, shared_info)

    # --- Select curriculum phase based on current_step ---
    if current_step < cc.phase_1_end:
        random_ratio = cc.phase_1_random_ratio
        phase_name = "Phase 1 (kickoffs only)"
    elif current_step < cc.phase_2_end:
        random_ratio = cc.phase_2_random_ratio
        phase_name = "Phase 2 (mixed)"
    else:
        random_ratio = cc.phase_3_random_ratio
        phase_name = "Phase 3 (mostly random)"

    # Build state mutator
    if random_ratio <= 0.0:
        position_mutator = KickoffMutator()
    else:
        position_mutator = WeightedSampleMutator([
            (KickoffMutator(), 1.0 - random_ratio),
            (RandomStateMutator(), random_ratio),
        ])

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_size, orange_size=orange_size),
        position_mutator,
    )

    # --- Build Environment ---
    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
    )

    return RLGymV2GymWrapper(rlgym_env)


# ============================================================================
# CHECKPOINT RESUME LOGIC
# ============================================================================

def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint directory across all checkpoint locations.

    Searches:
      1. checkpoint_dir itself (flat layout: checkpoint_dir/{step}/PPO_POLICY.pt)
      2. Any sibling directories matching checkpoint_dir-* (created by
         Learner's add_unix_timestamp=True in previous runs)

    Returns the path to the highest-numbered step directory, or None.
    """
    candidates = []  # list of (step_number, full_path)

    # Collect all directories to search
    search_roots = []
    if os.path.exists(checkpoint_dir):
        search_roots.append(checkpoint_dir)

    # Also search sibling dirs matching checkpoints-* pattern
    parent = os.path.dirname(checkpoint_dir)
    base = os.path.basename(checkpoint_dir)
    if os.path.exists(parent):
        for entry in os.listdir(parent):
            if entry.startswith(base + "-") and os.path.isdir(os.path.join(parent, entry)):
                search_roots.append(os.path.join(parent, entry))

    for root in search_roots:
        for entry in os.listdir(root):
            full_path = os.path.join(root, entry)
            if not os.path.isdir(full_path):
                continue

            # Check if this is a step directory (contains PPO_POLICY.pt)
            if os.path.isfile(os.path.join(full_path, "PPO_POLICY.pt")):
                try:
                    step = int(entry)
                    candidates.append((step, full_path))
                except ValueError:
                    continue
            else:
                # Could be a nested run dir — check its subdirectories
                for sub_entry in os.listdir(full_path):
                    sub_path = os.path.join(full_path, sub_entry)
                    if os.path.isdir(sub_path) and \
                       os.path.isfile(os.path.join(sub_path, "PPO_POLICY.pt")):
                        try:
                            step = int(sub_entry)
                            candidates.append((step, sub_path))
                        except ValueError:
                            continue

    if not candidates:
        return None

    best = max(candidates, key=lambda t: t[0])
    return best[1]


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train(config: VyrexConfig, resume: bool = False):
    """
    Main training entry point for VYREX.

    Sets up the PPO learner, WandB logging, and starts the training loop.
    """
    from rlgym_ppo import Learner
    import torch

    # =========================================================================
    # RTX 4070 Ti OPTIMIZATION: Enable TF32 for Ada Lovelace
    # =========================================================================
    # TF32 (TensorFloat-32) uses the Tensor Cores on Ada Lovelace GPUs to
    # accelerate float32 matrix multiplications with minimal precision loss.
    # This is a FREE ~2-3x speedup on matmul-heavy workloads like PPO.
    # No accuracy impact for RL training.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[VYREX] GPU Detected: {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"[VYREX] TF32 Enabled: matmul={torch.backends.cuda.matmul.allow_tf32}, "
              f"cudnn={torch.backends.cudnn.allow_tf32}")
    else:
        print("[VYREX] WARNING: No CUDA GPU detected. Training will be VERY slow.")
        print("[VYREX] Ensure PyTorch is installed with CUDA support.")

    # Print config summary
    print(config.summary())
    print()

    # Ensure directories exist
    os.makedirs(config.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(config.paths.log_dir, exist_ok=True)

    # --- Resume from checkpoint ---
    # rlgym-ppo Learner handles WandB init internally (including resume
    # from saved run_id in BOOK_KEEPING_VARS.json). We do NOT call
    # wandb.init() here — that would create a duplicate run.
    checkpoint_load_dir = None
    resume_step = 0
    if resume:
        checkpoint_load_dir = find_latest_checkpoint(config.paths.checkpoint_dir)
        if checkpoint_load_dir:
            print(f"[VYREX] Resuming from checkpoint: {checkpoint_load_dir}")
            # Extract step count from checkpoint directory name
            try:
                resume_step = int(os.path.basename(checkpoint_load_dir))
            except ValueError:
                # Fall back to reading BOOK_KEEPING_VARS.json
                bk_path = os.path.join(checkpoint_load_dir, "BOOK_KEEPING_VARS.json")
                if os.path.exists(bk_path):
                    import json
                    with open(bk_path) as f:
                        bk = json.load(f)
                    resume_step = int(bk.get("cumulative_timesteps", 0))
        else:
            print("[VYREX] No checkpoint found. Starting fresh training.")

    # --- Determine curriculum phase ---
    cc = config.curriculum
    if resume_step < cc.phase_1_end:
        phase_name = "Phase 1 (kickoffs only)"
        random_ratio = cc.phase_1_random_ratio
    elif resume_step < cc.phase_2_end:
        phase_name = "Phase 2 (mixed)"
        random_ratio = cc.phase_2_random_ratio
    else:
        phase_name = "Phase 3 (mostly random)"
        random_ratio = cc.phase_3_random_ratio
    print(f"[VYREX] Curriculum: {phase_name} (random_ratio={random_ratio:.0%}) at step {resume_step:,}")

    # --- Build the environment function ---
    # Use functools.partial instead of a closure so it can be pickled
    # for multiprocessing (rlgym-ppo spawns worker processes)
    env_fn = partial(build_vyrex_env, config, resume_step)

    # --- Create the Learner ---
    pc = config.ppo

    # Instantiate metrics logger (extends rlgym_ppo MetricsLogger ABC)
    metrics_logger = VyrexMetricsLogger(config)

    learner_kwargs = dict(
        env_create_function=env_fn,
        n_proc=pc.n_proc,
        min_inference_size=pc.min_inference_size,
        metrics_logger=metrics_logger,
        ppo_batch_size=pc.ppo_batch_size,
        ts_per_iteration=pc.ts_per_iteration,
        exp_buffer_size=pc.exp_buffer_size,
        ppo_minibatch_size=pc.ppo_minibatch_size,
        ppo_ent_coef=pc.ppo_ent_coef,
        ppo_epochs=pc.ppo_epochs,
        policy_lr=pc.policy_lr,
        critic_lr=pc.critic_lr,
        policy_layer_sizes=config.network.policy_layer_sizes,
        critic_layer_sizes=config.network.critic_layer_sizes,
        standardize_returns=pc.standardize_returns,
        standardize_obs=pc.standardize_obs,
        save_every_ts=pc.save_every_ts,
        timestep_limit=pc.timestep_limit,
        log_to_wandb=pc.log_to_wandb,
        wandb_project_name=config.paths.wandb_project,
        wandb_group_name="vyrex-2v2",
        wandb_run_name=f"vyrex-2v2-{int(time.time())}",
        checkpoints_save_folder=config.paths.checkpoint_dir,
        add_unix_timestamp=False,
        n_checkpoints_to_keep=5,
        render=pc.render,
        render_delay=pc.render_delay,
    )

    # Add checkpoint loading if resuming
    if checkpoint_load_dir:
        learner_kwargs["checkpoint_load_folder"] = checkpoint_load_dir

    learner = Learner(**learner_kwargs)

    print("\n" + "=" * 60)
    print("  VYREX Training Starting...")
    print(f"  Target: {pc.timestep_limit:,} total steps")
    print(f"  Parallel environments: {pc.n_proc}")
    print(f"  Batch size: {pc.ppo_batch_size:,}")
    print(f"  Minibatch size: {pc.ppo_minibatch_size:,}")
    print(f"  Saving every: {pc.save_every_ts:,} steps")
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
        total_mem = torch.cuda.mem_get_info()[1] / (1024**3)
        print(f"  GPU VRAM: {free_mem:.1f} GB free / {total_mem:.1f} GB total")
    print("=" * 60 + "\n")

    # --- Start Training ---
    learner.learn()

    # --- Training Complete ---
    print("\n[VYREX] Training complete!")

    # Export final model for RLBot deployment
    export_model_for_rlbot(config)


# ============================================================================
# MODEL EXPORT FOR RLBOT v5 DEPLOYMENT
# ============================================================================

def export_model_for_rlbot(config: VyrexConfig):
    """
    Export the trained policy model to the RLBot deployment directory.
    Copies the latest POLICY.pt to the rlbot_deploy/src/ folder.
    """
    checkpoint_dir = find_latest_checkpoint(config.paths.checkpoint_dir)
    if checkpoint_dir is None:
        print("[VYREX] No checkpoint found for export.")
        return

    policy_path = os.path.join(checkpoint_dir, "PPO_POLICY.pt")
    if not os.path.exists(policy_path):
        # Try alternative name
        for fname in os.listdir(checkpoint_dir):
            if "policy" in fname.lower() and fname.endswith(".pt"):
                policy_path = os.path.join(checkpoint_dir, fname)
                break

    if not os.path.exists(policy_path):
        print(f"[VYREX] Policy file not found in {checkpoint_dir}")
        return

    dest = os.path.join(config.paths.model_export_dir, "POLICY.pt")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(policy_path, dest)
    print(f"[VYREX] Model exported to: {dest}")
    print("[VYREX] You can now use this model with the RLBot v5 bot in rlbot_deploy/")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="VYREX — Devastating 2v2 Rocket League AI Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                       # Train with defaults
  python train.py --n_proc 32           # Use 32 parallel environments
  python train.py --render              # Enable visualization
  python train.py --resume              # Resume from latest checkpoint
  python train.py --lr 5e-5             # Custom learning rate
  python train.py --team_size 1         # Train for 1v1 instead
        """,
    )
    parser.add_argument("--n_proc", type=int, default=None,
                        help="Number of parallel RocketSim instances")
    parser.add_argument("--render", action="store_true",
                        help="Enable visual rendering (slows training)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate for both policy and critic")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="PPO batch size and ts_per_iteration")
    parser.add_argument("--team_size", type=int, default=None,
                        help="Players per team (1, 2, or 3)")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable WandB logging")
    parser.add_argument("--timestep_limit", type=int, default=None,
                        help="Total training timestep budget")
    return parser.parse_args()


def main():
    args = parse_args()
    config = VyrexConfig()

    # Apply CLI overrides
    if args.n_proc is not None:
        config.ppo.n_proc = args.n_proc
    if args.render:
        config.ppo.render = True
    if args.lr is not None:
        config.ppo.policy_lr = args.lr
        config.ppo.critic_lr = args.lr
    if args.batch_size is not None:
        config.ppo.ppo_batch_size = args.batch_size
        config.ppo.ts_per_iteration = args.batch_size
        config.ppo.exp_buffer_size = args.batch_size * 3
    if args.team_size is not None:
        config.env.team_size = args.team_size
    if args.no_wandb:
        config.ppo.log_to_wandb = False
    if args.timestep_limit is not None:
        config.ppo.timestep_limit = args.timestep_limit

    train(config, resume=args.resume)


if __name__ == "__main__":
    main()
