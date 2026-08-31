"""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║                        V Y R E X   v 4                                   ║
║               Devastating 2v2 Rocket League AI                           ║
║                                                                          ║
║   Framework:   rlgym-learn + rlgym-learn-algos (PPO)                     ║
║   Rating:      TrueSkill (rlgym-learn-trueskill)                         ║
║   Simulation:  RLGym v2 + RocketSim (headless)                           ║
║   Monitoring:  Weights & Biases (wandb)                                  ║
║   Deployment:  RLBot v5                                                  ║
║                                                                          ║
║   Migration:   v3 (rlgym-ppo) → v4 (rlgym-learn)                        ║
║   Changes:     Framework only — all rewards, network, config preserved   ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝

Training Script — Run this to start training VYREX v4.

Usage:
    conda activate vyrex
    python train.py                    # Train with default config
    python train.py --n_proc 16        # Override parallel environments
    python train.py --render           # Enable visualization
    python train.py --resume           # Resume from latest checkpoint
"""

import os
import sys

# Prevent numpy from using multiple threads in env processes (causes throttling)
os.environ["OPENBLAS_NUM_THREADS"] = "1"


# ============================================================================
# ENVIRONMENT BUILDER
# ============================================================================

def build_vyrex_env():
    """
    Build a 2v2 RLGym v2 environment for VYREX training.

    This function is called once per parallel worker process.
    It must import everything it needs inside the function body
    because each worker is a separate process.

    Returns:
        Raw RLGym environment (rlgym-learn does NOT use RLGymV2GymWrapper).
    """
    # All imports inside function — required by multiprocessing
    import random
    import numpy as np
    from rlgym.api import RLGym, SharedInfoProvider
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import (
        GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition
    )
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import (
        MutatorSequence, FixedTeamSizeMutator, KickoffMutator
    )
    from rlgym.rocket_league.api import GameState as GS_Type
    from rlgym.api import StateMutator
    from rlgym.rocket_league import common_values

    from config import DEFAULT_CONFIG
    from rewards import build_vyrex_reward

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
        """Randomize ball and car positions within the field."""

        def apply(self, state, shared_info):
            bx = np.random.uniform(-3500, 3500)
            by = np.random.uniform(-4500, 4500)
            bz = np.random.uniform(common_values.BALL_RESTING_HEIGHT,
                                   common_values.CEILING_Z * 0.6)
            state.ball.position = np.array([bx, by, bz], dtype=np.float32)

            bvx = np.random.uniform(-2000, 2000)
            bvy = np.random.uniform(-2000, 2000)
            bvz = np.random.uniform(-500, 1000)
            state.ball.linear_velocity = np.array([bvx, bvy, bvz], dtype=np.float32)
            state.ball.angular_velocity = np.random.uniform(-3, 3, size=3).astype(np.float32)

            for car in state.cars.values():
                cx = np.random.uniform(-3800, 3800)
                cy = np.random.uniform(-4800, 4800)
                cz = 17.0
                car.physics.position = np.array([cx, cy, cz], dtype=np.float32)
                yaw = np.random.uniform(-np.pi, np.pi)
                car.physics.euler_angles = np.array([0, yaw, 0], dtype=np.float32)
                cvx = np.random.uniform(-500, 500)
                cvy = np.random.uniform(-500, 500)
                car.physics.linear_velocity = np.array([cvx, cvy, 0], dtype=np.float32)
                car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
                car.boost_amount = np.random.uniform(20, 100)

    class WeightedSampleMutator(StateMutator):
        """Probabilistically choose between multiple state mutators."""

        def __init__(self, mutators_with_weights):
            self.mutators = [m for m, _ in mutators_with_weights]
            self.weights = [w for _, w in mutators_with_weights]

        def apply(self, state, shared_info):
            chosen = random.choices(self.mutators, weights=self.weights, k=1)[0]
            chosen.apply(state, shared_info)

    # At 18.4B+ steps, always in Phase 3 (mostly random)
    random_ratio = cc.phase_3_random_ratio

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

    # =====================================================================
    # GAME METRICS SHARED INFO PROVIDER
    # =====================================================================

    class VyrexMetricsProvider(SharedInfoProvider):
        """Collects per-step game metrics and exposes them via shared_info."""

        METRICS_KEY = "vyrex_game_metrics"

        def create(self, shared_info):
            shared_info[self.METRICS_KEY] = None
            self._prev_touches = {}
            return shared_info

        def set_state(self, agents, initial_state, shared_info):
            shared_info[self.METRICS_KEY] = None
            self._prev_touches = {
                aid: initial_state.cars[aid].ball_touches for aid in agents
            }
            return shared_info

        def step(self, agents, state, shared_info):
            cars = state.cars
            n_agents = len(cars)
            if n_agents == 0:
                shared_info[self.METRICS_KEY] = None
                return shared_info

            ball_pos = state.ball.position
            ball_vel = state.ball.linear_velocity
            ball_speed = float(np.linalg.norm(ball_vel))

            total_touches = 0
            total_aerial = 0
            total_demos = 0
            total_boost = 0.0
            total_speed = 0.0
            n_airborne = 0
            n_zero_boost = 0
            n_supersonic = 0
            n_own_half = 0
            n_near_wall = 0
            n_has_flip = 0
            n_is_boosting = 0
            total_ball_dist = 0.0
            touch_ball_speed_sum = 0.0
            n_touch_agents = 0
            total_boost_collected = 0.0
            aerial_seq_touches = 0

            blue_positions = []
            orange_positions = []

            for aid, car in cars.items():
                # Touches
                prev = self._prev_touches.get(aid, 0)
                new_touches = max(car.ball_touches - prev, 0)
                self._prev_touches[aid] = car.ball_touches
                total_touches += new_touches

                # Aerial touches
                if new_touches > 0 and not car.on_ground:
                    total_aerial += new_touches
                    aerial_seq_touches += new_touches

                # Touch ball speed
                if new_touches > 0:
                    touch_ball_speed_sum += ball_speed
                    n_touch_agents += 1

                # Demos
                if car.is_demoed:
                    total_demos += 1

                # Boost
                total_boost += car.boost_amount
                if car.boost_amount < 1.0:
                    n_zero_boost += 1
                if car.is_boosting:
                    n_is_boosting += 1

                # Speed
                speed = float(np.linalg.norm(car.physics.linear_velocity))
                total_speed += speed
                if car.is_supersonic:
                    n_supersonic += 1

                # Position-based
                if not car.on_ground:
                    n_airborne += 1
                if car.has_flip:
                    n_has_flip += 1

                pos = car.physics.position
                ball_dist = float(np.linalg.norm(pos - ball_pos))
                total_ball_dist += ball_dist

                # Own half: blue y<0, orange y>0
                if car.team_num == 0:
                    if pos[1] < 0:
                        n_own_half += 1
                    blue_positions.append(pos)
                else:
                    if pos[1] > 0:
                        n_own_half += 1
                    orange_positions.append(pos)

                # Near wall
                if abs(pos[0]) > 3800 or abs(pos[1]) > 4800:
                    n_near_wall += 1

            # Goals
            goal_blue = 1 if state.goal_scored and state.scoring_team == 0 else 0
            goal_orange = 1 if state.goal_scored and state.scoring_team == 1 else 0

            # Teammate distance
            avg_teammate_dist = 0.0
            for team_positions in [blue_positions, orange_positions]:
                if len(team_positions) >= 2:
                    for i in range(len(team_positions)):
                        for j in range(i + 1, len(team_positions)):
                            avg_teammate_dist += float(
                                np.linalg.norm(team_positions[i] - team_positions[j])
                            )

            # Double commit: 2+ teammates within 500uu of ball
            n_double_commit = 0
            for team_positions in [blue_positions, orange_positions]:
                near_ball = sum(
                    1 for p in team_positions
                    if np.linalg.norm(p - ball_pos) < 500.0
                )
                if near_ball >= 2:
                    n_double_commit += 1

            metrics = np.zeros(23, dtype=np.float32)
            metrics[0] = total_touches
            metrics[1] = total_aerial
            metrics[2] = total_demos
            metrics[3] = goal_blue
            metrics[4] = goal_orange
            metrics[5] = total_boost / n_agents
            metrics[6] = total_speed / n_agents
            metrics[7] = n_airborne
            metrics[8] = n_agents
            metrics[9] = avg_teammate_dist
            metrics[10] = n_zero_boost
            metrics[11] = n_supersonic
            metrics[12] = total_ball_dist / n_agents
            metrics[13] = ball_speed
            metrics[14] = n_own_half
            metrics[15] = n_double_commit
            metrics[16] = touch_ball_speed_sum
            metrics[17] = n_touch_agents
            metrics[18] = total_boost_collected
            metrics[19] = n_near_wall
            metrics[20] = n_has_flip
            metrics[21] = n_is_boosting
            metrics[22] = aerial_seq_touches

            shared_info[self.METRICS_KEY] = metrics
            return shared_info

    # --- Build Environment ---
    # NOTE: rlgym-learn works with raw RLGym env — no RLGymV2GymWrapper needed!
    return RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        shared_info_provider=VyrexMetricsProvider(),
    )


# ============================================================================
# MAIN TRAINING ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    import json
    import shutil
    import time
    import threading
    from typing import Tuple

    import numpy as np
    import torch

    from rlgym_learn_algos.logging import (
        WandbMetricsLogger,
        WandbMetricsLoggerConfigModel,
    )
    from rlgym_learn_algos.ppo import (
        BasicCritic,
        DiscreteFF,
        ExperienceBufferConfigModel,
        GAETrajectoryProcessor,
        GAETrajectoryProcessorConfigModel,
        NumpyExperienceBuffer,
        PPOAgentController,
        PPOAgentControllerConfigModel,
        PPOLearnerConfigModel,
        PPOMetricsLogger,
    )

    from rlgym_learn import (
        BaseConfigModel,
        LearningCoordinator,
        LearningCoordinatorConfigModel,
        ProcessConfigModel,
        PyAnySerdeType,
        SerdeTypesModel,
        generate_config,
    )

    from rlgym.rocket_league.action_parsers import RepeatAction, LookupTableAction
    from rlgym.rocket_league.obs_builders import DefaultObs

    from config import VyrexConfig, DEFAULT_CONFIG
    from metrics_logger import VyrexMetricsLogger

    # TrueSkill components
    from utils.trueskill import ModelVersionManager, TrueSkillTracker, EvalRunner

    # ================================================================
    # CLI ARGUMENTS
    # ================================================================

    parser = argparse.ArgumentParser(
        description="VYREX v4 — Devastating 2v2 Rocket League AI Training (rlgym-learn)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                       # Train with defaults
  python train.py --n_proc 32           # Use 32 parallel environments
  python train.py --render              # Enable visualization
  python train.py --resume              # Resume from latest checkpoint
  python train.py --lr 5e-5             # Custom learning rate
  python train.py --no_wandb            # Disable WandB logging
  python train.py --no_trueskill        # Disable TrueSkill tracking
        """,
    )
    parser.add_argument("--n_proc", type=int, default=None,
                        help="Number of parallel RocketSim instances")
    parser.add_argument("--render", action="store_true",
                        help="Enable visual rendering (slows training)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate for both actor and critic")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="PPO batch size and timesteps_per_iteration")
    parser.add_argument("--team_size", type=int, default=None,
                        help="Players per team (1, 2, or 3)")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable WandB logging")
    parser.add_argument("--no_trueskill", action="store_true",
                        help="Disable TrueSkill tracking")
    parser.add_argument("--timestep_limit", type=int, default=None,
                        help="Total training timestep budget")
    parser.add_argument("--migrate_v3", type=str, default=None, metavar="PATH",
                        help="Path to v3 checkpoint folder to migrate (e.g. ../v3/data/checkpoints/18400867652)")
    args = parser.parse_args()

    config = VyrexConfig()

    # Apply CLI overrides
    if args.n_proc is not None:
        config.ppo.n_proc = args.n_proc
    if args.render:
        config.ppo.render = True
    if args.lr is not None:
        config.ppo.actor_lr = args.lr
        config.ppo.critic_lr = args.lr
    if args.batch_size is not None:
        config.ppo.batch_size = args.batch_size
        config.ppo.timesteps_per_iteration = args.batch_size
        config.ppo.experience_buffer_max_size = args.batch_size * 3
    if args.team_size is not None:
        config.env.team_size = args.team_size
    if args.no_wandb:
        config.ppo.log_to_wandb = False
    if args.timestep_limit is not None:
        config.ppo.timestep_limit = args.timestep_limit

    # ================================================================
    # V3 → V4 CHECKPOINT MIGRATION
    # ================================================================
    # Converts a v3 (rlgym-ppo) checkpoint into v4 (rlgym-learn) format.
    # The model weights are identical — only the file/folder layout differs.
    #
    # v3 layout:   PPO_POLICY.pt, PPO_POLICY_OPTIMIZER.pt,
    #              PPO_VALUE_NET.pt, PPO_VALUE_NET_OPTIMIZER.pt,
    #              BOOK_KEEPING_VARS.json
    #
    # v4 layout:   ppo_learner/actor.pt, ppo_learner/actor_optimizer.pt,
    #              ppo_learner/critic.pt, ppo_learner/critic_optimizer.pt,
    #              ppo_learner/misc.json, ppo_agent.json

    migrated_checkpoint_folder = None
    if args.migrate_v3 is not None:
        v3_path = os.path.abspath(args.migrate_v3)
        if not os.path.isdir(v3_path):
            print(f"[VYREX v4] ERROR: v3 checkpoint not found: {v3_path}")
            sys.exit(1)

        # Validate required v3 files exist
        required_v3 = ["PPO_POLICY.pt", "PPO_POLICY_OPTIMIZER.pt",
                        "PPO_VALUE_NET.pt", "PPO_VALUE_NET_OPTIMIZER.pt"]
        for f in required_v3:
            if not os.path.isfile(os.path.join(v3_path, f)):
                print(f"[VYREX v4] ERROR: Missing v3 file: {os.path.join(v3_path, f)}")
                sys.exit(1)

        # Read v3 bookkeeping for cumulative timesteps
        v3_book = {}
        book_path = os.path.join(v3_path, "BOOK_KEEPING_VARS.json")
        if os.path.isfile(book_path):
            with open(book_path, "r") as f:
                v3_book = json.load(f)

        v3_cumulative_ts = v3_book.get("cumulative_timesteps", 0)
        v3_model_updates = v3_book.get("cumulative_model_updates", 0)

        # Create v4 checkpoint directory
        save_root = os.path.join(config.paths.project_root, "agent_controllers_checkpoints", "PPO")
        # Use a run subfolder name
        run_name = "vyrex-v4-2v2"
        v4_ckpt_dir = os.path.join(save_root, run_name, str(v3_cumulative_ts))
        learner_dir = os.path.join(v4_ckpt_dir, "ppo_learner")
        os.makedirs(learner_dir, exist_ok=True)

        # Copy and rename model files
        file_map = {
            "PPO_POLICY.pt": os.path.join(learner_dir, "actor.pt"),
            "PPO_POLICY_OPTIMIZER.pt": os.path.join(learner_dir, "actor_optimizer.pt"),
            "PPO_VALUE_NET.pt": os.path.join(learner_dir, "critic.pt"),
            "PPO_VALUE_NET_OPTIMIZER.pt": os.path.join(learner_dir, "critic_optimizer.pt"),
        }
        for src_name, dst_path in file_map.items():
            src_path = os.path.join(v3_path, src_name)
            shutil.copy2(src_path, dst_path)
            print(f"  {src_name} → {os.path.relpath(dst_path, config.paths.project_root)}")

        # Write ppo_learner/misc.json (model update count)
        misc_state = {"cumulative_model_updates": v3_model_updates}
        with open(os.path.join(learner_dir, "misc.json"), "w") as f:
            json.dump(misc_state, f, indent=2)

        # Write ppo_agent.json (cumulative timesteps for the agent controller)
        agent_state = {
            "cur_iteration": 0,
            "iteration_timesteps": 0,
            "cumulative_timesteps": v3_cumulative_ts,
            "iteration_start_time": time.time(),
            "timestep_collection_start_time": time.time(),
        }
        with open(os.path.join(v4_ckpt_dir, "ppo_agent.json"), "w") as f:
            json.dump(agent_state, f, indent=2)

        migrated_checkpoint_folder = v4_ckpt_dir

        print(f"\n[VYREX v4] v3 checkpoint migrated successfully!")
        print(f"  Source:              {v3_path}")
        print(f"  Destination:         {v4_ckpt_dir}")
        print(f"  Cumulative steps:    {v3_cumulative_ts:,}")
        print(f"  Model updates:       {v3_model_updates:,}")
        print(f"  Weights compatible:  YES (identical state_dict keys)")
        print()

    # ================================================================
    # RTX 4070 Ti OPTIMIZATION: Enable TF32
    # ================================================================
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[VYREX v4] GPU Detected: {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"[VYREX v4] TF32 Enabled: matmul={torch.backends.cuda.matmul.allow_tf32}, "
              f"cudnn={torch.backends.cudnn.allow_tf32}")
    else:
        print("[VYREX v4] WARNING: No CUDA GPU detected. Training will be VERY slow.")

    # Print config summary
    print(config.summary())
    print()

    # ================================================================
    # ACTOR & CRITIC FACTORIES
    # ================================================================
    # These mirror v3's [2048, 2048, 1024, 1024] architecture exactly.
    # rlgym-learn uses factory functions instead of layer_sizes lists.

    DefaultObsSpaceType = Tuple[str, int]
    DefaultActionSpaceType = Tuple[str, int]

    def actor_factory(
        obs_space: DefaultObsSpaceType,
        action_space: DefaultActionSpaceType,
        device: str,
    ):
        return DiscreteFF(
            obs_space[1],
            action_space[1],
            tuple(config.network.policy_layer_sizes),
            device,
        )

    def critic_factory(obs_space: DefaultObsSpaceType, device: str):
        return BasicCritic(
            obs_space[1],
            tuple(config.network.critic_layer_sizes),
            device,
        )

    # ================================================================
    # TRUESKILL SETUP
    # ================================================================

    trueskill_enabled = not args.no_trueskill
    trueskill_tracker = None
    version_manager = None

    VERSIONS_DIR = os.path.join(config.paths.project_root, "model_versions")
    TRUESKILL_PATH = os.path.join(config.paths.project_root, "trueskill_ratings.json")

    if trueskill_enabled:
        version_manager = ModelVersionManager(
            versions_dir=VERSIONS_DIR,
            max_versions=config.ppo.trueskill_max_versions,
            save_every_ts=config.ppo.trueskill_version_save_interval,
        )

        trueskill_tracker = TrueSkillTracker(min_sigma=1.0)
        if os.path.exists(TRUESKILL_PATH):
            trueskill_tracker.load(TRUESKILL_PATH)
            print(f"[VYREX v4] Loaded TrueSkill ratings from {TRUESKILL_PATH}")
            print(f"  Total versions tracked: {len(trueskill_tracker.version_ratings)}")

    # ================================================================
    # CUSTOM METRICS LOGGER
    # ================================================================

    vyrex_metrics = VyrexMetricsLogger(config, trueskill_tracker)

    class VyrexPPOMetricsLogger(PPOMetricsLogger):
        """Extended PPO metrics logger that includes VYREX custom metrics."""

        def collect_env_metrics(self, data):
            """Extract game metrics from shared_info provided by VyrexMetricsProvider."""
            for env_shared_info in data:
                if env_shared_info is None:
                    continue
                metrics_array = env_shared_info.get("vyrex_game_metrics")
                if metrics_array is not None:
                    vyrex_metrics.accumulate_game_metrics(metrics_array)

        def get_metrics(self):
            base_metrics = super().get_metrics()

            # Get cumulative timesteps from the agent controller (set by _learn callback)
            cumulative_ts = getattr(self, '_cumulative_ts', 0)
            vyrex_custom = vyrex_metrics.get_metrics(cumulative_ts)

            return {**base_metrics, **vyrex_custom}

    vyrex_ppo_logger = VyrexPPOMetricsLogger()

    # ================================================================
    # RLGYM-LEARN CONFIGURATION
    # ================================================================

    pc = config.ppo

    # PPO Learner config — maps v3 hyperparameters exactly
    ppo_learner_config = PPOLearnerConfigModel(
        batch_size=pc.batch_size,
        n_minibatches=pc.n_minibatches,          # v3: 200K / 100K = 2
        n_epochs=pc.n_epochs,                     # v3: ppo_epochs = 2
        ent_coef=pc.ent_coef,                     # v3: ppo_ent_coef = 0.0005
        actor_lr=pc.actor_lr,                     # v3: policy_lr = 1e-4
        critic_lr=pc.critic_lr,                   # v3: critic_lr = 1e-4
        clip_range=pc.clip_range,                 # v3: ppo_clip_range = 0.2
    )

    # Experience buffer config — maps v3's buffer + GAE parameters
    experience_buffer_config = ExperienceBufferConfigModel(
        max_size=pc.experience_buffer_max_size,   # v3: exp_buffer_size = 600K
        trajectory_processor_config=GAETrajectoryProcessorConfigModel(
            gamma=pc.gamma,                       # v3: gamma = 0.995
            lmbda=pc.gae_lambda,                  # v3: gae_lambda = 0.95
            standardize_returns=pc.standardize_returns,  # v3: standardize_returns = True
        ),
    )

    # Wandb config
    wandb_config = None
    if pc.log_to_wandb:
        wandb_config = WandbMetricsLoggerConfigModel(
            group="vyrex-v4-2v2",
            run="vyrex-v4-2v2",
            project=config.paths.wandb_project,
        )

    # Resume checkpoint path
    checkpoint_load_folder = None
    if migrated_checkpoint_folder is not None:
        # Migration takes priority — load from the freshly converted checkpoint
        checkpoint_load_folder = migrated_checkpoint_folder
        print(f"[VYREX v4] Loading migrated v3 checkpoint: {migrated_checkpoint_folder}")
    elif args.resume:
        # rlgym-learn checkpoint structure: agent_controllers_checkpoints/PPO/<run_name>/<timestamp_ns>/
        # Checkpoint folders are named with time.time_ns(), not cumulative_timesteps.
        # The latest checkpoint is the folder with the highest timestamp.
        run_dir = os.path.join(
            config.paths.project_root, "agent_controllers_checkpoints", "PPO", "vyrex-v4-2v2"
        )
        if os.path.isdir(run_dir):
            best_ts = -1
            best_path = None
            for name in os.listdir(run_dir):
                sub = os.path.join(run_dir, name)
                if os.path.isdir(sub):
                    try:
                        ts = int(name)
                        if ts > best_ts:
                            best_ts = ts
                            best_path = sub
                    except ValueError:
                        continue
            if best_path:
                checkpoint_load_folder = best_path
                # Read actual cumulative_timesteps from ppo_agent.json
                agent_json = os.path.join(best_path, "ppo_agent.json")
                if os.path.isfile(agent_json):
                    with open(agent_json, "r") as f:
                        cts = json.load(f).get("cumulative_timesteps", "?")
                    print(f"[VYREX v4] Resuming from checkpoint: {best_path}")
                    print(f"  Cumulative timesteps: {cts:,}" if isinstance(cts, int) else f"  Cumulative timesteps: {cts}")
                else:
                    print(f"[VYREX v4] Resuming from checkpoint: {best_path}")
            else:
                print("[VYREX v4] No checkpoint found in run folder. Starting fresh training.")
        else:
            print("[VYREX v4] No run folder found. Starting fresh training.")

    # Agent controller config
    ac_kwargs = dict(
        learner_config=ppo_learner_config,
        experience_buffer_config=experience_buffer_config,
        timesteps_per_iteration=pc.timesteps_per_iteration,  # v3: ts_per_iteration = 200K
        save_every_ts=pc.save_every_ts,                       # v3: save_every_ts = 2M
        n_checkpoints_to_keep=5,                              # v3: n_checkpoints_to_keep = 5
        add_unix_timestamp=False,                             # v3: add_unix_timestamp = False
        run_name="vyrex-v4-2v2",
    )
    if wandb_config is not None:
        ac_kwargs["metrics_logger_config"] = wandb_config
    if checkpoint_load_folder is not None:
        ac_kwargs["checkpoint_load_folder"] = checkpoint_load_folder

    agent_controller_config = PPOAgentControllerConfigModel(**ac_kwargs)

    # Full learning coordinator config
    lc_config = LearningCoordinatorConfigModel(
        base_config=BaseConfigModel(
            serde_types=SerdeTypesModel(
                agent_id_serde_type=PyAnySerdeType.STRING(),
                action_serde_type=PyAnySerdeType.NUMPY(np.int64),
                obs_serde_type=PyAnySerdeType.NUMPY(np.float64),
                reward_serde_type=PyAnySerdeType.FLOAT(),
                obs_space_serde_type=PyAnySerdeType.TUPLE(
                    (PyAnySerdeType.STRING(), PyAnySerdeType.INT())
                ),
                action_space_serde_type=PyAnySerdeType.TUPLE(
                    (PyAnySerdeType.STRING(), PyAnySerdeType.INT())
                ),
            ),
            timestep_limit=pc.timestep_limit,
        ),
        process_config=ProcessConfigModel(
            n_proc=pc.n_proc,
            render=pc.render,
            render_delay=pc.render_delay,
        ),
        agent_controllers_config={
            "PPO": agent_controller_config,
        },
        agent_controllers_save_folder=os.path.join(
            config.paths.project_root, "agent_controllers_checkpoints"
        ),
    )

    # Generate config.json for reference
    generate_config(
        learning_coordinator_config=lc_config,
        config_location=os.path.join(config.paths.project_root, "config.json"),
        force_overwrite=True,
    )

    # ================================================================
    # TRUESKILL EVALUATION CALLBACK
    # ================================================================

    eval_lock = threading.Lock()
    eval_running = False
    last_eval_ts = 0

    def run_eval_background(actor_state_dict: dict, versions_to_eval: list):
        """Run TrueSkill evaluation in background thread."""
        global eval_running
        try:
            import numpy as _np
            from rlgym.rocket_league.action_parsers import RepeatAction as _RA, LookupTableAction as _LTA
            from rlgym.rocket_league.obs_builders import DefaultObs as _DO
            from rlgym.rocket_league import common_values as _cv

            print(f"\n[TrueSkill] Starting evaluation against {len(versions_to_eval)} versions...")
            print(f"[TrueSkill] Gamemodes: {config.ppo.trueskill_eval_gamemodes}")

            # Load version state dicts
            version_state_dicts = {}
            for vid in versions_to_eval:
                state_dict = version_manager.load_actor(vid, "cpu")
                if state_dict is not None:
                    version_state_dicts[vid] = state_dict

            if not version_state_dicts:
                print("[TrueSkill] No valid versions to evaluate against")
                return

            # Create eval obs/action that match training config exactly
            eval_obs = _DO(
                zero_padding=config.env.obs_zero_padding,
                pos_coef=_np.asarray([
                    1.0 / _cv.SIDE_WALL_X,
                    1.0 / _cv.BACK_NET_Y,
                    1.0 / _cv.CEILING_Z,
                ]),
                ang_coef=1.0 / _np.pi,
                lin_vel_coef=1.0 / _cv.CAR_MAX_SPEED,
                ang_vel_coef=1.0 / _cv.CAR_MAX_ANG_VEL,
                boost_coef=1.0 / 100.0,
            )
            eval_action = _RA(_LTA(), repeats=config.env.action_repeat)

            eval_runner = EvalRunner(
                policy_class=DiscreteFF,
                policy_kwargs={
                    "n_actions": 90,  # LookupTableAction action count
                    "layer_sizes": tuple(config.network.policy_layer_sizes),
                    # input_size inferred from state_dict in load_policy()
                },
                obs_builder=eval_obs,
                action_parser=eval_action,
                device="cpu",
                deterministic=True,
                game_length_seconds=300.0,
                max_overtime_seconds=300.0,
                no_touch_timeout_seconds=30.0,
            )

            eval_runner.run_eval_batch(
                current_state_dict=actor_state_dict,
                version_state_dicts=version_state_dicts,
                gamemodes=config.ppo.trueskill_eval_gamemodes,
                games_per_version=config.ppo.trueskill_eval_games_per_version,
                trueskill_tracker=trueskill_tracker,
            )

            trueskill_tracker.save(TRUESKILL_PATH)
            trueskill_tracker.print_leaderboard(deterministic=True)

        except Exception as e:
            print(f"[TrueSkill] Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            with eval_lock:
                eval_running = False

    def on_iteration_end(agent_controller):
        """Called after each training iteration for TrueSkill tracking."""
        global last_eval_ts, eval_running

        if not trueskill_enabled:
            return

        cumulative_ts = agent_controller.cumulative_timesteps

        # Update metrics logger timestep counter
        vyrex_ppo_logger._cumulative_ts = cumulative_ts

        # Save version checkpoint if needed
        if version_manager.should_save_version(cumulative_ts):
            actor_state_dict = agent_controller.learner.actor.state_dict()
            version = version_manager.save_version(
                actor_state_dict=actor_state_dict,
                cumulative_timesteps=cumulative_ts,
            )
            trueskill_tracker.promote_current_to_version(version.version_id)
            trueskill_tracker.save(TRUESKILL_PATH)

        # Run evaluation if needed (in background)
        if cumulative_ts - last_eval_ts >= config.ppo.trueskill_eval_interval:
            with eval_lock:
                if eval_running:
                    return
                eval_running = True

            all_versions = version_manager.get_version_ids()
            if len(all_versions) >= 1:
                versions_to_eval = all_versions[-config.ppo.trueskill_versions_to_eval:]

                actor_state_dict = {
                    k: v.cpu().clone()
                    for k, v in agent_controller.learner.actor.state_dict().items()
                }

                eval_thread = threading.Thread(
                    target=run_eval_background,
                    args=(actor_state_dict, versions_to_eval),
                    daemon=True,
                )
                eval_thread.start()
                last_eval_ts = cumulative_ts
            else:
                with eval_lock:
                    eval_running = False

    # ================================================================
    # CUSTOM AGENT CONTROLLER WITH TRUESKILL CALLBACK
    # ================================================================

    class VyrexAgentController(PPOAgentController):
        """PPOAgentController with TrueSkill evaluation callback."""

        def _learn(self):
            result = super()._learn()
            # Update cumulative_ts for metrics logger
            vyrex_ppo_logger._cumulative_ts = self.cumulative_timesteps
            on_iteration_end(self)
            return result

    # ================================================================
    # CREATE LEARNING COORDINATOR
    # ================================================================

    # Build the metrics logger with wandb wrapper
    if pc.log_to_wandb:
        metrics_logger = WandbMetricsLogger(vyrex_ppo_logger)
    else:
        metrics_logger = vyrex_ppo_logger

    learning_coordinator = LearningCoordinator(
        build_vyrex_env,
        agent_controllers={
            "PPO": VyrexAgentController(
                actor_factory=actor_factory,
                critic_factory=critic_factory,
                experience_buffer=NumpyExperienceBuffer(GAETrajectoryProcessor()),
                metrics_logger=metrics_logger,
                obs_standardizer=None,
            )
        },
        config=lc_config,
    )

    # ================================================================
    # START TRAINING
    # ================================================================

    print("\n" + "=" * 60)
    print("  VYREX v4 Training Starting (rlgym-learn)")
    print(f"  Target: {pc.timestep_limit:,} total steps")
    print(f"  Parallel environments: {pc.n_proc}")
    print(f"  Batch size: {pc.batch_size:,}")
    print(f"  Minibatches: {pc.n_minibatches} ({pc.batch_size // pc.n_minibatches:,} each)")
    print(f"  TrueSkill: {'ON' if trueskill_enabled else 'OFF'}")
    print(f"  WandB: {'ON' if pc.log_to_wandb else 'OFF'}")
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
        total_mem = torch.cuda.mem_get_info()[1] / (1024**3)
        print(f"  GPU VRAM: {free_mem:.1f} GB free / {total_mem:.1f} GB total")
    print("=" * 60 + "\n")

    learning_coordinator.start()

    # Close alive-progress bar before printing final output
    vyrex_metrics.close()

    # ================================================================
    # TRAINING COMPLETE
    # ================================================================

    print("\n[VYREX v4] Training complete!")

    # Export final model for RLBot deployment
    ckpt_dir = os.path.join(config.paths.project_root, "agent_controllers_checkpoints", "PPO")
    if os.path.isdir(ckpt_dir):
        best_step = -1
        best_path = None
        for name in os.listdir(ckpt_dir):
            sub = os.path.join(ckpt_dir, name)
            if os.path.isdir(sub):
                try:
                    step = int(name)
                    if step > best_step:
                        best_step = step
                        best_path = sub
                except ValueError:
                    continue
        if best_path:
            # Look for actor/policy model file
            for fname in os.listdir(best_path):
                if fname.endswith(".pt") and ("actor" in fname.lower() or "policy" in fname.lower()):
                    src = os.path.join(best_path, fname)
                    dest = os.path.join(config.paths.model_export_dir, "POLICY.pt")
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.copy2(src, dest)
                    print(f"[VYREX v4] Model exported to: {dest}")
                    break
