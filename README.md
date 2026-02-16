# VYREX

**Devastating 2v2 Rocket League AI — Trained with PPO via RLGym v2 + RocketSim**

---

## What is VYREX?

VYREX is a reinforcement learning bot for Rocket League, purpose-built for 2v2 domination.
It trains entirely in simulation (no Rocket League installation needed) using RocketSim at
hundreds of times real-time speed, then deploys to actual Rocket League matches via RLBot v5.

**Core Stack:**
- **Training:** RLGym v2 + RocketSim + rlgym-ppo (PPO)
- **Monitoring:** Weights & Biases (wandb)
- **Deployment:** RLBot v5
- **Environment:** Conda (Python 3.11 + CUDA 11.8)

---

## Quick Start

### Your Hardware
| Component | Spec | How VYREX Uses It |
|-----------|------|-------------------|
| **GPU** | RTX 4070 Ti (12GB) | PPO neural network training, TF32 auto-enabled |
| **CPU** | i7-14700K (20C/28T) | 20 parallel RocketSim environments |
| **RAM** | 48 GB DDR5 | Experience buffer (450K entries), comfortable headroom |

### 1. Setup Environment

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate vyrex

# Login to wandb for training dashboards
wandb login
```

> **Note on PyTorch:** The environment.yml installs PyTorch with CUDA 12.1, which is
> optimal for RTX 4070 Ti (Ada Lovelace, compute capability 8.9). If you have issues,
> edit `environment.yml` and change `pytorch-cuda=12.1` to `pytorch-cuda=11.8`.

### 2. Train VYREX

```bash
# Start training — pre-configured for your hardware
python train.py

# Your hardware config (auto-applied):
#   20 parallel RocketSim environments (1 per CPU core)
#   150K batch size (fits comfortably in 12GB VRAM)
#   75K minibatch size (GPU forward/backward pass)
#   TF32 auto-enabled for ~2-3x matmul speedup

# Useful overrides:
python train.py --render              # Watch the bot train (slower)
python train.py --resume              # Resume from checkpoint
python train.py --no_wandb            # Train without wandb

# If CPU isn't at 100%, try more processes:
python train.py --n_proc 22

# If you get CUDA out-of-memory errors, reduce minibatch in config.py:
#   ppo_minibatch_size: int = 50_000  (instead of 75_000)
```

### Expected Performance
| Metric | Expected Range | What It Means |
|--------|---------------|---------------|
| **SPS (Steps/sec)** | 2,000 — 3,500 | How fast you're collecting training data |
| **Time per 100M steps** | ~8 — 14 hours | Approximate wall clock time |
| **1B steps** | ~3 — 6 days | Bot should be competent at this point |
| **GPU VRAM Usage** | ~6 — 9 GB | Out of your 12GB (comfortable) |
| **RAM Usage** | ~8 — 12 GB | Out of your 48GB (no concern) |

### 3. Monitor Training

- **WandB Dashboard:** Opens automatically at wandb.ai after training starts
- **Console:** Metrics printed every iteration (reward, entropy, SPS)
- **Diagnostics:** Generated every 5M steps in `data/diagnostics/`

### 4. Get Optimization Advice

```bash
# Generate a diagnostic report
python report.py

# Copy the output and paste it to Claude with the prompt:
# "Here is my VYREX training report. Analyze it and give me specific
#  config.py changes to improve performance."
```

### 5. Deploy to Rocket League

```bash
# After training, export the model
# (This happens automatically at the end of training)

# Then in RLBot v5:
# 1. Open RLBot GUI
# 2. Click "Add" and select rlbot_deploy/src/bot.toml
# 3. Set up a 2v2 match and click "Start Match"
```

---

## Project Structure

```
vyrex/
├── environment.yml         # Conda environment (Python 3.11 + CUDA + all deps)
├── config.py               # All hyperparameters in one place
├── train.py                # Main training script (entry point)
├── rewards.py              # Custom reward functions (the heart of VYREX)
├── metrics_logger.py       # WandB logging + diagnostic report generation
├── report.py               # Read diagnostics and get optimization advice
├── data/                   # (Created during training)
│   ├── checkpoints/        # Model checkpoints (saved every 2M steps)
│   ├── diagnostics/        # JSON diagnostic reports
│   └── logs/               # Training logs
└── rlbot_deploy/           # RLBot v5 deployment package
    ├── match.toml          # 2v2 test match config
    └── src/
        ├── bot.toml        # RLBot v5 bot configuration
        ├── bot.py          # Inference bot (loads trained model)
        ├── loadout.toml    # Car customization
        └── POLICY.pt       # (Copied here after training)
```

---

## Architecture & Design Decisions

### Why PPO?
PPO (Proximal Policy Optimization) is the consensus choice for Rocket League AI.
Every major bot (Nexto, Necto, Seer, Lucy-SKG) uses PPO. It offers the best balance of
stability, sample efficiency, and ease of tuning. Alternatives like SAC or IMPALA have been
tried by the community with worse results in this domain.

### Why RLGym v2 + RocketSim?
RocketSim is a headless physics simulator that replicates Rocket League's physics exactly.
It runs 100-1000x faster than the actual game, requires no game installation, and works on
Linux/Mac/Windows. RLGym v2 wraps it in a standard Gym-style API.

### Why Discrete Actions (LookupTableAction)?
The community has extensively tested continuous vs. discrete action spaces.
Discrete actions via a lookup table of 90 meaningful input combinations consistently
outperform continuous action spaces for Rocket League. This is because many inputs are
naturally binary (jump/boost/handbrake are on/off, steering is full left/center/full right).

### Why 2v2 from the Start?
Many bots train 1v1 first and then struggle with teamplay. VYREX trains 2v2 from day one,
using a `team_spirit` curriculum that starts at 0 (individual rewards) and ramps to 0.8
(mostly shared team rewards). This produces agents that learn individual mechanics AND
cooperative rotation/passing.

### Network Architecture: [2048, 2048, 1024, 1024]
This is the proven architecture for competitive Rocket League bots. Smaller networks
(e.g., [256, 256]) plateau at low skill. Larger networks show diminishing returns.
The 4-layer design with decreasing width is standard in the community.

---

## Reward System

VYREX uses a multi-component reward system with carefully tuned weights:

| Reward | Weight | Type | Purpose |
|--------|--------|------|---------|
| GoalReward | 10.0 | Sparse | Score goals (primary objective) |
| ConcedePenalty | -10.0 | Sparse | Don't get scored on |
| VelocityBallToGoal | 0.15 | Dense | Hit ball toward opponent goal |
| SpeedTowardBall | 0.02 | Dense | Drive toward the ball |
| TouchBallReward | 0.5 | Sparse | Make contact with ball |
| SaveReward | 1.0 | Sparse | Make saves |
| DemoReward | 0.3 | Sparse | Demolish opponents |
| FaceBallReward | 0.005 | Dense | Face toward ball |
| InAirReward | 0.002 | Dense | Develop aerial ability |
| BoostConservation | 0.01 | Dense | Don't waste boost |
| TeammateSpacing | 0.01 | Dense | Don't ball-chase |
| RotationReward | 0.01 | Dense | Proper 2v2 positioning |

---

## Training Timeline (Estimated for Your Hardware)

| Steps | Wall Time | Expected Behavior |
|-------|-----------|-------------------|
| **10M** | ~1 hour | Bot drives toward ball, occasional touches |
| **50M** | ~5 hours | Consistent ball contact, basic shots |
| **100M** | ~12 hours | Scoring goals, basic positioning |
| **200M** | ~1 day | Kickoff strategy, boost collection |
| **500M** | ~3 days | Team rotation developing, aerial attempts |
| **1B** | ~6 days | Solid 2v2 play, consistent aerials, demos |
| **2B** | ~12 days | Advanced mechanics, passing plays, flicks |

> These estimates assume ~2,500 SPS average. Your actual SPS depends on
> CPU thermal throttling, background processes, and batch size tuning.

---

## Iterative Improvement Workflow

VYREX is designed for iterative optimization with Claude:

1. **Train** for a few million steps
2. **Run** `python report.py` to generate diagnostics
3. **Feed** the report to Claude with: *"Analyze this VYREX report and suggest config changes"*
4. **Apply** Claude's suggestions to `config.py`
5. **Resume** training with `python train.py --resume`
6. **Repeat** until satisfied

---

## Tuning Guide

### Performance Tuning (for your RTX 4070 Ti + i7-14700K)

**If SPS is below 2,000:**
- Open Task Manager → check CPU usage across all cores
- If CPU < 85%, increase `n_proc` in config.py (try 22, then 24)
- If GPU utilization is low (check with `nvidia-smi`), increase `ppo_minibatch_size` to 100,000
- Make sure you're NOT running other heavy programs during training

**If you get CUDA Out-of-Memory (OOM):**
- Reduce `ppo_minibatch_size` from 75,000 → 50,000 in config.py
- Close other GPU-using programs (browsers with hardware accel, Discord, etc.)
- Run `nvidia-smi` to check what's using your VRAM

**If SPS is above 3,000 and GPU util is low:**
- You're CPU-bottlenecked (good problem). Increase `ppo_batch_size` and
  `ts_per_iteration` to 200,000. Increase `exp_buffer_size` to 600,000.
- Try `ppo_epochs = 3` to extract more learning per iteration

### If the bot isn't scoring goals:
- Increase `velocity_ball_to_goal_weight` in config.py
- Ensure `goal_weight` is at least 10.0
- Check if the bot is touching the ball (touches_per_ep in diagnostics)

### If the bot ball-chases (both teammates chase):
- Increase `teammate_spacing_weight`
- Increase `rotation_reward_weight`
- Ensure team_spirit is ramping up (check diagnostics)

### If the bot wastes boost:
- Increase `boost_conservation_weight`
- Add a boost pickup reward

### If training is slow (low SPS):
- Increase `n_proc` (more parallel environments)
- Decrease `ppo_epochs` from 2 to 1
- Ensure you're using CUDA PyTorch (not CPU)

### If reward is flat / not improving:
- Check entropy in wandb — if too low (<1.5), increase `ppo_ent_coef`
- If too high (>3.0), decrease `ppo_ent_coef`
- Try reducing learning rate to 5e-5
- Check if any reward component is dominating (use wandb component tracking)

---

## Credits & References

- **RLGym:** https://rlgym.org/
- **RLBot:** https://rlbot.org/
- **RocketSim** by ZealanL: https://github.com/ZealanL/RocketSim
- **rlgym-ppo** by AechPro: https://github.com/AechPro/rlgym-ppo
- **Zealan's PPO Guide:** https://github.com/ZealanL/RLGym-PPO-Guide
- **Nexto/Necto:** https://github.com/Rolv-Arild/Necto
- **Lucy-SKG Paper:** https://arxiv.org/abs/2305.15801
- **Seer Thesis** by Neville Walo
- **Rlgym-v2-to-rlbot-v5:** https://github.com/Martico2432/Rlgym-v2-to-rlbot-v5

---

*VYREX — Fear the swarm.*
