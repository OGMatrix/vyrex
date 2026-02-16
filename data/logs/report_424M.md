# VYREX Training Report — 424M Steps
## Run 3: Post-Fix Resume (401M → 425M, ~23.6M new steps)
**Generated:** 2026-02-16  
**WandB Run:** `9ue9rans` (resumed)  
**Duration:** ~22 min at 17,500 avg SPS  

---

## Executive Summary

This run is the **first training session with all three critical changes active simultaneously**:

| Change | Old Value | New Value | Magnitude |
|--------|-----------|-----------|-----------|
| `grounded_weight` | 0.005 | **0.05** | 10× increase |
| `ppo_ent_coef` | 0.01 | **0.005** | 2× decrease |
| Curriculum Phase | Phase 2 (30% random) | **Phase 3 (70% random)** | New random distribution |

**Bottom line:** The behavioral fix worked — the perpetual-jumping pathology is **eliminated**. But the bot over-corrected from one extreme to the other: **airborne dropped from 72.5% to 3.2% within 5M steps**. Touch rate doubled, speed jumped 27%, goal frequency more than doubled. The policy experienced a violent but recoverable shock. Entropy found a new equilibrium. The value function is still catching up.

---

## 1. Behavioral Metrics (Diagnostics Snapshots)

Each diagnostic covers a 5M-step evaluation window. The 401M snapshot reflects the first moments after resume (still partially old behavior).

| Step | Airborne % | Touches/step | Aerial/step | Avg Boost | Avg Speed | Goals (B/O) | Spirit | Ent Coef |
|------|-----------|-------------|-------------|-----------|-----------|-------------|--------|----------|
| **401M** | **72.50** | 0.0258 | 0.0062 | 5.27 | 669.1 | 10 / 6 | 0.643 | 0.005 |
| **406M** | **3.26** | 0.0399 | 0.0220 | 6.62 | 828.4 | 22 / 25 | 0.651 | 0.005 |
| **411M** | **3.24** | 0.0417 | 0.0141 | 6.72 | 859.3 | 19 / 20 | 0.659 | 0.005 |
| **416M** | **3.47** | 0.0460 | 0.0180 | 6.93 | 869.1 | 22 / 16 | 0.667 | 0.005 |
| **421M** | **3.18** | 0.0421 | 0.0089 | 6.88 | 848.4 | 18 / 23 | 0.675 | 0.005 |

### Airborne Fraction: Complete Reversal

```
Timeline across ALL runs:

  251M (old):       85.5%  ████████████████████████████████████████████  
  358M (old):       77.2%  ██████████████████████████████████████████    
  399M (old cfg):   70.7%  ████████████████████████████████████          
  401M (new cfg):   72.5%  █████████████████████████████████████         <- first snapshot, transition in progress
  406M:              3.3%  ██                                            <- 5M steps later!
  411M:              3.2%  ██                                            
  416M:              3.5%  ██                                            
  421M:              3.2%  ██                                            <- stable at floor
```

**What happened:** The 10× increase in `grounded_weight` (0.005 → 0.05) made staying on the ground the single most profitable per-step action. At 0.05, the grounded reward contributes ~0.0485 reward per step (at 97% ground time) — this **dominates** the continuous reward signals like `speed_weight` (0.001) and `boost_weight` (0.001) by 50:1. The policy adapted within ~3-5M steps, eliminating virtually all jumping.

**Concern:** 3.2% airborne is **too low** for competitive Rocket League. Even ground-focused play requires 15-25% airborne time for:
- Dodge/flip mechanics (essential for speed and ball hits)
- Low aerials and wall touches
- Kickoff jumps
- Basic aerial challenges

The bot has gone from "always jumping pointlessly" to "never leaving the ground even when it should."

### Touch Rate: Dramatic Improvement

```
  251M:  0.0133  ████
  358M:  0.0253  ████████
  401M:  0.0258  ████████
  406M:  0.0399  ████████████
  416M:  0.0460  ██████████████   <- peak
  421M:  0.0421  █████████████
```

**+63% improvement** from 401M to 421M. The bot is engaging with the ball far more often. This is the natural result of staying grounded — the bot can actually drive toward and contact the ball instead of floating uselessly in the air.

### Speed: Substantial Increase

```
  251M:   502  ████████████████
  358M:   578  ███████████████████
  401M:   669  ██████████████████████
  406M:   828  ███████████████████████████
  416M:   869  █████████████████████████████  <- peak
  421M:   848  ████████████████████████████
```

**+27% from 401M to 421M.** Grounded bots can use dodge-cancels, powerslides, and sustained boosting for real movement. Being on the ground fundamentally enables effective locomotion mechanics.

### Goal Frequency: More Than Doubled

| Window | Blue Goals | Orange Goals | Total | Per 1M Steps |
|--------|-----------|-------------|-------|-------------|
| 401M   | 10        | 6           | 16    | 3.2         |
| 406M   | 22        | 25          | 47    | 9.4         |
| 411M   | 19        | 20          | 39    | 7.8         |
| 416M   | 22        | 16          | 38    | 7.6         |
| 421M   | 18        | 23          | 41    | 8.2         |

Goals per 1M steps jumped from 3.2 to 8.2 (**+156%**). The bot is actually playing Rocket League now — driving toward the ball, hitting it, and scoring. Blue vs Orange goals are roughly balanced (self-play as expected).

### Boost Management

Average boost climbed from 5.27 to 6.88 out of 100 (+31%). Still very low in absolute terms — the bot isn't actively collecting boost pads. This is expected at this training stage; boost pad routing is a higher-order behavior.

---

## 2. PPO Training Dynamics

### 2.1 The Policy Shock (Steps 401M → 403M)

The three simultaneous changes created a **significant policy shock** visible in the first ~9 iterations (~1.4M steps):

| Iteration | Step | KL Divergence | Clip Fraction | Note |
|-----------|------|--------------|---------------|------|
| 1 | 401.6M | 0.00028 | 0.00172 | Normal (loaded old policy) |
| 2 | 401.7M | **0.00781** | **0.05699** | **28× KL spike!** |
| 3 | 401.9M | **0.00725** | **0.04494** | Still extreme |
| 4 | 402.0M | **0.01192** | **0.06260** | **Peak — 43× normal KL** |
| 5 | 402.2M | **0.01046** | **0.05532** | Beginning descent |
| 6 | 402.3M | **0.00845** | **0.04413** | Recovering |
| 7 | 402.5M | **0.00744** | **0.04513** | Recovering |
| 8 | 402.6M | **0.00615** | **0.04301** | Approaching stability |
| 9 | 402.8M | **0.00511** | **0.03503** | Nearly settled |
| 10 | 402.9M | 0.00454 | 0.03334 | Below spike threshold |
| 15 | 403.7M | 0.00373 | 0.03068 | Stabilizing |
| 20 | 404.4M | 0.00341 | 0.02604 | Approaching equilibrium |

**Interpretation:** The first iteration used the old policy's rollout data — so KL was near zero. Starting from iteration 2, the policy began learning "don't jump, stay grounded" (from the 10× grounded_weight) simultaneously with the entropy pressure halving. This caused repeatedly large policy updates that triggered PPO's clipping mechanism.

**Was this dangerous?** A KL of 0.012 is extremely high for PPO. In many setups this would destabilize training. However:
- The KL *consistently decreased* — no runaway divergence
- The clip fraction tracked KL proportionally — PPO's safety mechanism worked
- By ~403M (just 1.5M steps), KL was back below 0.005
- By ~410M, KL settled to ~0.003 (still 10× the pre-fix baseline but stable)

The policy survived the shock, but this was riskier than necessary. **Applying three changes simultaneously was not ideal** — we cannot cleanly attribute effects.

### 2.2 Entropy Trajectory

```
Entropy over time (max possible = 4.500):

  401.6M: 4.441  ███████████████████████████████████████████████████  98.7%
  402.0M: 4.365  █████████████████████████████████████████████████    97.0%
  402.9M: 4.207  ███████████████████████████████████████████████      93.5%
  404.6M: 4.164  ██████████████████████████████████████████████       92.5%
  406.1M: 4.165  ██████████████████████████████████████████████       92.6%
  410.6M: 4.139  ██████████████████████████████████████████████       92.0%
  416.6M: 4.141  ██████████████████████████████████████████████       92.0%
  421.1M: 4.144  ██████████████████████████████████████████████       92.1%
  425.1M: 4.149  ██████████████████████████████████████████████       92.2%
```

**Phase breakdown:**
- **Rapid collapse (401M-403M):** 4.441 → 4.207. A drop of **0.234 in just 1.5M steps**. For comparison, the entire prior 115M-step run dropped entropy by only 0.02.
- **Settling (403M-406M):** 4.207 → 4.165. Decelerating.  
- **Equilibrium (406M-425M):** Oscillates 4.139 — 4.165. **Stable plateau at ~92% of max entropy.**

**Interpretation:** The halved `ent_coef` (0.01→0.005) did exactly what was intended: it allowed the policy to commit more decisively to learned behaviors instead of maintaining near-uniform randomness. The entropy found a natural equilibrium — it's NOT collapsing toward zero, which would indicate catastrophic specialization. At 92% of max, the policy retains substantial exploration capacity.

The entropy drop was **front-loaded** due to the simultaneous reward shock — the policy had a strong, clear signal ("stay on ground") that rapidly concentrated probability mass. Once that behavioral shift completed, entropy stabilized because the remaining action choices (steering, throttle, boost) still require diversity.

### 2.3 Reward Signal

| Window | Mean Reward | Std Dev | CV | Min | Max |
|--------|------------|---------|-----|-----|-----|
| W1 (401-407M) | 74.32 | 33.89 | 0.456 | 23.4 | 150.4 |
| W2 (407-413M) | 71.36 | 33.98 | 0.476 | 28.3 | 169.3 |
| W3 (413-419M) | 82.97 | 40.54 | 0.489 | 27.3 | 224.8 |
| W4 (419-425M) | 77.75 | 28.90 | 0.372 | 30.4 | 146.0 |

**Mean reward ~77** (up from ~20-30 in prior runs). However, a large portion of this increase is "mechanical" — the grounded reward at weight 0.05 contributes ~0.0485 per step just for staying on the ground. With 150K steps per batch, that's +7,275 reward per batch from grounding alone, which is ~48-50 reward per iteration on its own.

The **real gameplay improvement** is visible in the remaining ~27-30 points of reward coming from actual behaviors: touching the ball, scoring goals, maintaining speed, collecting boost.

**Variance is high** (CV ~0.45) with individual iterations ranging from 23 to 225. This is normal for RL training and reflects the stochastic nature of self-play outcomes — some batches have many goals, others don't.

### 2.4 Value Function Loss

| Window | Mean VF Loss | Max VF Loss |
|--------|-------------|-------------|
| W1 | 0.01672 | 0.01977 |
| W2 | 0.01769 | 0.02072 |
| W3 | 0.01811 | 0.01992 |
| W4 | 0.01823 | 0.02006 |

VF Loss is **3× higher** than the pre-fix baseline (~0.006) and shows a **slight upward trend**. This is expected: the reward distribution fundamentally changed (new grounded reward, more goals, faster speeds), and the value network must relearn the entire value landscape. The upward trend suggests the critic hasn't fully adapted yet.

This is not alarming — the VF loss magnitude correlates with reward magnitude, and rewards tripled. The ratio VFLoss/mean_reward ≈ 0.023% is comparable to before.

### 2.5 Policy & Value Update Magnitudes

| Metric | Start (401M) | End (425M) | Trend |
|--------|-------------|-----------|-------|
| Policy Update Mag | 0.194 | 0.350 | ↑ increasing |
| VF Update Mag | 0.148 | 0.418 | ↑ increasing |

Update magnitudes **increased** over the run, which is unusual after a shock (normally they decrease as the policy converges). This suggests the policy is still actively learning and adapting — the initial shock solved the "jumping" problem, but the bot is now exploring ground-based strategies with significant learning still happening. This is healthy at this stage.

---

## 3. Impact Attribution

Since three changes were applied simultaneously, let's try to disentangle:

### 3.1 grounded_weight 0.005 → 0.05 (10×)

**Strongest causal driver.** The airborne fraction collapse from 72.5% to 3.2% in 5M steps is almost entirely attributable to this. The reward signal is unambiguous and overwhelming: at 0.05, the grounded reward is the **dominant continuous reward** in the entire reward function.

| Reward Component | Weight | Typical per-step contribution |
|-----------------|--------|-------------------------------|
| **GroundedReward** | **0.05** | **~0.0485** (97% grounded) |
| SpeedReward | 0.001 | ~0.001 |
| BoostReward | 0.001 | ~0.0005 |
| InAirReward | 0.01 | ~0.00003 (3.2% air × low ball) |
| BallTouchReward | 3.0 | ~0.126 (but sparse) |

GroundedReward at 0.05 provides **50× more per-step signal than speed or boost**, making it the loudest signal in the reward function for any given timestep.

### 3.2 ppo_ent_coef 0.01 → 0.005 (halved)

**Accelerated but didn't cause** the behavioral shift. The entropy drop was concentrated in the first 1.5M steps (coinciding with the policy shock from grounded_weight), then stabilized. If only ent_coef had changed (without reward changes), we'd expect a gradual entropy decline, not a sudden cliff. The ent_coef change **amplified** the reward-driven shift by allowing the policy to commit to "stay grounded" faster.

The stable equilibrium at 92% entropy is healthy. The ent_coef is doing its job — enough exploration remains.

### 3.3 Phase 3 Curriculum (70% random states)

**Likely beneficial but hard to measure.** Random initial states force the bot to handle diverse game situations rather than just kickoffs. This should improve generalization. The slight SPS decrease (18,200 → 17,800) is consistent with more expensive random state generation. The high variance in rewards may partly reflect the more diverse scenarios.

---

## 4. Comparison: Historical Trajectory

| Metric | 251M | 358M | 401M | 421M | Δ (401→421) |
|--------|------|------|------|------|-------------|
| Airborne % | 85.5 | 77.2 | 72.5 | 3.2 | **-69.3 pp** |
| Touches/step | 0.013 | 0.025 | 0.026 | 0.042 | **+62%** |
| Avg Speed | 502 | 578 | 669 | 848 | **+27%** |
| Goals/5M | ~12 | ~20 | 16 | 41 | **+156%** |
| Avg Boost | 5.4 | 4.1 | 5.3 | 6.9 | **+30%** |
| Entropy | 4.452 | 4.454 | 4.441 | 4.149 | **-0.292** |
| KL (steady) | ~0.0003 | ~0.0005 | ~0.0003 | 0.0023 | elevated |

The 401M → 421M segment shows **more behavioral change in 20M steps than the prior 150M steps combined**. This demonstrates that the reward shaping was the binding constraint — the network had the capacity to learn ground play all along; it simply had no incentive to do so.

---

## 5. Diagnosis: Current State Assessment

### What's Working
1. **Jumping pathology eliminated** — The primary training objective is achieved
2. **Ball engagement dramatically improved** — Touch rate nearly doubled
3. **Locomotion improved** — Speed up 27%, bot actually drives
4. **Scoring emerged** — Goals more than doubled; real gameplay is happening
5. **Policy stable** — KL/Clip recovered from initial shock
6. **Entropy healthy** — 92% of max, not collapsing
7. **Training infrastructure solid** — Checkpointing, diagnostics, curriculum all working

### What Needs Attention
1. **Airborne 3.2% is a new pathology** — The bot won't jump/dodge even when it should
2. **grounded_weight dominates the reward** — At 50× the per-step signal of speed/boost, it crowds out nuance
3. **Value function still catching up** — VF loss 3× elevated and slowly rising
4. **Aerial play suppressed** — aerial_touch_rate dropped from 0.022 to 0.009
5. **SPS slightly reduced** — 18,200 → 17,800 (minor, likely Phase 3 cost)

---

## 6. Recommendations

### 6.1 IMMEDIATE: Reduce `grounded_weight` from 0.05 to 0.015

**Rationale:** At 0.05, the grounded reward is the loudest signal in the reward function, creating a new extreme. At 0.015, it would still be 15× the speed/boost rewards (strong enough to discourage pointless jumping) but would allow the policy to discover that jumping/dodging has tactical value when the ball is nearby.

Target airborne fraction: **15-25%** (healthy range for ground-focused play with appropriate aerial capability).

```python
grounded_weight: float = 0.015  # was 0.05, target 15-25% airborne
```

### 6.2 KEEP: `ppo_ent_coef` at 0.005

The entropy has stabilized at 92% of max — a healthy exploration rate. No change needed. The initial fast drop was from the reward shock, not entropy coefficient misconfiguration.

### 6.3 KEEP: Phase 3 Curriculum (70% random)

Working as designed. Diverse initial states are valuable for learning generalizable behaviors.

### 6.4 MONITOR: Value Function Convergence

The VF loss is still elevated and slowly rising. This is expected during reward restructuring but should **flatten within the next 20-30M steps** as the critic adapts to the new reward landscape. If VF loss exceeds 0.025, consider reducing the learning rate.

### 6.5 CONSIDER (future): Contextual Air Reward

Once ground play is solid (airborne stabilizes at 15-25%), consider adding a reward for being airborne **near the ball** (not the current conditional InAirReward which requires ball height > 300). This would teach purposeful aerial play, not just jumping.

### 6.6 AVOID: Simultaneous Changes

The three-way simultaneous change made attribution difficult and created a dangerous policy shock (KL 43× normal). Future adjustments should be made **one at a time** with 15-20M evaluation windows between changes.

---

## 7. Technical Notes

- **158 iterations** logged in 23.6M steps (~149K steps/iteration)
- **SPS:** 18,200 → 17,800 (slightly declining, within normal range)
- **Checkpoints retained:** 414M, 416M, 418M, 420M, 422M (5 kept per config)
- **Team spirit:** 0.643 → 0.675 (linear ramp toward 0.8 at 500M, on schedule)
- **Phase 3 curriculum confirmed active** (70% random states, 30% kickoff)
- **No NaN/Inf issues**, no crashes, no checkpoint corruption

---

## 8. Summary Scorecard

| Aspect | Score | Status |
|--------|-------|--------|
| Jumping Fix | **A+** | Eliminated 72.5% → 3.2% |
| Ball Engagement | **A** | Touch rate +63% |
| Ground Play | **A** | Speed +27%, goals +156% |
| Airborne Balance | **D** | Over-corrected to 3.2% (want 15-25%) |
| PPO Stability | **B** | Survived shock, KL recovering |
| Entropy Health | **A-** | Stabilized at 92%, healthy |
| Value Function | **C+** | 3× elevated, still settling |
| Reward Design | **B-** | grounded_weight too dominant |

**Overall: The critical behavioral fix succeeded. One config adjustment needed (grounded_weight 0.05 → 0.015) to reach the target equilibrium.**
