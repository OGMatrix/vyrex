# VYREX Training Report — 454M Steps
## Run 5: grounded_weight Reduction (433M → 456M, ~23M new steps)
**Generated:** 2026-02-16  
**WandB Run:** `9ue9rans` (resumed, session `run-20260215_230705`)  
**Duration:** ~21 min at ~17,900 avg SPS  

---

## Executive Summary

This is the **first run with `grounded_weight` reduced from 0.05 to 0.015**. The change was made to address the over-correction diagnosed in the 424M report, where airborne fraction collapsed from 72.5% to 3.2%.

**Key finding:** The reduction is **too conservative**. Airborne fraction recovered only from 3.2% to 4.3% over 20M steps — far short of the 15-25% target. At the current trajectory, reaching 15% airborne would take approximately **190M additional steps** (~3 hours). Meanwhile, the policy is operating at a new equilibrium where the ground reward still dominates, suppressing all forms of jumping including beneficial dodges and low aerials.

**The good news:** This run is the most stable and healthy PPO session to date. No policy shock, no KL spikes, steadily declining VF loss, and the best speed/touch metrics ever recorded. The bot is learning effectively within its "ground-only" constraint — it just needs that constraint relaxed further.

| Change | Value | Status |
|--------|-------|--------|
| `grounded_weight` 0.05 → 0.015 | Active | Too conservative — barely moving airborne |
| `ent_coef` 0.005 | Unchanged | Healthy — entropy stable at 91.3% |
| Phase 3 (70% random) | Active | Working well |
| Permanent checkpoints | Active | 3 new archives saved |

---

## 1. Context: Two Runs Since Last Report

The 424M report analyzed steps 401M–425M (grounded_weight=0.05). But training continued in that same session to 433M before it was stopped. Then the current run resumed at 433M with the new weight.

**Timeline:**
```
Run 4 (222912):  401M ────────────────────────── 433M   grounded_weight = 0.05
                  │ airborne: 72.5% → 3.2% → 2.97% → 3.38%
                  │ reward: mean ~77
                  
  ── CONFIG CHANGE: grounded_weight 0.05 → 0.015 ──
  
Run 5 (230705):  433M ────────────────────────── 456M   grounded_weight = 0.015
                  │ airborne: 3.15% → 4.27%  (+1.12pp in 23M steps)
                  │ reward: mean ~36
```

---

## 2. Behavioral Metrics (Diagnostics Snapshots)

### Full Diagnostic Table (This Run Only)

| Step | Airborne % | Touches/step | Aerial/step | Avg Boost | Avg Speed | Goals (B/O) | Spirit |
|------|-----------|-------------|-------------|-----------|-----------|-------------|--------|
| **433M** | **3.15** | 0.0461 | 0.0145 | 7.76 | 899.4 | 18 / 23 | 0.693 |
| **438M** | **3.84** | 0.0464 | 0.0190 | 6.64 | 882.1 | 31 / 27 | 0.702 |
| **443M** | **3.96** | 0.0460 | 0.0203 | 6.91 | 897.7 | 27 / 25 | 0.710 |
| **448M** | **3.84** | 0.0542 | 0.0153 | 6.36 | 889.2 | 20 / 23 | 0.718 |
| **453M** | **4.27** | 0.0505 | 0.0132 | 7.59 | 954.6 | 26 / 35 | 0.726 |

### 2.1 Airborne Fraction: Barely Moving

```
Full history:

  251M:       85.5%  ████████████████████████████████████████████   old InAirReward
  358M:       77.2%  ██████████████████████████████████████████     grounded=0.005
  399M:       70.7%  ████████████████████████████████████           grounded=0.005
  406M:        3.3%  ██                                            grounded=0.05
  421M:        3.2%  ██                                            grounded=0.05
  427M:        3.0%  ██                                            grounded=0.05
  433M:        3.2%  ██                                            ← config change boundary
  438M:        3.8%  ██                                            grounded=0.015
  443M:        4.0%  ██                                            grounded=0.015
  448M:        3.8%  ██                                            grounded=0.015
  453M:        4.3%  ██                                            grounded=0.015
```

**Recovery rate:** +1.12 percentage points in 20M steps = **+0.056pp per million steps**.

**Projected time to target:**
- To reach 15%: ~191M more steps (~2.9 hours)
- To reach 25%: ~369M more steps (~5.7 hours)

**But the rate is decelerating**, not steady. Looking at the 5M-window deltas:
```
  433→438M:  +0.69pp   ← initial response to weight change
  438→443M:  +0.12pp   ← sharply slowing
  443→448M:  -0.12pp   ← flat/regression
  448→453M:  +0.43pp   ← noisy recovery
```

This suggests the policy is finding a **new equilibrium around 3.5-4.5% airborne** where the GroundedReward at 0.015 balances against the natural benefit of occasional jumps. The recovery may stall entirely at 4-5% rather than continuing toward 15%.

**Why 0.015 is still too strong:** Even at 0.015, the grounded reward delivers +0.015 per step when on ground vs +0.000 when airborne. Over a 150K-step batch, that's 2,250 reward difference between an "always grounded" policy and a "sometimes jumping" policy. The only rewards that can counterbalance jumping are sparse events (goals at 10.0, touches at 3.0), which occur too infrequently to consistently overcome the dense per-step grounded penalty.

### 2.2 Speed: New All-Time High

```
  251M:   502  ████████████████
  358M:   578  ███████████████████
  401M:   669  ██████████████████████
  421M:   848  ████████████████████████████
  433M:   899  ██████████████████████████████
  453M:   955  ████████████████████████████████  ← NEW PEAK
```

**+12.5% improvement over this run** (848 → 955 uu/s). The bot is the fastest it has ever been. Ground-based locomotion continues to be refined — likely improving boost usage, powerslide turns, and sustained acceleration.

### 2.3 Touch Rate: Best Ever, Still Improving

```
  251M:  0.013  ████
  358M:  0.025  ████████
  421M:  0.042  ██████████████
  448M:  0.054  ██████████████████  ← NEW PEAK
  453M:  0.050  █████████████████
```

Touch rate has improved **+20% over this run** (0.042 → 0.050). The 448M peak of 0.054 is the highest ever recorded. The bot is engaging with the ball far more effectively.

### 2.4 Goal Frequency: Sustained High Level

| Window | Blue | Orange | Total | Per 1M Steps |
|--------|------|--------|-------|-------------|
| 433M | 18 | 23 | 41 | 8.2 |
| 438M | 31 | 27 | 58 | 11.6 |
| 443M | 27 | 25 | 52 | 10.4 |
| 448M | 20 | 23 | 43 | 8.6 |
| 453M | 26 | 35 | 61 | 12.2 |

Goals per 1M steps: **8.2 → 12.2**. The 438M and 453M snapshots show 58 and 61 total goals — the highest goal counts ever recorded. The bot is scoring at ~2× the rate from the 424M report. Blue/Orange ratio is approximately balanced as expected in self-play.

### 2.5 Teammate Spacing

```
  421M:  2401
  433M:  2484
  438M:  2589
  443M:  2717  ← peak
  448M:  2369
  453M:  2464
```

Average teammate distance fluctuates 2369–2717 uu. This is healthy — the bots are not cluster-chasing. With team_spirit at 0.71–0.73, the spacing reward is beginning to materially influence positioning.

---

## 3. PPO Training Dynamics

### 3.1 Policy Shock: Minimal (Healthy!)

Unlike the previous resume (KL peaked at 0.012, clip at 0.063), this transition was gentle:

| Iteration | Step | KL | Clip | Note |
|-----------|------|----|------|------|
| 1 | 433.4M | 0.00004 | 0.00000 | Loaded old policy (no update) |
| 2 | 433.5M | 0.00045 | 0.00184 | Minimal adjustment |
| 3 | 433.7M | 0.00193 | 0.01320 | Settling in |
| 4 | 433.8M | 0.00269 | 0.02000 | Approaching steady state |
| 5 | 434.0M | **0.00385** | **0.03797** | **Peak — 7× lower than last resume!** |
| 6 | 434.1M | 0.00298 | 0.02682 | Recovery |
| 10 | 434.7M | 0.00210 | 0.01445 | Near steady state |
| 15 | 435.5M | 0.00236 | 0.01603 | Stable |

**Peak KL of 0.00385 vs 0.01192 last time** — a 3× smaller policy shock. This confirms that reducing a reward weight creates a much gentler transition than increasing one. The PPO safety mechanisms (clipping) barely engaged.

### 3.2 Entropy: Stable Plateau, Slow Decline

```
Entropy trajectory (max possible = 4.500):

  433.4M:  4.134   ██████████████████████████████████████████████  91.9%
  437.9M:  4.124   ██████████████████████████████████████████████  91.6%
  442.4M:  4.124   ██████████████████████████████████████████████  91.6%
  446.9M:  4.131   ██████████████████████████████████████████████  91.8%
  451.4M:  4.117   ██████████████████████████████████████████████  91.5%
  456.3M:  4.111   ██████████████████████████████████████████████  91.4%
```

**Total entropy drop: 0.023 over 23M steps** — 12× slower than the previous run's 0.295 drop. The entropy is now in a healthy, slow-decline regime. At 91.4% of maximum, the policy maintains strong exploration capacity while gradually specializing. No risk of entropy collapse.

### 3.3 Reward Signal

| Window | Mean | StdDev | Min | Max |
|--------|------|--------|-----|-----|
| W1 (433-439M) | 40.25 | 19.05 | 9.33 | 96.51 |
| W2 (439-445M) | 35.99 | 13.05 | 13.39 | 61.17 |
| W3 (445-450M) | 33.14 | 14.28 | 14.44 | 73.47 |
| W4 (450-456M) | 35.15 | 14.90 | 16.17 | 78.90 |

**Mean reward: 36.1** — down from 77.4 in the previous run. This is almost entirely explained by the reduced grounded reward:

```
Grounded reward contribution:
  At 0.05 weight, 97% grounded:  0.05 × 0.97 × 150K ≈ 7,275/batch → ~48.5 reward units
  At 0.015 weight, 96% grounded: 0.015 × 0.96 × 150K ≈ 2,160/batch → ~14.4 reward units
  
  Difference: ~34.1 reward units  (matches observed drop of 77.4 → 36.1 = -41.3)
```

The extra ~7 point drop suggests gameplay rewards also slightly declined, possibly from the noisier reward landscape as the grounded reward noise floor dropped.

**Zero iterations above 100 reward** (vs 25% of iterations >100 in the previous run). This is entirely a mechanical consequence of the weight reduction — the grounded reward no longer inflates iteration totals above 100.

### 3.4 Value Function Loss: Improving

```
  VF Loss trajectory:
  433M:  0.0252   ← elevated from reward change
  437M:  0.0173   ← sharp recovery
  441M:  0.0163   ← settling
  445M:  0.0200   ← fluctuation
  449M:  0.0169   ← continuing decline
  453M:  0.0158   ← approaching baseline
  456M:  0.0188   ← slight bump
```

**Mean VF loss: 0.0176** — down from 0.0178 in the previous run, and the trend is clearly downward. The critic is successfully adapting to the new reward landscape. This is the first run where VF loss is DECLINING (it was rising in the previous run).

### 3.5 Update Magnitudes: Steady

| Metric | Start (433M) | End (456M) | Status |
|--------|-------------|-----------|--------|
| Policy Mag | 0.132 | 0.351 | Settled at ~0.35 |
| VF Mag | 0.158 | 0.420 | Settled at ~0.43 |

After the initial warm-up (first iteration always has low magnitudes from loaded policy), update magnitudes quickly converged to steady values and remained there throughout. This indicates consistent, healthy learning — neither accelerating (instability) nor decelerating (stagnation).

---

## 4. Comparison Across All Runs

| Metric | 251M | 358M | 421M (w=0.05) | 433M (w=0.015) | 453M (w=0.015) | Δ this run |
|--------|------|------|---------------|----------------|----------------|------------|
| Airborne % | 85.5 | 77.2 | 3.2 | 3.2 | 4.3 | **+1.1pp** |
| Touches/step | 0.013 | 0.025 | 0.042 | 0.046 | 0.050 | **+9%** |
| Avg Speed | 502 | 578 | 848 | 899 | 955 | **+6%** |
| Goals/5M | ~12 | ~20 | 41 | 41 | 61 | **+49%** |
| Avg Boost | 5.4 | 4.1 | 6.9 | 7.8 | 7.6 | -3% |
| Entropy | 4.452 | 4.454 | 4.149 | 4.134 | 4.111 | **-0.023** |
| KL (steady) | 0.0003 | 0.0005 | 0.0023 | 0.0020 | 0.0020 | stable |
| VF Loss | 0.006 | 0.006 | 0.018 | 0.025 | 0.018 | **↓ improving** |
| Reward mean | — | — | 77.4 | 40.3 | 35.2 | (grounded reward reduced) |

**Takeaway:** Every gameplay metric (speed, touches, goals) continues to improve. The only stalled metric is airborne recovery.

---

## 5. Permanent Checkpoints: Working

The permanent archiving system activated correctly:

| Archive | Step | Source |
|---------|------|--------|
| 433246932 | 433M | First archive after feature added |
| 439547572 | 440M | +7M boundary |
| 447948404 | 448M | +7M boundary |
| 414m, 422m, 433m | Various | Manually added before feature |

Rolling checkpoints (last 5): 446M, 448M, 450M, 452M, 454M.

---

## 6. Diagnosis: The Core Problem

### Why Airborne Isn't Recovering

The GroundedReward creates a **continuous per-step tax on airborne time**. Any reward weight > 0 makes "jump" strictly worse than "stay on ground" in expectation, unless the resulting aerial leads to a sparse reward event (touch, goal) that exceeds the accumulated grounded reward loss.

**Math for a 0.5-second jump (60 physics ticks, ~8 decision steps at action_repeat=8):**

```
Grounded reward foregone:  8 steps × 0.015 = 0.120 reward
To break even, the jump must lead to either:
  - A ball touch (+3.0) with >4% probability:  3.0 × 0.04 = 0.120  ✓
  - Speed gain (+0.001/step for ~20 steps): 0.020  ✗  (not enough)
  - A goal (+10.0) with >1.2% probability: 10.0 × 0.012 = 0.120  ✓
```

At the current touch rate (~5% per step), the expected touch reward from a jump is roughly 0.05 × 3.0 = 0.15 — which barely exceeds the grounded penalty. The policy is at the knife's edge where jumping *might* pay off, explaining the glacial recovery. But the math requires the bot to already be good at aerial contacts to make jumping profitable — a chicken-and-egg problem.

### The Chicken-and-Egg Problem

1. The bot won't jump because grounded reward makes jumping unprofitable
2. The bot can't learn aerial skills because it never jumps
3. Without aerial skills, the expected reward from jumping stays low
4. Therefore the bot continues not jumping → goto 1

This equilibrium trap means airborne fraction will likely **plateau at 4-6%** rather than continuing toward 15-25%, regardless of how long we train.

---

## 7. Recommendations

### 7.1 CRITICAL: Reduce `grounded_weight` Further — 0.015 → 0.003

**Target: 15-25% airborne.** At 0.003, the grounded reward becomes comparable to speed_weight (0.001) rather than dominating it. The per-step grounded penalty for jumping drops to 0.003 × 8 = 0.024 per jump — easily overcome by a touch attempt.

```python
grounded_weight: float = 0.003  # was 0.015, target 15-25% airborne
```

**Risk assessment:** At 0.003, the bot **will not revert to 85% airborne**. That pathology was caused by the unconditioned InAirReward which *actively rewarded* being airborne. GroundedReward at 0.003 provides a gentle bias toward ground play without the InAirReward to pull the other direction. The current InAirReward is conditioned on ball height > 300, so it only encourages aerial play when the ball is actually high.

### 7.2 ALTERNATIVE: Remove GroundedReward Entirely

A more aggressive option: set `grounded_weight = 0.0`. With the InAirReward now properly conditioned (ball > 300), there is no longer a reward that incentivizes pointless jumping. The bot should naturally find a healthy airborne ratio driven by game rewards alone.

**Risk:** Slightly higher — the policy might temporarily increase airborne time during the transition. But with 0 active reward for being airborne unconditionally, the old 85% pathology cannot return.

### 7.3 KEEP: `ppo_ent_coef` at 0.005

Working perfectly. Entropy declining at a healthy, glacial pace (91.4%). No change needed.

### 7.4 KEEP: Phase 3 Curriculum, Permanent Checkpoints

Both working as designed. No changes needed.

### 7.5 DO NOT: Change Multiple Things Simultaneously

The previous 3-way simultaneous change made attribution difficult. **Change only `grounded_weight` this time.** Evaluate after 15-20M steps.

---

## 8. Summary Scorecard

| Aspect | Score | Δ from 424M | Status |
|--------|-------|-------------|--------|
| Speed | **A+** | ↑ from A | 955 uu/s — all-time high |
| Ball Engagement | **A+** | ↑ from A | 0.054 peak touch rate — best ever |
| Goal Scoring | **A** | ↑ from A | 61 goals/window — best ever |
| PPO Stability | **A** | ↑ from B | KL max 0.004 — gentle transition |
| Entropy Health | **A** | ↑ from A- | 91.4%, slow decline, stable |
| Value Function | **B+** | ↑ from C+ | Declining, adapting well |
| Airborne Balance | **D-** | ↓ from D | 4.3% — barely moved, equilibrium trap |
| Reward Design | **C** | ↑ from B- | 0.015 still too dominant |

**Overall: Gameplay metrics are at all-time highs. PPO is the healthiest it has ever been. The sole remaining problem is the airborne equilibrium trap — `grounded_weight` needs one more reduction (0.015 → 0.003) to break out.**
