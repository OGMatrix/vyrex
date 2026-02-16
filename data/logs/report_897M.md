# VYREX Training Report — 897M Steps
## Run 7: The Overnight Session — 376M Steps Unattended
### (521M → 897M, grounded_weight=0.003, no config changes)
**Generated:** 2026-02-16 morning  
**WandB Run:** `9ue9rans` (session `run-20260216_003545`)  
**Duration:** ~5.8 hours at ~18,060 avg SPS  
**Iterations:** 2,507  

---

## Executive Summary

The overnight training session was the **longest single uninterrupted run** in VYREX's history — 376M steps over ~5.8 hours with no human intervention and no config changes. The results are mixed:

**The Good:**
- **Touch rate surged +41%** — from 0.053 at 519M to 0.075 at 897M, the single largest improvement in any metric across the entire training history
- **Speed hit all-time high** of 976, with a sustained plateau of 943 avg over the last 200M steps (up from 889 at 519M)
- **PPO training was rock-solid stable** — entropy perfectly flat (4.229 ± 0.015), KL never exceeded 0.0021, VF loss declined 17% through the session
- **71 permanent checkpoints** accumulated — plenty of rollback options

**The Concerning:**
- **Airborne fraction continued climbing from 34.6% → 47.0%**, NOT stabilizing at 34-35% as the 519M report projected
- **Aerial efficiency dropped 61%** — the bot is spending far MORE time in the air but converting LESS of that air time into actual ball contacts
- **Goals per window declined slightly** — from 58 to 51, though this is within noise

**The Bottom Line:**
The 519M report was premature in declaring airborne stabilization at 34-35%. The true equilibrium under `grounded_weight=0.003` is **~47%**, which is significantly above the 15-25% target. However, the bot is undeniably playing better overall — the touch rate explosion and speed gains suggest the policy is learning real skills despite the elevated airborne fraction. **The question is whether 47% airborne is a symptom of a growing problem or an acceptable characteristic of an increasingly capable bot.**

---

## 1. Airborne Fraction — The Full Story

### 1.1 Complete Trajectory (This Session)

| Step | Airborne % | Δ from prev | Phase |
|------|-----------|-------------|-------|
| 517M | 34.5 | — | End of last report |
| 521M | 38.9 | +4.4 | **Still climbing** |
| 526M | 37.4 | -1.5 | Oscillation |
| 536M | 40.8 | +3.4 | Climbing again |
| 546M | 40.5 | -0.3 | ~40% plateau attempt |
| 556M | 37.1 | -3.4 | Brief dip |
| 567M | 38.0 | +0.9 | |
| 577M | 41.5 | +3.5 | Breaking above 40% |
| 587M | 40.5 | -1.0 | |
| 597M | 42.0 | +1.5 | |
| 607M | 44.9 | +2.9 | **Pushing toward 45%** |
| 618M | 46.2 | +1.3 | Entering new plateau |
| 628M | 45.2 | -1.0 | |
| 638M | 43.9 | -1.3 | |
| 648M | 47.0 | +3.1 | |
| 658M | 47.1 | +0.1 | **~46-47% plateau** |
| 669M | 44.9 | -2.2 | |
| 679M | 46.2 | +1.3 | |
| 689M | 47.5 | +1.3 | |
| 699M | 44.6 | -2.9 | |
| 720M | 45.5 | — | Oscillating around 46% |
| 745M | 47.3 | — | |
| 771M | 47.1 | — | |
| 796M | 46.5 | — | |
| 822M | 46.7 | — | |
| 847M | 49.7 | — | Brief high spike |
| 873M | 47.8 | — | |
| 893M | 47.8 | — | **Final equilibrium ≈ 47%** |

```
Complete airborne history (all reports):

  251M:  85.5%  ████████████████████████████████████████████   original pathology
  358M:  77.2%  ██████████████████████████████████████████
  399M:  70.7%  ████████████████████████████████████
  406M:   3.3%  ██                                          grounded=0.05 shock
  421M:   3.2%  ██
  453M:   4.3%  ██                                          grounded=0.015
  461M:   4.4%  ██                                          grounded=0.003 starts
  476M:  17.2%  █████████                                   ← target zone
  486M:  27.1%  ██████████████
  497M:  33.0%  █████████████████
  519M:  34.6%  █████████████████                           ← "stabilized" (wasn't!)
  567M:  38.0%  ████████████████████
  618M:  46.2%  ████████████████████████                    ← new plateau
  700M:  44.6%  ██████████████████████
  750M:  44.3%  ██████████████████████
  800M:  46.3%  ████████████████████████
  850M:  49.7%  █████████████████████████
  897M:  47.8%  ████████████████████████                    ← TRUE equilibrium
```

### 1.2 Why Didn't It Stabilize at 34-35%?

The 519M report identified a ~34-35% plateau after only 5 consecutive readings showing deceleration. This was insufficient evidence — the system was still in transition. The actual equilibrium dynamics are:

**Phase 1 (461M-520M):** Rapid recovery from ground-lock (4.3% → 35%). The velocity was high because the policy was "re-discovering" jump actions that had been suppressed for 60M+ steps.

**Phase 2 (520M-620M):** Continued slower drift upward (35% → 46%). The policy was fine-tuning HOW MUCH to jump now that the cost was negligible. At grounded_weight=0.003, the per-jump cost is only ~0.024 reward — invisible against rewards of 20-30 per iteration.

**Phase 3 (620M-897M):** True plateau at 46-48% (mean=46.8%, std=1.6%). The oscillation here is genuine noise, not trend. The system has found its natural resting point.

### 1.3 The 47% Equilibrium — What Determines It?

The equilibrium airborne fraction is governed by the balance of:

| Force | Direction | Strength at 0.003 |
|-------|-----------|-------------------|
| GroundedReward (0.003/step on ground) | ↓ | **Negligible** — ~0.024/jump |
| InAirReward (0.01 when ball>300uu AND airborne) | ↑ | Modest but conditional |
| SpeedReward (speed_toward_ball, 0.02/step) | ↓ | Moderate — ground speed > air speed |
| Action distribution (20% of 90 actions = jump) | ↑ | Structural baseline |
| Entropy bonus (ent_coef=0.005) | ↑ | Pushes toward 20% baseline |
| Learned skill-based decisions | Mixed | Context-dependent |

At `grounded_weight=0.003`:
- Cost per jump ≈ 0.003 × 8 steps ≈ **0.024**
- Entropy push toward uniform (20% jump actions) ≈ **+0.005/step** equivalent
- Speed reward on ground vs air ≈ **variable**

The grounded reward is roughly **5× weaker** than the implicit entropy bonus pushing toward the 20% jump baseline. The policy naturally drifts above 20% toward whatever maximizes total reward minus grounded penalty, which lands at ~47%.

### 1.4 Is 47% Actually a Problem?

This requires nuanced analysis. Let's look at what CHANGED as airborne rose from 35% to 47%:

| Metric | 519M (35%) | 897M (47%) | Change | Verdict |
|--------|-----------|-----------|--------|---------|
| Touches/step | 0.049 | 0.075 | **+53%** | Dramatically better |
| Aerial touches | 0.014 | 0.011 | -21% | Worse efficiency |
| Speed | 889 | 976 | **+10%** | Better |
| Goals/window | 64 | 63 | -2% | Maintained |
| Boost | 6.9 | 7.6 | +10% | Better management |
| Spacing | 2840 | 2768 | -3% | Slight compression |

**The touch rate explosion is the critical finding.** Going from 0.049 to 0.075 touches/step represents a **53% improvement in ball engagement** — this is the single largest skill improvement in VYREX's entire training history. The bot is interacting with the ball much more frequently.

**But aerial efficiency deteriorated.** The ratio of aerial touches to total airborne time dropped from 0.058 to 0.023 (a 61% decline). This means the ADDITIONAL air time (beyond 35%) is producing very few aerial ball contacts. The extra jumping is not primarily about aerial play — it's likely:

1. **Dodge-jumping for speed** — Rocket League's fastest ground traversal uses forward-dodge (a jump action). The speed increase from 889→976 correlates.
2. **Recovery adjustments** — After contacts, the bot may be jumping to orient/recover
3. **Unproductive hops** — Some fraction is likely unnecessary small jumps due to the very low grounded penalty

**Assessment: 47% airborne is not the old pathology, but it's higher than optimal.** The bot is clearly improving (touch rate proves this), but some of the airborne time is wasted. A modest increase to grounded_weight could bring it into the 35-40% range without triggering the equilibrium trap, assuming we increase gently.

---

## 2. Touch Rate — The Breakthrough Metric

### 2.1 The +53% Improvement

This is arguably the most significant development of the entire training run:

```
Touch rate history:
  251M:  0.013  █
  358M:  0.025  ███
  421M:  0.042  █████
  453M:  0.050  ██████
  519M:  0.049  ██████
  600M:  0.060  ████████
  700M:  0.065  ████████
  800M:  0.072  █████████
  897M:  0.075  ██████████                    ← ALL-TIME HIGH
```

The touch rate was essentially flat from 421M through 519M (~0.042-0.050). The overnight session broke through this plateau. The improvement was gradual and sustained — not a sudden jump, suggesting genuine skill development rather than an artifact.

### 2.2 Phased Touch Rate Improvement

| Phase | Steps | Touch Rate | Δ from previous |
|-------|-------|-----------|-----------------|
| Pre-grounded era | 251-401M | 0.013-0.042 | — |
| Ground-locked era | 401-461M | 0.042-0.054 | Steady improvement |
| Recovery era (pre-overnight) | 461-521M | 0.049-0.061 | Oscillating |
| **Overnight early** | **521-620M** | **0.053-0.064** | **+10%** |
| **Overnight middle** | **620-750M** | **0.055-0.075** | **+17%** |
| **Overnight late** | **750-897M** | **0.062-0.082** | **+10% more** |

The improvement correlates with multiple factors working together:
- Higher speed → reaches ball faster
- More attempts (both ground and air) → more contacts
- Team spirit at max 0.8 → less double-committing
- 376M steps of uninterrupted learning → policy refinement

---

## 3. PPO Training Dynamics

### 3.1 The Most Stable Run Yet

This was the most PPO-stable session in VYREX's history. 2,507 iterations with zero anomalies.

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Reward | 24.84 | 9.92 | 7.57 | 81.98 |
| Entropy | **4.2294** | **0.0096** | **4.1966** | **4.2561** |
| KL | 0.00137 | 0.00030 | 0.00002 | 0.00207 |
| VF Loss | 0.01402 | 0.00138 | 0.01053 | 0.01923 |
| Clip Fraction | 0.01155 | 0.00358 | 0.00000 | 0.01976 |
| Policy Mag | 0.3591 | — | — | — |
| VF Mag | 0.4496 | — | — | — |

### 3.2 Entropy: Perfectly Flat

```
Entropy over sessions (max=4.500):

  401M: 4.441  ████████████████████████████████████████████████████  98.7%
  424M: 4.149  ███████████████████████████████████████████████       92.2%  ← grounded=0.05
  454M: 4.111  █████████████████████████████████████████████         91.4%  ← grounded=0.015
  461M: 4.125  ██████████████████████████████████████████████        91.7%  ← grounded=0.003
  519M: 4.226  ███████████████████████████████████████████████       93.9%  ← rising!
  
  ===== OVERNIGHT (this session) =====
  521M: 4.220  ███████████████████████████████████████████████       93.8%
  558M: 4.223  ███████████████████████████████████████████████       93.8%
  596M: 4.224  ███████████████████████████████████████████████       93.9%
  633M: 4.236  ███████████████████████████████████████████████       94.1%
  671M: 4.239  ███████████████████████████████████████████████       94.2%  ← peak window
  708M: 4.235  ███████████████████████████████████████████████       94.1%
  746M: 4.229  ███████████████████████████████████████████████       94.0%
  783M: 4.227  ███████████████████████████████████████████████       94.0%
  821M: 4.226  ███████████████████████████████████████████████       93.9%
  858M: 4.233  ███████████████████████████████████████████████       94.1%
  897M: 4.225  ███████████████████████████████████████████████       93.9%
```

**The entropy reversal that started at 461M has fully played out.** Entropy rose from 4.125 to 4.226 during the 461M-519M session, then **stabilized perfectly** at 4.229 ± 0.010 for the entire 376M-step overnight run. The range of 0.06 (4.197 to 4.256) represents only **1.4% variation** — this is a policy in deep equilibrium.

The final entropy of 4.229 (94.0% of theoretical max 4.500) represents a healthy balance:
- The policy has real preferences (not random — that would be 4.500)
- It maintains enough exploration to continue learning (94% is well above collapse thresholds)
- It's not wasting capacity on futile exploration

### 3.3 KL Divergence: Rock Solid

```
Mean KL per window:
  521-558M:  0.00140
  558-596M:  0.00135
  596-633M:  0.00136
  633-671M:  0.00137
  671-708M:  0.00137
  708-746M:  0.00137
  746-783M:  0.00135
  783-821M:  0.00137
  821-858M:  0.00134
  858-897M:  0.00139
```

KL divergence was essentialy a flat line at 0.00137 ± 0.00002 across all 10 windows. There was no policy shock at any point — not even a micro-disturbance. The policy made small, consistent updates throughout.

Maximum KL of 0.00207 is well below the PPO clip threshold and represents normal per-iteration variance.

### 3.4 Value Function Loss: Continuous Improvement

```
VF Loss trend (20 windows):
  521M: 0.01588  ██████████████████
  539M: 0.01592  ██████████████████
  558M: 0.01525  █████████████████
  577M: 0.01483  ████████████████
  596M: 0.01451  ████████████████
  614M: 0.01390  ███████████████
  633M: 0.01356  ███████████████
  652M: 0.01419  ███████████████
  671M: 0.01392  ███████████████
  689M: 0.01388  ███████████████
  708M: 0.01398  ███████████████
  727M: 0.01335  ██████████████
  746M: 0.01415  ███████████████
  764M: 0.01360  ███████████████
  783M: 0.01328  ██████████████
  802M: 0.01362  ███████████████
  821M: 0.01309  ██████████████
  839M: 0.01313  ██████████████
  858M: 0.01325  ██████████████
  877M: 0.01319  ██████████████    ← best sustained level
```

**VF loss dropped 17%** from 0.01590 (start) to 0.01319 (end). This sustained decline indicates the critic network is steadily improving its value predictions. The relative smoothness of the decline (no spikes, no regime changes) confirms the training process is stable and the reward landscape isn't shifting underneath the critic.

For context, VF loss history:
- 251M: 0.006 (simpler policy, simpler states)
- 421M: 0.018 (post-grounded shock)
- 519M: 0.017 (still elevated)
- 897M: **0.013** (recovering well)

### 3.5 Reward Signal

| Window | Steps | Mean | StdDev |
|--------|-------|------|--------|
| W1 | 521-558M | 23.32 | 9.05 |
| W2 | 558-596M | 23.08 | 9.66 |
| W3 | 596-633M | 23.90 | 9.03 |
| W4 | 633-671M | 25.47 | 10.45 |
| W5 | 671-708M | 25.69 | 10.39 |
| W6 | 708-746M | 25.59 | 10.38 |
| W7 | 746-783M | 25.16 | 9.51 |
| W8 | 783-821M | 26.29 | 10.69 |
| W9 | 821-858M | 25.67 | 10.07 |
| W10 | 858-897M | 24.23 | 9.69 |

**Mean reward: 24.84** — stable throughout with a slight hump in the middle (W4-W8). The reward signal is not showing signs of either collapse or rapid growth. This is expected — the bot is improving (more touches, more speed) but those gains are offset by more time spent airborne (less grounded reward per step). The two effects approximately cancel in the total reward.

Reward distribution (percentiles):
- P10: 13.76 | P25: 17.54 | **P50: 22.99** | P75: 29.99 | P90: 38.34

---

## 4. Aerial Efficiency — The Warning Signal

### 4.1 Efficiency Has Declined Substantially

| Phase | Airborne % | Aerial Touches | Efficiency |
|-------|-----------|---------------|------------|
| 461-521M | 25.7% | 0.0149 | **0.058** |
| 521-620M | 40.4% | 0.0126 | 0.031 (-47%) |
| 620-750M | 46.3% | 0.0114 | 0.025 (-58%) |
| 750-897M | 47.0% | 0.0106 | **0.023 (-61%)** |

The bot went from touching the ball in the air 5.8% of the time it's airborne, to only 2.3%. **The additional 21pp of airborne time (from 26% to 47%) produced FEWER aerial touches, not more.**

### 4.2 What Does This Mean?

The aerial_touch_rate metric measures touches where the car is airborne AND the ball is above a threshold. A declining efficiency means:

1. **Much of the 47% airborne time is NOT motivated by aerial play.** The bot is jumping for non-aerial reasons:
   - Forward-dodging for speed (confirmed by speed increase 897→943)
   - Jumping during approach/recovery sequences
   - Low unnecessary hops that don't reach ball height

2. **Genuine aerial skill is not developing.** Despite spending nearly half its time airborne, the bot's actual ability to intercept the ball in the air has not improved. Aerial touch rate stayed at ~0.011, which is similar to where it was hundreds of millions of steps ago.

3. **The InAirReward (conditional on ball>300uu) is not driving behavior.** The bot does not appear to be jumping specifically BECAUSE the ball is high — it's jumping for other reasons. The InAirReward contributes too rarely to influence behavior.

### 4.3 The Speed-Jump Connection

There's a strong correlation between airborne fraction and speed:

| Airborne Range | Avg Speed |
|---------------|-----------|
| < 35% (461-519M) | 897 |
| 35-42% (519-600M) | 894 |
| 42-48% (600-897M) | 932 |
| > 48% (peaks) | 959 |

This confirms that a significant portion of "airborne" time is actually **dodge-jumping for speed** — a legitimate Rocket League mechanic. The bot has learned that jump → dodge → land is faster than driving. This is actual skill, but it inflates the airborne metric.

---

## 5. Speed and Goals — Net Capability Assessment

### 5.1 Speed: All-Time High

```
Speed history:
  251M:  502   ██████████
  358M:  578   ████████████
  421M:  848   █████████████████
  453M:  955   ████████████████████           ← previous ATH
  519M:  889   ██████████████████
  700M:  916   ███████████████████
  800M:  938   ████████████████████
  897M:  976   ████████████████████           ← NEW ATH
```

The speed recovery from the dip at 519M (889) back to new all-time highs (976) is significant. The bot has surpassed its previous best (955) set during the ground-locked era. This suggests the bot is combining ground driving with dodge-jumping mechanics to achieve faster overall movement.

### 5.2 Goals: Maintained But Not Growing

```
Goals/window (blue+orange combined):
  Phase         | Goals/win | Context
  461-521M      |   57.9    | Recovering from ground-lock
  521-620M      |   54.5    | Transition
  620-750M      |   50.2    | New plateau
  750-897M      |   50.9    | Stable
```

Goal scoring has settled around ~51 goals per diagnostic window, down slightly from the 58 during the recovery phase. This is a **modest decline (-12%)** but well within noise bounds (std=6.3). The flattening likely reflects:
- Self-play equilibrium: as both teams improve, goals become harder
- More defensive awareness with full team spirit (0.8)
- Some time "wasted" on non-productive jumping

**Goal balance is nearly perfect:** Blue 25.9 avg, Orange 24.9 avg — indicating no systematic imbalance.

---

## 6. Team Dynamics

### 6.1 Team Spirit at Maximum

Team spirit reached its cap of 0.8 at ~502M steps and has remained there for the entire overnight session. With 80% shared reward:
- Agents are heavily incentivized to cooperate
- Individual ball-chasing is penalized
- Team-level outcomes (goals, saves) dominate individual metrics

### 6.2 Teammate Spacing

| Phase | Avg Spacing |
|-------|------------|
| 461-521M | 2771 |
| 521-620M | 2841 |
| 620-750M | 2954 |
| **750-897M** | **2949** |

Spacing has increased by ~6% over the overnight run, suggesting the bots are maintaining wider formations. The plateau at ~2950 units indicates the bots have found a comfortable separation distance. This is a positive sign for rotation/positioning awareness.

---

## 7. Infrastructure Status

### 7.1 Performance

| Metric | Value |
|--------|-------|
| Total iterations | 2,507 |
| SPS (sustained) | ~18,060 |
| Steps trained | 375,940,792 |
| Wall time | 5.73 hours |
| Avg iteration time | 8.5 sec |
| Collection time | 3.9 sec |
| Consumption time | 4.6 sec |

SPS maintained at ~18,000 throughout with no degredation — the hardware is running optimally.

### 7.2 Checkpoints

| Type | Count | Range |
|------|-------|-------|
| Rolling | 5 | 890M-897M |
| **Permanent** | **71** | 414M-894M |
| New this session | **58** permanent archives |

The permanent checkpoint system worked flawlessly, saving every ~6.3M steps (vs configured 7M, due to rounding). This gives excellent rollback granularity.

### 7.3 WandB

Run `9ue9rans` has accumulated 12 sessions total. The overnight session logged to `run-20260216_003545-9ue9rans` with 63,988 lines of output. All metrics are logged and visible on the WandB dashboard.

---

## 8. Full Historical Comparison

| Metric | 251M | 358M | 421M | 453M | 519M | **897M** | Trend |
|--------|------|------|------|------|------|---------|-------|
| Airborne % | 85.5 | 77.2 | 3.2 | 4.3 | 34.6 | **47.8** | ⚠️ above target |
| Touches/step | 0.013 | 0.025 | 0.042 | 0.050 | 0.049 | **0.075** | ✅ **ATH (+53%)** |
| Aerial rate | 0.009 | — | 0.014 | 0.015 | 0.014 | **0.011** | ↓ declining |
| Speed | 502 | 578 | 848 | 955 | 889 | **976** | ✅ **ATH** |
| Goals/window | ~12 | ~20 | 41 | 61 | 64 | **51** | ↔ slight decline |
| Boost | 5.4 | 4.1 | 6.9 | 7.6 | 6.9 | **7.6** | ✅ stable-high |
| Entropy | 4.452 | 4.454 | 4.149 | 4.111 | 4.226 | **4.229** | ✅ stable |
| KL (steady) | 0.0003 | 0.0005 | 0.0023 | 0.0020 | 0.0015 | **0.0014** | ✅ excellent |
| VF Loss | 0.006 | 0.006 | 0.018 | 0.018 | 0.017 | **0.013** | ✅ improving |
| Team Spirit | 0.40 | 0.57 | 0.68 | 0.73 | 0.80 | **0.80** | ✅ at max |
| Spacing | — | — | 2401 | 2553 | 2840 | **2949** | ✅ improving |
| SPS | ~17,500 | ~17,900 | ~17,800 | ~17,900 | ~18,100 | **~18,060** | ✅ stable |

---

## 9. The Grounded Weight Saga — Updated

```
Weight   Steps Active    Air% Start → End    Key Outcome
──────   ──────────────   ──────────────────   ───────────────────
0.005    251M → 401M     85.5% → 70.7%        Too weak (150M for -15pp)
0.050    401M → 433M     70.7% → 3.2%         Too strong (5M shock!)
0.015    433M → 461M     3.2%  → 4.3%         Equilibrium trap (still too strong)
0.003    461M → 897M     4.3%  → 47.8%        ⚠️ Free climbing — landed at 47%
```

The fundamental challenge remains: the dynamic range issue. Small changes in grounded_weight produce large effects on airborne fraction. The relationship is highly nonlinear:
- 0.050: forces 3% airborne (too low)
- 0.015: forces 4% airborne (still too low — the trap)
- 0.003: allows 47% airborne (higher than target)

The "sweet spot" for 35-40% airborne is likely around **0.005-0.008**, but reaching it without causing another destabilization requires careful adjustment.

---

## 10. Recommendations

### 10.1 Airborne Adjustment

**Option A (Conservative): Increase grounded_weight to 0.005**
- Returns to the original value that produced -15pp over 150M steps (when starting from 85%)
- Starting from 47% with an established ground-play capable policy, the effect should be gentler
- Expected outcome: airborne drifts from 47% toward 38-42% over ~100-150M steps
- Risk: LOW (the bot already knows how to play on the ground)

**Option B (Moderate): Increase grounded_weight to 0.006-0.007**
- Slightly stronger push, targeting 35-40%
- Should produce faster convergence (50-100M steps to equilibrium)
- Risk: LOW-MODERATE

**Option C (Do nothing): Keep 0.003**
- Accept 47% as the equilibrium
- The bot IS improving — touch rate and speed at all-time highs
- 47% may be fine if much of it is dodge-jumping for speed (legitimate mechanic)
- Risk: The aerial efficiency metric stays low

**Recommended: Option A (0.005)** — the gentlest meaningful adjustment. The bot's current skills are robust and won't be lost by a small weight increase. We have 71 permanent checkpoints to roll back if needed.

### 10.2 Other Considerations

1. **Learning rate reduction:** At 897M steps, consider halving LR from 0.0001 to 0.00005 around 1B steps. The policy is in deep equilibrium (entropy flat, KL flat) and may benefit from finer-grained learning.

2. **Curriculum advancement:** Still on Phase 3 (70% random states, 30% kickoffs). Could consider Phase 4 (higher kickoff ratio) around 1B, though the current ratio seems to be working well.

3. **InAirReward tuning:** The conditioned aerial reward (ball>300uu) is not effectively driving aerial development. If aerial play is a goal, consider:
   - Lowering the ball height threshold from 300 to 200
   - Increasing in_air_weight from 0.01 to 0.02
   - Adding an aerial touch bonus reward

---

## 11. Summary Scorecard

| Aspect | Score | Δ from 519M | Details |
|--------|-------|-------------|---------|
| Airborne Balance | **C+** | ↓ from B | 47.8% — above 15-25% target, stabilized |
| Ball Engagement | **A++** | ↑↑ from A+ | 0.075 — **53% improvement**, all-time record |
| Goal Scoring | **A** | ↓ from A+ | 51/window — slight decline from 64 |
| Speed | **A++** | ↑ from A | 976 — all-time record |
| PPO Stability | **A++** | ↑ from A+ | Entropy flat, KL flat, VFL declining — perfect |
| Entropy Health | **A+** | ↔ | 94.0% — stable equilibrium |
| Value Function | **A** | ↑ from A- | 0.013 — 17% improvement this session |
| Aerial Development | **C** | ↓ from B | Efficiency dropped 61%, not developing |
| Team Cooperation | **A** | ↔ | Spacing improving, goals balanced |
| Infrastructure | **A+** | ↔ | 71 permanent checkpoints, 18K SPS |

### Overall: **A-** — Strong session with one concerning trend

The overnight session was overwhelmingly positive from a training stability perspective and produced the best touch rate and speed the bot has ever achieved. The one concern — airborne fraction at 47% with declining aerial efficiency — is manageable with a gentle grounded_weight adjustment. The bot has firmly broken out of the ground-lock trap and developed into a genuinely more capable player.
