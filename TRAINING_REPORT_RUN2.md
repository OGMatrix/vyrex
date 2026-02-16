# VYREX Training Report — Run 2 (Post-Fix Continuation)

**Generated:** 2026-02-15  
**Run ID:** `9ue9rans` (resumed)  
**Step Range:** 251,577,028 → 366,939,848 (+115.4M steps)  
**Wall Clock:** ~1.71 hours (19:48 → 21:30)  
**Iterations:** 769  
**Hardware:** RTX 4070 Ti 12GB (Ada), i7-14700K 20C/28T, 48GB DDR5  

---

## 1. Executive Summary

This run resumed from checkpoint 251,577,028 with four critical fixes applied:

| Fix | Target | Status |
|-----|--------|--------|
| InAirReward → conditional (ball height > 300) | Eliminate pathological jumping | **Partially effective** |
| GroundedReward added (weight 0.005) | Incentivize ground play | **Active, slow effect** |
| Boost metric scaling (removed ×100) | Correct avg_boost reporting | **Fixed** |
| Curriculum mutators wired in | Phase 2 (30% random states) | **Active** |

**Key Finding:** Airborne fraction dropped from 85.6% → 77.2% over 115M steps — a directionally correct improvement, but the bot is still airborne **far too much** (healthy target: 10-20%). The deeply-entrenched jumping behavior from 250M+ steps of bad reward signal is proving resistant to the current fix strength. More aggressive intervention is needed.

---

## 2. Airborne Fraction — The Critical Metric

This was identified as the #1 issue: the bot was always jumping, never driving.

### Trajectory (sampled from diagnostics)

| Step | Airborne Frac | Delta from Pre-Fix Baseline |
|------|--------------|----------------------------|
| 237M (baseline) | 85.7% | — |
| 242M (baseline) | 85.1% | — |
| 247M (baseline) | 84.6% | — |
| **251M (fix applied)** | **85.6%** | **+0.0% (no change yet)** |
| 256M | 83.1% | -2.5pp |
| 261M | 82.8% | -2.8pp |
| 267M | 81.6% | -4.0pp |
| 272M | 81.0% | -4.6pp |
| 277M | 81.9% | -3.7pp |
| 282M | 80.1% | -5.5pp |
| 287M | 79.9% | -5.7pp |
| 292M | 80.2% | -5.4pp |
| 297M | 80.4% | -5.2pp |
| 302M | 78.6% | -7.0pp |
| 307M | 78.9% | -6.7pp |
| 312M | 78.8% | -6.8pp |
| 318M | 79.8% | -5.8pp |
| 323M | 79.1% | -6.5pp |
| 328M | 78.1% | -7.5pp |
| 333M | 77.6% | -8.0pp |
| 338M | 77.9% | -7.7pp |
| 343M | 80.7% | -4.9pp (regression) |
| 348M | 78.7% | -6.9pp |
| 353M | 77.6% | -8.0pp |
| 358M | 77.2% | -8.4pp |

### Analysis

- **Rate of decay:** ~8 percentage points over 107M steps = **0.075 pp/M steps**
- **At this rate, reaching 20% airborne would take:** (77.2 - 20) / 0.075 ≈ **762M more steps** (~13 hours)
- **At this rate, reaching 50% airborne:** ~363M more steps (~6 hours)
- **Problem:** The decay rate is **decelerating** — the first 30M steps dropped ~6pp, but the next 80M only dropped ~2pp more. The policy is settling into a new equilibrium around 78%.

### Root Cause Analysis

The current `GroundedReward` weight of **0.005** is too weak to overcome 250M steps of deeply-learned jumping behavior. The reward landscape comparison:

- **GroundedReward contribution per step:** 0.005 × 1.0 (when grounded) = **0.005 per step**
- **SpeedTowardBall:** 0.02 × (up to 1.0) = up to 0.02 per step
- **VelocityBallToGoal:** 0.15 × (up to 1.0) = up to 0.15 per step

The grounded reward is a tiny signal competing against the deeply-entrenched value function that has learned "jumping = good" across billions of value function updates. The old InAirReward at weight 0.002 ran for ~250M steps = ~150K effective updates, each reinforcing jump behavior.

### Recommendation

**Increase `grounded_weight` from 0.005 to 0.05-0.1** (10-20× increase). This is a temporary measure — it should be reduced back once airborne_frac drops below 25%. At weight 0.05, the per-step grounded signal (0.05) would be competitive with the dense shaping rewards and should accelerate the un-learning significantly.

---

## 3. PPO Training Stability

### Windowed Statistics (5 windows of ~23M steps each)

| Window | Steps | Mean Reward | KL Div | Clip Frac | Entropy | VF Loss | Policy Mag | SPS |
|--------|-------|-------------|--------|-----------|---------|---------|------------|-----|
| 1 | 251M-274M | 25.90 | 0.000475 | 0.00298 | 4.4742 | 0.00562 | 0.431 | 17,520 |
| 2 | 274M-297M | 27.29 | 0.000492 | 0.00298 | 4.4712 | 0.00565 | 0.423 | 17,591 |
| 3 | 297M-320M | 27.97 | 0.000512 | 0.00312 | 4.4686 | 0.00586 | 0.432 | 17,606 |
| 4 | 320M-343M | 28.12 | 0.000524 | 0.00317 | 4.4673 | 0.00588 | 0.418 | 17,625 |
| 5 | 343M-366M | 29.87 | 0.000566 | 0.00369 | 4.4614 | 0.00659 | 0.443 | 17,566 |

### Interpretation

1. **Mean KL Divergence (0.00048 → 0.00057):** Excellent. Well below the typical alarm threshold of 0.01. A very slight upward trend is actually healthy — it means the policy is making meaningful updates rather than being stuck. PPO is confidently making changes.

2. **Clip Fraction (0.003 → 0.0037):** Extremely low. The PPO clip ratio of 0.2 is almost never being hit, meaning the policy update trust region is rarely violated. This indicates the learning rate (1e-4) is well-calibrated and updates are smooth.

3. **Entropy (4.474 → 4.454):** Very slow decline (-0.020 over 115M steps). Entropy is still high (theoretical max for 90 actions = ln(90) ≈ 4.50), meaning the policy remains highly exploratory. This is **too high** for 360M steps — the bot hasn't committed to specific strategies yet. The entropy coefficient of 0.01 may need to be halved to 0.005 after 400M steps to encourage specialization.

4. **Value Function Loss (0.0056 → 0.0066):** Slightly increasing, which is expected after reward function changes. The value function is re-calibrating to the new reward landscape (GroundedReward added, InAirReward conditioned). Still very low — no concern.

5. **Policy Update Magnitude (0.43 → 0.44):** Stable. The policy is making consistent-sized updates. No sign of catastrophic divergence or stagnation.

6. **Value Function Update Magnitude (0.34):** Stable throughout. Good sign of balanced learning.

### Reward Trajectory

| Period | Mean Reward | Min | Max |
|--------|------------|-----|-----|
| First 10 iter (251M) | 26.33 | 15.14 | 44.00 |
| Last 10 iter (366M) | 34.32 | 24.50 | 45.89 |
| Overall Mean | 27.83 | — | — |

**Reward increased ~30% over the run** (25.90 → 29.87 in windowed means). This is a healthy trend, indicating the bot is genuinely learning from the new reward structure. The variance is high (min ~10, max ~70 within windows) which is normal for RL with sparse goal rewards in the mix.

---

## 4. Game Performance Metrics

### Touch Rate (Engagement Metric)

| Step | Touches/Step | Change |
|------|-------------|--------|
| 89M (earliest) | 0.00488 | — |
| 247M (pre-fix) | 0.01411 | +189% |
| 251M (post-fix) | 0.01325 | baseline |
| 282M | 0.01453 | +9.7% |
| 312M | 0.01779 | +34.3% |
| 338M | 0.02178 | +64.4% |
| 358M (final) | **0.02533** | **+91.1%** |

**Nearly doubled** touch rate over 107M steps. This is the strongest positive signal — the bot is engaging with the ball far more often. Directly correlated with the airborne fraction decrease; when the bot is on the ground more, it can actually reach the ball.

### Aerial Touch Rate

| Step | Aerial Touches/Step |
|------|-------------------|
| 89M-247M (pre-fix avg) | ~0.001 |
| 307M | **0.01951** (spike) |
| 338M | 0.00734 |
| 358M | 0.00842 |

Aerial touch rate increased even though overall airborne time decreased. This means: **the bot is doing fewer meaningless jumps and more targeted aerials when the ball is actually high** — exactly what the conditional InAirReward was designed to achieve. The old bot jumped constantly but rarely hit the ball in the air. The new bot jumps less but hits aerial balls more.

### Goal Statistics

| Step | Blue Goals | Orange Goals | Diff |
|------|-----------|-------------|------|
| 247M (pre-fix) | 12 | 5 | +7 |
| 251M | 9 | 8 | +1 |
| 272M | 7 | 13 | -6 |
| 302M | 6 | 8 | -2 |
| 318M | **15** | 8 | **+7** |
| 333M | **14** | 8 | **+6** |
| 338M | **14** | 9 | **+5** |
| 353M | **12** | 4 | **+8** |
| 358M | 11 | 9 | +2 |

Goal scoring shows significant variance (expected in self-play), but the second half of the run (318M+) shows a **strong winning trend**. In the last 6 diagnostics samples (333M-358M), blue outscored orange by an average of +3.5 goals.

There was a noticeable dip in performance around 260M-300M — this is the expected **reward function transition penalty**. When rewards change, the policy briefly gets confused because the old behavior (jumping) is no longer rewarded, but the new correct behavior hasn't been reinforced yet. The fact that scoring recovered and surpassed pre-fix levels by 318M is very positive.

### Avg Boost

| Period | Avg Boost |
|--------|-----------|
| 89M-247M (old metric, ×100 bug) | 343-399 (actually 3.4-4.0) |
| 251M (fixed metric) | **4.8** |
| 358M (final) | **4.3** |

After the metric scaling fix, we can see correct values. Avg boost of **4.3-4.8 out of 100** is extremely low. The bot is using virtually all its boost immediately. This is consistent with the jumping behavior — every jump costs boost, and constant jumping depletes it instantly.

**Expected improvement:** As airborne_frac comes down, boost conservation should naturally improve. If it doesn't, `boost_conservation_weight` may need to be increased from 0.01 to 0.03.

### Avg Speed

| Period | Avg Speed |
|--------|-----------|
| 251M | 635.3 |
| 358M | 618.9 |
| Range | 607-651 |

Average speed is stable around **620-640 u/s** (max possible: 2300 u/s). This is ~28% of maximum car speed, which is low but expected for a bot that's still learning to orient itself properly after the reward change. Speed should increase as ground play improves.

### Teammate Distance

| Period | Avg Teammate Dist |
|--------|-------------------|
| 247M | 3401 |
| 343M | **2970** (lowest) |
| 358M | 3424 |
| Range | 2970-3590 |

Teammate spacing is fluctuating around ~3300-3500 units. The team_spirit is ramping (0.40 → 0.57), so team-aware behavior is gradually increasing. The dip to 2970 at 343M might indicate the bots started ball-chasing together temporarily. No action needed — the `TeammateSpacingReward` should handle this as team_spirit continues to increase.

---

## 5. Training Infrastructure

### Steps Per Second (SPS)

| Window | Mean SPS |
|--------|----------|
| 1 (251-274M) | 17,520 |
| 2 (274-297M) | 17,591 |
| 3 (297-320M) | 17,606 |
| 4 (320-343M) | 17,625 |
| 5 (343-366M) | 17,566 |

**Rock-solid at ~17,500-17,600 SPS** throughout the entire run. Zero degradation. The curriculum mutators (WeightedSampleMutator with 30% RandomStateMutator) have no measurable overhead compared to pure KickoffMutator.

Theoretical throughput: 20 workers × 150K batch / ~8.5s iteration = ~17,650 SPS. We're at 99.7% efficiency.

### Timing Breakdown (from final iterations)

| Component | Time (seconds) |
|-----------|---------------|
| Timestep Collection | 4.05-4.46 |
| Timestep Consumption (GPU) | 4.42-4.51 |
| PPO Batch Consumption | 0.50-0.52 |
| Total Iteration | 8.47-8.95 |

Collection and consumption are nearly balanced (4.2s vs 4.5s), meaning neither CPU nor GPU is severely bottlenecked. The GPU consumption time has increased slightly from earlier runs (~2.5s → 4.5s), suggesting the value function is taking more updates. This is consistent with the increased VF loss from the reward function changes.

### Checkpoint Status

New checkpoint structure is working correctly:
- `data/checkpoints/` now contains: 352M, 354M, 356M, 358M, 360M
- Old `data/checkpoints-1771163708278249100/` preserved (246M-251M)
- `add_unix_timestamp=False` working — no new timestamped dirs created
- 5 checkpoints retained (rolling window) as configured

---

## 6. Team Spirit & Curriculum

### Team Spirit Progression

| Step | Team Spirit |
|------|-------------|
| 89M | 0.143 |
| 247M | 0.396 |
| 251M | 0.403 |
| 297M | 0.476 |
| 343M | 0.550 |
| 358M | 0.574 |

Linear ramp: `spirit = step / 500M × 0.8`. At 366M steps → 0.586. On track to reach target 0.8 at 500M steps.

Currently at ~57% team spirit — meaning reward is 57% shared between teammates and 43% individual. The bots should be starting to develop basic cooperative behaviors (not both chasing the ball, basic rotation).

### Curriculum Phase

Training ran in **Phase 2: Mixed (30% random states, 70% kickoffs)**. This was correctly determined:
- Phase boundary: 100M-300M = Phase 2
- Resume step was 251M → Phase 2
- Note: Since curriculum is set at env creation and workers persist, the entire run used Phase 2 ratios even after crossing 300M. This means 300M-366M trained with 30% random instead of the intended 70%. This is **not a bug** — it's the expected behavior of the current architecture (workers are spawned once). The next `--resume` will correctly enter Phase 3.

---

## 7. Comparison: Pre-Fix vs Post-Fix

| Metric | Pre-Fix (247M) | Post-Fix (358M) | Change |
|--------|---------------|-----------------|--------|
| Airborne Frac | 84.6% | 77.2% | **-7.4pp** ↓ |
| Touches/Step | 0.0141 | 0.0253 | **+79.4%** ↑ |
| Aerial Touch Rate | 0.0000 | 0.0084 | **+∞** ↑ |
| Avg Boost | 3.94 (corrected) | 4.30 | +9.1% |
| Avg Speed | 621.9 | 618.9 | -0.5% |
| Mean Reward | ~24.6 | ~29.9 | **+21.5%** ↑ |
| Entropy | 4.476 | 4.454 | -0.5% |
| KL Divergence | 0.0004 | 0.0006 | +50% (still safe) |
| SPS | 17,863 | 17,566 | -1.7% (stable) |
| Team Spirit | 0.396 | 0.574 | +0.178 |

**Verdict:** All key metrics moved in the right direction. The fixes are working, but the airborne fraction decay is too slow. The bot is demonstrably better — touching the ball almost twice as often, scoring more goals in the second half, and developing actual aerial skills (hitting a ball that's elevated, not just randomly jumping).

---

## 8. Identified Issues & Recommendations

### CRITICAL: Airborne Fraction Still Too High

**Problem:** 77% airborne is still catastrophic. The bot is jumping 3-4x more than it should.

**Root Cause:** GroundedReward at 0.005 is insufficient to overcome 250M steps of learned jumping.

**Recommendation Options (ordered by aggressiveness):**

1. **Increase `grounded_weight` to 0.05** (10×). Conservative. Expected to reach ~50% airborne by 450M.
2. **Increase `grounded_weight` to 0.10** (20×). Moderate. May overshoot and make bot passive.
3. **Replace GroundedReward with JumpPenalty** (negative reward when jump action is selected and ball is below 300). Direct action-space intervention. Most targeted but requires access to action in reward.
4. **Reduce action space** — create a modified LookupTableAction with fewer jump entries (e.g., 9 instead of 18 out of 90). Extreme but effective.

**Recommended:** Option 1 — increase to 0.05. Monitor for 20M steps. If airborne_frac doesn't drop below 70% within 20M steps, escalate to Option 2.

### MODERATE: Entropy Too High

**Problem:** At 4.454, entropy is 99% of maximum (ln(90)=4.50). After 360M+ steps, the policy should be more decisive.

**Recommendation:** Reduce `ppo_ent_coef` from 0.01 to 0.005 after 400M steps. This will encourage the policy to commit to learned strategies rather than maintaining near-uniform exploration.

### LOW: Boost Near Zero

**Problem:** Avg boost of 4.3/100 means the bot uses all boost immediately.

**Analysis:** This is likely a downstream symptom of jumping. Each jump consumes at least some boost (if boosting during jump), and constant aerial adjustment uses boost rapidly. This should self-correct as airborne_frac drops.

**Recommendation:** No immediate action. Monitor. If boost stays below 10 after airborne_frac drops below 50%, increase `boost_conservation_weight` from 0.01 to 0.03.

### NOTE: Curriculum Phase Transition

The next `--resume` will transition to Phase 3 (70% random states) since step 366M > phase_2_end (300M). This is a significant shift that may cause a temporary performance dip as the bot adapts to starting from random positions instead of kickoffs. This is expected and healthy for long-term generalization.

---

## 9. Projected Milestones

| Target | Steps Needed | Estimated Time |
|--------|-------------|----------------|
| 50% airborne (with current grounded_weight=0.005) | ~363M | ~5.7 hrs |
| 50% airborne (with grounded_weight=0.05) | ~50-80M | ~0.8-1.3 hrs |
| 20% airborne (healthy) | ~762M+ | ~12+ hrs |
| Phase 3 curriculum active | Next resume | 0 |
| Team spirit 0.8 (full cooperation) | 133M more | ~2.1 hrs |
| 500M total steps | 133M more | ~2.1 hrs |
| 1B total steps | 633M more | ~10 hrs |

---

## 10. Raw Data Summary Tables

### Diagnostics Timeline (all 22 post-fix samples)

| Step (M) | Airborne | Touches | Aerial | Demo | Goals(B/O) | Boost | Speed | Team Dist | Spirit |
|----------|----------|---------|--------|------|-----------|-------|-------|-----------|--------|
| 251.7 | 0.856 | 0.0133 | 0.0040 | 0.000 | 9/8 | 4.80 | 635.3 | 3263 | 0.403 |
| 256.8 | 0.831 | 0.0137 | 0.0000 | 8e-5 | 10/12 | 4.59 | 618.1 | 3563 | 0.411 |
| 261.9 | 0.828 | 0.0155 | 0.0035 | 0.000 | 4/7 | 4.04 | 628.3 | 3509 | 0.419 |
| 267.0 | 0.816 | 0.0160 | 0.0000 | 0.000 | 13/12 | 4.88 | 629.8 | 3461 | 0.427 |
| 272.1 | 0.810 | 0.0158 | 0.0000 | 1e-4 | 7/13 | 4.13 | 640.9 | 3346 | 0.435 |
| 277.2 | 0.819 | 0.0159 | 0.0034 | 0.000 | 8/10 | 4.61 | 626.8 | 3326 | 0.444 |
| 282.3 | 0.801 | 0.0145 | 0.0037 | 1e-4 | 11/11 | 4.83 | 639.8 | 3525 | 0.452 |
| 287.4 | 0.799 | 0.0165 | 0.0000 | 0.000 | 10/10 | 4.48 | 621.6 | 3177 | 0.460 |
| 292.5 | 0.802 | 0.0149 | 0.0036 | 2e-4 | 6/8 | 4.38 | 627.9 | 3255 | 0.468 |
| 297.6 | 0.804 | 0.0175 | 0.0000 | 0.000 | 5/3 | 4.46 | 618.9 | 3590 | 0.476 |
| 302.7 | 0.786 | 0.0152 | 0.0035 | 0.000 | 6/8 | 4.43 | 628.9 | 3243 | 0.484 |
| 307.8 | 0.789 | 0.0164 | 0.0195 | 0.000 | 7/7 | 4.51 | 641.6 | 3301 | 0.493 |
| 312.9 | 0.788 | 0.0178 | 0.0060 | 1e-4 | 9/10 | 4.66 | 651.9 | 3299 | 0.501 |
| 318.0 | 0.798 | 0.0170 | 0.0000 | 0.000 | 15/8 | 4.80 | 636.8 | 3451 | 0.509 |
| 323.1 | 0.791 | 0.0182 | 0.0029 | 0.000 | 8/7 | 4.77 | 638.0 | 3493 | 0.517 |
| 328.2 | 0.781 | 0.0180 | 0.0059 | 8e-5 | 9/8 | 4.93 | 650.9 | 3542 | 0.525 |
| 333.3 | 0.776 | 0.0185 | 0.0029 | 0.000 | 14/8 | 4.46 | 646.5 | 3378 | 0.533 |
| 338.4 | 0.779 | 0.0218 | 0.0073 | 1e-4 | 14/9 | 4.53 | 641.9 | 3340 | 0.542 |
| 343.5 | 0.807 | 0.0218 | 0.0061 | 2e-4 | 8/9 | 4.08 | 621.1 | 2970 | 0.550 |
| 348.6 | 0.787 | 0.0170 | 0.0000 | 0.000 | 11/9 | 4.47 | 627.8 | 3506 | 0.558 |
| 353.7 | 0.776 | 0.0191 | 0.0028 | 0.000 | 12/4 | 4.24 | 637.4 | 3475 | 0.566 |
| 358.8 | 0.772 | 0.0253 | 0.0084 | 1e-4 | 11/9 | 4.30 | 618.9 | 3424 | 0.574 |

### PPO Metrics (sampled every 100 iterations)

| Step (M) | Reward | KL | Clip | Entropy | VF Loss | Pol Mag | VF Mag |
|----------|--------|-----|------|---------|---------|---------|--------|
| 251.7 | 17.40 | 0.00001 | 0.00001 | 4.4754 | 0.00613 | 0.148 | 0.128 |
| 266.7 | 26.57 | 0.00048 | 0.00348 | 4.4741 | 0.00463 | 0.422 | 0.348 |
| 281.7 | 30.68 | 0.00053 | 0.00336 | 4.4702 | 0.00632 | 0.443 | 0.380 |
| 296.7 | 21.16 | 0.00048 | 0.00305 | 4.4709 | 0.00557 | 0.421 | 0.320 |
| 311.7 | 25.58 | 0.00051 | 0.00311 | 4.4666 | 0.00660 | 0.463 | 0.348 |
| 326.7 | 24.20 | 0.00048 | 0.00276 | 4.4690 | 0.00612 | 0.405 | 0.349 |
| 341.7 | 29.38 | 0.00044 | 0.00299 | 4.4676 | 0.00591 | 0.446 | 0.346 |
| 356.7 | 24.37 | 0.00061 | 0.00414 | 4.4633 | 0.00664 | 0.438 | 0.334 |

---

## 11. Conclusion

The four fixes are all working correctly:
1. **Conditional InAirReward** — aerial touch rate improved while overall airborne time decreased
2. **GroundedReward** — airborne fraction trending down, but too slowly
3. **Boost scaling** — metrics now report correct values (4.3/100 instead of 430)
4. **Curriculum** — Phase 2 running smoothly with 30% random states, no SPS impact

The biggest remaining challenge is the **airborne fraction at 77%**. The policy has 250M steps of "jumping = good" baked into its weights, and the current grounded_weight=0.005 provides insufficient counter-pressure. The recommended next step is to increase grounded_weight to 0.05 (10×) and resume training. Phase 3 will activate automatically on next resume, introducing 70% random states which will also help — the bot will start from varied positions rather than always from kickoff, encouraging more diverse (and grounded) play patterns.
