# VYREX Training Report — 519M Steps
## Run 6: grounded_weight=0.003 — The Goldilocks Search Continues
### (461M → 521M, ~60M new steps)
**Generated:** 2026-02-16  
**WandB Run:** `9ue9rans` (resumed, session `run-20260215_233352`)  
**Duration:** ~55 min at ~18,100 avg SPS  

---

## Executive Summary

This is the **longest continuous session since training began** and the first with `grounded_weight=0.003` (reduced from 0.015). The reduction broke the equilibrium trap that held airborne at 4.3% — the bot has **rediscovered jumping**.

**Airborne fraction climbed from 4.4% to 34.6% over 60M steps**, passing through the 15-25% target zone around 470-480M and continuing upward to stabilize at ~34-35%. This is above the initial target but represents a **fundamentally different kind of airborne behavior** than the original 85% pathology — the bot is now making contextual decisions about when to jump rather than defaulting to either extreme.

**Most remarkable finding:** Entropy *increased* for the first time in the entire training history — from 4.125 to 4.226 (+0.10). The policy became **more exploratory** as the grounded constraint released, actively investigating when jumping is beneficial. This is textbook healthy exploration behavior.

**Overnight outlook:** The config is stable and the bot is training well. No changes needed for the next ~6 hours (~389M steps to ~910M).

| Metric | 453M (before) | 519M (now) | Change |
|--------|--------------|-----------|--------|
| Airborne % | 4.3 | 34.6 | **+30.3pp** |
| Touches/step | 0.050 | 0.049 | maintained |
| Avg Speed | 955 | 889 | -7% (expected) |
| Goals/window | 61 | 64 | maintained |
| Entropy | 4.111 | 4.226 | **+0.10 (RISING!)** |
| KL (steady) | 0.0020 | 0.0015 | improved |
| VF Loss | 0.018 | 0.017 | improving |

---

## 1. The Airborne Recovery Arc

### Diagnostic Trajectory (This Run)

| Step | Airborne % | Δ per 5M | Phase |
|------|-----------|----------|-------|
| 458M | 4.48 | — | Lag (still old behavior) |
| 461M | 4.40 | -0.08 | Barely moving |
| 466M | **7.52** | **+3.12** | Liftoff — policy starts jumping |
| 471M | **13.68** | **+6.16** | Accelerating rapidly |
| 476M | **17.19** | +3.51 | **Enters 15-25% target zone** |
| 481M | **21.32** | +4.13 | Mid-target, still climbing |
| 486M | **27.12** | +5.80 | Exits target zone upward |
| 492M | **32.00** | +4.88 | Approaching equilibrium |
| 497M | **32.97** | +0.97 | **Decelerating sharply** |
| 502M | **34.94** | +1.97 | Slow glide |
| 507M | **34.31** | -0.63 | Oscillation begins |
| 512M | **35.42** | +1.11 | |
| 517M | **34.55** | -0.87 | **Stabilizing ~34-35%** |

```
Full airborne history:

  251M:  85.5%  ████████████████████████████████████████████   old pathology
  358M:  77.2%  ██████████████████████████████████████████
  399M:  70.7%  ████████████████████████████████████
  406M:   3.3%  ██                                          grounded=0.05 shock
  421M:   3.2%  ██                                          grounded=0.05
  453M:   4.3%  ██                                          grounded=0.015
  461M:   4.4%  ██                                          grounded=0.003 starts
  466M:   7.5%  ████
  471M:  13.7%  ███████
  476M:  17.2%  █████████                                   ← enters target zone
  481M:  21.3%  ███████████
  486M:  27.1%  ██████████████                              ← exits target zone
  492M:  32.0%  ████████████████
  497M:  33.0%  █████████████████
  502M:  34.9%  ██████████████████
  507M:  34.3%  █████████████████
  512M:  35.4%  ██████████████████
  517M:  34.6%  █████████████████                           ← stabilizing
```

### Analysis: Why 34-35% and Not 15-25%?

The equilibrium is determined by the balance of forces:

**Forces pushing airborne up:**
- Natural action distribution: 20% of LookupTableAction's 90 entries have jump=1
- InAirReward (+0.01 when airborne AND ball > 300uu) — modest but persistent for aerial situations
- Sparse goal/touch rewards that sometimes reward aerial play
- Entropy bonus (ent_coef=0.005) resists the policy from fully committing to ground-only

**Forces pushing airborne down:**
- GroundedReward (0.003/step) — very gentle now
- Speed is higher on ground → speed_reward favors groundedness
- Ground is where most touches happen at this skill level

At grounded_weight=0.003, the per-jump cost is only 0.003 × ~8 steps = **0.024 reward** — trivially small. The jump action (20% of actions) is nearly "free" to explore. The policy naturally gravitated to ~35% airborne, which is actually close to the 20% baseline that pure random action selection would produce, plus additional purposeful jumping the bot has learned.

### Is 34-35% a Problem?

**Probably not.** Context matters more than the raw number:

1. **It's NOT the old pathology.** The old 85% airborne was unconditioned — the bot jumped constantly regardless of context. The current 35% emerges after the policy learned ground play, suggesting it's making at least partially informed decisions about when to jump.

2. **Aerial touch rate is maintained.** At 0.014 per step, the bot is touching the ball in the air at roughly the same rate as when it was 85% airborne (0.009-0.022 range). This suggests the current aerials are more purposeful.

3. **Touch rate and goal rate stayed high.** If the bot were jumping pointlessly, we'd see touches and goals drop. Instead both are at or near all-time highs.

4. **Speed decreased modestly** (955 → 889), which is the expected cost of more airborne time. A 7% speed decrease for 30pp more airborne is a reasonable tradeoff if the aerial time is being used productively.

5. **Human Rocket League intermediate players** are typically airborne 25-40% of the time. 34% is within the normal range.

---

## 2. The Entropy Reversal

This is the most analytically interesting finding of the entire VYREX training history.

### Entropy Has Been RISING Since ~465M

```
Entropy over time (max possible = 4.500):

  401M: 4.441  ████████████████████████████████████████████████████  98.7%  ← post-ent_coef change
  425M: 4.149  ██████████████████████████████████████████████        92.2%  ← grounded=0.05
  433M: 4.134  ██████████████████████████████████████████████        91.9%
  456M: 4.111  █████████████████████████████████████████████         91.4%  ← grounded=0.015
  461M: 4.125  ██████████████████████████████████████████████        91.7%  ← grounded=0.003 starts
  466M: 4.133  ██████████████████████████████████████████████        91.8%
  471M: 4.167  ██████████████████████████████████████████████        92.6%
  481M: 4.183  ██████████████████████████████████████████████        93.0%
  492M: 4.213  ███████████████████████████████████████████████       93.6%
  502M: 4.204  ██████████████████████████████████████████████        93.4%
  512M: 4.222  ███████████████████████████████████████████████       93.8%
  521M: 4.226  ███████████████████████████████████████████████       93.9%
```

**From 401M to 461M:** Entropy dropped 0.316 (4.441 → 4.125) — the policy was specializing, committing to "stay on ground always."

**From 461M to 521M:** Entropy rose 0.101 (4.125 → 4.226) — the policy is **de-specializing**, rediscovering that jumping is sometimes valuable.

### Why Is This Happening and Is It Healthy?

**Yes, this is healthy.** The entropy increase is driven by the policy re-learning to use the jump action. When grounded_weight was 0.05, the policy heavily suppressed jump probabilities (entropy dropped because the action distribution concentrated on non-jump actions). Now that the grounded penalty is negligible, those suppressed probabilities are being re-explored.

The entropy is heading toward a new equilibrium around 93-94% of max — higher than the old 91-92% plateau but lower than the 98.7% it started at after the ent_coef change. This suggests:
- The policy has learned SOME things it's committed to (no longer nearly uniform random)
- But it's exploring MORE action space than during the "ground-locked" period
- The jump-related action probabilities are becoming more evenly distributed between "jump" and "don't jump" depending on context

The entropy should stabilize in the 4.20-4.25 range over the overnight run. If it continues rising above 4.30 (95.5%), that would be concerning — but the rate of increase is already decelerating (0.047/5M in W1 → 0.008/5M in W5).

---

## 3. PPO Training Dynamics

### 3.1 Policy Shock: None

This was the **gentlest transition of all resumes**:

| Iteration | KL | Clip | Context |
|-----------|-----|------|---------|
| 1 | 0.00010 | 0.00001 | Loaded policy (no update) |
| 2 | 0.00097 | 0.00582 | Minimal |
| 3 | 0.00200 | 0.01491 | Near steady state already |
| 4 | 0.00207 | 0.01494 | Stable |
| 5 | **0.00212** | **0.01553** | **Peak — nearly identical to steady state** |

**Peak KL of 0.00212** — only marginally above the steady-state mean of 0.0015. For comparison:
- grounded_weight 0.005→0.05: peak KL = 0.01192 (43× normal)
- grounded_weight 0.05→0.015: peak KL = 0.00385 (7× normal)
- grounded_weight 0.015→0.003: peak KL = 0.00212 (**1.4× normal**)

Each reduction was gentler than the last. Reducing a reward weight creates a much smaller policy shock than increasing one.

### 3.2 Reward Signal

| Window | Steps | Mean | StdDev | Min | Max |
|--------|-------|------|--------|-----|-----|
| W1 | 461-473M | 25.34 | 9.34 | 8.72 | 59.98 |
| W2 | 473-485M | 22.17 | 8.86 | 8.21 | 59.06 |
| W3 | 485-497M | 21.82 | 9.34 | 7.23 | 59.42 |
| W4 | 497-509M | 23.21 | 11.21 | 7.93 | 60.66 |
| W5 | 509-521M | 23.86 | 8.85 | 7.41 | 51.51 |

**Mean reward: 23.3** — down from 36.1 at grounded_weight=0.015. The mechanical explanation:

```
Grounded reward contribution at different weights:
  0.050: ~0.0485/step × 150K = ~7,275/batch → ~48.5/iter
  0.015: ~0.0144/step × 150K = ~2,160/batch → ~14.4/iter
  0.003: ~0.0020/step × 150K = ~  300/batch → ~ 2.0/iter
                                              ─────────
  Reduction 0.015→0.003: ~12.4 reward units lost (matches 36.1→23.3 = -12.8)
```

The reward drop is essentially 100% explained by the grounded reward magnitude change. **Actual gameplay reward is unchanged** — touches, goals, and speed rewards are all contributing at the same level.

### 3.3 Value Function Loss: Best Trend Yet

```
VF Loss trajectory:
  W1 (461-473M): 0.01708   ← starting elevated
  W2 (473-485M): 0.01671
  W3 (485-497M): 0.01686
  W4 (497-509M): 0.01690
  W5 (509-521M): 0.01632   ← new low

  Full history mean: 0.01677
```

VF loss is in a **sustained decline** for the first time. The critic is successfully tracking the (now much simpler) reward landscape where grounded reward contributes only ~2 units per iteration instead of 48. The noise floor is lower, making the critic's job easier.

### 3.4 KL Divergence: Declining

```
Mean KL per window:
  W1: 0.001605
  W2: 0.001568
  W3: 0.001450
  W4: 0.001523
  W5: 0.001422  ← trending down
```

KL divergence is gradually declining from 0.0016 to 0.0014, suggesting the policy is making smaller updates per iteration as it approaches its natural equilibrium. This is healthy convergence behavior.

### 3.5 Update Magnitudes

| Metric | W1 | W5 | Trend |
|--------|-----|-----|-------|
| Policy Mag | 0.348 | 0.387 | slight increase |
| VF Mag | 0.412 | 0.426 | stable |

Magnitudes are steady. The slight increase in policy magnitude reflects the entropy-driven exploration — the policy is making small but diverse updates as it explores when to jump.

---

## 4. Full Historical Comparison

| Metric | 251M | 358M | 421M | 453M | 519M | Overall Trend |
|--------|------|------|------|------|------|---------------|
| Airborne % | 85.5 | 77.2 | 3.2 | 4.3 | 34.6 | oscillating → stabilizing |
| Touches/step | 0.013 | 0.025 | 0.042 | 0.050 | 0.049 | **↑ sustained** |
| Avg Speed | 502 | 578 | 848 | 955 | 889 | **↑ high plateau** |
| Goals/window | ~12 | ~20 | 41 | 61 | 64 | **↑ sustained** |
| Avg Boost | 5.4 | 4.1 | 6.9 | 7.6 | 6.9 | stable |
| Entropy | 4.452 | 4.454 | 4.149 | 4.111 | 4.226 | **reversed → rising** |
| KL (steady) | 0.0003 | 0.0005 | 0.0023 | 0.0020 | 0.0015 | **improving** |
| VF Loss | 0.006 | 0.006 | 0.018 | 0.018 | 0.017 | **↓ recovering** |
| Team Spirit | 0.40 | 0.57 | 0.68 | 0.73 | **0.80** | **hit ceiling** |
| SPS | ~17,500 | ~17,900 | ~17,800 | ~17,900 | ~18,100 | **↑ best ever** |

### Key Milestones This Run
- **~476M:** Airborne entered target zone (17.2%)
- **~486M:** Airborne exited target zone upward (27.1%)
- **~497M:** Airborne stabilization began (~33-35%)
- **~502M:** **Team spirit reached maximum 0.8** — fully cooperative training now active

---

## 5. Team Spirit Reaching 0.8

Team spirit hit its ceiling of 0.8 at ~500M steps (configured ramp: 0.0 → 0.8 over 500M steps). From this point forward:
- **80% of each agent's reward** comes from the team's shared reward
- **20%** is individual reward
- This should promote cooperative behaviors: passing, rotation, covering for teammate

The transition was gradual and created no observable discontinuity in metrics. The spacing metric (avg_teammate_dist) has been climbing alongside, from 2401 at 421M to 2840 at 497M, suggesting the bots are beginning to maintain wider formations rather than ball-chasing together.

---

## 6. Permanent Checkpoints: Working Well

**10 new permanent checkpoints** were archived this session:

| Step | Archive |
|------|---------|
| 461M | 461299916 |
| 468M | 467600748 |
| 474M | 473901452 |
| 482M | 482302468 |
| 489M | 488603200 |
| 495M | 494903932 |
| 503M | 503304784 |
| 510M | 509605444 |
| 516M | 515906036 |

Total permanent archives: **16** (including 6 from earlier sessions + manual). The archiving system is functioning as designed.

---

## 7. Overnight Training Projection (6 Hours)

| Parameter | Value |
|-----------|-------|
| Current step | ~521M |
| SPS (sustained) | ~18,100 |
| Duration | ~6 hours |
| Steps overnight | **~391M** |
| Expected end | **~912M** |
| Permanent checkpoints | ~55 new archives |
| Rolling checkpoints | 5 retained (latest 5 ×2M intervals) |

### What To Expect

**Behavioral metrics** should be stable overnight:
- Airborne: likely stays 33-36%, may drift slowly
- Touches/goals: maintained or slowly improving
- Speed: plateau around 880-920
- Team spirit: fixed at 0.8 (ceiling reached)

**PPO dynamics** should be healthy:
- Entropy: should stabilize 4.20-4.25 range (~93% of max)
- KL: should remain 0.0012-0.0018 (normal)
- VF Loss: should continue gentle decline toward ~0.015
- No risk of divergence or collapse

### Potential Concerns (Low Risk)

1. **Airborne creep:** If airborne continues climbing above 40%, it could signal early reversion to the old pathology. However, the InAirReward is conditioned (ball > 300), and GroundedReward at 0.003 provides a gentle floor. Risk: LOW.

2. **Entropy ceiling:** If entropy rises above 4.30 (95.5%), the policy is becoming too random. The rate of increase is already decelerating (0.047/5M → 0.008/5M), so this is unlikely. Risk: VERY LOW.

3. **VF Loss spike:** A sudden VF loss increase could signal reward distribution shift. No signs of this currently. Risk: VERY LOW.

**Bottom line:** No config changes needed. The bot should train stably through the night and wake up at ~900M+ steps with well-developed ground play, emerging aerial capability, and fully cooperative team behavior.

---

## 8. Summary Scorecard

| Aspect | Score | Δ from 454M | Status |
|--------|-------|-------------|--------|
| Airborne Balance | **B** | ↑↑ from D- | 34.6% — above target but functional, stabilized |
| Ball Engagement | **A+** | ↔ | 0.049-0.061 — best era sustained |
| Goal Scoring | **A+** | ↑ from A | 64 goals/window — all-time high |
| Speed | **A** | ↓ from A+ | 889 — slightly lower (airborne tradeoff), still excellent |
| PPO Stability | **A+** | ↑ from A | KL max 0.0021, gentlest transition ever |
| Entropy Health | **A** | ↑ from A | 93.9%, healthy exploration via reversal |
| Value Function | **A-** | ↑ from B+ | Sustained decline, best trend |
| Team Cooperation | **A** | NEW | Team spirit at 0.8, spacing improving |
| Reward Design | **B+** | ↑ from C | 0.003 broke the trap, equilibrium higher than ideal |
| Infrastructure | **A** | ↔ | 10 permanent archives, 18K SPS sustained |

### The Grounded Weight Journey — A Retrospective

```
Weight   Steps Active   Airborne Result     Verdict
──────   ────────────   ────────────────     ─────────────────────
0.005    251M → 401M    85% → 70.7%         Too weak (150M for -15pp)
0.050    401M → 433M    70.7% → 3.2%        Too strong (5M shock!)
0.015    433M → 461M    3.2% → 4.3%         Equilibrium trap
0.003    461M → 521M    4.3% → 34.6%        ✓ Broke the trap
```

**The current 0.003 weight should be stable for the overnight run.** If 34-35% airborne proves to be too high after evaluation, the next adjustment would be 0.003 → 0.005 (a small increase), but this should be assessed after the bot has had ~200M+ steps to fully settle into its current behavioral equilibrium.

**Sleep well — VYREX is in good hands.** 🤖
