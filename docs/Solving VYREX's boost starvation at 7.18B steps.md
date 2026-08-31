# Solving VYREX's boost starvation at 7.18B steps

**The core problem is not reward weight — it's reward timing.** VYREX's BoostChangeReward fires only at the moment of pad pickup, but the *decision* to drive toward a pad occurs 5–20 steps earlier. Multiplying a temporally-displaced sparse signal by 7.5× makes the spike taller but doesn't help PPO bridge the credit assignment gap. Meanwhile, ball-related rewards like VelocityBallToGoalReward shape *every single tick*, creating a ~100× denser gradient signal for ball-chasing than for boost collection. The fix requires converting boost collection from a sparse event into a continuous dense signal — and no public Rocket League RL codebase has done this yet.

The research below draws on the Necto/Nexto codebase (Rolv-Arild), Lucy-SKG (arXiv:2305.15801), ZealanL's RLGym-PPO-Guide, the Seer thesis (ETH Zürich), RUDDER (NeurIPS 2019), Ng et al.'s PBRS (1999), and extensive community source analysis.

---

## What top bots actually do about boost (and what they don't)

**Necto/Nexto**, the community's most successful open-source bot (Diamond → Grand Champion), uses a surprisingly simple boost reward: `boost_gain_w=1.5` and `boost_lose_w=0.8`, both operating on per-step boost deltas. The lose penalty scales with car height — ground-level boost waste is penalized most, aerial usage is forgiven. Critically, Necto includes **all 34 boost pads as separate entities** in its observation space, processed through cross-attention. Each pad gets a position vector plus an is_active flag. This architectural choice — pads as first-class entities the attention mechanism can attend to — is likely why Necto learns boost pathing despite using only delta-based rewards. Even so, the Twitch clip titled "Nexto loves 0 boost" suggests imperfect boost management persists even at GC level.

**Lucy-SKG**, which beats both Necto and Nexto, takes a radically different approach: it **deliberately omits all boost pad entities from observations**. The authors argue pads are static objects the agent should discover implicitly. Lucy-SKG instead uses auxiliary prediction heads (state prediction, reward estimation) that regularize representations, plus a "Kinesthetic Reward Combination" that computes the geometric mean of reward components — forcing the agent to satisfy *all* objectives simultaneously to earn positive reward. No public code is available.

**ZealanL's RLGym-PPO-Guide** (the most authoritative community resource, written by Nexto's developer) recommends a `SaveBoostReward` using `sqrt(player.boost_amount)`, which makes **low boost disproportionately valuable** — the marginal reward of gaining 1% boost at 0% is much higher than at 50%. The guide explicitly warns: "Bots have a tendency to ignore small pads, so I recommend making the small pad pickup reward much stronger than just 12% of the big boost pickup reward." A custom reward distinguishing small pad pickups from big pad pickups is recommended as essential.

**No top bot uses an approach-based boost reward.** No `VelocityToNearestPadReward`, `SpeedTowardBoostReward`, or any continuous pad-proximity reward exists in any public repository — not in rlgym, rlgym-tools, Necto, or any community codebase found. This represents a gap in the ecosystem that directly causes the problem VYREX faces.

---

## Why BoostChangeReward is structurally broken

The fundamental issue is a **reward timing asymmetry**. Ball-oriented rewards provide continuous, every-tick gradient signal: VelocityPlayerToBallReward fires as the car moves toward the ball, VelocityBallToGoalReward fires as the ball rolls goalward. These rewards have **zero temporal displacement** — the decision to move toward the ball and the reward for doing so occur on the same tick. Boost collection rewards, by contrast, fire only at the instant of pad pickup, while the causal decision (turning toward the pad) happened **5–20 action steps earlier**.

With VYREX's configuration (γ=0.995, λ=0.95, 8-tick action hold), GAE preserves **(γλ)^n = 0.94525^n** of the signal across n steps. At 5 steps, **75%** of the signal survives — adequate. At 20 steps, only **32%** survives — marginal. But the deeper problem isn't the discount math; it's that **RUDDER (Arjona-Medina et al., NeurIPS 2019) formally proves TD methods require exponentially many updates to correct bias from delayed rewards**, where the exponent scales with delay length. The value function cannot reliably predict when boost pickups will occur, making GAE advantage estimates noisy and biased for boost-related state-action pairs.

Increasing the BoostChangeReward weight from 0.04 to 0.30 does not address this. A taller reward spike at the moment of pickup is still temporally displaced from the decision. The agent's value function still cannot attribute the pickup to the turn-toward-pad action that caused it. The signal-to-noise ratio for boost credit assignment remains poor regardless of weight. **This explains why the 7.5× weight increase produced zero measurable improvement.**

The `BoostChangeReward` implementation itself — rewarding `delta of sqrt(0.01 * boost_amount)` — compounds the issue through its sqrt transformation. Picking up a small pad (+12 boost) from zero generates a smaller reward than picking up a big pad (+100 boost), but the marginal utility curve means small pads have *lower absolute reward* despite being more abundant and strategically important for maintaining nonzero boost.

---

## The solution: potential-based reward shaping for pad proximity

**Potential-Based Reward Shaping (PBRS)** from Ng et al. (1999) allows adding F(s,s') = γ·φ(s') − φ(s) to any reward without changing the optimal policy — a mathematical guarantee, not a heuristic. The right potential function converts VYREX's delayed boost reward into an immediate, every-tick signal.

**Proposed potential function**: φ(s) = −min_distance_to_nearest_available_pad (or a weighted variant considering pad type and current boost level). When the car moves toward the nearest available pad, φ(s') > φ(s), producing a **positive reward on every step the agent approaches a pad**. When moving away, the signal is negative. This provides the same dense, continuous gradient that VelocityBallToGoalReward provides for ball play — but for boost collection.

Implementation considerations for VYREX:

- **Pad type weighting**: Use φ(s) = −min(d_big / w_big, d_small / w_small) where w_big and w_small weight big and small pads. Since small pads are more numerous and the agent ignores them, w_small should be higher than proportional (e.g., w_small = 0.8, w_big = 1.0 rather than w_small = 0.12).
- **Boost-conditional activation**: Only activate when boost is below a threshold (e.g., 30%). Above threshold, set φ(s) = 0 to avoid rewarding unnecessary pad detours. This prevents the "hoarding" behavior ZealanL warns about.
- **Pad availability**: φ must use pad timer data (already in VYREX's 34 boost_pad_timers observations) to only consider currently available pads.
- **Terminal state handling**: Set φ(terminal) = 0 for episodic training, required for the PBRS guarantee.
- **Discontinuity management**: When the nearest pad changes (collected or respawned), the potential function shifts. This is valid under PBRS theory as it's purely state-based, but may cause minor reward noise.

PBRS-MAXQ-0 (IJCAI 2015) tested this exact class of approach for resource collection problems and found it significantly outperformed baselines, even with imperfect heuristics. No Rocket League project has implemented this yet — it would be novel in the domain.

A concrete implementation skeleton:

```python
class BoostProximityPBRS(RewardFunction):
    def __init__(self, gamma=0.995, boost_threshold=0.3):
        self.gamma = gamma
        self.boost_threshold = boost_threshold
        self.prev_phi = {}

    def _phi(self, player, state):
        if player.boost_amount > self.boost_threshold:
            return 0.0
        min_dist = float('inf')
        for i, pad in enumerate(state.boost_pads):
            if pad.is_active:
                d = np.linalg.norm(player.car_data.position - pad.position)
                weight = 1.0 if pad.is_big else 0.8
                min_dist = min(min_dist, d / weight)
        return -min_dist / FIELD_DIAGONAL  # normalize

    def get_reward(self, player, state, previous_action):
        phi_now = self._phi(player, state)
        phi_prev = self.prev_phi.get(player.car_id, 0.0)
        self.prev_phi[player.car_id] = phi_now
        return self.gamma * phi_now - phi_prev
```

---

## Observation space gaps and what to add

VYREX's 132-feature DefaultObs includes the 34 boost_pad_timers, but these are **raw timer values in global coordinates** — the network must learn to cross-reference timer values with fixed pad positions from memory, then compute relative distances and directions. This is a substantial implicit computation burden for the policy network.

Necto solves this by treating each pad as a **separate entity with explicit position and availability**, processed through cross-attention. VYREX's feedforward architecture (2048→2048→1024→1024) cannot replicate this — it receives a flat feature vector and must learn spatial relationships purely from weights.

**Recommended observation additions** (add to a custom ObsBuilder extending DefaultObs):

- **Distance to nearest available small pad** (1 feature, normalized by field diagonal)
- **Distance to nearest available big pad** (1 feature, normalized)
- **Direction vector to nearest available pad** (3 features, unit vector in car-relative frame)
- **Current boost amount in [0,1]** (already included, but verify normalization)
- **Number of available pads within radius R** (1 feature, e.g., R = 2000 unreal units)

These 6 features make the boost-proximity information *explicit* rather than requiring the network to implicitly compute it from 34 timer values and memorized pad positions. This directly supports both the PBRS reward (the agent can see what the reward is shaping toward) and general boost-aware decision-making.

An even stronger approach would follow Necto's entity-based architecture: represent each pad as a (position, is_available, is_big) tuple and process them through attention. However, this requires architectural changes beyond observation engineering.

---

## Curriculum training and auxiliary tasks worth trying

**Boost-focused curriculum phases** are well-supported by the literature. Bansal et al. (2018) demonstrated curriculum learning that interpolates between dense exploration rewards and sparse competitive rewards, with exploration rewards gradually annealed. OpenAI Five's Dota 2 agent explicitly included resource collection rewards and noted these were "important for successful training" — not optional.

A practical curriculum for VYREX:

1. **Phase 1 (boost collection)**: Spawn with 0 boost, disable ball interaction, reward only boost collection with the PBRS approach-based signal. Train for ~100M steps until the agent reliably collects pads.
2. **Phase 2 (boost + ball)**: Re-enable ball, use full reward stack with PBRS boost shaping at high weight. Train until boost time-at-zero drops below 40%.
3. **Phase 3 (full competitive)**: Gradually reduce explicit boost reward weight as competitive rewards dominate. The PBRS guarantee means the reduction doesn't change the optimal policy — it only removes the learning scaffold.

**Auxiliary prediction heads** following Lucy-SKG's approach could further help. Adding a small head that predicts "distance to nearest available pad" or "will I collect a pad in the next K steps" forces the network to build boost-relevant internal representations. Lucy-SKG's auxiliary state prediction and reward estimation tasks were a key contributor to its SOTA performance over Necto/Nexto.

**Risk note**: A CS 238 project attempted curriculum learning for Rocket League kickoffs and reported negative results. However, their curriculum design was fundamentally different (not boost-focused), and their failure likely reflects poor curriculum structure rather than a fundamental limitation.

---

## Self-play creates a boost-devaluing equilibrium

Self-play can stabilize at a **local Nash equilibrium where neither agent collects boost**. The mechanism: if Agent A doesn't collect boost, Agent B faces no aerial threats, fast challenges, or long-range plays that require boost to counter. Agent B therefore has no competitive pressure to collect boost. The game degenerates into a slow, ground-based contest where boost is irrelevant — a stable but suboptimal equilibrium.

This is a well-documented phenomenon. The JMLR 2025 paper on GFXP proves that self-play can get stuck in local Nash equilibria and that any algorithm with "preference preservation" will not converge to the global NE with high probability. DeepMind's FTW and AlphaStar explicitly addressed this through population-based training and league play with diverse opponents.

**Three practical interventions for VYREX:**

- **Mixed opponent pools**: Train against checkpoints from different training stages. Include some checkpoints that DO use boost effectively (even from early training where boost behavior was better). This creates games where VYREX experiences being "out-boosted."
- **Asymmetric reward injection**: Give VYREX's opponent a scripted boost-collection behavior for some fraction of training games. This forces VYREX to deal with a boost-rich opponent and learn that boost matters competitively.
- **Intrinsic boost reward that self-play cannot eliminate**: The PBRS approach-based reward provides gradient signal for boost collection regardless of opponent behavior, breaking the symmetry that allows the boost-devaluing equilibrium to form.

---

## Concrete action plan ranked by expected impact

**Tier 1 — Implement immediately (highest expected impact):**

1. **Add a VelocityToNearestPadReward** using the PBRS formulation. This converts the 5–20 step delayed signal into a per-tick dense signal, directly addressing the temporal credit assignment failure. Weight it at ~0.01–0.02 (continuous rewards need low weights per ZealanL's guidance). This reward does not exist in any public codebase and must be built custom, adapting the `SpeedTowardBallReward` template from rlgym's official docs.

2. **Add SaveBoostReward with sqrt(boost_amount)** at weight ~0.1. This provides continuous pressure to *maintain* nonzero boost, complementing the approach-based signal. The sqrt makes the 0→12 boost range extremely high-gradient.

3. **Amplify small pad pickup rewards 4–5×** relative to proportional value. ZealanL explicitly recommends this as essential. Small pads are everywhere but bots systematically ignore them.

**Tier 2 — Implement after Tier 1 shows progress:**

4. **Add 6 explicit observation features**: nearest small pad distance, nearest big pad distance, direction-to-nearest-pad vector (3D), and available-pads-within-radius count. These reduce the implicit computation burden on the feedforward network.

5. **Reduce or remove the existing BoostChangeReward**. Once PBRS provides continuous approach signal and SaveBoostReward maintains boost-having pressure, the sparse delta-based reward becomes redundant noise.

6. **Introduce opponent diversity** in self-play. Mix in historical checkpoints and, if possible, scripted opponents with boost-collecting behavior.

**Tier 3 — Advanced techniques if Tier 1–2 are insufficient:**

7. **Boost-focused curriculum phase**: 100M steps of boost-only training with 0-boost spawn states before resuming full competitive training.

8. **Auxiliary prediction head** for "steps until next pad collection" — forces the network to build boost-aware representations, following Lucy-SKG's approach.

9. **Increase GAE λ from 0.95 to 0.97–0.98** to improve long-horizon credit assignment for any remaining sparse signals. This adds variance but helps the 15–20 step delay range.

10. **RUDDER-style reward redistribution** as a last resort — train an LSTM to identify which state-action pairs actually cause boost collection and redistribute the reward accordingly. This directly solves the temporal credit assignment problem but adds significant implementation complexity.

## Conclusion

VYREX's 75% zero-boost time is not a reward weight problem — it's a **reward architecture problem**. The entire Rocket League RL community relies on sparse, event-based boost rewards while providing dense, continuous rewards for ball play. This creates an order-of-magnitude signal asymmetry that PPO cannot overcome through weight scaling alone. The PBRS approach-based reward for pad proximity is the highest-leverage intervention: it's theoretically guaranteed to preserve the optimal policy, provides the same per-tick density as ball rewards, has precedent in resource collection RL (PBRS-MAXQ-0), and no public Rocket League codebase has implemented it yet. Combined with SaveBoostReward's sqrt-based continuous pressure and amplified small-pad pickup rewards, this three-part fix addresses the temporal credit assignment gap, the signal density asymmetry, and the small-pad neglect problem simultaneously.