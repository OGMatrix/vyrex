<script setup lang="ts">
/**
 * TechStack — Architecture & Technology Showcase
 * Displays the technologies used with categorized cards.
 */
import { ref } from 'vue'
import { useScrollAnimation } from '../composables/useScrollAnimation'

const sectionRef = ref<HTMLElement>()
const isVisible = useScrollAnimation(sectionRef)

const categories = [
  {
    title: 'Training',
    items: [
      { name: 'PPO', desc: 'Proximal Policy Optimization — the industry-standard algorithm for game AI', accent: '#22d3ee' },
      { name: 'RLGym v2', desc: 'Gym-style API wrapping RocketSim for standardized RL training', accent: '#60a5fa' },
      { name: 'RocketSim', desc: 'Headless physics engine running 100-1000× real-time speed', accent: '#a78bfa' },
    ],
  },
  {
    title: 'Infrastructure',
    items: [
      { name: 'PyTorch', desc: 'Neural network backbone with TF32 auto-enabled for Ada Lovelace', accent: '#f87171' },
      { name: '20× Parallel', desc: 'Concurrent RocketSim environments across all CPU cores', accent: '#fbbf24' },
      { name: 'Conda + CUDA', desc: 'Reproducible environment with Python 3.11 and CUDA 12.1', accent: '#34d399' },
    ],
  },
  {
    title: 'Monitoring',
    items: [
      { name: 'Weights & Biases', desc: 'Real-time training dashboards, metric tracking, and run comparison', accent: '#fbbf24' },
      { name: 'Custom Diagnostics', desc: 'JSON snapshots every 5M steps with behavioral profiling', accent: '#22d3ee' },
      { name: 'Live Dashboard', desc: 'This website — SSE-powered live metrics and transparent versioning', accent: '#60a5fa' },
    ],
  },
  {
    title: 'Deployment',
    items: [
      { name: 'RLBot v5', desc: 'Deploy trained models directly into real Rocket League matches', accent: '#34d399' },
      { name: '[2048,2048,1024,1024]', desc: 'Battle-tested 4-layer network architecture for competitive bots', accent: '#a78bfa' },
      { name: '2v2 Native', desc: 'Team spirit curriculum produces agents that cooperate, not just compete', accent: '#f87171' },
    ],
  },
]
</script>

<template>
  <section id="stack" ref="sectionRef" class="stack section">
    <!-- Angled top -->
    <div class="stack__divider-top" aria-hidden="true"></div>

    <div class="container">
      <div class="stack__header scroll-reveal" :class="{ visible: isVisible }">
        <span class="badge badge--cyan">Under the Hood</span>
        <h2 class="heading-lg">
          Tech <span class="text-gradient">Stack</span>
        </h2>
        <p class="stack__desc">
          Built on proven technology used by every major Rocket League AI.
          No experimental frameworks — battle-tested tools tuned for maximum throughput.
        </p>
      </div>

      <div class="stack__categories stagger-children">
        <div
          v-for="(cat, ci) in categories"
          :key="ci"
          class="stack__category scroll-reveal"
          :class="{ visible: isVisible }"
        >
          <h3 class="stack__cat-title">{{ cat.title }}</h3>
          <div class="stack__items">
            <div
              v-for="(item, ii) in cat.items"
              :key="ii"
              class="stack__item glass-card"
              :style="{ '--item-accent': item.accent }"
            >
              <div class="stack__item-name" :style="{ color: item.accent }">
                {{ item.name }}
              </div>
              <div class="stack__item-desc">{{ item.desc }}</div>
              <div class="stack__item-glow" aria-hidden="true"></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Decorative shape -->
    <div class="stack__shape" aria-hidden="true"></div>
  </section>
</template>

<style scoped>
.stack {
  position: relative;
  background: var(--bg-base);
  overflow: hidden;
}

.stack__divider-top {
  position: absolute;
  top: -1px;
  left: 0;
  right: 0;
  height: 80px;
  background: var(--bg-raised);
  clip-path: polygon(0 0, 100% 0, 0 100%);
}

.stack__header {
  text-align: center;
  max-width: 650px;
  margin: 0 auto 3rem;
}

.stack__header h2 {
  margin-top: 0.75rem;
}

.stack__desc {
  margin-top: 1rem;
  color: var(--text-2);
  font-size: var(--fs-lg);
}

.stack__categories {
  display: grid;
  gap: 2.5rem;
}

.stack__cat-title {
  font-size: var(--fs-sm);
  font-weight: 600;
  color: var(--text-3);
  text-transform: uppercase;
  letter-spacing: var(--tracking-wider);
  margin-bottom: 0.75rem;
}

.stack__items {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 0.75rem;
}

.stack__item {
  position: relative;
  padding: 1.25rem 1.5rem;
  overflow: hidden;
}

.stack__item:hover {
  border-color: color-mix(in srgb, var(--item-accent) 30%, transparent);
}

.stack__item-name {
  font-family: var(--font-mono);
  font-size: var(--fs-sm);
  font-weight: 700;
  margin-bottom: 0.4rem;
}

.stack__item-desc {
  font-size: var(--fs-sm);
  color: var(--text-2);
  line-height: var(--leading-normal);
}

.stack__item-glow {
  position: absolute;
  top: -50%;
  right: -30%;
  width: 120px;
  height: 120px;
  background: radial-gradient(circle, var(--item-accent), transparent 70%);
  opacity: 0;
  transition: opacity var(--dur-normal) var(--ease);
  pointer-events: none;
}

.stack__item:hover .stack__item-glow {
  opacity: 0.06;
}

/* Decorative */
.stack__shape {
  position: absolute;
  bottom: -50px;
  right: -50px;
  width: 300px;
  height: 300px;
  background: var(--gradient-brand);
  opacity: 0.02;
  clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
  pointer-events: none;
}
</style>
