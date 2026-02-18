<script setup lang="ts">
/**
 * VisionSection — Project Vision & Goals
 * Static content showcasing what VYREX is and why it exists.
 */
import { ref } from 'vue'
import { useScrollAnimation } from '../composables/useScrollAnimation'

const sectionRef = ref<HTMLElement>()
const isVisible = useScrollAnimation(sectionRef)

const features = [
  {
    icon: '🧠',
    title: 'Self-Learning',
    description: 'Trained entirely through self-play using Proximal Policy Optimization. No human demonstrations, no scripted behavior — pure emergent intelligence.',
  },
  {
    icon: '⚡',
    title: 'Superhuman Speed',
    description: 'RocketSim runs 100-1000× faster than real-time. VYREX processes 18,000+ game steps per second across 20 parallel environments.',
  },
  {
    icon: '🤝',
    title: 'Team Intelligence',
    description: 'Built for 2v2 from day one. Team spirit curriculum teaches individual mechanics first, then gradually shapes cooperative rotation and passing.',
  },
  {
    icon: '🔬',
    title: 'Transparent Process',
    description: 'Every training decision is documented. Every config change is versioned. Every metric is tracked. Full transparency into the development journey.',
  },
]
</script>

<template>
  <section id="vision" ref="sectionRef" class="vision section">
    <div class="container">
      <div class="vision__header scroll-reveal" :class="{ visible: isVisible }">
        <span class="badge badge--cyan">The Vision</span>
        <h2 class="heading-lg">
          Building a <span class="text-gradient">devastating</span> 2v2 AI
        </h2>
        <p class="vision__desc">
          VYREX isn't just another Rocket League bot. It's a deep reinforcement learning agent
          designed to master the most complex team mode — 2v2 — through billions of steps of
          self-play simulation. Every decision, every aerial, every rotation emerges from pure learning.
        </p>
      </div>

      <div class="vision__grid stagger-children">
        <div
          v-for="(feature, i) in features"
          :key="i"
          class="vision__card glass-card scroll-reveal"
          :class="{ visible: isVisible }"
        >
          <div class="vision__card-icon">{{ feature.icon }}</div>
          <h3 class="vision__card-title">{{ feature.title }}</h3>
          <p class="vision__card-desc">{{ feature.description }}</p>
        </div>
      </div>
    </div>

    <!-- Decorative shape -->
    <div class="vision__accent" aria-hidden="true"></div>
  </section>
</template>

<style scoped>
.vision {
  position: relative;
  background: var(--bg-raised);
}

.vision__header {
  text-align: center;
  max-width: 700px;
  margin: 0 auto 4rem;
}

.vision__header h2 {
  margin-top: 0.75rem;
}

.vision__desc {
  margin-top: 1rem;
  color: var(--text-2);
  font-size: var(--fs-lg);
  line-height: var(--leading-normal);
}

.vision__grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 1.25rem;
}

.vision__card {
  padding: 2rem 1.75rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.vision__card-icon {
  font-size: 2rem;
  width: 52px;
  height: 52px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--gradient-subtle);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-1);
}

.vision__card-title {
  font-size: var(--fs-lg);
  font-weight: 600;
}

.vision__card-desc {
  color: var(--text-2);
  font-size: var(--fs-sm);
  line-height: var(--leading-normal);
}

/* Decorative accent shape */
.vision__accent {
  position: absolute;
  top: 20%;
  right: -80px;
  width: 200px;
  height: 200px;
  background: var(--gradient-brand);
  opacity: 0.03;
  border-radius: 40% 60% 65% 35% / 40% 45% 55% 60%;
  transform: rotate(15deg);
  pointer-events: none;
}
</style>
