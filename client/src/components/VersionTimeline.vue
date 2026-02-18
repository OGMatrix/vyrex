<script setup lang="ts">
/**
 * VersionTimeline — Full Training Journey Timeline
 * Vertical timeline with alternating VersionCards.
 * Step scale on the center line, scroll-animated reveals.
 */
import { ref, computed } from 'vue'
import { useScrollAnimation } from '../composables/useScrollAnimation'
import { formatNumber } from '../composables/useAnimatedNumber'
import VersionCard from './VersionCard.vue'
import type { Version } from '../types'

const props = defineProps<{
  versions: Version[]
  isLoading: boolean
}>()

const sectionRef = ref<HTMLElement>()
const isVisible = useScrollAnimation(sectionRef)

const sortedVersions = computed(() =>
  [...props.versions].sort((a, b) => b.stepRange[1] - a.stepRange[1])
)

const totalSteps = computed(() => {
  if (props.versions.length === 0) return 0
  return Math.max(...props.versions.map((v) => v.stepRange[1]))
})
</script>

<template>
  <section id="journey" ref="sectionRef" class="journey section">
    <div class="container">
      <div class="journey__header scroll-reveal" :class="{ visible: isVisible }">
        <span class="badge badge--cyan">Transparent Development</span>
        <h2 class="heading-lg">
          The <span class="text-gradient">Journey</span>
        </h2>
        <p class="journey__desc">
          Every training decision, every config change, every breakthrough and setback — documented
          with full transparency. Click any version to see exactly what changed and what it did.
        </p>
      </div>

      <!-- Summary Stats -->
      <div class="journey__stats scroll-reveal" :class="{ visible: isVisible }">
        <div class="journey__stat">
          <span class="journey__stat-value font-mono text-gradient">{{ sortedVersions.length }}</span>
          <span class="journey__stat-label">Versions</span>
        </div>
        <div class="journey__stat-sep" aria-hidden="true"></div>
        <div class="journey__stat">
          <span class="journey__stat-value font-mono text-gradient">{{ formatNumber(totalSteps, 'compact') }}</span>
          <span class="journey__stat-label">Total Steps</span>
        </div>
        <div class="journey__stat-sep" aria-hidden="true"></div>
        <div class="journey__stat">
          <span class="journey__stat-value font-mono text-gradient">{{ sortedVersions.filter(v => v.changes.length > 0).length }}</span>
          <span class="journey__stat-label">Config Changes</span>
        </div>
      </div>

      <!-- Timeline -->
      <div class="journey__timeline">
        <!-- Center Line -->
        <div class="journey__line" aria-hidden="true"></div>

        <!-- Version Cards -->
        <div
          v-for="(version, i) in sortedVersions"
          :key="version.id"
          class="journey__entry scroll-reveal"
          :class="{
            visible: isVisible,
            'journey__entry--left': i % 2 === 0,
            'journey__entry--right': i % 2 === 1,
          }"
          :style="{ transitionDelay: `${i * 100}ms` }"
        >
          <!-- Timeline Dot -->
          <div class="journey__dot" :class="{ 'journey__dot--latest': i === 0 }">
            <div class="journey__dot-inner"></div>
          </div>

          <!-- Step Label -->
          <div class="journey__step-label font-mono">
            {{ formatNumber(version.stepRange[1], 'compact') }}
          </div>

          <!-- Card -->
          <VersionCard
            :version="version"
            :index="i"
            :is-latest="i === 0"
          />
        </div>

        <!-- Loading Skeletons -->
        <template v-if="isLoading">
          <div
            v-for="i in 3"
            :key="'skel-' + i"
            class="journey__entry"
            :class="i % 2 === 1 ? 'journey__entry--left' : 'journey__entry--right'"
          >
            <div class="journey__dot">
              <div class="journey__dot-inner"></div>
            </div>
            <div class="glass-card" style="padding: 1.5rem; width: 100%;">
              <div class="skeleton" style="width: 60px; height: 20px; margin-bottom: 0.75rem;"></div>
              <div class="skeleton" style="width: 150px; height: 22px; margin-bottom: 0.5rem;"></div>
              <div class="skeleton" style="width: 100%; height: 14px; margin-bottom: 0.25rem;"></div>
              <div class="skeleton" style="width: 80%; height: 14px;"></div>
            </div>
          </div>
        </template>
      </div>
    </div>

    <!-- Decorative shapes -->
    <div class="journey__shape-1" aria-hidden="true"></div>
    <div class="journey__shape-2" aria-hidden="true"></div>
  </section>
</template>

<style scoped>
.journey {
  position: relative;
  background: var(--bg-raised);
  overflow: hidden;
}

.journey__header {
  text-align: center;
  max-width: 680px;
  margin: 0 auto 2rem;
}

.journey__header h2 {
  margin-top: 0.75rem;
}

.journey__desc {
  margin-top: 1rem;
  color: var(--text-2);
  font-size: var(--fs-lg);
}

/* Stats */
.journey__stats {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 2rem;
  margin-bottom: 3.5rem;
  flex-wrap: wrap;
}

.journey__stat {
  text-align: center;
}

.journey__stat-value {
  display: block;
  font-size: var(--fs-4xl);
  font-weight: 800;
  line-height: 1;
}

.journey__stat-label {
  font-size: var(--fs-xs);
  color: var(--text-3);
  text-transform: uppercase;
  letter-spacing: var(--tracking-wider);
  font-weight: 500;
  margin-top: 0.25rem;
}

.journey__stat-sep {
  width: 1px;
  height: 40px;
  background: var(--border-1);
}

/* Timeline */
.journey__timeline {
  position: relative;
  max-width: 900px;
  margin: 0 auto;
}

.journey__line {
  position: absolute;
  top: 0;
  bottom: 0;
  left: 50%;
  width: 2px;
  background: linear-gradient(
    to bottom,
    var(--accent),
    var(--border-1) 20%,
    var(--border-1) 80%,
    transparent
  );
  transform: translateX(-50%);
}

/* Entry */
.journey__entry {
  position: relative;
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  gap: 1.5rem;
  align-items: start;
  margin-bottom: 1.5rem;
}

.journey__entry--left .v-card {
  grid-column: 1;
  grid-row: 1;
}

.journey__entry--left .journey__dot {
  grid-column: 2;
  grid-row: 1;
}

.journey__entry--left .journey__step-label {
  grid-column: 3;
  grid-row: 1;
  text-align: left;
  padding-left: 0.5rem;
}

.journey__entry--right .v-card,
.journey__entry--right .glass-card {
  grid-column: 3;
  grid-row: 1;
}

.journey__entry--right .journey__dot {
  grid-column: 2;
  grid-row: 1;
}

.journey__entry--right .journey__step-label {
  grid-column: 1;
  grid-row: 1;
  text-align: right;
  padding-right: 0.5rem;
}

/* Dot */
.journey__dot {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  margin-top: 1.5rem;
  z-index: 1;
}

.journey__dot-inner {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: var(--bg-base);
  border: 2px solid var(--border-2);
  transition: all var(--dur-normal) var(--ease);
}

.journey__dot--latest .journey__dot-inner {
  background: var(--accent);
  border-color: var(--accent);
  box-shadow: 0 0 12px rgba(34, 211, 238, 0.4);
}

/* Step label */
.journey__step-label {
  font-size: var(--fs-xs);
  color: var(--text-3);
  margin-top: 1.65rem;
  white-space: nowrap;
}

/* Decorative shapes */
.journey__shape-1 {
  position: absolute;
  top: 10%;
  left: -120px;
  width: 250px;
  height: 250px;
  background: var(--gradient-brand);
  opacity: 0.02;
  clip-path: polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%);
  pointer-events: none;
}

.journey__shape-2 {
  position: absolute;
  bottom: 5%;
  right: -80px;
  width: 200px;
  height: 200px;
  background: var(--gradient-brand-r);
  opacity: 0.025;
  border-radius: 30% 70% 70% 30% / 30% 30% 70% 70%;
  pointer-events: none;
}

/* Responsive: collapse to single column */
@media (max-width: 768px) {
  .journey__line {
    left: 16px;
  }

  .journey__entry {
    grid-template-columns: auto 1fr;
    gap: 1rem;
  }

  .journey__entry--left .v-card,
  .journey__entry--right .v-card,
  .journey__entry--left .glass-card,
  .journey__entry--right .glass-card {
    grid-column: 2;
    grid-row: 1;
  }

  .journey__entry--left .journey__dot,
  .journey__entry--right .journey__dot {
    grid-column: 1;
    grid-row: 1;
  }

  .journey__entry--left .journey__step-label,
  .journey__entry--right .journey__step-label {
    display: none;
  }

  .journey__stats {
    gap: 1.5rem;
  }

  .journey__stat-sep {
    display: none;
  }
}
</style>
