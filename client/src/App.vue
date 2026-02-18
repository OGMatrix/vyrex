<script setup lang="ts">
/**
 * App.vue — Root Orchestrator
 * Initializes API connection, manages global state,
 * and passes data down to all section components.
 */
import { computed } from 'vue'
import { useApi } from './composables/useApi'
import { useTheme } from './composables/useTheme'

import NavBar from './components/NavBar.vue'
import HeroSection from './components/HeroSection.vue'
import VisionSection from './components/VisionSection.vue'
import LiveMetrics from './components/LiveMetrics.vue'
import VersionTimeline from './components/VersionTimeline.vue'
import TechStack from './components/TechStack.vue'
import FooterSection from './components/FooterSection.vue'

// Initialize theme (applies data-theme to <html>)
useTheme()

// Initialize API (singleton — shared across all components)
const {
  connectionState,
  isLoading,
  isDemo,
  latestMetrics,
  runInfo,
  versions,
  metricHistory,
} = useApi()

const totalSteps = computed(() => {
  const s = latestMetrics.value?._step
  if (typeof s === 'number' && s > 0) return s
  return runInfo.value?.totalSteps ?? 0
})

const sps = computed(() => {
  const v = latestMetrics.value?.overall_sps
  return typeof v === 'number' ? v : 0
})
</script>

<template>
  <div class="app">
    <NavBar />

    <main>
      <HeroSection
        :total-steps="totalSteps"
        :sps="sps"
        :is-loading="isLoading"
        :connection-state="connectionState"
        :is-demo="isDemo"
      />

      <VisionSection />

      <LiveMetrics
        :metrics="latestMetrics"
        :metric-history="metricHistory"
        :is-loading="isLoading"
        :connection-state="connectionState"
        :is-demo="isDemo"
      />

      <VersionTimeline
        :versions="versions"
        :is-loading="isLoading"
      />

      <TechStack />
    </main>

    <FooterSection :run-info="runInfo" />

    <!-- Demo mode indicator -->
    <Transition name="demo-toast">
      <div v-if="isDemo" class="demo-toast">
        <span class="demo-toast__icon">🎮</span>
        <span>Demo Mode — Connect backend for live data</span>
      </div>
    </Transition>
  </div>
</template>

<style scoped>
.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

main {
  flex: 1;
}

/* Demo Toast */
.demo-toast {
  position: fixed;
  bottom: 1.5rem;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.6rem 1.2rem;
  background: var(--bg-overlay);
  border: 1px solid var(--border-accent);
  border-radius: var(--radius-full);
  font-size: var(--fs-xs);
  color: var(--text-2);
  z-index: var(--z-overlay);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  box-shadow: var(--shadow-lg);
  white-space: nowrap;
}

.demo-toast__icon {
  font-size: 1rem;
}

.demo-toast-enter-active {
  transition: all 0.5s var(--ease);
}
.demo-toast-leave-active {
  transition: all 0.3s var(--ease);
}
.demo-toast-enter-from {
  opacity: 0;
  transform: translateX(-50%) translateY(20px);
}
.demo-toast-leave-to {
  opacity: 0;
  transform: translateX(-50%) translateY(10px);
}
</style>
