<script setup lang="ts">
/**
 * HeroSection — Full-viewport Landing with Animated Grid & Step Counter
 * Features: animated grid background, glowing logo, live step counter,
 * status indicator, scroll-down arrow.
 */
import { ref, computed, onMounted } from 'vue'
import { useAnimatedNumber, formatNumber } from '../composables/useAnimatedNumber'
import type { ConnectionState } from '../types'

const props = defineProps<{
  totalSteps: number
  sps: number
  isLoading: boolean
  connectionState: ConnectionState
  isDemo: boolean
}>()

const sectionRef = ref<HTMLElement>()
const targetSteps = computed(() => props.totalSteps || 0)
const displaySteps = useAnimatedNumber(targetSteps, { duration: 2000 })

const formattedSteps = computed(() => {
  const n = displaySteps.value
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B'
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M'
  if (n >= 1e3) return (n / 1e3).toFixed(0) + 'K'
  return n.toFixed(0)
})

const statusText = computed(() => {
  if (props.isDemo) return 'Demo Mode'
  if (props.connectionState === 'connected') return 'Training Live'
  return 'Last Session'
})

const statusColor = computed(() => {
  if (props.connectionState === 'connected') return 'green'
  if (props.isDemo) return 'yellow'
  return 'ghost'
})

const loaded = ref(false)
onMounted(() => {
  requestAnimationFrame(() => { loaded.value = true })
})

function scrollDown() {
  document.querySelector('#vision')?.scrollIntoView({ behavior: 'smooth' })
}
</script>

<template>
  <section ref="sectionRef" class="hero">
    <!-- Animated Grid Background -->
    <div class="hero__grid" aria-hidden="true"></div>
    <div class="hero__glow" aria-hidden="true"></div>

    <!-- Diagonal Decorative Shapes -->
    <div class="hero__shape hero__shape--1" aria-hidden="true"></div>
    <div class="hero__shape hero__shape--2" aria-hidden="true"></div>

    <div class="hero__content container" :class="{ loaded }">
      <!-- Status Badge -->
      <div class="hero__status">
        <span class="badge" :class="`badge--${statusColor}`">
          <span class="pulse-dot" :class="{ 'pulse-dot--disconnected': connectionState !== 'connected' }"></span>
          {{ statusText }}
        </span>
      </div>

      <!-- Logo / Title -->
      <h1 class="hero__title">
        <span class="hero__title-v">V</span>YREX
      </h1>

      <p class="hero__subtitle">
        Devastating <strong>2v2</strong> Rocket League AI
      </p>

      <!-- Animated Step Counter -->
      <div class="hero__counter" v-if="!isLoading">
        <div class="hero__counter-number font-mono">
          {{ formattedSteps }}
        </div>
        <div class="hero__counter-label">
          training steps
        </div>
        <div class="hero__counter-sps font-mono" v-if="sps > 0">
          {{ formatNumber(sps, 'number') }} steps/sec
        </div>
      </div>

      <!-- Skeleton Counter -->
      <div class="hero__counter" v-else>
        <div class="skeleton" style="width: 240px; height: 60px; margin: 0 auto;"></div>
        <div class="skeleton" style="width: 120px; height: 16px; margin: 8px auto 0;"></div>
      </div>

      <!-- CTAs -->
      <div class="hero__actions">
        <a href="#journey" class="btn btn--primary" @click.prevent="document.querySelector('#journey')?.scrollIntoView({ behavior: 'smooth' })">
          View Journey
        </a>
        <a href="#metrics" class="btn btn--ghost" @click.prevent="document.querySelector('#metrics')?.scrollIntoView({ behavior: 'smooth' })">
          Live Metrics
        </a>
      </div>
    </div>

    <!-- Scroll Indicator -->
    <button class="hero__scroll" @click="scrollDown" aria-label="Scroll down">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="6 9 12 15 18 9"/>
      </svg>
    </button>

    <!-- Angled Bottom Divider -->
    <div class="hero__divider" aria-hidden="true"></div>
  </section>
</template>

<style scoped>
.hero {
  position: relative;
  min-height: 100vh;
  min-height: 100dvh;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

/* Animated Grid */
.hero__grid {
  position: absolute;
  inset: -60px;
  background-image:
    linear-gradient(rgba(34, 211, 238, 0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(34, 211, 238, 0.035) 1px, transparent 1px);
  background-size: 60px 60px;
  mask-image: radial-gradient(ellipse 70% 50% at 50% 45%, black 20%, transparent 70%);
  -webkit-mask-image: radial-gradient(ellipse 70% 50% at 50% 45%, black 20%, transparent 70%);
  animation: grid-drift 30s linear infinite;
  pointer-events: none;
}

/* Glow behind title */
.hero__glow {
  position: absolute;
  width: min(700px, 90vw);
  height: min(700px, 90vw);
  background: radial-gradient(circle, rgba(34, 211, 238, 0.08) 0%, transparent 65%);
  top: 50%;
  left: 50%;
  transform: translate(-50%, -55%);
  pointer-events: none;
  animation: glow-pulse 6s ease-in-out infinite;
}

/* Decorative Irregular Shapes */
.hero__shape {
  position: absolute;
  pointer-events: none;
  opacity: 0.04;
}

.hero__shape--1 {
  top: 10%;
  right: -5%;
  width: 400px;
  height: 400px;
  background: var(--gradient-brand);
  clip-path: polygon(50% 0%, 100% 38%, 82% 100%, 18% 100%, 0% 38%);
  animation: float 12s ease-in-out infinite;
}

.hero__shape--2 {
  bottom: 15%;
  left: -3%;
  width: 300px;
  height: 300px;
  background: var(--gradient-brand-r);
  clip-path: polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%);
  animation: float 15s ease-in-out infinite reverse;
}

/* Content */
.hero__content {
  position: relative;
  z-index: 1;
  text-align: center;
  padding-top: 5rem;
  opacity: 0;
  transform: translateY(20px);
  transition: opacity 0.8s var(--ease), transform 1s var(--ease);
}

.hero__content.loaded {
  opacity: 1;
  transform: none;
}

.hero__status {
  margin-bottom: 1.5rem;
}

.hero__title {
  font-size: var(--fs-hero);
  font-weight: 900;
  letter-spacing: 0.15em;
  line-height: 1;
  margin-bottom: 0.5rem;
}

.hero__title-v {
  background: var(--gradient-brand);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  font-size: 1.1em;
}

.hero__subtitle {
  font-size: var(--fs-xl);
  color: var(--text-2);
  font-weight: 400;
  margin-bottom: 2.5rem;
  letter-spacing: var(--tracking-wide);
}

.hero__subtitle strong {
  color: var(--accent);
  font-weight: 600;
}

/* Counter */
.hero__counter {
  margin-bottom: 2.5rem;
}

.hero__counter-number {
  font-size: clamp(2.5rem, 7vw, 4.5rem);
  font-weight: 700;
  letter-spacing: var(--tracking-tight);
  background: var(--gradient-brand);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  line-height: 1.1;
}

.hero__counter-label {
  font-size: var(--fs-sm);
  color: var(--text-3);
  text-transform: uppercase;
  letter-spacing: var(--tracking-wider);
  font-weight: 500;
  margin-top: 0.25rem;
}

.hero__counter-sps {
  font-size: var(--fs-xs);
  color: var(--text-3);
  margin-top: 0.5rem;
  opacity: 0.7;
}

/* Actions */
.hero__actions {
  display: flex;
  gap: 0.75rem;
  justify-content: center;
  flex-wrap: wrap;
}

/* Scroll Indicator */
.hero__scroll {
  position: absolute;
  bottom: 2.5rem;
  left: 50%;
  transform: translateX(-50%);
  color: var(--text-3);
  animation: float 3s ease-in-out infinite;
  z-index: 1;
  padding: 0.5rem;
  border-radius: var(--radius-full);
  transition: color var(--dur-fast) var(--ease);
}

.hero__scroll:hover {
  color: var(--accent);
}

/* Bottom Divider */
.hero__divider {
  position: absolute;
  bottom: -1px;
  left: 0;
  right: 0;
  height: 100px;
  background: var(--bg-base);
  clip-path: polygon(0 60%, 100% 0%, 100% 100%, 0% 100%);
}

[data-theme="light"] .hero__grid {
  background-image:
    linear-gradient(rgba(6, 182, 212, 0.06) 1px, transparent 1px),
    linear-gradient(90deg, rgba(6, 182, 212, 0.06) 1px, transparent 1px);
}

[data-theme="light"] .hero__glow {
  background: radial-gradient(circle, rgba(6, 182, 212, 0.06) 0%, transparent 65%);
}

@media (max-width: 768px) {
  .hero__shape { display: none; }
}
</style>
