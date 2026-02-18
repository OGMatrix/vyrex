<script setup lang="ts">
/**
 * LiveMetrics — Real-time Training Dashboard
 * Grid of MetricCards + TimeSeriesChart for selected metric.
 * Connects to SSE for live updates. Falls back to demo data.
 */
import { ref, computed, watch } from 'vue'
import { useScrollAnimation } from '../composables/useScrollAnimation'
import MetricCard from './MetricCard.vue'
import TimeSeriesChart from './TimeSeriesChart.vue'
import type { MetricSnapshot, ConnectionState, MetricDef } from '../types'

const props = defineProps<{
  metrics: MetricSnapshot
  metricHistory: Record<string, number[]>
  isLoading: boolean
  connectionState: ConnectionState
  isDemo: boolean
}>()

const sectionRef = ref<HTMLElement>()
const isVisible = useScrollAnimation(sectionRef)

const selectedMetric = ref('reward')

const metricDefs: MetricDef[] = [
  { key: 'reward', label: 'Mean Reward', format: 'decimal', decimals: 2, icon: '🏆', color: '#22d3ee', description: 'Average reward per iteration', goodDirection: 'up' },
  { key: 'overall_sps', label: 'Steps/Second', format: 'number', icon: '⚡', color: '#60a5fa', description: 'Training throughput (steps per second)', goodDirection: 'up' },
  { key: 'entropy', label: 'Entropy', format: 'decimal', decimals: 3, icon: '🎲', color: '#a78bfa', description: 'Policy entropy — exploration level', goodDirection: 'stable' },
  { key: 'game/touches_per_step', label: 'Touch Rate', format: 'decimal', decimals: 4, icon: '🎯', color: '#34d399', description: 'Ball touches per game step', goodDirection: 'up' },
  { key: 'game/avg_speed', label: 'Avg Speed', format: 'number', suffix: ' uu/s', icon: '💨', color: '#fbbf24', description: 'Average car speed in unreal units', goodDirection: 'up' },
  { key: 'game/airborne_fraction', label: 'Airborne', format: 'percent', decimals: 1, icon: '🚀', color: '#f87171', description: 'Fraction of time spent airborne', goodDirection: 'stable' },
  { key: 'vf_loss', label: 'VF Loss', format: 'decimal', decimals: 4, icon: '📉', color: '#fb923c', description: 'Value function loss — critic accuracy', goodDirection: 'down' },
  { key: 'kl_divergence', label: 'KL Divergence', format: 'decimal', decimals: 5, icon: '📏', color: '#94a3b8', description: 'KL divergence — policy stability', goodDirection: 'down' },
]

function getMetricValue(key: string): number | null {
  const v = props.metrics[key]
  if (v == null || isNaN(v as number)) return null
  return v as number
}

function getSparkData(key: string): number[] {
  const hist = props.metricHistory[key]
  if (!hist || hist.length === 0) return []
  // Take last 50 points for sparkline
  return hist.slice(-50)
}

const selectedDef = computed(() =>
  metricDefs.find((d) => d.key === selectedMetric.value) ?? metricDefs[0]
)

const chartDatasets = computed(() => {
  const def = selectedDef.value
  const data = props.metricHistory[def.key]
  if (!data) return []
  return [
    {
      key: def.key,
      label: def.label,
      data,
      color: def.color,
    },
  ]
})

const chartSteps = computed(() => {
  return props.metricHistory._step ?? []
})
</script>

<template>
  <section id="metrics" ref="sectionRef" class="metrics section">
    <!-- Angled top divider -->
    <div class="metrics__divider-top" aria-hidden="true"></div>

    <div class="container">
      <div class="metrics__header scroll-reveal" :class="{ visible: isVisible }">
        <div class="metrics__header-left">
          <span class="badge badge--cyan">Live Dashboard</span>
          <h2 class="heading-lg">Training <span class="text-gradient">Metrics</span></h2>
        </div>
        <div class="metrics__header-right">
          <div class="metrics__status">
            <span
              class="pulse-dot"
              :class="{ 'pulse-dot--disconnected': connectionState !== 'connected' }"
            ></span>
            <span class="metrics__status-text">
              {{ connectionState === 'connected' ? 'Connected' : isDemo ? 'Demo Data' : 'Offline' }}
            </span>
          </div>
        </div>
      </div>

      <!-- Metric Cards Grid -->
      <div class="metrics__grid stagger-children">
        <MetricCard
          v-for="def in metricDefs"
          :key="def.key"
          :value="getMetricValue(def.key)"
          :label="def.label"
          :format="def.format"
          :decimals="def.decimals"
          :suffix="def.suffix"
          :spark-data="getSparkData(def.key)"
          :color="def.color"
          :icon="def.icon"
          :is-loading="isLoading"
          :description="def.description"
          :selected="selectedMetric === def.key"
          class="scroll-reveal"
          :class="{ visible: isVisible }"
          @select="selectedMetric = def.key"
        />
      </div>

      <!-- Time Series Chart -->
      <div class="metrics__chart glass-card scroll-reveal" :class="{ visible: isVisible }">
        <div class="metrics__chart-header">
          <h3 class="heading-sm">
            <span class="metrics__chart-dot" :style="{ background: selectedDef.color }"></span>
            {{ selectedDef.label }}
          </h3>
          <span class="metrics__chart-desc">{{ selectedDef.description }}</span>
        </div>
        <TimeSeriesChart
          v-if="chartSteps.length > 0 && chartDatasets.length > 0"
          :steps="chartSteps"
          :datasets="chartDatasets"
          :height="280"
        />
        <div v-else class="metrics__chart-empty">
          <div class="skeleton" style="width: 100%; height: 200px;"></div>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.metrics {
  position: relative;
  background: var(--bg-base);
}

.metrics__divider-top {
  position: absolute;
  top: -1px;
  left: 0;
  right: 0;
  height: 100px;
  background: var(--bg-raised);
  clip-path: polygon(0 0, 100% 0, 100% 40%, 0 100%);
}

.metrics__header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 2rem;
  flex-wrap: wrap;
}

.metrics__header-left h2 {
  margin-top: 0.5rem;
}

.metrics__status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.4rem 0.8rem;
  background: var(--bg-card);
  border: 1px solid var(--border-1);
  border-radius: var(--radius-full);
}

.metrics__status-text {
  font-size: var(--fs-xs);
  font-weight: 500;
  color: var(--text-2);
}

.metrics__grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 0.75rem;
  margin-bottom: 1.5rem;
}

.metrics__chart {
  padding: 1.5rem;
}

.metrics__chart-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1rem;
  flex-wrap: wrap;
}

.metrics__chart-header h3 {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.metrics__chart-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  flex-shrink: 0;
}

.metrics__chart-desc {
  font-size: var(--fs-xs);
  color: var(--text-3);
}

.metrics__chart-empty {
  padding: 2rem 0;
}

@media (max-width: 640px) {
  .metrics__grid {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
