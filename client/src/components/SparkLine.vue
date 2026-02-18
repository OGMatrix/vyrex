<script setup lang="ts">
/**
 * SparkLine — Lightweight SVG Mini-Chart
 * Renders a smooth line chart with optional gradient fill.
 * Zero dependencies, GPU-accelerated via SVG.
 */
import { computed } from 'vue'

const props = withDefaults(
  defineProps<{
    data: number[]
    color?: string
    width?: number
    height?: number
    filled?: boolean
    strokeWidth?: number
  }>(),
  {
    color: 'var(--accent)',
    width: 120,
    height: 36,
    filled: true,
    strokeWidth: 1.5,
  }
)

const uid = `spark-${Math.random().toString(36).slice(2, 9)}`

const pathD = computed(() => {
  const pts = props.data
  if (!pts || pts.length < 2) return ''

  const len = pts.length
  const filtered = pts.filter((v) => v != null && !isNaN(v))
  if (filtered.length < 2) return ''

  const minY = Math.min(...filtered)
  const maxY = Math.max(...filtered)
  const range = maxY - minY || 1

  const pad = 2
  const w = props.width - pad * 2
  const h = props.height - pad * 2

  const points = pts.map((v, i) => {
    const x = pad + (i / (len - 1)) * w
    const y = pad + h - ((((v ?? minY) - minY) / range) * h)
    return `${x.toFixed(1)},${y.toFixed(1)}`
  })

  return `M${points.join(' L')}`
})

const fillPathD = computed(() => {
  if (!pathD.value || !props.filled) return ''
  const h = props.height
  const pts = props.data
  const len = pts.length
  const pad = 2
  const w = props.width - pad * 2
  const lastX = pad + w
  const firstX = pad
  return `${pathD.value} L${lastX.toFixed(1)},${h} L${firstX.toFixed(1)},${h} Z`
})
</script>

<template>
  <svg
    :width="width"
    :height="height"
    :viewBox="`0 0 ${width} ${height}`"
    class="sparkline"
    preserveAspectRatio="none"
  >
    <defs>
      <linearGradient :id="`${uid}-grad`" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" :stop-color="color" stop-opacity="0.25" />
        <stop offset="100%" :stop-color="color" stop-opacity="0" />
      </linearGradient>
    </defs>

    <!-- Fill area -->
    <path
      v-if="fillPathD"
      :d="fillPathD"
      :fill="`url(#${uid}-grad)`"
    />

    <!-- Line -->
    <path
      v-if="pathD"
      :d="pathD"
      fill="none"
      :stroke="color"
      :stroke-width="strokeWidth"
      stroke-linecap="round"
      stroke-linejoin="round"
      vector-effect="non-scaling-stroke"
    />
  </svg>
</template>

<style scoped>
.sparkline {
  display: block;
  overflow: visible;
}
</style>
