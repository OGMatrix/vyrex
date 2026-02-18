<script setup lang="ts">
/**
 * TimeSeriesChart — Custom SVG Time Series Visualization
 * Renders a smooth, responsive line chart using pure SVG.
 * Supports hover tooltip, theme awareness, multiple datasets.
 */
import { ref, computed, onMounted, onUnmounted } from 'vue'

interface Dataset {
  key: string
  label: string
  data: number[]
  color: string
}

const props = defineProps<{
  steps: number[]
  datasets: Dataset[]
  height?: number
}>()

const containerRef = ref<HTMLElement>()
const svgWidth = ref(800)
const svgHeight = computed(() => props.height ?? 240)
const hoverIndex = ref(-1)
const mouseX = ref(0)

const padding = { top: 20, right: 20, bottom: 30, left: 60 }

// ResizeObserver
let resizeObs: ResizeObserver | null = null
onMounted(() => {
  if (containerRef.value) {
    svgWidth.value = containerRef.value.clientWidth
    resizeObs = new ResizeObserver(([entry]) => {
      svgWidth.value = entry.contentRect.width
    })
    resizeObs.observe(containerRef.value)
  }
})
onUnmounted(() => resizeObs?.disconnect())

const plotW = computed(() => svgWidth.value - padding.left - padding.right)
const plotH = computed(() => svgHeight.value - padding.top - padding.bottom)

function clamp(v: number, min: number, max: number) {
  return Math.max(min, Math.min(max, v))
}

// Compute Y-axis bounds per dataset (each dataset gets its own scale)
const yBounds = computed(() => {
  const bounds: Record<string, { min: number; max: number }> = {}
  for (const ds of props.datasets) {
    const valid = ds.data.filter((v) => v != null && !isNaN(v))
    if (valid.length === 0) {
      bounds[ds.key] = { min: 0, max: 1 }
      continue
    }
    let mn = Math.min(...valid)
    let mx = Math.max(...valid)
    if (mn === mx) { mn -= 0.5; mx += 0.5 }
    // Add 10% padding
    const pad = (mx - mn) * 0.1
    bounds[ds.key] = { min: mn - pad, max: mx + pad }
  }
  return bounds
})

// Primary dataset for Y-axis labels
const primaryBounds = computed(() => {
  if (props.datasets.length === 0) return { min: 0, max: 1 }
  return yBounds.value[props.datasets[0].key] ?? { min: 0, max: 1 }
})

function toX(i: number): number {
  const len = props.steps.length
  if (len <= 1) return padding.left
  return padding.left + (i / (len - 1)) * plotW.value
}

function toY(v: number, key: string): number {
  const b = yBounds.value[key] ?? primaryBounds.value
  const range = b.max - b.min || 1
  return padding.top + plotH.value - ((v - b.min) / range) * plotH.value
}

// SVG paths
const paths = computed(() => {
  return props.datasets.map((ds) => {
    const points = ds.data
      .map((v, i) => {
        if (v == null || isNaN(v)) return null
        return `${toX(i).toFixed(1)},${toY(v, ds.key).toFixed(1)}`
      })
      .filter(Boolean)

    return {
      key: ds.key,
      label: ds.label,
      color: ds.color,
      d: points.length > 1 ? `M${points.join(' L')}` : '',
    }
  })
})

// Y-axis labels
const yLabels = computed(() => {
  const b = primaryBounds.value
  const count = 5
  const labels = []
  for (let i = 0; i <= count; i++) {
    const val = b.min + (i / count) * (b.max - b.min)
    const y = padding.top + plotH.value - (i / count) * plotH.value
    labels.push({
      y,
      text: formatAxisLabel(val),
    })
  }
  return labels
})

// X-axis labels
const xLabels = computed(() => {
  const len = props.steps.length
  if (len === 0) return []
  const count = Math.min(5, len)
  const labels = []
  for (let i = 0; i < count; i++) {
    const idx = Math.round((i / (count - 1)) * (len - 1))
    labels.push({
      x: toX(idx),
      text: formatStepLabel(props.steps[idx]),
    })
  }
  return labels
})

function formatAxisLabel(v: number): string {
  if (Math.abs(v) >= 1e6) return (v / 1e6).toFixed(1) + 'M'
  if (Math.abs(v) >= 1e3) return (v / 1e3).toFixed(1) + 'K'
  if (Math.abs(v) < 0.01 && v !== 0) return v.toExponential(1)
  if (Math.abs(v) < 10) return v.toFixed(3)
  return v.toFixed(1)
}

function formatStepLabel(v: number): string {
  if (v >= 1e9) return (v / 1e9).toFixed(1) + 'B'
  if (v >= 1e6) return (v / 1e6).toFixed(0) + 'M'
  if (v >= 1e3) return (v / 1e3).toFixed(0) + 'K'
  return v.toFixed(0)
}

// Hover handling
function onMouseMove(e: MouseEvent) {
  const rect = containerRef.value?.getBoundingClientRect()
  if (!rect) return
  const x = e.clientX - rect.left
  mouseX.value = x

  const relX = x - padding.left
  const len = props.steps.length
  if (len === 0) return

  const idx = Math.round((relX / plotW.value) * (len - 1))
  hoverIndex.value = clamp(idx, 0, len - 1)
}

function onMouseLeave() {
  hoverIndex.value = -1
}

const tooltipStyle = computed(() => {
  if (hoverIndex.value < 0) return {}
  const x = toX(hoverIndex.value)
  const flip = x > svgWidth.value * 0.7
  return {
    left: flip ? 'auto' : `${x + 12}px`,
    right: flip ? `${svgWidth.value - x + 12}px` : 'auto',
    top: `${padding.top + 8}px`,
  }
})
</script>

<template>
  <div
    ref="containerRef"
    class="ts-chart"
    @mousemove="onMouseMove"
    @mouseleave="onMouseLeave"
  >
    <svg
      :width="svgWidth"
      :height="svgHeight"
      :viewBox="`0 0 ${svgWidth} ${svgHeight}`"
    >
      <!-- Grid lines -->
      <line
        v-for="(label, i) in yLabels"
        :key="'grid-' + i"
        :x1="padding.left"
        :y1="label.y"
        :x2="svgWidth - padding.right"
        :y2="label.y"
        class="ts-chart__grid"
      />

      <!-- Data lines -->
      <path
        v-for="p in paths"
        :key="p.key"
        :d="p.d"
        fill="none"
        :stroke="p.color"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
      />

      <!-- Hover vertical line -->
      <line
        v-if="hoverIndex >= 0"
        :x1="toX(hoverIndex)"
        :y1="padding.top"
        :x2="toX(hoverIndex)"
        :y2="svgHeight - padding.bottom"
        class="ts-chart__hover-line"
      />

      <!-- Hover dots -->
      <template v-if="hoverIndex >= 0">
        <circle
          v-for="ds in datasets"
          :key="'dot-' + ds.key"
          :cx="toX(hoverIndex)"
          :cy="toY(ds.data[hoverIndex], ds.key)"
          r="4"
          :fill="ds.color"
          stroke="var(--bg-base)"
          stroke-width="2"
        />
      </template>

      <!-- Y-axis labels -->
      <text
        v-for="(label, i) in yLabels"
        :key="'y-' + i"
        :x="padding.left - 8"
        :y="label.y + 4"
        class="ts-chart__label"
        text-anchor="end"
      >
        {{ label.text }}
      </text>

      <!-- X-axis labels -->
      <text
        v-for="(label, i) in xLabels"
        :key="'x-' + i"
        :x="label.x"
        :y="svgHeight - 6"
        class="ts-chart__label"
        text-anchor="middle"
      >
        {{ label.text }}
      </text>
    </svg>

    <!-- Tooltip -->
    <Transition name="tooltip">
      <div v-if="hoverIndex >= 0" class="ts-chart__tooltip" :style="tooltipStyle">
        <div class="ts-chart__tooltip-step font-mono">
          Step {{ formatStepLabel(steps[hoverIndex]) }}
        </div>
        <div
          v-for="ds in datasets"
          :key="ds.key"
          class="ts-chart__tooltip-row"
        >
          <span class="ts-chart__tooltip-dot" :style="{ background: ds.color }"></span>
          <span class="ts-chart__tooltip-label">{{ ds.label }}</span>
          <span class="ts-chart__tooltip-val font-mono">
            {{ formatAxisLabel(ds.data[hoverIndex]) }}
          </span>
        </div>
      </div>
    </Transition>
  </div>
</template>

<style scoped>
.ts-chart {
  position: relative;
  width: 100%;
  overflow: hidden;
}

.ts-chart__grid {
  stroke: var(--border-1);
  stroke-dasharray: 4 4;
}

.ts-chart__hover-line {
  stroke: var(--text-3);
  stroke-width: 1;
  stroke-dasharray: 3 3;
  opacity: 0.6;
}

.ts-chart__label {
  font-family: var(--font-mono);
  font-size: 10px;
  fill: var(--text-3);
}

/* Tooltip */
.ts-chart__tooltip {
  position: absolute;
  background: var(--bg-overlay);
  border: 1px solid var(--border-2);
  border-radius: var(--radius-sm);
  padding: 0.5rem 0.75rem;
  pointer-events: none;
  z-index: var(--z-tooltip);
  min-width: 140px;
  backdrop-filter: blur(12px);
}

.ts-chart__tooltip-step {
  font-size: var(--fs-xs);
  color: var(--text-3);
  margin-bottom: 0.35rem;
  padding-bottom: 0.3rem;
  border-bottom: 1px solid var(--border-1);
}

.ts-chart__tooltip-row {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: var(--fs-xs);
  padding: 0.1rem 0;
}

.ts-chart__tooltip-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.ts-chart__tooltip-label {
  color: var(--text-2);
  flex: 1;
}

.ts-chart__tooltip-val {
  color: var(--text-1);
  font-weight: 600;
}

.tooltip-enter-active,
.tooltip-leave-active {
  transition: opacity var(--dur-fast);
}
.tooltip-enter-from,
.tooltip-leave-to {
  opacity: 0;
}
</style>
