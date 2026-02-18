<script setup lang="ts">
/**
 * MetricCard — Single Metric Display with SparkLine & Skeleton
 * Shows: value, label, optional sparkline, optional trend indicator.
 */
import { computed } from 'vue'
import SparkLine from './SparkLine.vue'
import SkeletonLoader from './SkeletonLoader.vue'
import { formatNumber } from '../composables/useAnimatedNumber'

const props = defineProps<{
  value: number | null | undefined
  label: string
  format?: 'number' | 'percent' | 'compact' | 'decimal'
  decimals?: number
  suffix?: string
  sparkData?: number[]
  color?: string
  icon?: string
  isLoading?: boolean
  description?: string
  selected?: boolean
}>()

defineEmits<{
  select: []
}>()

const formattedValue = computed(() => {
  if (props.value == null) return '—'
  return formatNumber(props.value, props.format ?? 'number', props.decimals) + (props.suffix ?? '')
})

const cardColor = computed(() => props.color ?? 'var(--accent)')
</script>

<template>
  <button
    class="metric-card glass-card"
    :class="{ 'metric-card--selected': selected, 'metric-card--loading': isLoading }"
    :style="{ '--card-accent': cardColor }"
    @click="$emit('select')"
    :title="description"
  >
    <!-- Loading State -->
    <template v-if="isLoading">
      <div class="metric-card__top">
        <SkeletonLoader width="24px" height="24px" radius="6px" />
      </div>
      <SkeletonLoader width="80%" height="28px" />
      <SkeletonLoader width="60%" height="14px" />
      <SkeletonLoader width="100%" height="32px" />
    </template>

    <!-- Loaded State -->
    <template v-else>
      <div class="metric-card__top">
        <span class="metric-card__icon" v-if="icon">{{ icon }}</span>
      </div>

      <div class="metric-card__value font-mono" :style="{ color: cardColor }">
        {{ formattedValue }}
      </div>

      <div class="metric-card__label">{{ label }}</div>

      <div class="metric-card__spark" v-if="sparkData && sparkData.length > 1">
        <SparkLine :data="sparkData" :color="cardColor" :width="140" :height="32" />
      </div>
    </template>

    <!-- Active indicator line -->
    <div class="metric-card__indicator" v-if="selected"></div>
  </button>
</template>

<style scoped>
.metric-card {
  position: relative;
  padding: 1.25rem 1.25rem 1rem;
  text-align: left;
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  cursor: pointer;
  overflow: hidden;
  transition: all var(--dur-normal) var(--ease);
}

.metric-card:hover {
  border-color: color-mix(in srgb, var(--card-accent) 30%, transparent);
}

.metric-card--selected {
  border-color: color-mix(in srgb, var(--card-accent) 40%, transparent);
  background: color-mix(in srgb, var(--card-accent) 4%, var(--bg-card));
  box-shadow: 0 0 30px color-mix(in srgb, var(--card-accent) 8%, transparent);
}

.metric-card__top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.25rem;
}

.metric-card__icon {
  font-size: 1.2rem;
}

.metric-card__value {
  font-size: var(--fs-3xl);
  font-weight: 700;
  line-height: 1.1;
  letter-spacing: var(--tracking-tight);
}

.metric-card__label {
  font-size: var(--fs-xs);
  color: var(--text-3);
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: var(--tracking-wide);
}

.metric-card__spark {
  margin-top: 0.5rem;
  opacity: 0.8;
}

.metric-card__indicator {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: var(--card-accent);
}

/* Loading shimmer for the card */
.metric-card--loading {
  pointer-events: none;
}
</style>
