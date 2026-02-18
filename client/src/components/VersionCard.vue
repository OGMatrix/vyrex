<script setup lang="ts">
/**
 * VersionCard — Single Version Entry in the Timeline
 * Expandable card showing config changes, impact metrics, and assessment.
 */
import { ref, computed } from 'vue'
import type { Version } from '../types'
import { formatNumber } from '../composables/useAnimatedNumber'

const props = defineProps<{
  version: Version
  index: number
  isLatest: boolean
}>()

const expanded = ref(false)

const gradeColor = computed(() => {
  const g = props.version.assessment.grade
  if (g.startsWith('A')) return 'var(--green)'
  if (g.startsWith('B')) return 'var(--cyan-400)'
  if (g.startsWith('C')) return 'var(--yellow)'
  return 'var(--red)'
})

const statusBadge = computed(() => {
  if (props.version.status === 'active') return 'badge--green'
  if (props.version.status === 'planned') return 'badge--ghost'
  return 'badge--cyan'
})

const stepRangeText = computed(() => {
  const [start, end] = props.version.stepRange
  return `${formatNumber(start, 'compact')} → ${formatNumber(end, 'compact')}`
})

const stepsInVersion = computed(() => {
  const [start, end] = props.version.stepRange
  return formatNumber(end - start, 'compact')
})

function trendIcon(trend: 'up' | 'down' | 'stable'): string {
  if (trend === 'up') return '↑'
  if (trend === 'down') return '↓'
  return '→'
}
</script>

<template>
  <div
    class="v-card glass-card"
    :class="{
      'v-card--expanded': expanded,
      'v-card--latest': isLatest,
      'v-card--left': index % 2 === 0,
      'v-card--right': index % 2 === 1,
    }"
  >
    <!-- Header (always visible) -->
    <button class="v-card__header" @click="expanded = !expanded">
      <div class="v-card__header-top">
        <span class="v-card__version badge" :class="statusBadge">
          {{ version.id }}
        </span>
        <span
          class="v-card__grade font-mono"
          :style="{ color: gradeColor }"
        >
          {{ version.assessment.grade }}
        </span>
      </div>

      <h3 class="v-card__name">{{ version.name }}</h3>

      <div class="v-card__meta">
        <span class="v-card__steps font-mono">{{ stepRangeText }}</span>
        <span class="v-card__duration">{{ stepsInVersion }} steps</span>
      </div>

      <p class="v-card__summary">{{ version.summary }}</p>

      <div class="v-card__expand-hint">
        <svg
          :class="{ rotated: expanded }"
          width="16" height="16" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" stroke-width="2" stroke-linecap="round"
        >
          <polyline points="6 9 12 15 18 9"/>
        </svg>
        <span>{{ expanded ? 'Less' : 'Details' }}</span>
      </div>
    </button>

    <!-- Expandable Details -->
    <div class="v-card__details" :class="{ 'v-card__details--open': expanded }">
      <div class="v-card__details-inner">
        <!-- Config Changes -->
        <div v-if="version.changes.length > 0" class="v-card__section">
          <h4 class="v-card__section-title">
            <span>⚙️</span> Config Changes
          </h4>
          <div class="v-card__changes">
            <div
              v-for="(change, i) in version.changes"
              :key="i"
              class="v-card__change"
            >
              <div class="v-card__change-param font-mono">{{ change.parameter }}</div>
              <div class="v-card__change-values">
                <span class="v-card__change-old">{{ change.oldValue }}</span>
                <span class="v-card__change-arrow">→</span>
                <span class="v-card__change-new">{{ change.newValue }}</span>
              </div>
              <div class="v-card__change-rationale">{{ change.rationale }}</div>
            </div>
          </div>
        </div>

        <div v-else class="v-card__section">
          <h4 class="v-card__section-title">
            <span>⚙️</span> Config Changes
          </h4>
          <p class="v-card__no-changes">No configuration changes — continuation run</p>
        </div>

        <!-- Impact Metrics -->
        <div v-if="version.impact.length > 0" class="v-card__section">
          <h4 class="v-card__section-title">
            <span>📊</span> Impact
          </h4>
          <div class="v-card__impacts">
            <div
              v-for="(imp, i) in version.impact"
              :key="i"
              class="v-card__impact"
            >
              <span class="v-card__impact-metric">{{ imp.metric }}</span>
              <span class="v-card__impact-values font-mono">
                <template v-if="imp.before != null">
                  {{ imp.before }}{{ imp.unit }}
                </template>
                <template v-else>—</template>
                <span
                  class="v-card__impact-arrow"
                  :class="{
                    'v-card__impact-arrow--good': imp.isGood,
                    'v-card__impact-arrow--bad': !imp.isGood,
                  }"
                >
                  {{ trendIcon(imp.trend) }}
                </span>
                {{ imp.after }}{{ imp.unit }}
              </span>
            </div>
          </div>
        </div>

        <!-- Assessment -->
        <div class="v-card__section">
          <h4 class="v-card__section-title">
            <span>📋</span> Assessment
          </h4>
          <p class="v-card__assessment">{{ version.assessment.summary }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.v-card {
  position: relative;
  overflow: hidden;
  transition: all var(--dur-normal) var(--ease);
}

.v-card--latest {
  border-color: var(--border-accent);
}

.v-card--expanded {
  border-color: var(--border-2);
}

/* Header */
.v-card__header {
  width: 100%;
  text-align: left;
  padding: 1.25rem 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.v-card__header-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.v-card__grade {
  font-size: var(--fs-xl);
  font-weight: 800;
}

.v-card__name {
  font-size: var(--fs-xl);
  font-weight: 700;
  line-height: var(--leading-snug);
}

.v-card__meta {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  flex-wrap: wrap;
}

.v-card__steps {
  font-size: var(--fs-xs);
  color: var(--accent);
  background: rgba(34, 211, 238, 0.08);
  padding: 0.15em 0.5em;
  border-radius: var(--radius-sm);
}

.v-card__duration {
  font-size: var(--fs-xs);
  color: var(--text-3);
}

.v-card__summary {
  font-size: var(--fs-sm);
  color: var(--text-2);
  line-height: var(--leading-normal);
  margin-top: 0.25rem;
}

.v-card__expand-hint {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  font-size: var(--fs-xs);
  color: var(--text-3);
  margin-top: 0.5rem;
  transition: color var(--dur-fast) var(--ease);
}

.v-card__header:hover .v-card__expand-hint {
  color: var(--accent);
}

.v-card__expand-hint svg {
  transition: transform var(--dur-normal) var(--ease);
}

.v-card__expand-hint svg.rotated {
  transform: rotate(180deg);
}

/* Expandable Details */
.v-card__details {
  display: grid;
  grid-template-rows: 0fr;
  transition: grid-template-rows var(--dur-slow) var(--ease);
}

.v-card__details--open {
  grid-template-rows: 1fr;
}

.v-card__details-inner {
  overflow: hidden;
  padding: 0 1.5rem;
}

.v-card__details--open .v-card__details-inner {
  padding-bottom: 1.5rem;
}

.v-card__section {
  padding-top: 1rem;
  border-top: 1px solid var(--border-1);
  margin-top: 0.75rem;
}

.v-card__section-title {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: var(--fs-sm);
  font-weight: 600;
  margin-bottom: 0.75rem;
  color: var(--text-2);
}

/* Changes */
.v-card__changes {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.v-card__change {
  padding: 0.75rem;
  background: var(--bg-card);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-1);
}

.v-card__change-param {
  font-size: var(--fs-xs);
  color: var(--accent);
  font-weight: 600;
  margin-bottom: 0.25rem;
}

.v-card__change-values {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: var(--fs-sm);
  font-family: var(--font-mono);
  margin-bottom: 0.35rem;
}

.v-card__change-old {
  color: var(--text-3);
  text-decoration: line-through;
  opacity: 0.7;
}

.v-card__change-arrow {
  color: var(--text-3);
}

.v-card__change-new {
  color: var(--green);
  font-weight: 600;
}

.v-card__change-rationale {
  font-size: var(--fs-xs);
  color: var(--text-3);
  font-style: italic;
}

.v-card__no-changes {
  font-size: var(--fs-sm);
  color: var(--text-3);
  font-style: italic;
}

/* Impact */
.v-card__impacts {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.v-card__impact {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.35rem 0.6rem;
  background: var(--bg-card);
  border-radius: var(--radius-sm);
  font-size: var(--fs-sm);
}

.v-card__impact-metric {
  color: var(--text-2);
  font-weight: 500;
}

.v-card__impact-values {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  font-size: var(--fs-xs);
}

.v-card__impact-arrow {
  font-weight: 700;
}

.v-card__impact-arrow--good {
  color: var(--green);
}

.v-card__impact-arrow--bad {
  color: var(--red);
}

/* Assessment */
.v-card__assessment {
  font-size: var(--fs-sm);
  color: var(--text-2);
  line-height: var(--leading-normal);
}
</style>
