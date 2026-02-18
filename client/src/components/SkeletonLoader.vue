<script setup lang="ts">
/**
 * SkeletonLoader — Reusable Shimmer Loading Placeholder
 * Matches the exact dimensions of the content it replaces.
 */
defineProps<{
  width?: string
  height?: string
  radius?: string
  circle?: boolean
}>()
</script>

<template>
  <div
    class="skeleton"
    :class="{ 'skeleton--circle': circle }"
    :style="{
      width: circle ? height : (width ?? '100%'),
      height: height ?? '1em',
      borderRadius: circle ? '50%' : (radius ?? 'var(--radius-sm)'),
    }"
    aria-hidden="true"
  />
</template>

<style scoped>
.skeleton {
  flex-shrink: 0;
}
.skeleton--circle {
  aspect-ratio: 1;
}
</style>
