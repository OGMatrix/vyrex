/* ============================================================
   useAnimatedNumber — Smooth Number Counter Animation
   ============================================================
   Uses requestAnimationFrame for buttery-smooth counting.
   Returns a display ref that animates from current to target.
   ============================================================ */

import { ref, watch, onUnmounted, type Ref } from 'vue'

interface AnimatedNumberOptions {
  duration?: number
  decimals?: number
}

export function useAnimatedNumber(
  target: Ref<number>,
  options: AnimatedNumberOptions = {}
) {
  const { duration = 1200, decimals = 0 } = options
  const display = ref(0)
  let animId: number | null = null
  let currentFrom = 0

  function animate(from: number, to: number) {
    if (animId !== null) cancelAnimationFrame(animId)

    currentFrom = from
    const diff = to - from
    if (Math.abs(diff) < 0.001) {
      display.value = to
      return
    }

    const startTime = performance.now()

    function tick(now: number) {
      const elapsed = now - startTime
      const progress = Math.min(elapsed / duration, 1)
      // Ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3)

      const value = from + diff * eased
      display.value = Number(value.toFixed(decimals))

      if (progress < 1) {
        animId = requestAnimationFrame(tick)
      } else {
        display.value = Number(to.toFixed(decimals))
        animId = null
      }
    }

    animId = requestAnimationFrame(tick)
  }

  watch(
    target,
    (newVal, oldVal) => {
      const from = oldVal !== undefined ? oldVal : 0
      animate(from, newVal)
    },
    { immediate: true }
  )

  onUnmounted(() => {
    if (animId !== null) cancelAnimationFrame(animId)
  })

  return display
}

/**
 * Format a number for display with appropriate units.
 */
export function formatNumber(n: number | null | undefined, style: 'compact' | 'number' | 'percent' | 'decimal' = 'number', decimals?: number): string {
  if (n == null || isNaN(n)) return '—'

  if (style === 'compact') {
    if (n >= 1_000_000_000) return `${(n / 1e9).toFixed(decimals ?? 2)}B`
    if (n >= 1_000_000) return `${(n / 1e6).toFixed(decimals ?? 1)}M`
    if (n >= 1_000) return `${(n / 1e3).toFixed(decimals ?? 1)}K`
    return n.toFixed(decimals ?? 0)
  }

  if (style === 'percent') {
    const pct = n > 1 ? n : n * 100 // Handle both 0.47 and 47.0
    return `${pct.toFixed(decimals ?? 1)}%`
  }

  if (style === 'decimal') {
    return n.toFixed(decimals ?? 4)
  }

  // 'number' — with thousands separators
  return n.toLocaleString('en-US', {
    maximumFractionDigits: decimals ?? 0,
  })
}
