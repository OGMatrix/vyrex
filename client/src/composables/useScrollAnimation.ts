/* ============================================================
   useScrollAnimation — IntersectionObserver-based Scroll Reveal
   ============================================================
   Adds 'visible' class to elements when they enter the viewport.
   Supports threshold and once (default: true) options.
   ============================================================ */

import { ref, onMounted, onUnmounted, type Ref } from 'vue'

interface ScrollAnimationOptions {
  threshold?: number
  once?: boolean
  rootMargin?: string
}

export function useScrollAnimation(
  target: Ref<HTMLElement | undefined>,
  options: ScrollAnimationOptions = {}
) {
  const isVisible = ref(false)
  let observer: IntersectionObserver | null = null

  const {
    threshold = 0.12,
    once = true,
    rootMargin = '0px 0px -40px 0px',
  } = options

  onMounted(() => {
    if (!target.value) return

    observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          isVisible.value = true
          if (once && observer) {
            observer.disconnect()
          }
        } else if (!once) {
          isVisible.value = false
        }
      },
      { threshold, rootMargin }
    )

    observer.observe(target.value)
  })

  onUnmounted(() => {
    observer?.disconnect()
  })

  return isVisible
}

/**
 * Observe multiple children of a container for staggered animations.
 * Returns a Set of indices that are visible.
 */
export function useStaggeredReveal(
  container: Ref<HTMLElement | undefined>,
  childSelector: string = '.scroll-reveal',
  options: ScrollAnimationOptions = {}
) {
  const visibleIndices = ref<Set<number>>(new Set())
  const observers: IntersectionObserver[] = []

  const {
    threshold = 0.1,
    once = true,
    rootMargin = '0px 0px -20px 0px',
  } = options

  onMounted(() => {
    if (!container.value) return

    const children = container.value.querySelectorAll(childSelector)
    children.forEach((child, index) => {
      const obs = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) {
            visibleIndices.value = new Set([...visibleIndices.value, index])
            ;(child as HTMLElement).classList.add('visible')
            if (once) obs.disconnect()
          }
        },
        { threshold, rootMargin }
      )
      obs.observe(child)
      observers.push(obs)
    })
  })

  onUnmounted(() => {
    observers.forEach((obs) => obs.disconnect())
  })

  return visibleIndices
}
