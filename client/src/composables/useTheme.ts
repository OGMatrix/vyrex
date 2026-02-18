/* ============================================================
   useTheme — Dark / Light Mode Composable
   ============================================================
   Persists to localStorage, respects prefers-color-scheme,
   applies data-theme attribute on <html>.
   ============================================================ */

import { ref, watch, onMounted } from 'vue'

export type Theme = 'dark' | 'light'

const STORAGE_KEY = 'vyrex-theme'
const theme = ref<Theme>('dark')
let initialized = false

function applyTheme(t: Theme) {
  document.documentElement.setAttribute('data-theme', t)
  document.querySelector('meta[name="theme-color"]')?.setAttribute(
    'content',
    t === 'dark' ? '#07070C' : '#F8F9FC'
  )
}

function detectSystemTheme(): Theme {
  if (typeof window === 'undefined') return 'dark'
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
}

export function useTheme() {
  if (!initialized) {
    initialized = true

    // Read from localStorage or system preference
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored === 'dark' || stored === 'light') {
      theme.value = stored
    } else {
      theme.value = detectSystemTheme()
    }

    applyTheme(theme.value)

    // Watch for changes
    watch(theme, (t) => {
      applyTheme(t)
      localStorage.setItem(STORAGE_KEY, t)
    })

    // Listen for system theme changes
    if (typeof window !== 'undefined') {
      window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', (e) => {
        // Only auto-switch if user hasn't manually chosen
        if (!localStorage.getItem(STORAGE_KEY)) {
          theme.value = e.matches ? 'light' : 'dark'
        }
      })
    }
  }

  function toggleTheme() {
    theme.value = theme.value === 'dark' ? 'light' : 'dark'
  }

  return {
    theme,
    toggleTheme,
    isDark: () => theme.value === 'dark',
  }
}
