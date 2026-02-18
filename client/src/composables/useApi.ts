/* ============================================================
   useApi — Centralized API Client with SSE & Demo Fallback
   ============================================================
   Singleton composable: shared state across all components.
   Falls back to embedded demo data if the backend is unreachable.
   ============================================================ */

import { reactive, toRefs, shallowRef, readonly } from 'vue'
import type { RunInfo, Version, MetricSnapshot, ConnectionState } from '../types'

// ============================================================================
// Demo Data — Realistic fallback from training reports
// ============================================================================

const DEMO_RUN_INFO: RunInfo = {
  id: '9ue9rans',
  name: 'VYREX-2v2-PPO',
  state: 'finished',
  createdAt: '2026-02-15T11:55:49',
  heartbeatAt: '2026-02-16T09:30:00',
  tags: ['2v2', 'ppo', 'rlgym-v2', 'rocketsim'],
  totalSteps: 897_000_000,
}

const DEMO_LATEST: MetricSnapshot = {
  _step: 897_000_000,
  reward: 24.84,
  entropy: 4.229,
  kl_divergence: 0.00137,
  vf_loss: 0.013,
  clip_fraction: 0.01155,
  policy_update_magnitude: 0.3591,
  overall_sps: 18060,
  'game/touches_per_step': 0.075,
  'game/aerial_touches_per_step': 0.011,
  'game/avg_speed': 976,
  'game/airborne_fraction': 0.478,
  'game/avg_boost': 7.6,
  'game/goals_blue': 25.9,
  'game/goals_orange': 24.9,
  'game/avg_teammate_dist': 2949,
}

function generateDemoHistory(): Record<string, number[]> {
  const steps = Array.from({ length: 100 }, (_, i) => Math.round(i * 8_970_000))
  const reward = steps.map((s) => {
    if (s < 251e6) return 5 + Math.random() * 8 + (s / 251e6) * 12
    if (s < 401e6) return 18 + Math.random() * 10
    if (s < 433e6) return 40 + Math.random() * 30
    return 20 + Math.random() * 15 + (s - 433e6) / 500e6 * 8
  })
  const entropy = steps.map((s) => {
    if (s < 401e6) return 4.45 + Math.random() * 0.02
    if (s < 461e6) return 4.15 - (s - 401e6) / 60e6 * 0.04 + Math.random() * 0.02
    return 4.22 + Math.random() * 0.015
  })
  const airborne = steps.map((s) => {
    if (s < 251e6) return 0.85 + Math.random() * 0.02
    if (s < 401e6) return 0.85 - (s - 251e6) / 150e6 * 0.15 + Math.random() * 0.03
    if (s < 433e6) return 0.032 + Math.random() * 0.01
    if (s < 461e6) return 0.043 + Math.random() * 0.008
    if (s < 521e6) return 0.044 + (s - 461e6) / 60e6 * 0.3 + Math.random() * 0.03
    return 0.42 + Math.random() * 0.06
  })
  const touches = steps.map((s) => {
    if (s < 251e6) return 0.005 + (s / 251e6) * 0.008 + Math.random() * 0.002
    if (s < 401e6) return 0.013 + (s - 251e6) / 150e6 * 0.013 + Math.random() * 0.003
    if (s < 521e6) return 0.042 + (s - 401e6) / 120e6 * 0.01 + Math.random() * 0.005
    return 0.05 + (s - 521e6) / 376e6 * 0.025 + Math.random() * 0.005
  })
  const speed = steps.map((s) => {
    if (s < 251e6) return 200 + (s / 251e6) * 300 + Math.random() * 50
    if (s < 401e6) return 500 + (s - 251e6) / 150e6 * 170 + Math.random() * 40
    if (s < 461e6) return 800 + Math.random() * 80
    return 880 + (s - 461e6) / 436e6 * 100 + Math.random() * 40
  })
  const sps = steps.map(() => 17500 + Math.random() * 1200)

  return { _step: steps, reward, entropy, 'game/airborne_fraction': airborne, 'game/touches_per_step': touches, 'game/avg_speed': speed, overall_sps: sps }
}

// ============================================================================
// Shared State (module-level singleton)
// ============================================================================

interface ApiState {
  connectionState: ConnectionState
  isLoading: boolean
  error: string | null
  isDemo: boolean
  latestMetrics: MetricSnapshot
  runInfo: RunInfo | null
  versions: Version[]
}

const state = reactive<ApiState>({
  connectionState: 'connecting',
  isLoading: true,
  error: null,
  isDemo: false,
  latestMetrics: {},
  runInfo: null,
  versions: [],
})

const metricHistory = shallowRef<Record<string, number[]>>({})
let eventSource: EventSource | null = null
let initialized = false

// ============================================================================
// Internal Helpers
// ============================================================================

async function fetchJson<T>(url: string, timeout = 8000): Promise<T> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeout)
  try {
    const res = await fetch(url, { signal: controller.signal })
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
    return await res.json()
  } finally {
    clearTimeout(timer)
  }
}

function connectSSE() {
  if (eventSource) {
    eventSource.close()
  }

  eventSource = new EventSource('/api/metrics/stream')

  eventSource.onopen = () => {
    state.connectionState = 'connected'
  }

  eventSource.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data)
      if (data._error) {
        state.connectionState = 'error'
        return
      }
      state.latestMetrics = { ...state.latestMetrics, ...data }
      state.connectionState = 'connected'
    } catch {
      // Ignore parse errors
    }
  }

  eventSource.onerror = () => {
    state.connectionState = 'disconnected'
    eventSource?.close()
    // Reconnect with backoff
    setTimeout(() => {
      if (!state.isDemo) connectSSE()
    }, 10_000)
  }
}

function loadDemoData() {
  state.isDemo = true
  state.runInfo = DEMO_RUN_INFO
  state.latestMetrics = DEMO_LATEST
  metricHistory.value = generateDemoHistory()
  state.isLoading = false
  state.connectionState = 'disconnected'
}

async function init() {
  if (initialized) return
  initialized = true

  state.isLoading = true
  state.error = null

  try {
    // Try fetching versions (always served, even without wandb)
    const [versions, health] = await Promise.all([
      fetchJson<Version[]>('/api/versions'),
      fetchJson<{ status: string; wandb_configured: boolean }>('/api/health'),
    ])

    state.versions = versions

    if (!health.wandb_configured) {
      // Server running but wandb not configured — use demo metrics
      loadDemoData()
      state.versions = versions // Keep real versions
      return
    }

    // Fetch live data
    const [runInfo, latest] = await Promise.all([
      fetchJson<RunInfo>('/api/run'),
      fetchJson<MetricSnapshot>('/api/metrics/latest'),
    ])

    if ((runInfo as { error?: string }).error || (latest as { error?: string }).error) {
      throw new Error('WandB returned error')
    }

    state.runInfo = runInfo
    state.latestMetrics = latest
    state.isLoading = false

    // Start SSE
    connectSSE()

    // Fetch history in background
    fetchHistory()
  } catch {
    // Backend unreachable — fall back to demo
    loadDemoData()
  }
}

async function fetchHistory(
  keys: string[] = ['reward', 'entropy', 'game/airborne_fraction', 'game/touches_per_step', 'game/avg_speed', 'overall_sps'],
  samples: number = 500
) {
  try {
    const params = new URLSearchParams({
      keys: keys.join(','),
      samples: String(samples),
    })
    const data = await fetchJson<Record<string, number[]>>(
      `/api/metrics/history?${params}`
    )
    if (!(data as { error?: string }).error) {
      metricHistory.value = data
    }
  } catch {
    // Use demo history if fetch fails
    if (Object.keys(metricHistory.value).length === 0) {
      metricHistory.value = generateDemoHistory()
    }
  }
}

async function refresh() {
  try {
    const latest = await fetchJson<MetricSnapshot>('/api/metrics/latest')
    if (!(latest as { error?: string }).error) {
      state.latestMetrics = { ...state.latestMetrics, ...latest }
    }
  } catch {
    // Ignore refresh errors
  }
}

// ============================================================================
// Public Composable
// ============================================================================

export function useApi() {
  init() // idempotent

  return {
    ...toRefs(readonly(state)),
    metricHistory: readonly(metricHistory),
    fetchHistory,
    refresh,
  }
}
