/* ============================================================
   VYREX Dashboard — TypeScript Interfaces
   ============================================================ */

export interface RunInfo {
  id: string
  name: string
  state: string
  createdAt: string
  heartbeatAt: string | null
  tags: string[]
  totalSteps: number
  config?: Record<string, unknown>
  error?: string
}

export interface MetricSnapshot {
  [key: string]: number | null
}

export interface MetricHistory {
  _step: number[]
  [key: string]: number[]
}

export interface ConfigChange {
  parameter: string
  oldValue: string
  newValue: string
  rationale: string
}

export interface MetricImpact {
  metric: string
  before: number | null
  after: number
  unit: string
  trend: 'up' | 'down' | 'stable'
  isGood: boolean
}

export interface VersionAssessment {
  grade: string
  summary: string
}

export interface Version {
  id: string
  name: string
  stepRange: [number, number]
  date: string
  status: 'completed' | 'active' | 'planned'
  summary: string
  changes: ConfigChange[]
  impact: MetricImpact[]
  assessment: VersionAssessment
}

/** Definition for a metric card in the live dashboard */
export interface MetricDef {
  key: string
  label: string
  format: 'number' | 'percent' | 'compact' | 'decimal'
  decimals?: number
  suffix?: string
  icon: string
  color: string
  description: string
  goodDirection: 'up' | 'down' | 'stable'
}

/** SSE connection state */
export type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'error'
