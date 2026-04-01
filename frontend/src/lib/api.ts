// Empty string = use Vite proxy (/api → http://localhost:8000). Set VITE_API_BASE_URL only for production.
const BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? ''

// ─── Types ────────────────────────────────────────────────────────────────────

export interface Repo {
  name: string
  path?: string
  analyzed_at?: string
  status?: 'ready' | 'analyzing' | 'failed'
  has_context?: boolean
  has_summary?: boolean
  repo_path?: string
}

export interface JobStatus {
  job_id: string
  status: 'pending' | 'running' | 'done' | 'completed' | 'failed'
  progress: number
  stage: string
  message?: string
  repo_name?: string
  repo?: string
  error?: string
}

export interface GraphStats {
  repo: string
  total_nodes: number
  total_edges: number
  files: number
  classes: number
  functions: number
  lines_of_code: number
  languages: Record<string, number>
  node_type_counts: Record<string, number>
  tech_stack: string[]
}

export interface Hotspot {
  id: string
  name: string
  type: string
  file?: string
  pagerank: number
  rank: number
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

type RequestOptions = RequestInit & { timeout?: number }

async function request<T>(path: string, options?: RequestOptions): Promise<T> {
  const { timeout = 30_000, ...fetchOptions } = options ?? {}
  const url = `${BASE_URL}${path}`

  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeout)

  try {
    const res = await fetch(url, {
      headers: { 'Content-Type': 'application/json', ...fetchOptions.headers },
      signal: controller.signal,
      ...fetchOptions,
    })

    if (!res.ok) {
      let message = `HTTP ${res.status}`
      try {
        const body = await res.json()
        message = body.detail ?? body.message ?? message
      } catch {
        // ignore parse errors
      }
      throw new Error(message)
    }

    return res.json() as Promise<T>
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') {
      throw new Error('Request timed out — the server took too long to respond')
    }
    throw e
  } finally {
    clearTimeout(timeoutId)
  }
}

// ─── Repos (Overview page) ───────────────────────────────────────────────────

export function listRepos(): Promise<Repo[]> {
  return request<Repo[]>('/api/repos')
}

export function analyzeRepo(path: string, name?: string): Promise<{ job_id: string }> {
  return request<{ job_id: string }>('/api/analyze', {
    method: 'POST',
    body: JSON.stringify({ repo_path: path, repo_name: name }),
  })
}

export function deleteRepo(repoName: string): Promise<{ deleted: string; cleaned: string[] }> {
  return request<{ deleted: string; cleaned: string[] }>(`/api/repos/${encodeURIComponent(repoName)}`, {
    method: 'DELETE',
  })
}

export function getJobStatus(jobId: string): Promise<JobStatus> {
  return request<JobStatus>(`/api/status/${jobId}`)
}

export function getGraphStats(repo: string): Promise<GraphStats> {
  return request<GraphStats>(`/api/graph/stats?repo=${encodeURIComponent(repo)}`)
}

export function getHotspots(repo: string, topN = 20): Promise<Hotspot[]> {
  return request<Hotspot[]>(
    `/api/graph/hotspots?repo=${encodeURIComponent(repo)}&top_n=${topN}`
  )
}

// ─── Knowledge Base (Knowledge page) ─────────────────────────────────────────

export interface KnowledgeQuestion {
  id: string
  function_id: string
  file: string
  condition: string
  condition_type: string
  explanation: string
  question: string
  suggested_answers: string[]
  answered: boolean
  answer: string
  rule_id: string
}

export interface KnowledgeRule {
  id: string
  description: string
  rule_type: string
  severity: string
  source: string
  function_id: string
  file: string
  constraint: string
  created_at: string
}

export interface KnowledgeStats {
  questions: number
  answered: number
  unanswered: number
  rules: number
  coverage: number
  by_severity: { critical: number; high: number; medium: number; low: number }
}

export interface GraphNodeSummary {
  id: string
  name: string
  type: 'Function' | 'Class' | 'File' | string
  file: string
}

export function getKnowledgeQuestions(repo: string, unansweredOnly = false): Promise<KnowledgeQuestion[]> {
  const params = new URLSearchParams({ unanswered_only: String(unansweredOnly) })
  return request<KnowledgeQuestion[]>(`/api/knowledge/${encodeURIComponent(repo)}/questions?${params}`)
}

export function submitAnswer(repo: string, questionId: string, answer: string, ruleType = 'policy', severity = 'medium'): Promise<{ rule_id: string }> {
  return request<{ rule_id: string }>(`/api/knowledge/${encodeURIComponent(repo)}/answer`, {
    method: 'POST',
    body: JSON.stringify({ question_id: questionId, answer, rule_type: ruleType, severity }),
  })
}

export function getKnowledgeRules(repo: string): Promise<KnowledgeRule[]> {
  return request<KnowledgeRule[]>(`/api/knowledge/${encodeURIComponent(repo)}/rules`)
}

export function searchGraphNodes(repo: string, q: string, nodeType?: string): Promise<GraphNodeSummary[]> {
  const params = new URLSearchParams({ q, limit: '20' })
  if (nodeType) params.set('node_type', nodeType)
  return request<GraphNodeSummary[]>(`/api/knowledge/${encodeURIComponent(repo)}/nodes/search?${params}`)
}

export function addRule(repo: string, rule: {
  description: string
  rule_type: string
  severity: string
  file?: string
  constraint?: string
  node_id?: string
  node_type?: string
  node_name?: string
}): Promise<{ rule_id: string }> {
  return request<{ rule_id: string }>(`/api/knowledge/${encodeURIComponent(repo)}/rules`, {
    method: 'POST',
    body: JSON.stringify(rule),
  })
}

export function getKnowledgeStats(repo: string): Promise<KnowledgeStats> {
  return request<KnowledgeStats>(`/api/knowledge/${encodeURIComponent(repo)}/stats`)
}

// ─── Agent Pipeline (Agent page) ─────────────────────────────────────────────

export interface AgentTicket {
  ticket_id: string
  title: string
  description: string
  repo_name: string
  priority: string
  affected_component?: string
  comments: string[]
}

export interface AgentJobStatus {
  job_id: string
  status: 'pending' | 'running' | 'done' | 'failed' | 'escalated'
  stage: string
  iteration_count: number
  result: {
    intent?: Record<string, unknown>
    localization?: {
      fault_files: string[]
      fault_functions: string[]
      root_cause_hypothesis: string
      confidence: number
      evidence: string[]
    }
    repair?: {
      patches: { file_path: string; original_code: string; patched_code: string; explanation: string }[]
      test_patches?: { file_path: string; original_code: string; patched_code: string; explanation: string }[]
      explanation: string
      tests_added: string[]
    }
    review?: {
      verdict: string
      confidence: number
      checks: { name: string; status: string; comment: string }[]
      feedback: string
    }
    pr_url?: string
    context_nodes?: number
    test_result?: string
  } | null
  error: string
  debug?: boolean
}

export interface TraceEvent {
  index: number
  timestamp: number
  wall_time: string
  event_type: 'stage_start' | 'stage_end' | 'llm_request' | 'llm_response' | 'tool_call' | 'tool_result' | 'patch_candidate' | 'test_output' | 'error' | 'info'
  stage: string
  data: Record<string, unknown>
}

export function listAgentTickets(): Promise<AgentTicket[]> {
  return request<AgentTicket[]>('/api/agent/tickets')
}

export function runMockTicket(ticketId: string, debug = false): Promise<{ job_id: string; status: string; debug?: boolean }> {
  const params = debug ? '?debug=true' : ''
  return request<{ job_id: string; status: string; debug?: boolean }>(`/api/agent/run-mock/${ticketId}${params}`, {
    method: 'POST',
  })
}

export function runAgentTicket(ticket: {
  ticket_id?: string
  title: string
  description: string
  repo_name: string
  repo_path?: string
  priority?: string
  affected_component?: string
  debug?: boolean
}): Promise<{ job_id: string; status: string; debug?: boolean }> {
  return request<{ job_id: string; status: string; debug?: boolean }>('/api/agent/run', {
    method: 'POST',
    body: JSON.stringify(ticket),
    timeout: 10_000,
  })
}

export function subscribeToTrace(
  jobId: string,
  onEvent: (event: TraceEvent) => void,
  onDone: () => void,
  onError?: (err: Event) => void,
): () => void {
  const url = `${BASE_URL}/api/agent/trace/${jobId}`
  const eventSource = new EventSource(url)

  eventSource.onmessage = (e) => {
    try {
      onEvent(JSON.parse(e.data) as TraceEvent)
    } catch {
      // ignore parse errors
    }
  }

  eventSource.addEventListener('done', () => {
    onDone()
    eventSource.close()
  })

  eventSource.onerror = (e) => {
    if (onError) onError(e)
    eventSource.close()
  }

  return () => eventSource.close()
}

export function getAgentJobStatus(jobId: string): Promise<AgentJobStatus> {
  return request<AgentJobStatus>(`/api/agent/status/${jobId}`)
}

export function listAgentJobs(): Promise<AgentJobStatus[]> {
  return request<AgentJobStatus[]>('/api/agent/jobs')
}
