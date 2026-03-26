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

export interface GraphNode {
  id: string
  name: string
  type: 'File' | 'Class' | 'Function' | 'BusinessRule' | 'DomainConcept' | 'DecisionPoint'
  file?: string
  summary?: string
  pagerank?: number
  layer?: string
  language?: string
  line_start?: number
  line_end?: number
  [key: string]: unknown
}

export interface GraphEdge {
  id?: string
  source: string
  target: string
  type: 'CONTAINS' | 'CALLS' | 'IMPORTS' | 'INHERITS' | 'RELATED_TO' | string
  weight?: number
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
  repo: string
  layer?: string
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
  type: GraphNode['type']
  file?: string
  pagerank: number
  rank: number
}

export interface ContextLayer {
  layer: number
  name: string
  description: string
  node_count: number
  token_estimate: number
  completeness: number
}

export interface ContextLayersResponse {
  repo: string
  layers: ContextLayer[]
  total_tokens: number
  token_budget: number
}

export interface SearchResult {
  id: string
  name: string
  type: GraphNode['type']
  file?: string
  snippet?: string
  score: number
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${BASE_URL}${path}`
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
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
}

// ─── API Functions ────────────────────────────────────────────────────────────

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

export function getRepoDetail(repoName: string): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>(`/api/repos/${encodeURIComponent(repoName)}`)
}

export function getJobStatus(jobId: string): Promise<JobStatus> {
  return request<JobStatus>(`/api/status/${jobId}`)
}

// ─── Knowledge Base (Business Rules) ─────────────────────────────────────────

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

export function addRule(repo: string, rule: {
  description: string
  rule_type: string
  severity: string
  file?: string
  constraint?: string
}): Promise<{ rule_id: string }> {
  return request<{ rule_id: string }>(`/api/knowledge/${encodeURIComponent(repo)}/rules`, {
    method: 'POST',
    body: JSON.stringify(rule),
  })
}

export function getKnowledgeStats(repo: string): Promise<KnowledgeStats> {
  return request<KnowledgeStats>(`/api/knowledge/${encodeURIComponent(repo)}/stats`)
}

export function getGraph(repo: string, layer?: string, limit = 5000): Promise<GraphData> {
  const params = new URLSearchParams({ repo, limit: String(limit) })
  if (layer) params.set('layer', layer)
  return request<GraphData>(`/api/graph?${params}`)
}

export function getGraphStats(repo: string): Promise<GraphStats> {
  return request<GraphStats>(`/api/graph/stats?repo=${encodeURIComponent(repo)}`)
}

export function getHotspots(repo: string, topN = 20): Promise<Hotspot[]> {
  return request<Hotspot[]>(
    `/api/graph/hotspots?repo=${encodeURIComponent(repo)}&top_n=${topN}`
  )
}

export function getContextLayers(repo: string): Promise<ContextLayersResponse> {
  return request<ContextLayersResponse>(
    `/api/context/layers?repo=${encodeURIComponent(repo)}`
  )
}

export function getContextFull(repo: string): Promise<{ content: string }> {
  return request<{ content: string }>(
    `/api/context/full?repo=${encodeURIComponent(repo)}`
  )
}

export function search(repo: string, q: string): Promise<SearchResult[]> {
  return request<SearchResult[]>(
    `/api/search?repo=${encodeURIComponent(repo)}&q=${encodeURIComponent(q)}`
  )
}

// ─── Chat ────────────────────────────────────────────────────────────────────

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface ChatResponse {
  answer: string
  model: string
  usage: { input_tokens: number; output_tokens: number }
}

export function chatAsk(
  repo: string,
  question: string,
  history: ChatMessage[] = [],
): Promise<ChatResponse> {
  return request<ChatResponse>('/api/chat', {
    method: 'POST',
    body: JSON.stringify({ repo, question, history }),
  })
}

// ─── Agent Pipeline ─────────────────────────────────────────────────────────

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
      patches: { file_path: string; explanation: string }[]
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
}

export function listAgentTickets(): Promise<AgentTicket[]> {
  return request<AgentTicket[]>('/api/agent/tickets')
}

export function runMockTicket(ticketId: string): Promise<{ job_id: string; status: string }> {
  return request<{ job_id: string; status: string }>(`/api/agent/run-mock/${ticketId}`, {
    method: 'POST',
  })
}

export function runAgentTicket(ticket: {
  title: string
  description: string
  repo_name: string
  repo_path?: string
  priority?: string
}): Promise<{ job_id: string; status: string }> {
  return request<{ job_id: string; status: string }>('/api/agent/run', {
    method: 'POST',
    body: JSON.stringify(ticket),
  })
}

export function getAgentJobStatus(jobId: string): Promise<AgentJobStatus> {
  return request<AgentJobStatus>(`/api/agent/status/${jobId}`)
}

export function listAgentJobs(): Promise<AgentJobStatus[]> {
  return request<AgentJobStatus[]>('/api/agent/jobs')
}
