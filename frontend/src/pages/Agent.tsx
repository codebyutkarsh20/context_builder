import { useState, useEffect, useCallback, useRef } from 'react'
import {
  Cpu, Play, Loader2, CheckCircle2, XCircle, AlertTriangle,
  ArrowRight, FileCode, Bug, Wrench, Eye, GitPullRequest,
  Target, Brain, Shield, Clock, Search,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { useRepo } from '../lib/RepoContext'
import {
  listAgentTickets, runMockTicket, runAgentTicket, getAgentJobStatus, listAgentJobs,
  type AgentTicket, type AgentJobStatus,
} from '../lib/api'

const MAX_ITERATIONS = 3
const MAX_POLL_COUNT = 900 // 30 min × (2s interval)

// ─── Pipeline stage config ──────────────────────────────────────────────────

const STAGES = [
  { key: 'intake', label: 'Intake', icon: Bug, color: 'text-blue-400', bg: 'bg-blue-500' },
  { key: 'exploring', label: 'Explore', icon: Search, color: 'text-teal-400', bg: 'bg-teal-500' },
  { key: 'context_assembly', label: 'Context', icon: Brain, color: 'text-purple-400', bg: 'bg-purple-500' },
  { key: 'localizing', label: 'Localize', icon: Target, color: 'text-amber-400', bg: 'bg-amber-500' },
  { key: 'reading_source', label: 'Read Code', icon: FileCode, color: 'text-yellow-400', bg: 'bg-yellow-500' },
  { key: 'repairing', label: 'Repair', icon: Wrench, color: 'text-orange-400', bg: 'bg-orange-500' },
  { key: 'reviewing', label: 'Review', icon: Eye, color: 'text-cyan-400', bg: 'bg-cyan-500' },
  { key: 'testing', label: 'Test', icon: Shield, color: 'text-lime-400', bg: 'bg-lime-500' },
  { key: 'pr_creating', label: 'PR', icon: GitPullRequest, color: 'text-green-400', bg: 'bg-green-500' },
]

function getStageIndex(status: string): number {
  if (status === 'pending' || status === 'Queued' || status === 'Starting pipeline') return 0
  const idx = STAGES.findIndex(s => s.key === status)
  if (status === 'done' || status === 'escalated') return STAGES.length
  return idx >= 0 ? idx : -1
}

const isTerminal = (status: string) =>
  status === 'done' || status === 'failed' || status === 'escalated'

// ─── Sub-components ─────────────────────────────────────────────────────────

function TicketCard({ ticket, onRun, isRunning, isDisabled }: {
  ticket: AgentTicket
  onRun: (id: string) => void
  isRunning: boolean
  isDisabled: boolean
}) {
  const priorityColors: Record<string, string> = {
    critical: 'bg-red-500/20 text-red-300 border-red-500/30',
    high: 'bg-orange-500/20 text-orange-300 border-orange-500/30',
    medium: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
    low: 'bg-green-500/20 text-green-300 border-green-500/30',
  }

  return (
    <div className={cn(
      "p-4 rounded-xl bg-zinc-900 border transition-all",
      isRunning ? 'border-rose-500/40 bg-rose-950/10' : 'border-zinc-800 hover:border-zinc-700'
    )}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-mono text-zinc-600">{ticket.ticket_id}</span>
            <span className={cn('text-[10px] px-1.5 py-0.5 rounded-full border font-medium',
              priorityColors[ticket.priority] || priorityColors.medium)}>
              {ticket.priority}
            </span>
          </div>
          <p className="text-sm font-medium text-zinc-200 mb-1">{ticket.title}</p>
          <p className="text-xs text-zinc-500 line-clamp-2">{ticket.description}</p>
        </div>
        <button
          onClick={() => onRun(ticket.ticket_id)}
          disabled={isDisabled}
          className={cn(
            'flex-shrink-0 flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium transition-all',
            isRunning
              ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30 cursor-wait'
              : isDisabled
              ? 'bg-zinc-800 text-zinc-600 cursor-not-allowed'
              : 'bg-rose-500/20 text-rose-300 hover:bg-rose-500/30 border border-rose-500/30'
          )}
        >
          {isRunning ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
          {isRunning ? 'Running...' : 'Run Agent'}
        </button>
      </div>
    </div>
  )
}

function PipelineProgress({ status, iterationCount }: { status: string; iterationCount: number }) {
  const currentIdx = getStageIndex(status)
  const isDone = status === 'done'
  const isEscalated = status === 'escalated'
  const isFailed = status === 'failed'

  return (
    <div className="flex items-center gap-1">
      {STAGES.map((stage, idx) => {
        const Icon = stage.icon
        const isActive = idx === currentIdx
        const isComplete = idx < currentIdx
        return (
          <div key={stage.key} className="flex items-center gap-1">
            <div className={cn(
              'flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all',
              isActive ? `${stage.bg}/20 ${stage.color} border border-current/30` :
              isComplete ? 'bg-zinc-800 text-zinc-400' :
              'bg-zinc-900 text-zinc-700'
            )}>
              {isActive && !isDone && !isEscalated ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : isComplete || isDone ? (
                <CheckCircle2 className="w-3 h-3 text-green-400" />
              ) : (
                <Icon className="w-3 h-3" />
              )}
              <span className="hidden sm:inline">{stage.label}</span>
            </div>
            {idx < STAGES.length - 1 && (
              <ArrowRight className={cn('w-3 h-3 flex-shrink-0', isComplete ? 'text-zinc-600' : 'text-zinc-800')} />
            )}
          </div>
        )
      })}
      {iterationCount > 1 && (
        <span className="ml-2 text-[10px] text-zinc-600 font-mono">
          iter {iterationCount}/{MAX_ITERATIONS}
        </span>
      )}
      {isDone && <CheckCircle2 className="ml-2 w-4 h-4 text-green-400" />}
      {isEscalated && <AlertTriangle className="ml-2 w-4 h-4 text-yellow-400" />}
      {isFailed && <XCircle className="ml-2 w-4 h-4 text-red-400" />}
    </div>
  )
}

function ReviewChecks({ checks }: { checks: { name: string; status: string; comment: string }[] }) {
  if (!checks?.length) return null
  const statusIcon = (s: string) =>
    s === 'PASS' ? <CheckCircle2 className="w-3.5 h-3.5 text-green-400" /> :
    s === 'FAIL' ? <XCircle className="w-3.5 h-3.5 text-red-400" /> :
    <AlertTriangle className="w-3.5 h-3.5 text-yellow-400" />

  return (
    <div className="space-y-1.5">
      {checks.map((check) => (
        <div key={check.name} className="flex items-start gap-2 px-3 py-2 rounded-lg bg-zinc-900/60 border border-zinc-800/40">
          {statusIcon(check.status)}
          <div className="flex-1 min-w-0">
            <span className="text-xs font-bold text-zinc-400">{check.name}</span>
            <p className="text-xs text-zinc-500 mt-0.5">{check.comment}</p>
          </div>
        </div>
      ))}
    </div>
  )
}

// ─── Status helpers ─────────────────────────────────────────────────────────

function getOutcomeDisplay(activeJob: AgentJobStatus) {
  const result = activeJob.result
  const status = activeJob.status
  const reviewVerdict = result?.review?.verdict

  // Pipeline-level status takes precedence
  if (status === 'escalated') {
    return { label: 'Escalated to Human', icon: AlertTriangle, iconColor: 'text-yellow-400', bg: 'bg-yellow-950/30 border-yellow-800/40' }
  }
  if (status === 'failed') {
    return { label: 'Pipeline Failed', icon: XCircle, iconColor: 'text-red-400', bg: 'bg-red-950/30 border-red-800/40' }
  }

  // Then check review verdict for done jobs
  if (status === 'done' && reviewVerdict === 'APPROVE') {
    return { label: 'Fix Approved', icon: CheckCircle2, iconColor: 'text-green-400', bg: 'bg-green-950/30 border-green-800/40' }
  }
  if (status === 'done' && (reviewVerdict === 'ESCALATE' || reviewVerdict === 'CHANGES_REQUESTED')) {
    return { label: reviewVerdict === 'ESCALATE' ? 'Escalated to Human' : 'Changes Requested', icon: AlertTriangle, iconColor: 'text-yellow-400', bg: 'bg-yellow-950/30 border-yellow-800/40' }
  }
  // Done with unexpected/missing verdict — still show as completed
  if (status === 'done') {
    return { label: 'Completed', icon: CheckCircle2, iconColor: 'text-green-400', bg: 'bg-green-950/30 border-green-800/40' }
  }

  // Still running
  return { label: 'Running...', icon: Loader2, iconColor: 'text-blue-400', bg: 'bg-zinc-900 border-zinc-800' }
}

// ─── Main Page ──────────────────────────────────────────────────────────────

const STORAGE_KEY = 'agent_active_job'

const RESULT_STORAGE_KEY = 'agent_last_result'

function saveJobToStorage(jobId: string, ticketId: string) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify({ jobId, ticketId }))
}

function saveResultToStorage(job: AgentJobStatus) {
  try { localStorage.setItem(RESULT_STORAGE_KEY, JSON.stringify(job)) } catch {}
}

function loadResultFromStorage(): AgentJobStatus | null {
  try {
    const raw = localStorage.getItem(RESULT_STORAGE_KEY)
    return raw ? JSON.parse(raw) : null
  } catch { return null }
}

function loadJobFromStorage(): { jobId: string; ticketId: string } | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : null
  } catch { return null }
}

function clearJobStorage() {
  localStorage.removeItem(STORAGE_KEY)
}

export default function AgentPage() {
  const { activeRepo, activeRepoData } = useRepo()
  const [tickets, setTickets] = useState<AgentTicket[]>([])
  const [activeJob, setActiveJob] = useState<AgentJobStatus | null>(() => loadResultFromStorage())
  const [runningTicketId, setRunningTicketId] = useState<string | null>(() => {
    // Initialize from localStorage so UI doesn't flash empty on refresh
    const saved = loadJobFromStorage()
    return saved ? saved.ticketId : null
  })
  const [error, setError] = useState<string | null>(null)
  const [pastJobs, setPastJobs] = useState<AgentJobStatus[]>([])
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const pollCountRef = useRef(0)

  // Custom ticket form
  const [customTicketId, setCustomTicketId] = useState('')
  const [customTitle, setCustomTitle] = useState('')
  const [customDesc, setCustomDesc] = useState('')
  const [customComponent, setCustomComponent] = useState('')
  const [customRepoPath, setCustomRepoPath] = useState('')
  const [showCustom, setShowCustom] = useState(false)

  useEffect(() => {
    listAgentTickets().then(setTickets).catch((e: Error) => setError(`Failed to load tickets: ${e.message}`))
  }, [])

  useEffect(() => {
    listAgentJobs().then(setPastJobs).catch(() => {})
  }, [])

  // Cleanup poll on unmount (but don't clear storage — job keeps running)
  useEffect(() => {
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [])

  const pollJob = useCallback((id: string) => {
    if (pollRef.current) clearInterval(pollRef.current)
    pollCountRef.current = 0
    pollRef.current = setInterval(async () => {
      pollCountRef.current += 1
      if (pollCountRef.current > MAX_POLL_COUNT) {
        if (pollRef.current) clearInterval(pollRef.current)
        pollRef.current = null
        setRunningTicketId(null)
        clearJobStorage()
        setError('Agent pipeline timed out after 30 minutes — check backend logs')
        return
      }
      try {
        const status = await getAgentJobStatus(id)
        setActiveJob(status)
        if (isTerminal(status.status)) {
          if (pollRef.current) clearInterval(pollRef.current)
          pollRef.current = null
          setRunningTicketId(null)
          clearJobStorage()
          saveResultToStorage(status)
          listAgentJobs().then(setPastJobs).catch(() => {})
        }
      } catch {
        if (pollRef.current) clearInterval(pollRef.current)
        pollRef.current = null
        setRunningTicketId(null)
        clearJobStorage()
        setError('Lost connection to agent job — the pipeline may still be running in the backend')
      }
    }, 2000)
  }, [])

  // On mount: resume polling if there's an active job from a previous visit
  useEffect(() => {
    const saved = loadJobFromStorage()
    if (saved) {
      setRunningTicketId(saved.ticketId)
      // Fetch current status immediately, then start polling
      getAgentJobStatus(saved.jobId).then((status) => {
        setActiveJob(status)
        if (isTerminal(status.status)) {
          setRunningTicketId(null)
          clearJobStorage()
          saveResultToStorage(status)
        } else {
          pollJob(saved.jobId)
        }
      }).catch(() => {
        // Job no longer exists on backend (server restarted?)
        clearJobStorage()
        setRunningTicketId(null)
      })
    }
  }, [pollJob])

  const handleRunMock = async (ticketId: string) => {
    setError(null)
    setRunningTicketId(ticketId)
    setActiveJob(null)
    try {
      const res = await runMockTicket(ticketId)
      saveJobToStorage(res.job_id, ticketId)
      setActiveJob({ job_id: res.job_id, status: 'pending', stage: 'Queued', iteration_count: 0, result: null, error: '' })
      pollJob(res.job_id)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to start agent')
      setRunningTicketId(null)
    }
  }

  const handleRunCustom = async () => {
    if (!customTitle.trim() || !activeRepo) return
    setError(null)
    setRunningTicketId('custom')
    setActiveJob(null)
    try {
      const res = await runAgentTicket({
        ticket_id: customTicketId.trim() || undefined,
        title: customTitle,
        description: customDesc,
        repo_name: activeRepo,
        repo_path: customRepoPath.trim() || activeRepoData?.repo_path,
        affected_component: customComponent.trim() || undefined,
      })
      saveJobToStorage(res.job_id, 'custom')
      setActiveJob({ job_id: res.job_id, status: 'pending', stage: 'Queued', iteration_count: 0, result: null, error: '' })
      pollJob(res.job_id)
      setCustomTicketId('')
      setCustomTitle('')
      setCustomDesc('')
      setCustomComponent('')
      setCustomRepoPath('')
      setShowCustom(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to start agent')
      setRunningTicketId(null)
    }
  }

  const result = activeJob?.result
  const isRunning = runningTicketId !== null

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex-shrink-0 px-6 py-4 border-b border-zinc-700/50 bg-zinc-900 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-rose-500/20 flex items-center justify-center">
            <Cpu className="w-4 h-4 text-rose-400" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-zinc-100 flex items-center gap-2">
              AI Deploy Agent
              {activeRepo && <span className="text-zinc-500 font-normal font-mono text-xs">-- {activeRepo}</span>}
            </h1>
            <p className="text-[10px] text-zinc-600">{'Jira ticket \u2192 Fix \u2192 Review \u2192 PR'}</p>
          </div>
        </div>
        {activeJob && (
          <PipelineProgress status={activeJob.stage || activeJob.status} iterationCount={activeJob.iteration_count} />
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Error */}
          {error && (
            <div className="flex items-center gap-2 p-3 rounded-xl bg-red-950/30 border border-red-900/40 text-red-400 text-sm">
              <XCircle className="w-4 h-4 flex-shrink-0" />{error}
            </div>
          )}

          {/* Pipeline Result */}
          {result && (
            <div className="space-y-4">
              {/* Status banner */}
              {(() => {
                const outcome = getOutcomeDisplay(activeJob!)
                const OutcomeIcon = outcome.icon
                return (
                  <div className={cn('p-4 rounded-xl border', outcome.bg)}>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <OutcomeIcon className={cn('w-5 h-5', outcome.iconColor, outcome.label === 'Running...' && 'animate-spin')} />
                        <span className="text-sm font-bold text-zinc-200">{outcome.label}</span>
                      </div>
                      <div className="flex items-center gap-3 text-xs text-zinc-500">
                        {result.context_nodes ? (
                          <span className="flex items-center gap-1"><Brain className="w-3 h-3" /> {result.context_nodes} nodes</span>
                        ) : null}
                        <span className="flex items-center gap-1"><Clock className="w-3 h-3" /> {activeJob?.iteration_count || 0} iterations</span>
                        {result.review?.confidence ? (
                          <span className="font-mono">{Math.round(Number(result.review.confidence) * 100)}% confidence</span>
                        ) : null}
                      </div>
                    </div>
                    {activeJob?.error && (
                      <p className="text-xs text-red-400 mt-1">{activeJob.error}</p>
                    )}
                  </div>
                )
              })()}

              {/* Localization */}
              {result.localization && (
                <div className="p-4 rounded-xl bg-zinc-900 border border-zinc-800">
                  <div className="flex items-center gap-2 mb-3">
                    <Target className="w-4 h-4 text-amber-400" />
                    <h3 className="text-xs font-bold text-zinc-400 uppercase tracking-wider">Fault Localization</h3>
                    {result.localization.confidence > 0 && (
                      <span className="ml-auto text-[10px] font-mono text-zinc-600">
                        {Math.round(result.localization.confidence * 100)}% confidence
                      </span>
                    )}
                  </div>
                  {result.localization.root_cause_hypothesis && (
                    <p className="text-sm text-zinc-300 mb-3">{result.localization.root_cause_hypothesis}</p>
                  )}
                  {result.localization.fault_files?.length > 0 && (
                    <div className="flex flex-wrap gap-1.5">
                      {result.localization.fault_files.map((f: string) => (
                        <span key={f} className="flex items-center gap-1 px-2 py-1 rounded-md bg-zinc-800 border border-zinc-700/50 text-xs text-zinc-300 font-mono">
                          <FileCode className="w-3 h-3 text-amber-400" />{f}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Repair */}
              {result.repair?.explanation && (
                <div className="p-4 rounded-xl bg-zinc-900 border border-zinc-800">
                  <div className="flex items-center gap-2 mb-3">
                    <Wrench className="w-4 h-4 text-orange-400" />
                    <h3 className="text-xs font-bold text-zinc-400 uppercase tracking-wider">Proposed Fix</h3>
                  </div>
                  <p className="text-sm text-zinc-300 mb-3">{result.repair.explanation}</p>
                  {result.repair.patches?.length > 0 && (
                    <div className="space-y-2">
                      {result.repair.patches.map((p: { file_path: string; explanation: string }) => (
                        <div key={p.file_path || p.explanation} className="px-3 py-2 rounded-lg bg-zinc-800/60 border border-zinc-700/30">
                          <p className="text-xs font-mono text-zinc-400">{p.file_path}</p>
                          <p className="text-xs text-zinc-500 mt-1">{p.explanation}</p>
                        </div>
                      ))}
                    </div>
                  )}
                  {result.repair.tests_added?.length > 0 && (
                    <div className="mt-3">
                      <p className="text-[10px] font-bold text-zinc-600 uppercase mb-1">Tests Added</p>
                      <ul className="space-y-0.5">
                        {result.repair.tests_added.map((t: string, i: number) => (
                          <li key={i} className="text-xs text-zinc-500 flex items-center gap-1">
                            <Shield className="w-3 h-3 text-green-500" />{t}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {/* Review */}
              {result.review?.checks?.length ? (
                <div className="p-4 rounded-xl bg-zinc-900 border border-zinc-800">
                  <div className="flex items-center gap-2 mb-3">
                    <Eye className="w-4 h-4 text-cyan-400" />
                    <h3 className="text-xs font-bold text-zinc-400 uppercase tracking-wider">Review Checks</h3>
                  </div>
                  <ReviewChecks checks={result.review.checks} />
                  {result.review.feedback && (
                    <p className="text-xs text-zinc-500 mt-3 italic">{result.review.feedback}</p>
                  )}
                </div>
              ) : null}

              {/* PR */}
              {result.pr_url && (
                <div className="p-4 rounded-xl bg-green-950/20 border border-green-800/30">
                  <div className="flex items-center gap-2">
                    <GitPullRequest className="w-4 h-4 text-green-400" />
                    <span className="text-sm text-green-300 font-medium">PR Created</span>
                    <a href={result.pr_url} target="_blank" rel="noopener noreferrer" className="text-xs text-green-500 font-mono ml-auto hover:underline">{result.pr_url}</a>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Running indicator (no result yet) */}
          {isRunning && !result && (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <Loader2 className="w-8 h-8 text-rose-400 animate-spin mb-4" />
              <p className="text-sm text-zinc-400">Agent pipeline running...</p>
              <p className="text-xs text-zinc-600 mt-1">This takes 1-2 minutes (5 LLM calls)</p>
            </div>
          )}

          {/* Ticket Selection */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-bold text-zinc-300">Bug Tickets</h2>
              <button
                onClick={() => setShowCustom(!showCustom)}
                className="text-xs text-rose-400 hover:text-rose-300 transition-colors"
              >
                {showCustom ? 'Hide custom' : '+ Custom ticket'}
              </button>
            </div>

            {/* Custom ticket form */}
            {showCustom && (
              <div className="p-4 rounded-xl bg-zinc-900 border border-zinc-800 mb-3 space-y-3">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={customTicketId}
                    onChange={(e) => setCustomTicketId(e.target.value)}
                    placeholder="Ticket ID (e.g. PROJ-1001)"
                    className="w-40 px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-sm text-zinc-200 placeholder-zinc-600 focus:border-rose-500/50 focus:outline-none font-mono"
                  />
                  <input
                    type="text"
                    value={customTitle}
                    onChange={(e) => setCustomTitle(e.target.value)}
                    placeholder="Bug title (e.g., 500 error on checkout) *"
                    className="flex-1 px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-sm text-zinc-200 placeholder-zinc-600 focus:border-rose-500/50 focus:outline-none"
                  />
                </div>
                <textarea
                  value={customDesc}
                  onChange={(e) => setCustomDesc(e.target.value)}
                  placeholder="Full bug description — include reproduction steps and expected vs actual behavior..."
                  rows={4}
                  className="w-full px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-sm text-zinc-200 placeholder-zinc-600 focus:border-rose-500/50 focus:outline-none resize-none"
                />
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={customComponent}
                    onChange={(e) => setCustomComponent(e.target.value)}
                    placeholder="Affected file/component (e.g. agent/feature_flags.py)"
                    className="flex-1 px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-sm text-zinc-200 placeholder-zinc-600 focus:border-rose-500/50 focus:outline-none font-mono"
                  />
                  <input
                    type="text"
                    value={customRepoPath}
                    onChange={(e) => setCustomRepoPath(e.target.value)}
                    placeholder={activeRepoData?.repo_path ?? 'Repo path (overrides default)'}
                    className="flex-1 px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-sm text-zinc-200 placeholder-zinc-600 focus:border-rose-500/50 focus:outline-none font-mono"
                  />
                </div>
                <button
                  onClick={handleRunCustom}
                  disabled={isRunning || !customTitle.trim() || !activeRepo}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-rose-500/20 text-rose-300 hover:bg-rose-500/30 border border-rose-500/30 text-xs font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Play className="w-3.5 h-3.5" /> Run Agent on Custom Ticket
                </button>
              </div>
            )}

            {/* Mock tickets */}
            <div className="space-y-2">
              {tickets.map((ticket) => (
                <TicketCard
                  key={ticket.ticket_id}
                  ticket={ticket}
                  onRun={handleRunMock}
                  isRunning={runningTicketId === ticket.ticket_id}
                  isDisabled={isRunning}
                />
              ))}
              {tickets.length === 0 && (
                <div className="flex flex-col items-center py-8 px-4 rounded-xl bg-zinc-900 border border-zinc-800 border-dashed">
                  <Bug className="w-8 h-8 text-zinc-700 mb-3" />
                  <p className="text-sm font-medium text-zinc-400 mb-1">No tickets yet</p>
                  <p className="text-xs text-zinc-600 mb-4">Create your first bug ticket to run the agent pipeline</p>
                  <button
                    onClick={() => setShowCustom(true)}
                    className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-rose-500/20 text-rose-300 hover:bg-rose-500/30 border border-rose-500/30 text-xs font-medium transition-all"
                  >
                    <Play className="w-3.5 h-3.5" /> Create your first ticket
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Past Runs */}
          {pastJobs.length > 0 && (
            <div>
              <h2 className="text-sm font-bold text-zinc-300 mb-3">Past Runs</h2>
              <div className="space-y-2">
                {pastJobs.map((job) => (
                  <div
                    key={job.job_id}
                    onClick={() => { setActiveJob(job); setRunningTicketId(null) }}
                    className={cn(
                      "p-3 rounded-xl border cursor-pointer transition-all",
                      activeJob?.job_id === job.job_id
                        ? "bg-zinc-800 border-rose-500/40"
                        : "bg-zinc-900 border-zinc-800 hover:border-zinc-700"
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {job.status === 'done' && <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />}
                        {job.status === 'failed' && <XCircle className="w-3.5 h-3.5 text-red-400" />}
                        {job.status === 'escalated' && <AlertTriangle className="w-3.5 h-3.5 text-yellow-400" />}
                        {(job.status === 'running' || job.status === 'pending') && <Loader2 className="w-3.5 h-3.5 text-blue-400 animate-spin" />}
                        <span className="text-xs font-mono text-zinc-400">{job.job_id.slice(0, 8)}</span>
                        {job.result?.review?.verdict && (
                          <span className={cn('text-[10px] px-1.5 py-0.5 rounded border font-medium',
                            job.result.review.verdict === 'APPROVE' ? 'bg-green-500/10 text-green-400 border-green-500/20' :
                            job.result.review.verdict === 'ESCALATE' ? 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20' :
                            'bg-zinc-700 text-zinc-400 border-zinc-600'
                          )}>{job.result.review.verdict}</span>
                        )}
                      </div>
                      <span className="text-[10px] text-zinc-600 font-mono">{job.status}</span>
                    </div>
                    {job.result?.localization?.root_cause_hypothesis && (
                      <p className="text-xs text-zinc-500 mt-1 truncate">{job.result.localization.root_cause_hypothesis}</p>
                    )}
                    {job.result?.repair?.explanation && !job.result?.localization?.root_cause_hypothesis && (
                      <p className="text-xs text-zinc-500 mt-1 truncate">{job.result.repair.explanation}</p>
                    )}
                    {job.result?.pr_url && (
                      <a
                        href={job.result.pr_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        onClick={(e) => e.stopPropagation()}
                        className="mt-1.5 flex items-center gap-1 text-[10px] text-green-500 hover:text-green-400 font-mono hover:underline"
                      >
                        <GitPullRequest className="w-3 h-3" />
                        {job.result.pr_url}
                      </a>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
