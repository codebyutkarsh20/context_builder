import { useState, useEffect, useCallback, useRef } from 'react'
import {
  Cpu, Play, Loader2, CheckCircle2, XCircle,
  AlertTriangle, Bug, Terminal, FileCode, GitPullRequest, AlertCircle,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { useRepo } from '../lib/RepoContext'
import {
  listAgentTickets, runMockTicket, runAgentTicket, getAgentJobStatus, listAgentJobs,
  subscribeToTrace, listRepos,
  type AgentTicket, type AgentJobStatus, type TraceEvent, type Repo,
} from '../lib/api'
import {
  TicketCard, StatusBadge, EndToEndSummary, TraceLogPanel, LiveActivityFeed,
} from '../components/agent'

const MAX_POLL_COUNT = 900 // 30 min

const isTerminal = (status: string) =>
  status === 'done' || status === 'failed' || status === 'escalated'

// ─── Local storage helpers ──────────────────────────────────────────────────

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

// ─── Main Page ──────────────────────────────────────────────────────────────

export default function AgentPage() {
  const { activeRepo } = useRepo()
  const [tickets, setTickets] = useState<AgentTicket[]>([])
  const [activeJob, setActiveJob] = useState<AgentJobStatus | null>(() => loadResultFromStorage())
  const [runningTicketId, setRunningTicketId] = useState<string | null>(() => {
    const saved = loadJobFromStorage()
    return saved ? saved.ticketId : null
  })
  const [error, setError] = useState<string | null>(null)
  const [pastJobs, setPastJobs] = useState<AgentJobStatus[]>([])
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const pollCountRef = useRef(0)

  // Trace (always-on when running)
  const [showDebugTrace, setShowDebugTrace] = useState(false)
  const [traceEvents, setTraceEvents] = useState<TraceEvent[]>([])
  const [traceIsLive, setTraceIsLive] = useState(false)
  const traceUnsubRef = useRef<(() => void) | null>(null)

  // Custom ticket form
  const [customTicketId, setCustomTicketId] = useState('')
  const [customTitle, setCustomTitle] = useState('')
  const [customDesc, setCustomDesc] = useState('')
  const [customComponent, setCustomComponent] = useState('')
  const [customRepoPath, setCustomRepoPath] = useState('')

  // Repo selection
  const [repos, setRepos] = useState<Repo[]>([])
  const [selectedRepoName, setSelectedRepoName] = useState<string>('')
  const [reposLoading, setReposLoading] = useState(false)

  useEffect(() => {
    listAgentTickets().then(setTickets).catch((e: Error) => setError(`Failed to load tickets: ${e.message}`))
  }, [])

  useEffect(() => {
    listAgentJobs().then(setPastJobs).catch(() => {})
  }, [])

  useEffect(() => {
    setReposLoading(true)
    listRepos()
      .then(setRepos)
      .catch((e: Error) => console.warn('Failed to load repos:', e.message))
      .finally(() => setReposLoading(false))
  }, [])

  // Auto-select active repo
  useEffect(() => {
    if (activeRepo && !selectedRepoName && repos.length > 0) {
      const repo = repos.find(r => r.name === activeRepo)
      if (repo) setSelectedRepoName(activeRepo)
    }
  }, [activeRepo, repos, selectedRepoName])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
      if (traceUnsubRef.current) traceUnsubRef.current()
    }
  }, [])

  const startTraceSubscription = useCallback((jobId: string) => {
    if (traceUnsubRef.current) traceUnsubRef.current()
    setTraceEvents([])
    setTraceIsLive(true)
    traceUnsubRef.current = subscribeToTrace(
      jobId,
      (evt) => setTraceEvents(prev => {
        if (prev.length > 0 && evt.index <= prev[prev.length - 1].index) return prev
        return [...prev, evt]
      }),
      () => setTraceIsLive(false),
      () => setTraceIsLive(false),
    )
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
        setError('Agent pipeline timed out after 30 minutes -- check backend logs')
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
        setError('Lost connection to agent job -- the pipeline may still be running in the backend')
      }
    }, 2000)
  }, [])

  // Resume polling on mount
  useEffect(() => {
    const saved = loadJobFromStorage()
    if (saved) {
      setRunningTicketId(saved.ticketId)
      startTraceSubscription(saved.jobId)
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
        clearJobStorage()
        setRunningTicketId(null)
      })
    }
  }, [pollJob, startTraceSubscription])

  const startRun = (jobId: string, ticketId: string) => {
    saveJobToStorage(jobId, ticketId)
    setActiveJob({ job_id: jobId, status: 'pending', stage: 'Queued', iteration_count: 0, result: null, error: '' })
    pollJob(jobId)
    startTraceSubscription(jobId)  // Always subscribe to trace
  }

  const handleRunMock = async (ticketId: string) => {
    setError(null)
    setRunningTicketId(ticketId)
    setActiveJob(null)
    setTraceEvents([])
    try {
      const res = await runMockTicket(ticketId, true)
      startRun(res.job_id, ticketId)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to start agent')
      setRunningTicketId(null)
    }
  }

  const handleRunCustom = async () => {
    if (!customTitle.trim() || !selectedRepoName) {
      setError('Please select a repository and enter a bug title')
      return
    }

    const selectedRepo = repos.find(r => r.name === selectedRepoName)
    if (!selectedRepo) {
      setError('Selected repository not found')
      return
    }

    if (!selectedRepo.has_context) {
      setError(`Repository "${selectedRepoName}" needs to be analyzed first. Go to Overview and click Analyze.`)
      return
    }

    setError(null)
    setRunningTicketId('custom')
    setActiveJob(null)
    setTraceEvents([])
    try {
      const res = await runAgentTicket({
        ticket_id: customTicketId.trim() || undefined,
        title: customTitle,
        description: customDesc,
        repo_name: selectedRepoName,
        repo_path: customRepoPath.trim() || selectedRepo.repo_path,
        affected_component: customComponent.trim() || undefined,
        debug: true,
      })
      startRun(res.job_id, 'custom')
      setCustomTicketId('')
      setCustomTitle('')
      setCustomDesc('')
      setCustomComponent('')
      setCustomRepoPath('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to start agent')
      setRunningTicketId(null)
    }
  }

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
        <div className="flex items-center gap-3">
          {activeJob && (
            <StatusBadge status={activeJob.stage || activeJob.status} isRunning={isRunning} />
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Error */}
          {error && (
            <div className="flex items-center gap-2 p-3 rounded-xl bg-red-950/30 border border-red-900/40 text-red-400 text-sm">
              <XCircle className="w-4 h-4 flex-shrink-0" />{error}
            </div>
          )}

          {/* Live Activity Feed — always visible when running or has events */}
          {(isRunning || traceEvents.length > 0) && (
            <div className="space-y-2">
              <LiveActivityFeed events={traceEvents} isLive={traceIsLive} />

              {/* Debug trace toggle */}
              <div className="flex justify-end">
                <button
                  onClick={() => setShowDebugTrace(!showDebugTrace)}
                  className={cn(
                    'flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[10px] font-medium transition-colors',
                    showDebugTrace
                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                      : 'text-zinc-600 hover:text-zinc-400'
                  )}
                >
                  <Terminal className="w-3 h-3" />
                  {showDebugTrace ? 'Hide debug trace' : 'Show debug trace'}
                  <span className="font-mono text-zinc-600">{traceEvents.length}</span>
                </button>
              </div>

              {showDebugTrace && (
                <TraceLogPanel events={traceEvents} isLive={traceIsLive} />
              )}
            </div>
          )}

          {/* End-to-End Summary (after completion) */}
          {activeJob?.result && (
            <EndToEndSummary job={activeJob} />
          )}

          {/* Custom Ticket Form */}
          <div className="p-4 rounded-xl bg-zinc-900 border border-rose-500/20">
            <div className="flex items-center gap-2 mb-3">
              <Bug className="w-4 h-4 text-rose-400" />
              <h2 className="text-sm font-bold text-zinc-300">Submit Bug Ticket</h2>
            </div>

            {repos.length === 0 && !reposLoading && (
              <div className="mb-3 flex items-center gap-2 p-3 rounded-lg bg-amber-950/30 border border-amber-900/40 text-amber-400 text-sm">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>No repositories analyzed. Go to Overview and analyze a repo first.</span>
              </div>
            )}

            {selectedRepoName && repos.find(r => r.name === selectedRepoName) && (
              <div className="mb-3 flex items-center gap-2 p-3 rounded-lg bg-zinc-800/50 border border-zinc-700/50 text-xs">
                <span className="text-zinc-400">Selected:</span>
                <span className="font-mono text-zinc-200">{selectedRepoName}</span>
                {repos.find(r => r.name === selectedRepoName)?.has_context ? (
                  <span className="flex items-center gap-1 text-emerald-400 ml-auto">
                    <CheckCircle2 className="w-3.5 h-3.5" /> Ready
                  </span>
                ) : (
                  <span className="flex items-center gap-1 text-yellow-400 ml-auto">
                    <AlertTriangle className="w-3.5 h-3.5" /> Not analyzed
                  </span>
                )}
              </div>
            )}

            <div className="space-y-3">
              <select
                value={selectedRepoName}
                onChange={(e) => setSelectedRepoName(e.target.value)}
                disabled={reposLoading || repos.length === 0}
                className="w-full px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-sm text-zinc-200 focus:border-rose-500/50 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <option value="">Select repository *</option>
                {repos.map((repo) => (
                  <option key={repo.name} value={repo.name}>
                    {repo.name} {repo.has_context ? '\u2713' : '(needs analysis)'}
                  </option>
                ))}
              </select>

              <div className="flex gap-2">
                <input
                  type="text"
                  value={customTicketId}
                  onChange={(e) => setCustomTicketId(e.target.value)}
                  placeholder="Ticket ID (e.g. BUG-001)"
                  className="w-40 px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-sm text-zinc-200 placeholder-zinc-600 focus:border-rose-500/50 focus:outline-none font-mono"
                />
                <input
                  type="text"
                  value={customTitle}
                  onChange={(e) => setCustomTitle(e.target.value)}
                  placeholder="Bug title *"
                  className="flex-1 px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-sm text-zinc-200 placeholder-zinc-600 focus:border-rose-500/50 focus:outline-none"
                />
              </div>
              <textarea
                value={customDesc}
                onChange={(e) => setCustomDesc(e.target.value)}
                placeholder="Describe the bug: expected behavior, actual behavior, reproduction steps, affected file/function..."
                rows={3}
                className="w-full px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-sm text-zinc-200 placeholder-zinc-600 focus:border-rose-500/50 focus:outline-none resize-none"
              />
              <div className="flex gap-2 items-center">
                <input
                  type="text"
                  value={customComponent}
                  onChange={(e) => setCustomComponent(e.target.value)}
                  placeholder="Affected file (e.g. backend/rag/retriever.py) -- optional"
                  className="flex-1 px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-sm text-zinc-200 placeholder-zinc-600 focus:border-rose-500/50 focus:outline-none font-mono"
                />
                <button
                  onClick={handleRunCustom}
                  disabled={isRunning || !customTitle.trim() || !selectedRepoName}
                  className="flex items-center gap-1.5 px-5 py-2 rounded-lg bg-rose-600 text-white hover:bg-rose-500 text-xs font-bold transition-all disabled:opacity-40 disabled:cursor-not-allowed flex-shrink-0"
                >
                  {isRunning && runningTicketId === 'custom' ? (
                    <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Running...</>
                  ) : (
                    <><Play className="w-3.5 h-3.5" /> Run Agent</>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Sample Tickets — only show if there are any */}
          {tickets.length > 0 && (
            <div>
              <h2 className="text-sm font-bold text-zinc-300 mb-3">Sample Tickets</h2>
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
              </div>
            </div>
          )}

          {/* Past Runs */}
          {pastJobs.length > 0 && (
            <div>
              <h2 className="text-sm font-bold text-zinc-300 mb-3">Past Runs</h2>
              <div className="space-y-2">
                {pastJobs.map((job) => {
                  const patches = job.result?.repair?.patches?.length ?? 0
                  const checks = job.result?.review?.checks ?? []
                  const passCount = checks.filter((c: { status: string }) => c.status === 'PASS').length
                  const totalChecks = checks.length
                  return (
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
                          <span className="text-xs font-mono text-zinc-500">{job.job_id.slice(0, 8)}</span>
                          {job.result?.review?.verdict && (
                            <span className={cn('text-[10px] px-1.5 py-0.5 rounded border font-bold',
                              job.result.review.verdict === 'APPROVE' ? 'bg-green-500/10 text-green-400 border-green-500/20' :
                              job.result.review.verdict === 'ESCALATE' ? 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20' :
                              'bg-zinc-700 text-zinc-400 border-zinc-600'
                            )}>{job.result.review.verdict}</span>
                          )}
                        </div>
                        <div className="flex items-center gap-3 text-[10px] text-zinc-600 font-mono">
                          {patches > 0 && <span>{patches} patch{patches > 1 ? 'es' : ''}</span>}
                          {totalChecks > 0 && <span>{passCount}/{totalChecks} checks</span>}
                          <span>iter {job.iteration_count}</span>
                        </div>
                      </div>
                      {job.result?.repair?.explanation && (
                        <p className="text-xs text-zinc-400 mt-1.5 truncate">{job.result.repair.explanation}</p>
                      )}
                      {!job.result?.repair?.explanation && job.error && (
                        <p className="text-xs text-zinc-600 mt-1.5 truncate">{job.error}</p>
                      )}
                      <div className="flex items-center gap-3 mt-1.5">
                        {job.result?.localization?.fault_files?.slice(0, 2).map((f: string) => (
                          <span key={f} className="text-[10px] font-mono text-zinc-600 flex items-center gap-1">
                            <FileCode className="w-2.5 h-2.5" />{f.split('/').pop()}
                          </span>
                        ))}
                        {job.result?.pr_url && (
                          <a
                            href={job.result.pr_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()}
                            className="flex items-center gap-1 text-[10px] text-green-500 hover:text-green-400 font-mono hover:underline ml-auto"
                          >
                            <GitPullRequest className="w-2.5 h-2.5" /> PR
                          </a>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
