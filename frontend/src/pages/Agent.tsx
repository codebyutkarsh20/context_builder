import { useState, useEffect, useCallback, useRef } from 'react'
import {
  Cpu, Play, Loader2, CheckCircle2, XCircle, AlertTriangle,
  ArrowRight, FileCode, Bug, Wrench, Eye, GitPullRequest,
  Target, Brain, Shield, Search, ChevronDown, ChevronRight,
  Terminal, Filter, Zap, MessageSquare, TestTube, Layers,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { useRepo } from '../lib/RepoContext'
import {
  listAgentTickets, runMockTicket, runAgentTicket, getAgentJobStatus, listAgentJobs,
  subscribeToTrace,
  type AgentTicket, type AgentJobStatus, type TraceEvent,
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

// ─── Code Diff Viewer ──────────────────────────────────────────────────────

function CodeDiff({ original, patched, filePath }: { original: string; patched: string; filePath: string }) {
  const [view, setView] = useState<'diff' | 'original' | 'patched'>('diff')

  const origLines = original.split('\n')
  const patchedLines = patched.split('\n')

  // Simple line-by-line diff
  const maxLen = Math.max(origLines.length, patchedLines.length)
  const diffLines: { type: 'same' | 'removed' | 'added' | 'changed'; orig: string; patched: string; lineNum: number }[] = []

  for (let i = 0; i < maxLen; i++) {
    const o = origLines[i] ?? ''
    const p = patchedLines[i] ?? ''
    if (o === p) {
      diffLines.push({ type: 'same', orig: o, patched: p, lineNum: i + 1 })
    } else if (i >= origLines.length) {
      diffLines.push({ type: 'added', orig: '', patched: p, lineNum: i + 1 })
    } else if (i >= patchedLines.length) {
      diffLines.push({ type: 'removed', orig: o, patched: '', lineNum: i + 1 })
    } else {
      diffLines.push({ type: 'changed', orig: o, patched: p, lineNum: i + 1 })
    }
  }

  return (
    <div className="rounded-lg bg-zinc-950 border border-zinc-800 overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b border-zinc-800 bg-zinc-900/50">
        <span className="text-xs font-mono text-zinc-400 flex items-center gap-1.5">
          <FileCode className="w-3 h-3 text-orange-400" />
          {filePath}
        </span>
        <div className="flex gap-1">
          {(['diff', 'original', 'patched'] as const).map(v => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={cn('px-2 py-0.5 rounded text-[10px] font-medium transition-colors',
                view === v ? 'bg-zinc-700 text-zinc-200' : 'text-zinc-600 hover:text-zinc-400'
              )}
            >{v === 'diff' ? 'Diff' : v === 'original' ? 'Before' : 'After'}</button>
          ))}
        </div>
      </div>
      <div className="max-h-80 overflow-y-auto overflow-x-auto">
        {view === 'diff' ? (
          <div className="text-[11px] font-mono leading-5">
            {diffLines.filter(l => l.type !== 'same' || diffLines.filter(d => d.type !== 'same').length < 20).map((line, i) => {
              if (line.type === 'same') return (
                <div key={i} className="px-3 py-0 text-zinc-600 flex">
                  <span className="w-8 text-right text-zinc-700 select-none mr-3 flex-shrink-0">{line.lineNum}</span>
                  <span className="whitespace-pre">{line.orig}</span>
                </div>
              )
              if (line.type === 'removed' || line.type === 'changed') return (
                <div key={`r-${i}`} className="px-3 py-0 bg-red-950/30 text-red-300 flex">
                  <span className="w-8 text-right text-red-800 select-none mr-3 flex-shrink-0">{line.lineNum}</span>
                  <span className="select-none text-red-600 mr-1">-</span>
                  <span className="whitespace-pre">{line.orig}</span>
                </div>
              )
              return null
            })}
            {diffLines.filter(l => l.type !== 'same').map((line, i) => {
              if (line.type === 'added' || line.type === 'changed') return (
                <div key={`a-${i}`} className="px-3 py-0 bg-green-950/30 text-green-300 flex">
                  <span className="w-8 text-right text-green-800 select-none mr-3 flex-shrink-0">{line.lineNum}</span>
                  <span className="select-none text-green-600 mr-1">+</span>
                  <span className="whitespace-pre">{line.patched}</span>
                </div>
              )
              return null
            })}
          </div>
        ) : (
          <pre className="px-3 py-2 text-[11px] font-mono text-zinc-400 whitespace-pre overflow-x-auto">
            {view === 'original' ? original : patched}
          </pre>
        )}
      </div>
    </div>
  )
}

function PatchCard({ patch }: { patch: { file_path: string; original_code: string; patched_code: string; explanation: string } }) {
  const [showDiff, setShowDiff] = useState(false)
  return (
    <div className="rounded-lg bg-zinc-800/60 border border-zinc-700/30 overflow-hidden">
      <div
        className="flex items-center justify-between px-3 py-2 cursor-pointer hover:bg-zinc-800/80 transition-colors"
        onClick={() => setShowDiff(!showDiff)}
      >
        <div className="flex items-center gap-2 min-w-0">
          <FileCode className="w-3.5 h-3.5 text-orange-400 flex-shrink-0" />
          <span className="text-xs font-mono text-zinc-300 truncate">{patch.file_path}</span>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="text-[10px] text-zinc-600">
            {patch.original_code ? `${patch.original_code.split('\n').length} lines` : 'new file'}
          </span>
          {showDiff ? <ChevronDown className="w-3 h-3 text-zinc-600" /> : <ChevronRight className="w-3 h-3 text-zinc-600" />}
        </div>
      </div>
      {patch.explanation && (
        <p className="px-3 pb-2 text-xs text-zinc-500">{patch.explanation}</p>
      )}
      {showDiff && patch.original_code && patch.patched_code && (
        <CodeDiff original={patch.original_code} patched={patch.patched_code} filePath={patch.file_path} />
      )}
      {showDiff && !patch.original_code && patch.patched_code && (
        <pre className="px-3 py-2 text-[11px] font-mono text-green-400 bg-green-950/20 max-h-60 overflow-y-auto whitespace-pre">
          {patch.patched_code.slice(0, 3000)}
        </pre>
      )}
    </div>
  )
}

// ─── Trace Log Panel ────────────────────────────────────────────────────────

type TraceFilter = 'all' | 'llm' | 'tools' | 'tests' | 'stages' | 'guardrails'

const EVENT_STYLES: Record<string, { icon: typeof Zap; color: string; label: string }> = {
  stage_start: { icon: Layers, color: 'text-blue-400', label: 'Stage' },
  stage_end: { icon: Layers, color: 'text-blue-300', label: 'Stage End' },
  llm_request: { icon: MessageSquare, color: 'text-purple-400', label: 'LLM Req' },
  llm_response: { icon: MessageSquare, color: 'text-purple-300', label: 'LLM Resp' },
  tool_call: { icon: Wrench, color: 'text-orange-400', label: 'Tool Call' },
  tool_result: { icon: Wrench, color: 'text-orange-300', label: 'Tool Result' },
  guardrail_event: { icon: Shield, color: 'text-red-400', label: 'Guardrail' },
  state_transition: { icon: ArrowRight, color: 'text-cyan-400', label: 'Phase' },
  context_compaction: { icon: Brain, color: 'text-yellow-400', label: 'Context' },
  prompt_build: { icon: FileCode, color: 'text-green-400', label: 'Prompt' },
  run_outcome: { icon: Target, color: 'text-emerald-400', label: 'Outcome' },
  patch_candidate: { icon: FileCode, color: 'text-green-400', label: 'Patch' },
  test_output: { icon: TestTube, color: 'text-lime-400', label: 'Test' },
  error: { icon: XCircle, color: 'text-red-400', label: 'Error' },
  info: { icon: Zap, color: 'text-zinc-400', label: 'Info' },
}

function matchesFilter(evt: TraceEvent, filter: TraceFilter): boolean {
  if (filter === 'all') return true
  if (filter === 'llm') return evt.event_type === 'llm_request' || evt.event_type === 'llm_response'
  if (filter === 'tools') return evt.event_type === 'tool_call' || evt.event_type === 'tool_result'
  if (filter === 'tests') return evt.event_type === 'test_output' || (evt.event_type === 'tool_call' && evt.data?.tool_name === 'run_tests')
  if (filter === 'stages') return evt.event_type === 'stage_start' || evt.event_type === 'stage_end' || evt.event_type === 'state_transition' || evt.event_type === 'run_outcome'
  if (filter === 'guardrails') return evt.event_type === 'guardrail_event' || evt.event_type === 'context_compaction'
  return true
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function TraceEventRow({ evt }: { evt: TraceEvent }) {
  const [expanded, setExpanded] = useState(false)
  const style = EVENT_STYLES[evt.event_type] || EVENT_STYLES.info
  const Icon = style.icon
  const data = evt.data

  // Build one-line summary
  let summary = ''
  if (evt.event_type === 'stage_start') summary = `→ ${data.stage}`
  else if (evt.event_type === 'stage_end') summary = `✓ ${data.stage} (${formatDuration(data.duration_ms as number || 0)})`
  else if (evt.event_type === 'llm_request') summary = `${data.model || 'llm'} [${data.phase || '?'}] ~${data.context_tokens || 0} tokens (${data.context_pct || 0}% ctx)`
  else if (evt.event_type === 'llm_response') summary = `${data.model || 'llm'} in=${data.input_tokens || 0} out=${data.output_tokens || 0} $${Number(data.cost_usd || 0).toFixed(3)} (total $${Number(data.cumulative_cost_usd || 0).toFixed(3)})`
  else if (evt.event_type === 'tool_call') summary = `[${data.phase || '?'}] ${data.tool_name}(${JSON.stringify(data.args || {}).slice(0, 80)})`
  else if (evt.event_type === 'tool_result') summary = `${data.tool_name} (${formatDuration(data.duration_ms as number || 0)}) ${String(data.result_preview || '').slice(0, 60)}`
  else if (evt.event_type === 'guardrail_event') summary = `${data.action === 'block' ? 'BLOCKED' : 'WARN'} ${data.tool_name}: ${String(data.message || '').slice(0, 80)}`
  else if (evt.event_type === 'state_transition') summary = `${data.from_phase} → ${data.to_phase} (at call #${data.at_call}, $${Number(data.cost_usd_at_transition || 0).toFixed(3)})`
  else if (evt.event_type === 'context_compaction') summary = `${data.action}: ${data.tokens_before} → ${data.tokens_after} tokens (saved ${data.tokens_saved})`
  else if (evt.event_type === 'prompt_build') summary = `system=${data.system_prompt_tokens_approx || 0} tokens, kickstart=${data.kickstart_chars || 0} chars`
  else if (evt.event_type === 'run_outcome') summary = `${String(data.outcome || '').toUpperCase()} — ${data.tool_call_count} calls, $${Number(data.cost_usd || 0).toFixed(3)}, ${data.elapsed_seconds || 0}s`
  else if (evt.event_type === 'patch_candidate') summary = `${data.file_path}`
  else if (evt.event_type === 'test_output') summary = (data.passed ? 'PASSED' : 'FAILED') + ` (${data.patches_applied} patches)`
  else if (evt.event_type === 'error') summary = String(data.message || '').slice(0, 120)
  else if (evt.event_type === 'info') summary = String(data.message || '').slice(0, 120)

  // Expandable detail content
  const hasDetail = [
    'llm_request', 'llm_response', 'tool_call', 'tool_result',
    'test_output', 'patch_candidate', 'guardrail_event', 'run_outcome',
    'prompt_build', 'context_compaction',
  ].includes(evt.event_type)

  let detailContent = ''
  if (expanded) {
    if (evt.event_type === 'llm_request') detailContent = `Model: ${data.model}\nPhase: ${data.phase}\nContext tokens: ${data.context_tokens} (${data.context_pct}%)\nMessages: ${data.message_count}\nCost so far: $${data.cost_usd_so_far}`
    else if (evt.event_type === 'llm_response') detailContent = `Input: ${data.input_tokens} tokens\nOutput: ${data.output_tokens} tokens\nCache creation: ${data.cache_creation_tokens || 0}\nCache read: ${data.cache_read_tokens || 0}\nCall cost: $${data.cost_usd}\nTotal cost: $${data.cumulative_cost_usd}`
    else if (evt.event_type === 'tool_call') detailContent = `Phase: ${data.phase}\nCall #${data.call_number}\n\nArgs:\n${JSON.stringify(data.args, null, 2)}${data.reasoning ? `\n\nAgent reasoning:\n${data.reasoning}` : ''}`
    else if (evt.event_type === 'tool_result') detailContent = String(data.result_preview || '')
    else if (evt.event_type === 'guardrail_event') detailContent = `Tool: ${data.tool_name}\nAction: ${data.action}\nCall #${data.call_number}\n\n${data.message}`
    else if (evt.event_type === 'run_outcome') detailContent = JSON.stringify(data, null, 2)
    else if (evt.event_type === 'prompt_build') detailContent = `System prompt: ${data.system_prompt_chars} chars (~${data.system_prompt_tokens_approx} tokens)\nTask message: ${data.task_message_chars} chars\nKickstart context: ${data.kickstart_chars} chars\nConventions: ${data.conventions_chars} chars\nBusiness rules: ${data.business_rules_chars} chars\nHint files: ${JSON.stringify(data.hint_files)}`
    else if (evt.event_type === 'context_compaction') detailContent = `Action: ${data.action}\nBefore: ${data.tokens_before} tokens\nAfter: ${data.tokens_after} tokens\nSaved: ${data.tokens_saved} tokens\nAt call: #${data.at_call}`
    else if (evt.event_type === 'test_output') detailContent = String(data.result || '')
    else if (evt.event_type === 'patch_candidate') detailContent = `File: ${data.file_path}\nExplanation: ${data.explanation}`
    else detailContent = JSON.stringify(data, null, 2)
  }

  return (
    <div className="group">
      <div
        className={cn(
          'flex items-center gap-2 px-3 py-1.5 text-xs font-mono hover:bg-zinc-800/50 transition-colors',
          hasDetail && 'cursor-pointer',
        )}
        onClick={() => hasDetail && setExpanded(!expanded)}
      >
        <span className="text-zinc-600 w-14 flex-shrink-0 text-right">{evt.timestamp.toFixed(1)}s</span>
        <span className={cn('w-4 flex-shrink-0 flex items-center', style.color)}>
          {hasDetail ? (expanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />) : <Icon className="w-3 h-3" />}
        </span>
        <span className={cn('w-20 flex-shrink-0 text-[10px] font-bold uppercase', style.color)}>{style.label}</span>
        <span className="text-zinc-400 truncate flex-1">{summary}</span>
        <span className="text-zinc-700 text-[10px] flex-shrink-0">{evt.stage}</span>
      </div>
      {expanded && detailContent && (
        <pre className="mx-3 mb-2 px-3 py-2 rounded-lg bg-zinc-950 border border-zinc-800 text-[11px] text-zinc-400 overflow-x-auto max-h-60 overflow-y-auto whitespace-pre-wrap">
          {detailContent.slice(0, 5000)}
          {detailContent.length > 5000 && '\n... (truncated)'}
        </pre>
      )}
    </div>
  )
}

function TraceLogPanel({ events, isLive }: { events: TraceEvent[]; isLive: boolean }) {
  const [filter, setFilter] = useState<TraceFilter>('all')
  const [collapsed, setCollapsed] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const autoScrollRef = useRef(true)

  const filtered = events.filter(e => matchesFilter(e, filter))

  // Auto-scroll to bottom when new events arrive
  useEffect(() => {
    if (autoScrollRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [filtered.length])

  // Detect manual scroll
  const handleScroll = () => {
    if (!scrollRef.current) return
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current
    autoScrollRef.current = scrollHeight - scrollTop - clientHeight < 40
  }

  // Compute stage timings
  const stageTimings: { stage: string; duration_ms: number }[] = []
  for (const evt of events) {
    if (evt.event_type === 'stage_end' && evt.data.duration_ms) {
      stageTimings.push({ stage: String(evt.data.stage), duration_ms: evt.data.duration_ms as number })
    }
  }

  const FILTERS: { key: TraceFilter; label: string; icon: typeof Zap }[] = [
    { key: 'all', label: 'All', icon: Filter },
    { key: 'stages', label: 'Phases', icon: Layers },
    { key: 'llm', label: 'LLM', icon: MessageSquare },
    { key: 'tools', label: 'Tools', icon: Wrench },
    { key: 'tests', label: 'Tests', icon: TestTube },
    { key: 'guardrails', label: 'Guards', icon: Shield },
  ]

  if (collapsed) {
    return (
      <div
        className="flex items-center gap-2 px-4 py-2 rounded-xl bg-zinc-900 border border-zinc-800 cursor-pointer hover:border-zinc-700 transition-colors"
        onClick={() => setCollapsed(false)}
      >
        <Terminal className="w-3.5 h-3.5 text-emerald-400" />
        <span className="text-xs font-medium text-zinc-400">Pipeline Trace</span>
        <span className="text-[10px] text-zinc-600 font-mono">{events.length} events</span>
        {isLive && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />}
        <ChevronRight className="w-3 h-3 text-zinc-600 ml-auto" />
      </div>
    )
  }

  return (
    <div className="rounded-xl bg-zinc-900 border border-zinc-800 overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-zinc-800">
        <div className="flex items-center gap-2 cursor-pointer" onClick={() => setCollapsed(true)}>
          <Terminal className="w-3.5 h-3.5 text-emerald-400" />
          <span className="text-xs font-bold text-zinc-300">Pipeline Trace</span>
          {isLive && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />}
          <ChevronDown className="w-3 h-3 text-zinc-600" />
        </div>
        <span className="text-[10px] text-zinc-600 font-mono">{filtered.length}/{events.length} events</span>
        <div className="flex items-center gap-1 ml-auto">
          {FILTERS.map(f => (
            <button
              key={f.key}
              onClick={() => setFilter(f.key)}
              className={cn(
                'flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium transition-colors',
                filter === f.key
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                  : 'text-zinc-600 hover:text-zinc-400'
              )}
            >
              <f.icon className="w-2.5 h-2.5" />
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {/* Stage timing bar */}
      {stageTimings.length > 0 && (
        <div className="flex items-center gap-0.5 px-4 py-1.5 border-b border-zinc-800/50 overflow-x-auto">
          {stageTimings.map((st, i) => {
            const totalMs = stageTimings.reduce((s, t) => s + t.duration_ms, 0)
            const pct = totalMs > 0 ? Math.max(3, (st.duration_ms / totalMs) * 100) : 0
            return (
              <div
                key={i}
                className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-zinc-800/60 text-[9px] font-mono"
                style={{ flex: `${pct} 0 0` }}
                title={`${st.stage}: ${formatDuration(st.duration_ms)}`}
              >
                <span className="text-zinc-500 truncate">{st.stage}</span>
                <span className="text-zinc-600 flex-shrink-0">{formatDuration(st.duration_ms)}</span>
              </div>
            )
          })}
        </div>
      )}

      {/* Events */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="max-h-96 overflow-y-auto divide-y divide-zinc-800/30"
      >
        {filtered.length === 0 ? (
          <div className="flex items-center justify-center py-8 text-xs text-zinc-600">
            {events.length === 0 ? 'Waiting for trace events...' : 'No events match this filter'}
          </div>
        ) : (
          filtered.map((evt, i) => <TraceEventRow key={`${evt.index}-${i}`} evt={evt} />)
        )}
      </div>
    </div>
  )
}

// ─── Status helpers ─────────────────────────────────────────────────────────

function getOutcomeDisplay(activeJob: AgentJobStatus) {
  const result = activeJob.result
  const status = activeJob.status
  const reviewVerdict = result?.review?.verdict

  if (status === 'escalated') {
    return { label: 'Escalated to Human', icon: AlertTriangle, iconColor: 'text-yellow-400', bg: 'bg-yellow-950/30 border-yellow-800/40' }
  }
  if (status === 'failed') {
    return { label: 'Pipeline Failed', icon: XCircle, iconColor: 'text-red-400', bg: 'bg-red-950/30 border-red-800/40' }
  }
  if (status === 'done' && reviewVerdict === 'APPROVE') {
    return { label: 'Fix Approved', icon: CheckCircle2, iconColor: 'text-green-400', bg: 'bg-green-950/30 border-green-800/40' }
  }
  if (status === 'done' && (reviewVerdict === 'ESCALATE' || reviewVerdict === 'CHANGES_REQUESTED')) {
    return { label: reviewVerdict === 'ESCALATE' ? 'Escalated to Human' : 'Changes Requested', icon: AlertTriangle, iconColor: 'text-yellow-400', bg: 'bg-yellow-950/30 border-yellow-800/40' }
  }
  if (status === 'done') {
    return { label: 'Completed', icon: CheckCircle2, iconColor: 'text-green-400', bg: 'bg-green-950/30 border-green-800/40' }
  }
  return { label: 'Running...', icon: Loader2, iconColor: 'text-blue-400', bg: 'bg-zinc-900 border-zinc-800' }
}

// ─── Result Flow Tabs ──────────────────────────────────────────────────────

type ResultTab = 'localization' | 'changes' | 'tests' | 'review' | 'pr'

const RESULT_TABS: { key: ResultTab; label: string; icon: typeof Target }[] = [
  { key: 'localization', label: 'Localization', icon: Target },
  { key: 'changes', label: 'Code Changes', icon: Wrench },
  { key: 'tests', label: 'Tests', icon: TestTube },
  { key: 'review', label: 'Review', icon: Eye },
  { key: 'pr', label: 'PR & Deploy', icon: GitPullRequest },
]

function ResultFlowView({ job }: { job: AgentJobStatus }) {
  const [activeTab, setActiveTab] = useState<ResultTab>('localization')
  const result = job.result
  if (!result) return null

  const outcome = getOutcomeDisplay(job)
  const OutcomeIcon = outcome.icon
  const patches = result.repair?.patches ?? []
  const testPatches = result.repair?.test_patches ?? []
  const checks = result.review?.checks ?? []
  const passCount = checks.filter(c => c.status === 'PASS').length

  // Tab status indicators
  const tabStatus = (key: ResultTab): 'done' | 'warn' | 'fail' | 'none' => {
    if (key === 'localization') return result.localization?.confidence ? (result.localization.confidence > 0.5 ? 'done' : 'warn') : 'none'
    if (key === 'changes') return patches.length > 0 ? 'done' : 'fail'
    if (key === 'tests') return result.test_result?.includes('passed') ? 'done' : testPatches.length > 0 ? 'warn' : 'none'
    if (key === 'review') return result.review?.verdict === 'APPROVE' ? 'done' : result.review?.verdict ? 'warn' : 'none'
    if (key === 'pr') return result.pr_url ? 'done' : 'none'
    return 'none'
  }

  const statusDot = (s: ReturnType<typeof tabStatus>) =>
    s === 'done' ? 'bg-green-400' : s === 'warn' ? 'bg-yellow-400' : s === 'fail' ? 'bg-red-400' : 'bg-zinc-700'

  return (
    <div className="space-y-4">
      {/* Status Banner */}
      <div className={cn('p-4 rounded-xl border flex items-center justify-between', outcome.bg)}>
        <div className="flex items-center gap-3">
          <OutcomeIcon className={cn('w-6 h-6', outcome.iconColor, outcome.label === 'Running...' && 'animate-spin')} />
          <div>
            <span className="text-base font-bold text-zinc-100">{outcome.label}</span>
            {job.error && <p className="text-xs text-red-400 mt-0.5 max-w-xl truncate">{job.error}</p>}
          </div>
        </div>
        <div className="flex items-center gap-4 text-xs text-zinc-400">
          {result.localization?.confidence != null && (
            <div className="text-center">
              <div className="text-lg font-bold text-zinc-200">{Math.round(result.localization.confidence * 100)}%</div>
              <div className="text-[10px] text-zinc-600">localization</div>
            </div>
          )}
          {patches.length > 0 && (
            <div className="text-center">
              <div className="text-lg font-bold text-zinc-200">{patches.length}</div>
              <div className="text-[10px] text-zinc-600">patches</div>
            </div>
          )}
          {checks.length > 0 && (
            <div className="text-center">
              <div className="text-lg font-bold text-zinc-200">{passCount}/{checks.length}</div>
              <div className="text-[10px] text-zinc-600">checks</div>
            </div>
          )}
          <div className="text-center">
            <div className="text-lg font-bold text-zinc-200">{job.iteration_count}</div>
            <div className="text-[10px] text-zinc-600">iterations</div>
          </div>
        </div>
      </div>

      {/* Flow Tabs */}
      <div className="flex gap-1 bg-zinc-900 rounded-xl p-1 border border-zinc-800">
        {RESULT_TABS.map((tab, idx) => {
          const Icon = tab.icon
          const status = tabStatus(tab.key)
          const isActive = activeTab === tab.key
          return (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={cn(
                'flex-1 flex items-center justify-center gap-2 px-3 py-2.5 rounded-lg text-xs font-medium transition-all relative',
                isActive
                  ? 'bg-zinc-800 text-zinc-100 shadow-sm'
                  : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/50'
              )}
            >
              <Icon className="w-3.5 h-3.5" />
              <span className="hidden md:inline">{tab.label}</span>
              <span className={cn('w-1.5 h-1.5 rounded-full absolute top-1.5 right-1.5', statusDot(status))} />
              {idx < RESULT_TABS.length - 1 && !isActive && (
                <ArrowRight className="w-3 h-3 text-zinc-800 absolute -right-2 z-10" />
              )}
            </button>
          )
        })}
      </div>

      {/* Tab Content */}
      <div className="min-h-[300px]">
        {activeTab === 'localization' && (
          <div className="space-y-4">
            {result.localization ? (
              <div className="p-5 rounded-xl bg-zinc-900 border border-zinc-800">
                <div className="flex items-center gap-2 mb-4">
                  <Target className="w-5 h-5 text-amber-400" />
                  <h3 className="text-sm font-bold text-zinc-200">Root Cause Analysis</h3>
                  {result.localization.confidence > 0 && (
                    <span className="ml-auto px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 text-xs font-bold border border-amber-500/20">
                      {Math.round(result.localization.confidence * 100)}% confident
                    </span>
                  )}
                </div>
                <p className="text-sm text-zinc-300 leading-relaxed mb-4">{result.localization.root_cause_hypothesis}</p>
                <div>
                  <p className="text-[10px] font-bold text-zinc-600 uppercase mb-2">Fault Files</p>
                  <div className="flex flex-wrap gap-2">
                    {result.localization.fault_files?.map((f: string) => (
                      <span key={f} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-800 border border-zinc-700/50 text-xs text-zinc-300 font-mono">
                        <FileCode className="w-3.5 h-3.5 text-amber-400" />{f}
                      </span>
                    ))}
                  </div>
                </div>
                {result.localization.fault_functions?.length > 0 && (
                  <div className="mt-3">
                    <p className="text-[10px] font-bold text-zinc-600 uppercase mb-2">Fault Functions</p>
                    <div className="flex flex-wrap gap-2">
                      {result.localization.fault_functions.map((f: string) => (
                        <span key={f} className="px-3 py-1.5 rounded-lg bg-orange-500/10 border border-orange-500/20 text-xs text-orange-300 font-mono">{f}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center justify-center py-12 text-sm text-zinc-600">No localization data</div>
            )}
          </div>
        )}

        {activeTab === 'changes' && (
          <div className="space-y-4">
            {patches.length > 0 ? (
              <>
                {result.repair?.explanation && (
                  <div className="p-4 rounded-xl bg-zinc-900 border border-zinc-800">
                    <p className="text-sm text-zinc-300">{result.repair.explanation}</p>
                  </div>
                )}
                <div className="space-y-3">
                  {patches.map((p) => (
                    <PatchCard key={p.file_path || p.explanation} patch={p} />
                  ))}
                </div>
              </>
            ) : (
              <div className="flex items-center justify-center py-12 text-sm text-zinc-600">No code changes generated</div>
            )}
          </div>
        )}

        {activeTab === 'tests' && (
          <div className="space-y-4">
            {/* Test patches */}
            {testPatches.length > 0 && (
              <div className="space-y-3">
                <h4 className="text-xs font-bold text-zinc-400 uppercase">Generated Tests</h4>
                {testPatches.map((p) => (
                  <PatchCard key={p.file_path || 'test'} patch={p} />
                ))}
              </div>
            )}
            {/* Test execution results */}
            {result.test_result ? (
              <div className={cn('p-5 rounded-xl border',
                result.test_result.includes('passed') ? 'bg-green-950/20 border-green-800/30' : 'bg-red-950/20 border-red-800/30'
              )}>
                <div className="flex items-center gap-2 mb-3">
                  <Shield className={cn('w-5 h-5', result.test_result.includes('passed') ? 'text-green-400' : 'text-red-400')} />
                  <h3 className="text-sm font-bold text-zinc-200">Test Execution</h3>
                </div>
                <pre className="text-xs font-mono text-zinc-300 whitespace-pre-wrap bg-zinc-950/50 rounded-lg p-3">{result.test_result}</pre>
              </div>
            ) : testPatches.length === 0 ? (
              <div className="flex items-center justify-center py-12 text-sm text-zinc-600">No test data</div>
            ) : null}
          </div>
        )}

        {activeTab === 'review' && (
          <div className="space-y-4">
            {checks.length > 0 ? (
              <div className="p-5 rounded-xl bg-zinc-900 border border-zinc-800">
                <div className="flex items-center gap-2 mb-4">
                  <Eye className="w-5 h-5 text-cyan-400" />
                  <h3 className="text-sm font-bold text-zinc-200">
                    Review: <span className={cn(
                      result.review?.verdict === 'APPROVE' ? 'text-green-400' :
                      result.review?.verdict === 'CHANGES_REQUESTED' ? 'text-yellow-400' : 'text-red-400'
                    )}>{result.review?.verdict}</span>
                  </h3>
                  {result.review?.confidence != null && (
                    <span className="ml-auto text-xs font-mono text-zinc-500">{Math.round(Number(result.review.confidence) * 100)}% confidence</span>
                  )}
                </div>
                <ReviewChecks checks={checks} />
                {result.review?.feedback && (
                  <div className="mt-4 p-3 rounded-lg bg-zinc-800/50 border border-zinc-700/30">
                    <p className="text-[10px] font-bold text-zinc-600 uppercase mb-1">Reviewer Feedback</p>
                    <p className="text-xs text-zinc-400 leading-relaxed">{result.review.feedback}</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center justify-center py-12 text-sm text-zinc-600">No review data</div>
            )}
          </div>
        )}

        {activeTab === 'pr' && (
          <div className="space-y-4">
            {result.pr_url ? (
              <div className="p-6 rounded-xl bg-green-950/20 border border-green-800/30 text-center">
                <GitPullRequest className="w-10 h-10 text-green-400 mx-auto mb-3" />
                <h3 className="text-lg font-bold text-green-300 mb-2">Pull Request Created</h3>
                <a
                  href={result.pr_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-green-400 font-mono hover:underline"
                >{result.pr_url}</a>
                <p className="text-xs text-zinc-600 mt-3">Review and merge on GitHub</p>
              </div>
            ) : job.status === 'done' || job.status === 'escalated' ? (
              <div className="p-6 rounded-xl bg-zinc-900 border border-zinc-800 text-center">
                <GitPullRequest className="w-10 h-10 text-zinc-700 mx-auto mb-3" />
                <h3 className="text-sm font-bold text-zinc-400 mb-1">No PR Created</h3>
                <p className="text-xs text-zinc-600">{job.error || 'Pipeline did not reach PR creation stage'}</p>
              </div>
            ) : (
              <div className="flex items-center justify-center py-12 text-sm text-zinc-600">Waiting for pipeline to complete...</div>
            )}
          </div>
        )}
      </div>
    </div>
  )
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

  // Debug / trace
  const [debugMode, setDebugMode] = useState(false)
  const [traceEvents, setTraceEvents] = useState<TraceEvent[]>([])
  const [traceIsLive, setTraceIsLive] = useState(false)
  const traceUnsubRef = useRef<(() => void) | null>(null)

  // Custom ticket form
  const [customTicketId, setCustomTicketId] = useState('')
  const [customTitle, setCustomTitle] = useState('')
  const [customDesc, setCustomDesc] = useState('')
  const [customComponent, setCustomComponent] = useState('')
  const [customRepoPath, setCustomRepoPath] = useState('')
  // showCustom removed — form is always visible now

  useEffect(() => {
    listAgentTickets().then(setTickets).catch((e: Error) => setError(`Failed to load tickets: ${e.message}`))
  }, [])

  useEffect(() => {
    listAgentJobs().then(setPastJobs).catch(() => {})
  }, [])

  // Cleanup poll + trace on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
      if (traceUnsubRef.current) traceUnsubRef.current()
    }
  }, [])

  const startTraceSubscription = useCallback((jobId: string) => {
    // Clean up previous subscription
    if (traceUnsubRef.current) traceUnsubRef.current()
    setTraceEvents([])
    setTraceIsLive(true)
    traceUnsubRef.current = subscribeToTrace(
      jobId,
      (evt) => setTraceEvents(prev => {
        // Deduplicate by index — SSE catchup + live stream can overlap
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
    setTraceEvents([])
    try {
      const res = await runMockTicket(ticketId, debugMode)
      saveJobToStorage(res.job_id, ticketId)
      setActiveJob({ job_id: res.job_id, status: 'pending', stage: 'Queued', iteration_count: 0, result: null, error: '', debug: debugMode })
      pollJob(res.job_id)
      if (debugMode) startTraceSubscription(res.job_id)
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
    setTraceEvents([])
    try {
      const res = await runAgentTicket({
        ticket_id: customTicketId.trim() || undefined,
        title: customTitle,
        description: customDesc,
        repo_name: activeRepo,
        repo_path: customRepoPath.trim() || activeRepoData?.repo_path,
        affected_component: customComponent.trim() || undefined,
        debug: debugMode,
      })
      saveJobToStorage(res.job_id, 'custom')
      setActiveJob({ job_id: res.job_id, status: 'pending', stage: 'Queued', iteration_count: 0, result: null, error: '', debug: debugMode })
      pollJob(res.job_id)
      if (debugMode) startTraceSubscription(res.job_id)
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
        <div className="flex items-center gap-3">
          {activeJob && (
            <PipelineProgress status={activeJob.stage || activeJob.status} iterationCount={activeJob.iteration_count} />
          )}
          <label className="flex items-center gap-1.5 cursor-pointer select-none ml-2">
            <input
              type="checkbox"
              checked={debugMode}
              onChange={(e) => setDebugMode(e.target.checked)}
              className="w-3.5 h-3.5 rounded border-zinc-600 bg-zinc-800 text-emerald-500 focus:ring-emerald-500/30"
            />
            <span className="text-[10px] text-zinc-500 font-medium">Debug</span>
            {debugMode && <Terminal className="w-3 h-3 text-emerald-400" />}
          </label>
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

          {/* Trace Log (debug mode) */}
          {(traceEvents.length > 0 || (debugMode && isRunning)) && (
            <TraceLogPanel events={traceEvents} isLive={traceIsLive} />
          )}

          {/* Pipeline Result — Flow View */}
          {activeJob?.result && (
            <ResultFlowView job={activeJob} />
          )}

          {/* Running indicator (no result yet) */}
          {isRunning && !result && (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <Loader2 className="w-8 h-8 text-rose-400 animate-spin mb-4" />
              <p className="text-sm text-zinc-400">Agent pipeline running...</p>
              <p className="text-xs text-zinc-600 mt-1">This takes 1-2 minutes (5 LLM calls)</p>
            </div>
          )}

          {/* Custom Ticket Form — always visible */}
          <div className="p-4 rounded-xl bg-zinc-900 border border-rose-500/20">
            <div className="flex items-center gap-2 mb-3">
              <Bug className="w-4 h-4 text-rose-400" />
              <h2 className="text-sm font-bold text-zinc-300">Submit Bug Ticket</h2>
            </div>
            <div className="space-y-3">
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
                  placeholder="Affected file (e.g. backend/rag/retriever.py)"
                  className="flex-1 px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-sm text-zinc-200 placeholder-zinc-600 focus:border-rose-500/50 focus:outline-none font-mono"
                />
                <button
                  onClick={handleRunCustom}
                  disabled={isRunning || !customTitle.trim() || !activeRepo}
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

          {/* Ticket Selection */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-bold text-zinc-300">Sample Tickets</h2>
            </div>

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
                  <p className="text-sm font-medium text-zinc-400 mb-1">No sample tickets loaded</p>
                  <p className="text-xs text-zinc-600">Use the form above to submit a custom bug ticket</p>
                </div>
              )}
            </div>
          </div>

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
