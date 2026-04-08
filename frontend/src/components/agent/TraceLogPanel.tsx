import { useState, useEffect, useRef } from 'react'
import {
  Zap, Wrench, ArrowRight, FileCode, Brain, Shield,
  ChevronDown, ChevronRight, Terminal, Filter,
  MessageSquare, TestTube, Layers, XCircle, Target,
} from 'lucide-react'
import { cn } from '../../lib/utils'
import type { TraceEvent } from '../../lib/api'

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

  let summary = ''
  if (evt.event_type === 'stage_start') summary = `-> ${data.stage}`
  else if (evt.event_type === 'stage_end') summary = `done ${data.stage} (${formatDuration(data.duration_ms as number || 0)})`
  else if (evt.event_type === 'llm_request') summary = `${data.model || 'llm'} [${data.phase || '?'}] ~${data.context_tokens || 0} tokens (${data.context_pct || 0}% ctx)`
  else if (evt.event_type === 'llm_response') summary = `${data.model || 'llm'} in=${data.input_tokens || 0} out=${data.output_tokens || 0} $${Number(data.cost_usd || 0).toFixed(3)} (total $${Number(data.cumulative_cost_usd || 0).toFixed(3)})`
  else if (evt.event_type === 'tool_call') summary = `[${data.phase || '?'}] ${data.tool_name}(${JSON.stringify(data.args || {}).slice(0, 80)})`
  else if (evt.event_type === 'tool_result') summary = `${data.tool_name} (${formatDuration(data.duration_ms as number || 0)}) ${String(data.result_preview || '').slice(0, 60)}`
  else if (evt.event_type === 'guardrail_event') summary = `${data.action === 'block' ? 'BLOCKED' : 'WARN'} ${data.tool_name}: ${String(data.message || '').slice(0, 80)}`
  else if (evt.event_type === 'state_transition') summary = `${data.from_phase} -> ${data.to_phase} (at call #${data.at_call}, $${Number(data.cost_usd_at_transition || 0).toFixed(3)})`
  else if (evt.event_type === 'context_compaction') summary = `${data.action}: ${data.tokens_before} -> ${data.tokens_after} tokens (saved ${data.tokens_saved})`
  else if (evt.event_type === 'prompt_build') summary = `system=${data.system_prompt_tokens_approx || 0} tokens, kickstart=${data.kickstart_chars || 0} chars`
  else if (evt.event_type === 'run_outcome') summary = `${String(data.outcome || '').toUpperCase()} -- ${data.tool_call_count} calls, $${Number(data.cost_usd || 0).toFixed(3)}, ${data.elapsed_seconds || 0}s`
  else if (evt.event_type === 'patch_candidate') summary = `${data.file_path}`
  else if (evt.event_type === 'test_output') summary = (data.passed ? 'PASSED' : 'FAILED') + ` (${data.patches_applied} patches)`
  else if (evt.event_type === 'error') summary = String(data.message || '').slice(0, 120)
  else if (evt.event_type === 'info') summary = String(data.message || '').slice(0, 120)

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

export function TraceLogPanel({ events, isLive }: { events: TraceEvent[]; isLive: boolean }) {
  const [filter, setFilter] = useState<TraceFilter>('all')
  const [collapsed, setCollapsed] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const autoScrollRef = useRef(true)

  const filtered = events.filter(e => matchesFilter(e, filter))

  useEffect(() => {
    if (autoScrollRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [filtered.length])

  const handleScroll = () => {
    if (!scrollRef.current) return
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current
    autoScrollRef.current = scrollHeight - scrollTop - clientHeight < 40
  }

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
