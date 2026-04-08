import { useEffect, useRef, useMemo } from 'react'
import {
  Search, FileCode, Wrench, TestTube, Eye, GitPullRequest,
  ArrowRight, DollarSign, Loader2, Shield, Brain, Zap,
  CheckCircle2, XCircle, AlertTriangle,
} from 'lucide-react'
import { cn } from '../../lib/utils'
import type { TraceEvent } from '../../lib/api'

// ─── Human-readable message generation ───────────────────────────────────────

interface ActivityItem {
  id: string
  message: string
  icon: typeof Zap
  iconColor: string
  timestamp: number
  type: 'action' | 'result' | 'phase' | 'outcome' | 'warning'
  detail?: string
}

const TOOL_LABELS: Record<string, { verb: string; icon: typeof Zap; iconColor: string }> = {
  grep_repo:         { verb: 'Searching for',         icon: Search,   iconColor: 'text-blue-400' },
  grep_codebase:     { verb: 'Searching codebase for', icon: Search,   iconColor: 'text-blue-400' },
  read_file:         { verb: 'Reading',                icon: FileCode, iconColor: 'text-cyan-400' },
  list_directory:    { verb: 'Browsing',               icon: FileCode, iconColor: 'text-cyan-400' },
  string_replace:    { verb: 'Editing',                icon: Wrench,   iconColor: 'text-orange-400' },
  insert_text:       { verb: 'Adding code to',         icon: Wrench,   iconColor: 'text-orange-400' },
  create_file:       { verb: 'Creating',               icon: Wrench,   iconColor: 'text-orange-400' },
  run_tests:         { verb: 'Running tests',          icon: TestTube, iconColor: 'text-lime-400' },
  run_command:       { verb: 'Running command',        icon: TestTube, iconColor: 'text-lime-400' },
  get_callers:       { verb: 'Finding callers of',     icon: Search,   iconColor: 'text-purple-400' },
  get_callees:       { verb: 'Finding callees of',     icon: Search,   iconColor: 'text-purple-400' },
  get_blast_radius:  { verb: 'Checking blast radius of', icon: Search, iconColor: 'text-purple-400' },
  get_test_coverage: { verb: 'Checking test coverage', icon: TestTube, iconColor: 'text-lime-400' },
  query_graph:       { verb: 'Querying knowledge graph', icon: Brain,  iconColor: 'text-purple-400' },
  semantic_search:   { verb: 'Semantic search for',    icon: Search,   iconColor: 'text-blue-400' },
  submit_pr:         { verb: 'Creating pull request',  icon: GitPullRequest, iconColor: 'text-green-400' },
  request_review:    { verb: 'Requesting code review', icon: Eye,      iconColor: 'text-cyan-400' },
}

const PHASE_LABELS: Record<string, string> = {
  explore: 'Exploring codebase',
  edit: 'Applying fixes',
  test: 'Testing changes',
  review: 'Reviewing code',
  submit: 'Submitting PR',
}

function formatToolMessage(toolName: string, args: Record<string, unknown>): string {
  const label = TOOL_LABELS[toolName]
  const verb = label?.verb ?? `Using ${toolName}`

  // Build a human-readable target from the args
  if (toolName === 'grep_repo' || toolName === 'grep_codebase') {
    const pattern = args.pattern || args.query || args.search_term || ''
    const path = args.file_path || args.path || ''
    return path ? `${verb} "${pattern}" in ${shortenPath(String(path))}` : `${verb} "${pattern}"`
  }
  if (toolName === 'read_file') {
    return `${verb} ${shortenPath(String(args.file_path || args.path || ''))}`
  }
  if (toolName === 'list_directory') {
    return `${verb} ${shortenPath(String(args.path || args.directory || ''))}`
  }
  if (toolName === 'string_replace' || toolName === 'insert_text') {
    return `${verb} ${shortenPath(String(args.file_path || args.path || ''))}`
  }
  if (toolName === 'create_file') {
    return `${verb} ${shortenPath(String(args.file_path || args.path || ''))}`
  }
  if (toolName === 'run_tests') {
    const cmd = args.command || args.test_command || ''
    return cmd ? `${verb}: ${String(cmd).slice(0, 60)}` : verb
  }
  if (toolName === 'run_command') {
    return `${verb}: ${String(args.command || '').slice(0, 60)}`
  }
  if (toolName === 'get_callers' || toolName === 'get_callees' || toolName === 'get_blast_radius') {
    return `${verb} ${args.function_name || args.symbol || ''}`
  }
  if (toolName === 'query_graph' || toolName === 'semantic_search') {
    return `${verb} "${args.query || args.pattern || ''}"`
  }
  if (toolName === 'submit_pr') {
    return verb
  }
  if (toolName === 'request_review') {
    return verb
  }

  // Fallback: show tool name + first string arg
  const firstArg = Object.values(args).find(v => typeof v === 'string')
  return firstArg ? `${verb}: ${String(firstArg).slice(0, 50)}` : verb
}

function formatToolResult(toolName: string, data: Record<string, unknown>): string | null {
  const preview = String(data.result_preview || '').trim()
  const durationMs = data.duration_ms as number | undefined

  if (toolName === 'grep_repo' || toolName === 'grep_codebase') {
    // Extract match count from preview
    const matchCount = preview.match(/(\d+)\s*match/i)
    if (matchCount) return `Found ${matchCount[1]} matches`
    if (preview.includes('No matches') || preview.includes('0 results')) return 'No matches found'
    return preview ? `Found results` : null
  }
  if (toolName === 'run_tests') {
    if (data.passed) return `Tests passed`
    if (preview.includes('FAILED') || preview.includes('failed')) return `Tests failed`
    if (preview.includes('passed') || preview.includes('PASSED')) return `Tests passed`
    return preview ? preview.slice(0, 80) : null
  }
  if (toolName === 'get_callers' || toolName === 'get_callees') {
    const count = preview.match(/(\d+)/)
    if (count) return `Found ${count[1]} references`
    return null
  }

  // Don't show results for file reads / edits (too noisy)
  if (['read_file', 'list_directory', 'string_replace', 'insert_text', 'create_file'].includes(toolName)) {
    if (durationMs && durationMs > 2000) return `Done (${(durationMs / 1000).toFixed(1)}s)`
    return null
  }

  return preview ? preview.slice(0, 80) : null
}

function shortenPath(path: string): string {
  if (!path) return ''
  // Show last 2-3 path segments
  const parts = path.split('/')
  if (parts.length <= 3) return path
  return '.../' + parts.slice(-3).join('/')
}

function traceToActivity(events: TraceEvent[]): ActivityItem[] {
  const items: ActivityItem[] = []

  for (const evt of events) {
    const { event_type, data, timestamp, index } = evt

    if (event_type === 'tool_call') {
      const toolName = String(data.tool_name || '')
      const args = (data.args || {}) as Record<string, unknown>
      const label = TOOL_LABELS[toolName]
      items.push({
        id: `tc-${index}`,
        message: formatToolMessage(toolName, args),
        icon: label?.icon ?? Wrench,
        iconColor: label?.iconColor ?? 'text-orange-400',
        timestamp,
        type: 'action',
      })
    }

    if (event_type === 'tool_result') {
      const toolName = String(data.tool_name || '')
      const msg = formatToolResult(toolName, data)
      if (msg) {
        items.push({
          id: `tr-${index}`,
          message: msg,
          icon: CheckCircle2,
          iconColor: 'text-zinc-500',
          timestamp,
          type: 'result',
        })
      }
    }

    if (event_type === 'state_transition') {
      const toPhase = String(data.to_phase || '')
      const label = PHASE_LABELS[toPhase] || `Phase: ${toPhase}`
      items.push({
        id: `st-${index}`,
        message: label,
        icon: ArrowRight,
        iconColor: 'text-cyan-400',
        timestamp,
        type: 'phase',
      })
    }

    if (event_type === 'test_output') {
      const passed = data.passed as boolean
      const patchCount = data.patches_applied as number | undefined
      items.push({
        id: `to-${index}`,
        message: passed
          ? `All tests passed${patchCount ? ` (${patchCount} files patched)` : ''}`
          : `Tests failed${patchCount ? ` (${patchCount} files patched)` : ''}`,
        icon: passed ? CheckCircle2 : XCircle,
        iconColor: passed ? 'text-green-400' : 'text-red-400',
        timestamp,
        type: 'outcome',
        detail: String(data.result || '').slice(0, 200),
      })
    }

    if (event_type === 'patch_candidate') {
      items.push({
        id: `pc-${index}`,
        message: `Generated fix for ${shortenPath(String(data.file_path || ''))}`,
        icon: Wrench,
        iconColor: 'text-green-400',
        timestamp,
        type: 'action',
        detail: String(data.explanation || ''),
      })
    }

    if (event_type === 'guardrail_event') {
      const action = data.action === 'block' ? 'Blocked' : 'Warning'
      items.push({
        id: `ge-${index}`,
        message: `${action}: ${String(data.message || '').slice(0, 80)}`,
        icon: Shield,
        iconColor: data.action === 'block' ? 'text-red-400' : 'text-yellow-400',
        timestamp,
        type: 'warning',
      })
    }

    if (event_type === 'run_outcome') {
      const outcome = String(data.outcome || '').toLowerCase()
      const cost = Number(data.cost_usd || 0)
      const calls = data.tool_call_count as number | undefined
      const elapsed = data.elapsed_seconds as number | undefined
      const isSuccess = outcome === 'submitted' || outcome === 'success' || outcome === 'done'
      items.push({
        id: `ro-${index}`,
        message: isSuccess
          ? `Done -- ${calls || 0} tool calls, $${cost.toFixed(2)}${elapsed ? `, ${elapsed}s` : ''}`
          : `${outcome || 'finished'} -- ${calls || 0} tool calls, $${cost.toFixed(2)}`,
        icon: isSuccess ? CheckCircle2 : AlertTriangle,
        iconColor: isSuccess ? 'text-green-400' : 'text-yellow-400',
        timestamp,
        type: 'outcome',
      })
    }

    if (event_type === 'error') {
      items.push({
        id: `err-${index}`,
        message: String(data.message || 'Unknown error').slice(0, 100),
        icon: XCircle,
        iconColor: 'text-red-400',
        timestamp,
        type: 'warning',
      })
    }
  }

  return items
}

// ─── Cost tracker ────────────────────────────────────────────────────────────

function extractCost(events: TraceEvent[]): { current: number; toolCalls: number } {
  let current = 0
  let toolCalls = 0
  for (const evt of events) {
    if (evt.event_type === 'llm_response' && evt.data.cumulative_cost_usd) {
      current = Number(evt.data.cumulative_cost_usd)
    }
    if (evt.event_type === 'tool_call') {
      toolCalls++
    }
  }
  return { current, toolCalls }
}

// ─── Component ───────────────────────────────────────────────────────────────

export function LiveActivityFeed({ events, isLive }: { events: TraceEvent[]; isLive: boolean }) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const autoScrollRef = useRef(true)

  const items = useMemo(() => traceToActivity(events), [events])
  const { current: cost, toolCalls } = useMemo(() => extractCost(events), [events])

  // Current phase
  const currentPhase = useMemo(() => {
    for (let i = events.length - 1; i >= 0; i--) {
      if (events[i].event_type === 'state_transition') {
        return String(events[i].data.to_phase || '')
      }
    }
    return ''
  }, [events])

  useEffect(() => {
    if (autoScrollRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [items.length])

  const handleScroll = () => {
    if (!scrollRef.current) return
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current
    autoScrollRef.current = scrollHeight - scrollTop - clientHeight < 40
  }

  // Get the last item to show "currently doing"
  const actionItems = items.filter(i => i.type === 'action')
  const lastAction = actionItems[actionItems.length - 1]

  return (
    <div className="rounded-xl bg-zinc-900 border border-zinc-800 overflow-hidden">
      {/* Header with live status */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-zinc-800">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          {isLive ? (
            <>
              <Loader2 className="w-4 h-4 text-rose-400 animate-spin flex-shrink-0" />
              <span className="text-sm font-medium text-zinc-200 truncate">
                {lastAction?.message || (PHASE_LABELS[currentPhase] || 'Starting agent...')}
              </span>
            </>
          ) : (
            <>
              <CheckCircle2 className="w-4 h-4 text-green-400 flex-shrink-0" />
              <span className="text-sm font-medium text-zinc-300">Agent finished</span>
            </>
          )}
        </div>

        {/* Cost + tool call counter */}
        <div className="flex items-center gap-3 flex-shrink-0">
          {toolCalls > 0 && (
            <span className="text-[10px] font-mono text-zinc-500 flex items-center gap-1">
              <Wrench className="w-3 h-3" />
              {toolCalls} calls
            </span>
          )}
          {cost > 0 && (
            <span className="text-[10px] font-mono text-zinc-400 flex items-center gap-1 px-2 py-0.5 rounded-full bg-zinc-800 border border-zinc-700">
              <DollarSign className="w-3 h-3 text-emerald-400" />
              ${cost.toFixed(3)}
            </span>
          )}
          {isLive && <span className="w-2 h-2 rounded-full bg-rose-400 animate-pulse" />}
        </div>
      </div>

      {/* Activity items */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="max-h-72 overflow-y-auto"
      >
        {items.length === 0 ? (
          <div className="flex items-center justify-center py-8 text-xs text-zinc-600">
            <Loader2 className="w-3.5 h-3.5 animate-spin mr-2" />
            Waiting for agent to start...
          </div>
        ) : (
          <div className="py-1">
            {items.map((item) => (
              <div
                key={item.id}
                className={cn(
                  'flex items-start gap-2.5 px-4 py-1.5 text-sm',
                  item.type === 'result' && 'text-zinc-500 text-xs',
                  item.type === 'phase' && 'py-2 mt-1',
                  item.type === 'outcome' && 'py-2',
                )}
              >
                <item.icon className={cn(
                  'flex-shrink-0 mt-0.5',
                  item.iconColor,
                  item.type === 'result' ? 'w-3 h-3' : 'w-3.5 h-3.5',
                )} />
                <span className={cn(
                  'min-w-0',
                  item.type === 'phase' && 'font-bold text-cyan-300 text-xs uppercase tracking-wide',
                  item.type === 'action' && 'text-zinc-300 text-xs',
                  item.type === 'result' && 'text-zinc-500 text-xs',
                  item.type === 'outcome' && 'font-medium text-zinc-200 text-xs',
                  item.type === 'warning' && 'text-yellow-300 text-xs',
                )}>
                  {item.message}
                </span>
              </div>
            ))}
            {isLive && (
              <div className="flex items-center gap-2.5 px-4 py-1.5">
                <Loader2 className="w-3.5 h-3.5 text-zinc-600 animate-spin flex-shrink-0" />
                <span className="text-xs text-zinc-600 animate-pulse">Thinking...</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
