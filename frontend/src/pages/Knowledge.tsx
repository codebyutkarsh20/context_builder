import { useState, useEffect, useRef, useCallback } from 'react'
import {
  BookOpen, HelpCircle, CheckCircle2, Shield,
  ChevronDown, ChevronRight, FileCode, Send, Loader2, BarChart3, Plus, X, AlertCircle,
  Search, Link2, FunctionSquare, Boxes, File,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { useRepo } from '../lib/RepoContext'
import {
  getKnowledgeQuestions, getKnowledgeRules, getKnowledgeStats,
  submitAnswer, addRule, searchGraphNodes,
  type KnowledgeQuestion, type KnowledgeRule, type KnowledgeStats, type GraphNodeSummary,
} from '../lib/api'

const SEVERITY_COLORS: Record<string, string> = {
  critical: 'bg-red-500/20 text-red-300 border-red-500/30',
  high: 'bg-orange-500/20 text-orange-300 border-orange-500/30',
  medium: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
  low: 'bg-green-500/20 text-green-300 border-green-500/30',
}

const RULE_TYPE_COLORS: Record<string, string> = {
  legal: 'text-red-400',
  contractual: 'text-orange-400',
  policy: 'text-yellow-400',
  architectural: 'text-blue-400',
}

function StatsCard({ stats }: { stats: KnowledgeStats }) {
  const pct = stats.coverage
  return (
    <div className="grid grid-cols-4 gap-3 mb-6">
      <div className="p-3 rounded-xl bg-zinc-900 border border-zinc-800">
        <p className="text-[10px] text-zinc-600 uppercase font-bold">Questions</p>
        <p className="text-xl font-bold text-zinc-200">{stats.questions}</p>
      </div>
      <div className="p-3 rounded-xl bg-zinc-900 border border-zinc-800">
        <p className="text-[10px] text-zinc-600 uppercase font-bold">Answered</p>
        <p className="text-xl font-bold text-green-400">{stats.answered}</p>
      </div>
      <div className="p-3 rounded-xl bg-zinc-900 border border-zinc-800">
        <p className="text-[10px] text-zinc-600 uppercase font-bold">Rules</p>
        <p className="text-xl font-bold text-purple-400">{stats.rules}</p>
      </div>
      <div className="p-3 rounded-xl bg-zinc-900 border border-zinc-800">
        <p className="text-[10px] text-zinc-600 uppercase font-bold">Coverage</p>
        <div className="flex items-end gap-1">
          <p className="text-xl font-bold text-zinc-200">{pct}%</p>
          <div className="flex-1 h-1.5 bg-zinc-800 rounded-full mb-1.5">
            <div className="h-full bg-emerald-500 rounded-full transition-all" style={{ width: `${pct}%` }} />
          </div>
        </div>
      </div>
    </div>
  )
}

function QuestionCard({ q, repo, onAnswered }: {
  q: KnowledgeQuestion
  repo: string
  onAnswered: () => void
}) {
  const [expanded, setExpanded] = useState(false)
  const [answer, setAnswer] = useState('')
  const [ruleType, setRuleType] = useState('policy')
  const [severity, setSeverity] = useState('medium')
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)

  const handleSubmit = async () => {
    if (!answer.trim()) return
    setSubmitting(true)
    setSubmitError(null)
    try {
      await submitAnswer(repo, q.id, answer, ruleType, severity)
      onAnswered()
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : 'Failed to submit')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className={cn(
      'rounded-xl border transition-all',
      q.answered ? 'bg-zinc-900/50 border-zinc-800/50' : 'bg-zinc-900 border-zinc-800 hover:border-zinc-700'
    )}>
      <button onClick={() => setExpanded(!expanded)} className="w-full p-4 text-left">
        <div className="flex items-start gap-3">
          <div className="mt-0.5">
            {q.answered
              ? <CheckCircle2 className="w-4 h-4 text-green-400" />
              : <HelpCircle className="w-4 h-4 text-amber-400" />}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm text-zinc-200 font-medium">{q.question}</p>
            <div className="flex items-center gap-2 mt-1">
              <span className="text-[10px] font-mono text-zinc-600">{q.file}</span>
              <span className={cn('text-[10px] px-1.5 py-0.5 rounded-full border',
                q.condition_type === 'threshold' ? 'bg-red-500/10 text-red-400 border-red-500/20' :
                q.condition_type === 'role_check' ? 'bg-blue-500/10 text-blue-400 border-blue-500/20' :
                'bg-zinc-800 text-zinc-500 border-zinc-700'
              )}>{q.condition_type}</span>
            </div>
            {q.answered && q.answer && (
              <p className="text-xs text-green-400/70 mt-1 italic">Answered: {q.answer.slice(0, 100)}</p>
            )}
          </div>
          {expanded ? <ChevronDown className="w-4 h-4 text-zinc-600" /> : <ChevronRight className="w-4 h-4 text-zinc-600" />}
        </div>
      </button>

      {expanded && (
        <div className="px-4 pb-4 border-t border-zinc-800/50 pt-3 ml-7 space-y-3">
          {q.explanation && (
            <div>
              <p className="text-[10px] font-bold text-zinc-600 uppercase mb-1">What the AI sees</p>
              <p className="text-xs text-zinc-400">{q.explanation}</p>
            </div>
          )}
          {q.condition && (
            <div>
              <p className="text-[10px] font-bold text-zinc-600 uppercase mb-1">Code condition</p>
              <code className="text-xs text-amber-300 bg-zinc-800 px-2 py-1 rounded font-mono block">{q.condition}</code>
            </div>
          )}

          {!q.answered && (
            <div className="space-y-2 pt-2">
              <textarea
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                placeholder="Explain the business reason behind this code..."
                rows={3}
                className="w-full px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-sm text-zinc-200 placeholder-zinc-600 focus:border-purple-500/50 focus:outline-none resize-none"
              />
              {q.suggested_answers?.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {q.suggested_answers.map((sa, i) => (
                    <button key={i} onClick={() => setAnswer(sa)}
                      className="text-[10px] px-2 py-1 rounded-lg bg-zinc-800 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700 transition-colors border border-zinc-700/50">
                      {sa.slice(0, 60)}
                    </button>
                  ))}
                </div>
              )}
              {submitError && (
                <div className="flex items-center gap-2 p-2 rounded-lg bg-red-950/30 border border-red-800/40 text-red-400 text-xs">
                  <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />{submitError}
                </div>
              )}
              <div className="flex items-center gap-3">
                <select value={ruleType} onChange={(e) => setRuleType(e.target.value)}
                  className="text-xs bg-zinc-800 border border-zinc-700 text-zinc-300 rounded-lg px-2 py-1.5">
                  <option value="legal">Legal</option>
                  <option value="contractual">Contractual</option>
                  <option value="policy">Policy</option>
                  <option value="architectural">Architectural</option>
                </select>
                <select value={severity} onChange={(e) => setSeverity(e.target.value)}
                  className="text-xs bg-zinc-800 border border-zinc-700 text-zinc-300 rounded-lg px-2 py-1.5">
                  <option value="critical">Critical</option>
                  <option value="high">High</option>
                  <option value="medium">Medium</option>
                  <option value="low">Low</option>
                </select>
                <button onClick={handleSubmit} disabled={!answer.trim() || submitting}
                  className="ml-auto flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-purple-500/20 text-purple-300 hover:bg-purple-500/30 border border-purple-500/30 text-xs font-medium transition-all disabled:opacity-50">
                  {submitting ? <Loader2 className="w-3 h-3 animate-spin" /> : <Send className="w-3 h-3" />}
                  Store Rule
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ─── Node type helpers ────────────────────────────────────────────────────────

const NODE_TYPE_META: Record<string, { color: string; icon: React.ReactNode; label: string }> = {
  Function: { color: 'text-orange-400 bg-orange-500/10 border-orange-500/20', icon: <FunctionSquare className="w-3 h-3" />, label: 'Function' },
  Class:    { color: 'text-green-400 bg-green-500/10 border-green-500/20',    icon: <Boxes className="w-3 h-3" />,           label: 'Class' },
  File:     { color: 'text-blue-400 bg-blue-500/10 border-blue-500/20',       icon: <File className="w-3 h-3" />,            label: 'File' },
}

function NodeTypeBadge({ type }: { type: string }) {
  const m = NODE_TYPE_META[type]
  if (!m) return <span className="text-[10px] px-1.5 py-0.5 rounded-full border bg-zinc-800 text-zinc-500 border-zinc-700">{type}</span>
  return (
    <span className={cn('flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded-full border font-medium', m.color)}>
      {m.icon}{m.label}
    </span>
  )
}

// ─── Node search picker ────────────────────────────────────────────────────────

function NodeSearchPicker({
  repo,
  value,
  onChange,
}: {
  repo: string
  value: GraphNodeSummary | null
  onChange: (node: GraphNodeSummary | null) => void
}) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<GraphNodeSummary[]>([])
  const [searching, setSearching] = useState(false)
  const [open, setOpen] = useState(false)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const doSearch = useCallback((q: string) => {
    if (!q.trim()) { setResults([]); setOpen(false); return }
    setSearching(true)
    searchGraphNodes(repo, q)
      .then((res) => { setResults(res); setOpen(true) })
      .catch(() => {})
      .finally(() => setSearching(false))
  }, [repo])

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => doSearch(query), 280)
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current) }
  }, [query, doSearch])

  // close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  if (value) {
    return (
      <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-zinc-800 border border-purple-500/40">
        <Link2 className="w-3.5 h-3.5 text-purple-400 flex-shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 flex-wrap">
            <NodeTypeBadge type={value.type} />
            <span className="text-xs font-mono text-zinc-200 font-semibold truncate">{value.name}</span>
          </div>
          {value.file && value.file !== value.id && (
            <p className="text-[10px] font-mono text-zinc-600 mt-0.5 truncate">{value.file}</p>
          )}
        </div>
        <button onClick={() => onChange(null)} className="text-zinc-600 hover:text-red-400 transition-colors flex-shrink-0">
          <X className="w-3.5 h-3.5" />
        </button>
      </div>
    )
  }

  return (
    <div ref={containerRef} className="relative">
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-zinc-600 pointer-events-none" />
        {searching && <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-zinc-600 animate-spin" />}
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => results.length > 0 && setOpen(true)}
          placeholder="Search function, class or file…"
          className="w-full pl-9 pr-9 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-sm text-zinc-200 placeholder-zinc-600 focus:border-purple-500/50 focus:outline-none"
        />
      </div>

      {open && results.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl z-50 max-h-56 overflow-y-auto">
          {results.map((node) => (
            <button
              key={node.id}
              onMouseDown={(e) => { e.preventDefault(); onChange(node); setQuery(''); setOpen(false) }}
              className="w-full flex items-start gap-2.5 px-3 py-2.5 hover:bg-zinc-800 transition-colors text-left border-b border-zinc-800/60 last:border-0"
            >
              <div className="mt-0.5 flex-shrink-0">
                {NODE_TYPE_META[node.type]?.icon ?? <FileCode className="w-3 h-3 text-zinc-500" />}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5 flex-wrap">
                  <NodeTypeBadge type={node.type} />
                  <span className="text-xs font-mono text-zinc-200 font-medium truncate">{node.name}</span>
                </div>
                {node.file && <p className="text-[10px] font-mono text-zinc-600 mt-0.5 truncate">{node.file}</p>}
              </div>
            </button>
          ))}
        </div>
      )}

      {open && !searching && results.length === 0 && query.trim() && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-zinc-900 border border-zinc-700 rounded-xl shadow-xl z-50 px-3 py-3 text-xs text-zinc-600 text-center">
          No nodes found for "{query}"
        </div>
      )}
    </div>
  )
}

// ─── Add Rule Modal ───────────────────────────────────────────────────────────

function AddRuleModal({ repo, onClose, onAdded }: { repo: string; onClose: () => void; onAdded: () => void }) {
  const [description, setDescription] = useState('')
  const [ruleType, setRuleType] = useState('policy')
  const [severity, setSeverity] = useState('medium')
  const [constraint, setConstraint] = useState('')
  const [linkedNode, setLinkedNode] = useState<GraphNodeSummary | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [modalError, setModalError] = useState<string | null>(null)

  const handleSubmit = async () => {
    if (!description.trim()) return
    setSubmitting(true)
    setModalError(null)
    try {
      await addRule(repo, {
        description,
        rule_type: ruleType,
        severity,
        constraint,
        node_id: linkedNode?.id ?? '',
        node_type: linkedNode?.type ?? '',
        node_name: linkedNode?.name ?? '',
        file: linkedNode?.file ?? '',
      })
      onAdded()
      onClose()
    } catch (e) {
      setModalError(e instanceof Error ? e.message : 'Failed to add rule')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
      <div className="w-full max-w-lg bg-zinc-900 border border-zinc-700 rounded-2xl shadow-2xl">
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800">
          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4 text-purple-400" />
            <h2 className="text-sm font-bold text-zinc-100">Add Business Rule</h2>
          </div>
          <button onClick={onClose} className="text-zinc-600 hover:text-zinc-300 transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-5 space-y-4">
          {/* Description */}
          <div>
            <label className="text-[10px] font-bold text-zinc-500 uppercase block mb-1.5">Rule Description *</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="e.g. Late fees cannot exceed 10% of principal per RBI guidelines"
              rows={3}
              className="w-full px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-sm text-zinc-200 placeholder-zinc-600 focus:border-purple-500/50 focus:outline-none resize-none"
              autoFocus
            />
          </div>

          {/* Constraint */}
          <div>
            <label className="text-[10px] font-bold text-zinc-500 uppercase block mb-1.5">Constraint / Enforcement Detail</label>
            <input
              value={constraint}
              onChange={(e) => setConstraint(e.target.value)}
              placeholder="e.g. max_late_fee = principal * 0.10"
              className="w-full px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-sm text-zinc-200 placeholder-zinc-600 focus:border-purple-500/50 focus:outline-none"
            />
          </div>

          {/* Node picker */}
          <div>
            <label className="text-[10px] font-bold text-zinc-500 uppercase block mb-1.5">
              Attach to Node
              <span className="ml-1.5 text-zinc-700 normal-case font-normal">(links this rule to a function, class or file in the graph)</span>
            </label>
            <NodeSearchPicker repo={repo} value={linkedNode} onChange={setLinkedNode} />
            {!linkedNode && (
              <p className="text-[10px] text-zinc-700 mt-1.5 flex items-center gap-1">
                <AlertCircle className="w-3 h-3" />
                Without a node, the rule is stored but won't appear in the graph as an edge.
              </p>
            )}
          </div>

          {/* Type + Severity */}
          <div className="flex gap-3">
            <div className="flex-1">
              <label className="text-[10px] font-bold text-zinc-500 uppercase block mb-1.5">Type</label>
              <select value={ruleType} onChange={(e) => setRuleType(e.target.value)}
                className="w-full text-xs bg-zinc-800 border border-zinc-700 text-zinc-300 rounded-lg px-2 py-2">
                <option value="legal">Legal</option>
                <option value="contractual">Contractual</option>
                <option value="policy">Policy</option>
                <option value="architectural">Architectural</option>
              </select>
            </div>
            <div className="flex-1">
              <label className="text-[10px] font-bold text-zinc-500 uppercase block mb-1.5">Severity</label>
              <select value={severity} onChange={(e) => setSeverity(e.target.value)}
                className="w-full text-xs bg-zinc-800 border border-zinc-700 text-zinc-300 rounded-lg px-2 py-2">
                <option value="critical">Critical</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
              </select>
            </div>
          </div>
        </div>

        {modalError && (
          <div className="mx-5 mb-3 flex items-center gap-2 p-2.5 rounded-lg bg-red-950/30 border border-red-800/40 text-red-400 text-xs">
            <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />{modalError}
          </div>
        )}

        <div className="flex items-center justify-end gap-2 px-5 py-4 border-t border-zinc-800">
          <button onClick={onClose} className="px-4 py-1.5 text-xs text-zinc-500 hover:text-zinc-300 transition-colors">
            Cancel
          </button>
          <button onClick={handleSubmit} disabled={!description.trim() || submitting}
            className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-purple-500/20 text-purple-300 hover:bg-purple-500/30 border border-purple-500/30 text-xs font-medium transition-all disabled:opacity-50">
            {submitting ? <Loader2 className="w-3 h-3 animate-spin" /> : <Plus className="w-3 h-3" />}
            {linkedNode ? 'Add Rule + Link to Graph' : 'Add Rule'}
          </button>
        </div>
      </div>
    </div>
  )
}

function RuleCard({ rule }: { rule: KnowledgeRule }) {
  return (
    <div className="p-3 rounded-xl bg-zinc-900 border border-zinc-800">
      <div className="flex items-start gap-2">
        <Shield className={cn('w-4 h-4 mt-0.5', RULE_TYPE_COLORS[rule.rule_type] || 'text-zinc-400')} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className={cn('text-[10px] px-1.5 py-0.5 rounded-full border font-medium',
              SEVERITY_COLORS[rule.severity] || SEVERITY_COLORS.medium)}>
              {rule.severity}
            </span>
            <span className="text-[10px] text-zinc-600">{rule.rule_type}</span>
            <span className="text-[10px] text-zinc-700 ml-auto font-mono">{rule.id}</span>
          </div>
          <p className="text-sm text-zinc-300">{rule.description}</p>
          <div className="flex items-center gap-2 mt-1.5">
            <FileCode className="w-3 h-3 text-zinc-600" />
            <span className="text-[10px] font-mono text-zinc-600">{rule.file || rule.function_id}</span>
            <span className="text-[10px] text-zinc-700 ml-auto">{rule.source} | {rule.created_at}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function KnowledgePage() {
  const { activeRepo } = useRepo()
  const [tab, setTab] = useState<'questions' | 'rules'>('questions')
  const [questions, setQuestions] = useState<KnowledgeQuestion[]>([])
  const [rules, setRules] = useState<KnowledgeRule[]>([])
  const [stats, setStats] = useState<KnowledgeStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [showAnswered, setShowAnswered] = useState(false)
  const [showAddRule, setShowAddRule] = useState(false)

  const loadData = () => {
    if (!activeRepo) return
    setLoading(true)
    Promise.all([
      getKnowledgeQuestions(activeRepo, !showAnswered),
      getKnowledgeRules(activeRepo),
      getKnowledgeStats(activeRepo),
    ]).then(([q, r, s]) => {
      setQuestions(q)
      setRules(r)
      setStats(s)
    }).catch(() => {}).finally(() => setLoading(false))
  }

  useEffect(() => { loadData() }, [activeRepo, showAnswered])

  if (!activeRepo) {
    return <div className="flex items-center justify-center h-full text-zinc-600 text-sm">Select a repository</div>
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {showAddRule && activeRepo && (
        <AddRuleModal repo={activeRepo} onClose={() => setShowAddRule(false)} onAdded={loadData} />
      )}
      <div className="flex-shrink-0 px-6 py-4 border-b border-zinc-700/50 bg-zinc-900 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center">
            <BookOpen className="w-4 h-4 text-purple-400" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-zinc-100">Knowledge Base</h1>
            <p className="text-[10px] text-zinc-600">Business rules & decision points — the agent gets smarter with every answer</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => setTab('questions')}
            className={cn('px-3 py-1.5 rounded-lg text-xs font-medium transition-all',
              tab === 'questions' ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30' : 'text-zinc-500 hover:text-zinc-300')}>
            <HelpCircle className="w-3 h-3 inline mr-1" />Questions {stats ? `(${stats.unanswered})` : ''}
          </button>
          <button onClick={() => setTab('rules')}
            className={cn('px-3 py-1.5 rounded-lg text-xs font-medium transition-all',
              tab === 'rules' ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30' : 'text-zinc-500 hover:text-zinc-300')}>
            <Shield className="w-3 h-3 inline mr-1" />Rules {stats ? `(${stats.rules})` : ''}
          </button>
          <button onClick={() => setShowAddRule(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-purple-600 hover:bg-purple-500 text-white text-xs font-medium transition-all">
            <Plus className="w-3 h-3" />Add Rule
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-4xl mx-auto">
          {stats && <StatsCard stats={stats} />}

          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-6 h-6 text-purple-400 animate-spin" />
            </div>
          ) : tab === 'questions' ? (
            <div>
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-sm font-bold text-zinc-300">
                  Decision Point Questions
                </h2>
                <label className="flex items-center gap-2 text-xs text-zinc-500 cursor-pointer">
                  <input type="checkbox" checked={showAnswered} onChange={(e) => setShowAnswered(e.target.checked)}
                    className="rounded border-zinc-700 bg-zinc-800" />
                  Show answered
                </label>
              </div>
              {questions.length === 0 ? (
                <div className="flex flex-col items-center py-12 text-center">
                  <BarChart3 className="w-8 h-8 text-zinc-700 mb-3" />
                  <p className="text-sm text-zinc-400">
                    {showAnswered ? 'No questions found' : 'All questions answered!'}
                  </p>
                  <p className="text-xs text-zinc-600 mt-1">
                    {showAnswered ? 'Analyze a repo to generate decision point questions' : 'The agent has all the business context it needs'}
                  </p>
                </div>
              ) : (
                <div className="space-y-2">
                  {questions.map((q) => (
                    <QuestionCard key={q.id} q={q} repo={activeRepo} onAnswered={loadData} />
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div>
              <h2 className="text-sm font-bold text-zinc-300 mb-3">Business Rules</h2>
              {rules.length === 0 ? (
                <div className="flex flex-col items-center py-12 text-center">
                  <Shield className="w-8 h-8 text-zinc-700 mb-3" />
                  <p className="text-sm text-zinc-400">No business rules yet</p>
                  <p className="text-xs text-zinc-600 mt-1">Answer decision point questions to create rules</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {rules.map((r) => <RuleCard key={r.id} rule={r} />)}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
