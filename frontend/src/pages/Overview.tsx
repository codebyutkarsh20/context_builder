import { useState, useEffect, useCallback, useRef } from 'react'
import {
  FolderSearch, Loader2, CheckCircle2, XCircle, Clock,
  Files, Braces, Code2, AlignLeft, RefreshCw, AlertCircle,
  Zap, GitGraph, Database, TrendingUp, ChevronRight,
} from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell,
} from 'recharts'
import { cn, formatNumber } from '../lib/utils'
import { analyzeRepo, getGraphStats, listRepos, getJobStatus, getHotspots } from '../lib/api'
import type { GraphStats, Repo, JobStatus, Hotspot } from '../lib/api'
import { useRepo } from '../lib/RepoContext'
import { useNavigate } from 'react-router-dom'

// ─── Constants ────────────────────────────────────────────────────────────────

const NODE_TYPE_COLORS: Record<string, string> = {
  File: '#3b82f6',
  Class: '#22c55e',
  Function: '#f97316',
  BusinessRule: '#a855f7',
  DomainConcept: '#ec4899',
}

const LANG_COLORS = [
  '#3b82f6', '#22c55e', '#f97316', '#a855f7', '#ec4899',
  '#06b6d4', '#eab308', '#ef4444', '#8b5cf6', '#14b8a6',
]

// ─── Sub-components ───────────────────────────────────────────────────────────

interface StatCardProps {
  label: string
  value: string | number
  icon: React.ReactNode
  accent: string
  sub?: string
  loading?: boolean
}

function StatCard({ label, value, icon, accent, sub, loading }: StatCardProps) {
  return (
    <div className="relative p-5 rounded-2xl bg-zinc-900 border border-zinc-800 overflow-hidden group hover:border-zinc-700 transition-all">
      <div className={cn('absolute inset-0 opacity-5 group-hover:opacity-10 transition-opacity', accent)} />
      <div className="relative flex items-start justify-between">
        <div>
          <p className="text-xs font-medium text-zinc-500 mb-1">{label}</p>
          {loading ? (
            <div className="h-8 w-24 bg-zinc-800 rounded-lg animate-pulse" />
          ) : (
            <p className="text-3xl font-bold text-zinc-50 font-mono tracking-tight">
              {typeof value === 'number' ? formatNumber(value) : value}
            </p>
          )}
          {sub && !loading && <p className="text-xs text-zinc-500 mt-1">{sub}</p>}
        </div>
        <div className={cn('w-10 h-10 rounded-xl flex items-center justify-center', accent)}>
          {icon}
        </div>
      </div>
    </div>
  )
}

function jobStatusBadge(status: JobStatus['status']) {
  switch (status) {
    case 'completed':
    case 'done':
      return <span className="flex items-center gap-1.5 text-emerald-400 text-xs font-medium"><CheckCircle2 className="w-3.5 h-3.5" />Completed</span>
    case 'failed':
      return <span className="flex items-center gap-1.5 text-red-400 text-xs font-medium"><XCircle className="w-3.5 h-3.5" />Failed</span>
    case 'running':
      return <span className="flex items-center gap-1.5 text-blue-400 text-xs font-medium"><Loader2 className="w-3.5 h-3.5 animate-spin" />Running</span>
    default:
      return <span className="flex items-center gap-1.5 text-zinc-400 text-xs font-medium"><Clock className="w-3.5 h-3.5" />Pending</span>
  }
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function Overview() {
  const { activeRepo, refresh: refreshRepos } = useRepo()
  const navigate = useNavigate()
  const [repoPath, setRepoPath] = useState('')
  const [repoName, setRepoName] = useState('')
  const [analyzing, setAnalyzing] = useState(false)
  const [analyzeError, setAnalyzeError] = useState<string | null>(null)
  const [currentJob, setCurrentJob] = useState<JobStatus | null>(null)

  const [stats, setStats] = useState<GraphStats | null>(null)
  const [statsLoading, setStatsLoading] = useState(false)
  const [statsError, setStatsError] = useState<string | null>(null)

  const [repos, setRepos] = useState<Repo[]>([])
  const [hotspots, setHotspots] = useState<Hotspot[]>([])
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const pollCountRef = useRef(0)

  // Cleanup polling interval on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
    }
  }, [])

  const pollJob = useCallback((jobId: string) => {
    // Clear any existing interval
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
    pollCountRef.current = 0
    pollIntervalRef.current = setInterval(async () => {
      pollCountRef.current += 1
      // Timeout after ~10 minutes (400 polls × 1.5s)
      if (pollCountRef.current > 400) {
        if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
        pollIntervalRef.current = null
        setAnalyzing(false)
        setCurrentJob(prev => prev ? { ...prev, status: 'failed', error: 'Timed out after 10 minutes' } : null)
        return
      }
      try {
        const status = await getJobStatus(jobId)
        setCurrentJob(status)
        if (status.status === 'done' || status.status === 'completed' || status.status === 'failed') {
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
          pollIntervalRef.current = null
          setAnalyzing(false)
          refreshRepos()
        }
      } catch {
        if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
        pollIntervalRef.current = null
        setAnalyzing(false)
      }
    }, 1500)
  }, [refreshRepos])

  const handleAnalyze = async () => {
    if (!repoPath.trim()) return
    setAnalyzing(true)
    setAnalyzeError(null)
    setCurrentJob(null)
    try {
      const { job_id } = await analyzeRepo(repoPath.trim(), repoName.trim() || undefined)
      pollJob(job_id)
    } catch (e) {
      setAnalyzeError(e instanceof Error ? e.message : 'Analysis failed')
      setAnalyzing(false)
    }
  }

  useEffect(() => {
    if (!activeRepo) return
    setStatsLoading(true)
    setStatsError(null)
    getGraphStats(activeRepo)
      .then(setStats)
      .catch((e: Error) => setStatsError(e.message))
      .finally(() => setStatsLoading(false))
    getHotspots(activeRepo, 5).then(setHotspots).catch((e: Error) => console.warn('Hotspots unavailable:', e.message))
  }, [activeRepo])

  useEffect(() => {
    listRepos().then(setRepos).catch((e: Error) => console.warn('Failed to list repos:', e.message))
  }, [activeRepo])

  const nodeTypeData = stats?.node_type_counts
    ? Object.entries(stats.node_type_counts)
        .filter(([, v]) => v > 0)
        .map(([name, value]) => ({ name, value }))
    : []

  const langData = stats?.languages
    ? Object.entries(stats.languages).sort((a, b) => b[1] - a[1]).slice(0, 6)
        .map(([name, count]) => ({ name, count }))
    : []

  return (
    <div className="h-full overflow-y-auto">
      <div className="p-6 max-w-7xl mx-auto space-y-6">

        {/* ── Header ── */}
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-2xl font-bold text-zinc-50">Dashboard</h1>
            <p className="text-sm text-zinc-500 mt-0.5">Analyze repos and explore their code intelligence graph</p>
          </div>
          {activeRepo && (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-800 border border-zinc-700">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-xs font-medium text-zinc-300 font-mono">{activeRepo}</span>
            </div>
          )}
        </div>

        {/* ── Analyze form ── */}
        <div className="p-5 rounded-2xl bg-zinc-900 border border-zinc-800">
          <div className="flex items-center gap-2 mb-4">
            <div className="w-7 h-7 rounded-lg bg-blue-500/20 flex items-center justify-center">
              <FolderSearch className="w-4 h-4 text-blue-400" />
            </div>
            <h2 className="text-sm font-semibold text-zinc-200">Analyze Repository</h2>
          </div>

          <div className="flex flex-col sm:flex-row gap-2.5">
            <input
              type="text"
              value={repoPath}
              onChange={(e) => setRepoPath(e.target.value)}
              placeholder="/path/to/your/repository"
              className="flex-1 px-4 py-2.5 rounded-xl bg-zinc-800 border border-zinc-700 text-zinc-200 placeholder-zinc-600 text-sm focus:outline-none focus:border-blue-500/70 focus:bg-zinc-800 transition-all font-mono"
              onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
            />
            <input
              type="text"
              value={repoName}
              onChange={(e) => setRepoName(e.target.value)}
              placeholder="Name (optional)"
              className="w-44 px-4 py-2.5 rounded-xl bg-zinc-800 border border-zinc-700 text-zinc-200 placeholder-zinc-600 text-sm focus:outline-none focus:border-blue-500/70 transition-all"
            />
            <button
              onClick={handleAnalyze}
              disabled={analyzing || !repoPath.trim()}
              className={cn(
                'flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold transition-all whitespace-nowrap',
                analyzing || !repoPath.trim()
                  ? 'bg-zinc-800 text-zinc-600 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-500/20'
              )}
            >
              {analyzing ? <><Loader2 className="w-4 h-4 animate-spin" />Analyzing...</> : <><Zap className="w-4 h-4" />Analyze</>}
            </button>
          </div>

          {currentJob && (
            <div className="mt-4 p-3 rounded-xl bg-zinc-800/60 border border-zinc-700/60">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-zinc-400">{currentJob.stage || currentJob.message}</span>
                {jobStatusBadge(currentJob.status)}
              </div>
              <div className="h-1.5 bg-zinc-700 rounded-full overflow-hidden">
                <div
                  className={cn('h-full rounded-full transition-all duration-500',
                    currentJob.status === 'failed' ? 'bg-red-500' :
                    (currentJob.status === 'done' || currentJob.status === 'completed') ? 'bg-emerald-500' : 'bg-blue-500')}
                  style={{ width: `${currentJob.progress}%` }}
                />
              </div>
              <p className="text-xs text-zinc-600 mt-1">{currentJob.progress}% complete</p>
            </div>
          )}

          {analyzeError && (
            <div className="mt-3 flex items-center gap-2 text-red-400 text-sm">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />{analyzeError}
            </div>
          )}
        </div>

        {/* ── Stats section ── */}
        {activeRepo && (
          <>
            {statsError && (
              <div className="flex items-center gap-3 p-4 rounded-2xl bg-red-950/30 border border-red-900/40 text-red-400">
                <AlertCircle className="w-5 h-5 flex-shrink-0" />
                <p className="text-sm">{statsError}</p>
              </div>
            )}

            {/* Stat cards */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <StatCard label="Files" value={stats?.files ?? 0}
                icon={<Files className="w-5 h-5 text-blue-300" />}
                accent="bg-blue-500" sub="Source files" loading={statsLoading} />
              <StatCard label="Functions" value={stats?.functions ?? 0}
                icon={<Braces className="w-5 h-5 text-orange-300" />}
                accent="bg-orange-500" sub="Defined functions" loading={statsLoading} />
              <StatCard label="Classes" value={stats?.classes ?? 0}
                icon={<Code2 className="w-5 h-5 text-emerald-300" />}
                accent="bg-emerald-500" sub="Class definitions" loading={statsLoading} />
              <StatCard label="Lines of Code" value={stats?.lines_of_code ?? 0}
                icon={<AlignLeft className="w-5 h-5 text-purple-300" />}
                accent="bg-purple-500" sub="Total across repo" loading={statsLoading} />
            </div>

            {/* Tech stack */}
            {stats?.tech_stack && stats.tech_stack.length > 0 && (
              <div className="p-4 rounded-2xl bg-zinc-900 border border-zinc-800">
                <p className="text-xs font-semibold text-zinc-500 uppercase tracking-widest mb-3">Tech Stack</p>
                <div className="flex flex-wrap gap-2">
                  {stats.tech_stack.map((tech) => (
                    <span key={tech}
                      className="px-3 py-1 rounded-full bg-zinc-800 text-zinc-300 text-xs font-medium border border-zinc-700 hover:border-zinc-600 transition-colors">
                      {tech}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Charts + hotspots */}
            {!statsLoading && stats && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                {/* Node types pie */}
                <div className="p-5 rounded-2xl bg-zinc-900 border border-zinc-800">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold text-zinc-200">Node Distribution</h3>
                    <Database className="w-4 h-4 text-zinc-600" />
                  </div>
                  {nodeTypeData.length > 0 ? (
                    <>
                      <ResponsiveContainer width="100%" height={160}>
                        <PieChart>
                          <Pie data={nodeTypeData} cx="50%" cy="50%"
                            outerRadius={65} innerRadius={38} dataKey="value" paddingAngle={3}>
                            {nodeTypeData.map((entry, i) => (
                              <Cell key={entry.name}
                                fill={NODE_TYPE_COLORS[entry.name] ?? LANG_COLORS[i % LANG_COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 10, color: '#e4e4e7', fontSize: 12 }} />
                        </PieChart>
                      </ResponsiveContainer>
                      <div className="grid grid-cols-3 gap-1.5 mt-2">
                        {nodeTypeData.map((entry) => (
                          <div key={entry.name} className="flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full flex-shrink-0"
                              style={{ backgroundColor: NODE_TYPE_COLORS[entry.name] ?? '#6b7280' }} />
                            <span className="text-xs text-zinc-500 truncate">{entry.name}</span>
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    <div className="flex items-center justify-center h-40 text-zinc-600 text-sm">No data</div>
                  )}
                </div>

                {/* Language bar */}
                <div className="p-5 rounded-2xl bg-zinc-900 border border-zinc-800">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold text-zinc-200">Files by Language</h3>
                    <Code2 className="w-4 h-4 text-zinc-600" />
                  </div>
                  {langData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={190}>
                      <BarChart data={langData} margin={{ top: 0, right: 0, left: -22, bottom: 0 }}>
                        <XAxis dataKey="name" tick={{ fill: '#71717a', fontSize: 10 }} axisLine={false} tickLine={false} />
                        <YAxis tick={{ fill: '#52525b', fontSize: 10 }} axisLine={false} tickLine={false} />
                        <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 10, color: '#e4e4e7', fontSize: 12 }} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
                        <Bar dataKey="count" radius={[5, 5, 0, 0]}>
                          {langData.map((_, i) => <Cell key={i} fill={LANG_COLORS[i % LANG_COLORS.length]} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="flex items-center justify-center h-48 text-zinc-600 text-sm">No data</div>
                  )}
                </div>

                {/* Top hotspots */}
                <div className="p-5 rounded-2xl bg-zinc-900 border border-zinc-800">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold text-zinc-200">Top Hotspots</h3>
                    <button onClick={() => navigate('/graph')}
                      className="text-xs text-zinc-500 hover:text-zinc-300 flex items-center gap-1 transition-colors">
                      View all <ChevronRight className="w-3 h-3" />
                    </button>
                  </div>
                  {hotspots.length > 0 ? (
                    <div className="space-y-2">
                      {hotspots.slice(0, 5).map((h, i) => (
                        <div key={h.id} className="flex items-center gap-3 p-2.5 rounded-xl bg-zinc-800/60 hover:bg-zinc-800 transition-colors">
                          <span className="text-xs font-bold text-zinc-600 font-mono w-4 text-center">{i + 1}</span>
                          <div className="flex-1 min-w-0">
                            <p className="text-xs font-medium text-zinc-200 truncate font-mono">{h.name}</p>
                            <div className="flex items-center gap-1.5 mt-0.5">
                              <span className="text-[10px] font-medium px-1.5 py-0.5 rounded"
                                style={{ backgroundColor: (NODE_TYPE_COLORS[h.type] ?? '#6b7280') + '25', color: NODE_TYPE_COLORS[h.type] ?? '#9ca3af' }}>
                                {h.type}
                              </span>
                            </div>
                          </div>
                          <span className="text-xs font-mono text-zinc-500 shrink-0">{(h.pagerank ?? 0).toFixed(3)}</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center h-40 gap-2">
                      <TrendingUp className="w-8 h-8 text-zinc-700" />
                      <p className="text-sm text-zinc-600">No hotspot data</p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </>
        )}

        {/* ── Repositories table ── */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-zinc-300 flex items-center gap-2">
              <GitGraph className="w-4 h-4 text-zinc-500" />
              Repositories
            </h2>
            <button onClick={() => { listRepos().then(setRepos).catch(console.error); refreshRepos() }}
              className="flex items-center gap-1.5 text-xs text-zinc-500 hover:text-zinc-300 transition-colors">
              <RefreshCw className="w-3.5 h-3.5" />Refresh
            </button>
          </div>

          {repos.length === 0 ? (
            <div className="p-10 rounded-2xl bg-zinc-900 border border-zinc-800 flex flex-col items-center gap-3">
              <div className="w-14 h-14 rounded-2xl bg-zinc-800 flex items-center justify-center">
                <FolderSearch className="w-7 h-7 text-zinc-600" />
              </div>
              <p className="text-sm font-medium text-zinc-400">No repositories analyzed yet</p>
              <p className="text-xs text-zinc-600">Enter a path above to build your first knowledge graph</p>
            </div>
          ) : (
            <div className="rounded-2xl border border-zinc-800 overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-zinc-900 border-b border-zinc-800">
                    <th className="text-left px-4 py-3 text-xs font-semibold text-zinc-500 uppercase tracking-wider">Repository</th>
                    <th className="text-left px-4 py-3 text-xs font-semibold text-zinc-500 uppercase tracking-wider hidden md:table-cell">Path</th>
                    <th className="text-left px-4 py-3 text-xs font-semibold text-zinc-500 uppercase tracking-wider">Status</th>
                    <th className="text-left px-4 py-3 text-xs font-semibold text-zinc-500 uppercase tracking-wider hidden sm:table-cell">Analyzed</th>
                  </tr>
                </thead>
                <tbody className="bg-zinc-950">
                  {repos.map((repo, i) => (
                    <tr key={repo.name}
                      className={cn('border-b border-zinc-900 hover:bg-zinc-900/60 transition-colors cursor-pointer',
                        i === repos.length - 1 && 'border-b-0')}>
                      <td className="px-4 py-3">
                        <span className="font-semibold text-zinc-200 font-mono text-sm">{repo.name}</span>
                      </td>
                      <td className="px-4 py-3 hidden md:table-cell">
                        <span className="text-zinc-500 font-mono text-xs">{repo.path ?? '/tmp/context_builder/' + repo.name}</span>
                      </td>
                      <td className="px-4 py-3">
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-950/50 border border-emerald-800/40 text-emerald-400 text-xs font-medium">
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                          {repo.has_context ? 'Ready' : (repo.status ?? 'Unknown')}
                        </span>
                      </td>
                      <td className="px-4 py-3 hidden sm:table-cell">
                        <span className="text-zinc-600 text-xs">
                          {repo.analyzed_at ? new Date(repo.analyzed_at).toLocaleString() : '—'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

      </div>
    </div>
  )
}
