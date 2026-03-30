import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { X, GitGraph, AlertCircle, ChevronRight, FileCode, Hash, Network, ArrowDownToLine, ArrowUpFromLine, Search } from 'lucide-react'
import { cn, formatNumber, truncate } from '../lib/utils'
import KnowledgeGraph, { NODE_COLORS, EDGE_COLORS, type KnowledgeGraphControls } from '../components/KnowledgeGraph'
import { getGraph } from '../lib/api'
import type { GraphNode, GraphEdge, GraphData } from '../lib/api'
import { useRepo } from '../lib/RepoContext'

type LayerFilter = 'all' | 'code' | 'business'

const LAYER_OPTIONS: { id: LayerFilter; label: string }[] = [
  { id: 'all', label: 'All Layers' },
  { id: 'code', label: 'Code Layer' },
  { id: 'business', label: 'Business Layer' },
]

// ─── Node Detail Panel ────────────────────────────────────────────────────────

interface NodeDetailProps {
  node: GraphNode
  allNodes: GraphNode[]
  allEdges: GraphEdge[]
  onClose: () => void
  onJump: (node: GraphNode) => void
}

const EDGE_SECTION_CONFIG: Record<string, { inLabel: string; outLabel: string }> = {
  CALLS:        { inLabel: 'Called By',      outLabel: 'Calls' },
  CONTAINS:     { inLabel: 'Contained In',   outLabel: 'Contains' },
  IMPORTS:      { inLabel: 'Imported By',    outLabel: 'Imports' },
  INHERITS:     { inLabel: 'Inherited By',   outLabel: 'Inherits From' },
  RELATED_TO:   { inLabel: 'Related To',     outLabel: 'Related To' },
  HAS_DECISION: { inLabel: 'Decision Of',    outLabel: 'Has Decision' },
  GOVERNED_BY:  { inLabel: 'Governs',        outLabel: 'Governed By' },
  REPRESENTS:   { inLabel: 'Represented By', outLabel: 'Represents' },
}

function NodeConnList({
  label, edgeColor, nodes: list, overflow, onJump,
}: {
  label: string
  edgeColor: string
  nodes: GraphNode[]
  overflow: number
  onJump: (n: GraphNode) => void
}) {
  if (list.length === 0 && overflow === 0) return null
  return (
    <div>
      <div className="flex items-center gap-1.5 mb-1.5">
        <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: edgeColor }} />
        <p className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">
          {label} ({list.length + overflow})
        </p>
      </div>
      <ul className="space-y-1">
        {list.map((n) => (
          <li key={n.id}>
            <button onClick={() => onJump(n)}
              className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-zinc-800/60 hover:border-zinc-700/60 transition-all group text-left">
              <span className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ backgroundColor: NODE_COLORS[n.type] ?? '#6b7280' }} />
              <span className="text-xs text-zinc-400 group-hover:text-zinc-200 font-mono truncate transition-colors flex-1">
                {truncate(String(n.name ?? n.id).split('::').pop() ?? '', 24)}
              </span>
              <span className="text-[10px] text-zinc-700 group-hover:text-zinc-500 flex-shrink-0 font-mono">
                {n.type}
              </span>
              <ChevronRight className="w-3 h-3 text-zinc-700 group-hover:text-zinc-400 flex-shrink-0 transition-colors" />
            </button>
          </li>
        ))}
        {overflow > 0 && (
          <p className="text-[10px] text-zinc-700 text-center pt-0.5">+{overflow} more</p>
        )}
      </ul>
    </div>
  )
}

function NodeDetail({ node, allNodes, allEdges, onClose, onJump }: NodeDetailProps) {
  const nodeMap = new Map(allNodes.map((n) => [n.id, n]))
  const outEdges = allEdges.filter((e) => String(e.source) === node.id)
  const inEdges  = allEdges.filter((e) => String(e.target) === node.id)
  const totalEdges = inEdges.length + outEdges.length

  // Group by edge type
  const allEdgeTypes = [...new Set([...inEdges, ...outEdges].map((e) => e.type))]
    .sort((a, b) => {
      const order = ['CALLS', 'CONTAINS', 'IMPORTS', 'INHERITS', 'RELATED_TO']
      return (order.indexOf(a) ?? 99) - (order.indexOf(b) ?? 99)
    })

  const nodeColor = NODE_COLORS[node.type] ?? '#6b7280'
  const filePath = node.file ?? (String(node.id).includes('/') ? String(node.id).split('::')[0] : null)

  return (
    <div className="w-full h-full bg-zinc-950/95 backdrop-blur-sm border border-zinc-800/80 rounded-2xl flex flex-col overflow-hidden shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800/60 rounded-t-2xl"
        style={{ borderTop: `2px solid ${nodeColor}` }}>
        <div className="flex items-center gap-2 min-w-0">
          <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: nodeColor }} />
          <span className="text-xs font-bold text-zinc-300 uppercase tracking-wider">{node.type}</span>
        </div>
        <button onClick={onClose}
          className="w-6 h-6 rounded-lg hover:bg-zinc-800 flex items-center justify-center transition-colors flex-shrink-0">
          <X className="w-3.5 h-3.5 text-zinc-500" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Name */}
        <div className="px-4 py-3 border-b border-zinc-800/40">
          <p className="text-sm font-bold text-zinc-100 font-mono break-all leading-snug">
            {String(node.name ?? node.id).split('::').pop() ?? node.id}
          </p>
          {String(node.name ?? '').includes('::') && (
            <p className="text-[11px] text-zinc-600 font-mono mt-0.5 break-all">
              {String(node.name ?? node.id)}
            </p>
          )}
        </div>

        <div className="px-4 py-3 space-y-4">
          {/* File + location */}
          {filePath && (
            <div>
              <p className="text-[10px] font-bold text-zinc-600 uppercase tracking-widest mb-1.5">File</p>
              <div className="flex items-start gap-2 px-2.5 py-2 rounded-lg bg-zinc-900 border border-zinc-800/60">
                <FileCode className="w-3.5 h-3.5 text-zinc-500 mt-0.5 flex-shrink-0" />
                <span className="text-xs text-zinc-300 font-mono break-all leading-relaxed">{filePath}</span>
              </div>
              {node.line_start !== undefined && (
                <div className="flex items-center gap-1.5 mt-1.5 px-2.5 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800/60">
                  <Hash className="w-3 h-3 text-zinc-600 flex-shrink-0" />
                  <span className="text-xs text-zinc-500 font-mono">
                    Lines {node.line_start}–{node.line_end ?? node.line_start}
                  </span>
                </div>
              )}
            </div>
          )}

          {/* PageRank */}
          {node.pagerank != null && (
            <div>
              <p className="text-[10px] font-bold text-zinc-600 uppercase tracking-widest mb-1.5">PageRank Score</p>
              <div className="flex items-center gap-3">
                <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                  <div className="h-full rounded-full transition-all"
                    style={{ width: `${Math.min(100, node.pagerank * 200)}%`, backgroundColor: nodeColor }} />
                </div>
                <span className="text-xs font-mono text-zinc-300 w-14 text-right shrink-0">
                  {node.pagerank.toFixed(5)}
                </span>
              </div>
            </div>
          )}

          {/* Summary */}
          {node.summary && (
            <div>
              <p className="text-[10px] font-bold text-zinc-600 uppercase tracking-widest mb-1.5">Summary</p>
              <p className="text-xs text-zinc-400 leading-relaxed bg-zinc-900 px-2.5 py-2 rounded-lg border border-zinc-800/60">
                {node.summary}
              </p>
            </div>
          )}

          {/* Docstring */}
          {node.docstring ? (
            <div>
              <p className="text-[10px] font-bold text-zinc-600 uppercase tracking-widest mb-1.5">Docstring</p>
              <p className="text-xs text-zinc-400 leading-relaxed bg-zinc-900 px-2.5 py-2 rounded-lg border border-zinc-800/60 font-mono whitespace-pre-wrap line-clamp-6">
                {String(node.docstring)}
              </p>
            </div>
          ) : null}

          {/* Parameters (Function nodes) */}
          {node.type === 'Function' && node.params ? (
            <div>
              <p className="text-[10px] font-bold text-zinc-600 uppercase tracking-widest mb-1.5">Parameters</p>
              <p className="text-xs text-zinc-400 font-mono bg-zinc-900 px-2.5 py-2 rounded-lg border border-zinc-800/60 break-all">
                {String(node.params)}
              </p>
            </div>
          ) : null}

          {/* Methods count (Class nodes) */}
          {node.type === 'Class' && node.methods !== undefined ? (
            <div className="flex items-center gap-2 px-2.5 py-2 rounded-lg bg-zinc-900 border border-zinc-800/60">
              <span className="text-[10px] font-bold text-zinc-600 uppercase tracking-widest">Methods</span>
              <span className="text-xs font-mono text-zinc-300">{String(node.methods)}</span>
            </div>
          ) : null}

          {/* External calls */}
          {Array.isArray(node.external_calls) && (node.external_calls as string[]).length > 0 ? (
            <div>
              <p className="text-[10px] font-bold text-zinc-600 uppercase tracking-widest mb-1.5">External Calls</p>
              <div className="flex flex-wrap gap-1">
                {(node.external_calls as string[]).slice(0, 10).map((call) => (
                  <span key={call} className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800/60 text-zinc-400 font-mono">
                    {call}
                  </span>
                ))}
              </div>
            </div>
          ) : null}

          {/* Data access */}
          {(Array.isArray(node.reads_from) || Array.isArray(node.writes_to)) &&
           ((node.reads_from as string[] || []).length > 0 || (node.writes_to as string[] || []).length > 0) ? (
            <div>
              <p className="text-[10px] font-bold text-zinc-600 uppercase tracking-widest mb-1.5">Data Access</p>
              <div className="flex flex-wrap gap-1">
                {(node.reads_from as string[] || []).map((r) => (
                  <span key={`r-${r}`} className="text-[10px] px-1.5 py-0.5 rounded bg-blue-950/40 border border-blue-800/30 text-blue-400 font-mono">
                    reads {r}
                  </span>
                ))}
                {(node.writes_to as string[] || []).map((w) => (
                  <span key={`w-${w}`} className="text-[10px] px-1.5 py-0.5 rounded bg-orange-950/40 border border-orange-800/30 text-orange-400 font-mono">
                    writes {w}
                  </span>
                ))}
              </div>
            </div>
          ) : null}

          {/* Connection stats */}
          <div>
            <p className="text-[10px] font-bold text-zinc-600 uppercase tracking-widest mb-1.5">Connections</p>
            <div className="grid grid-cols-3 gap-1.5">
              <div className="px-2 py-2 rounded-lg bg-zinc-900 border border-zinc-800/60 text-center">
                <p className="text-base font-bold font-mono text-zinc-200">{inEdges.length}</p>
                <div className="flex items-center justify-center gap-1 mt-0.5">
                  <ArrowDownToLine className="w-2.5 h-2.5 text-zinc-600" />
                  <p className="text-[10px] text-zinc-600">Incoming</p>
                </div>
              </div>
              <div className="px-2 py-2 rounded-lg bg-zinc-900 border border-zinc-800/60 text-center">
                <p className="text-base font-bold font-mono text-zinc-200">{outEdges.length}</p>
                <div className="flex items-center justify-center gap-1 mt-0.5">
                  <ArrowUpFromLine className="w-2.5 h-2.5 text-zinc-600" />
                  <p className="text-[10px] text-zinc-600">Outgoing</p>
                </div>
              </div>
              <div className="px-2 py-2 rounded-lg bg-zinc-900 border border-zinc-800/60 text-center">
                <p className="text-base font-bold font-mono text-zinc-200">{totalEdges}</p>
                <p className="text-[10px] text-zinc-600 mt-0.5">Total</p>
              </div>
            </div>
          </div>

          {/* Connections by type */}
          {allEdgeTypes.length === 0 ? (
            <p className="text-xs text-zinc-700 text-center py-2">
              No connections in loaded graph
            </p>
          ) : (
            <div className="space-y-4">
              {allEdgeTypes.map((edgeType) => {
                const cfg = EDGE_SECTION_CONFIG[edgeType] ?? { inLabel: `← ${edgeType}`, outLabel: `→ ${edgeType}` }
                const edgeColor = EDGE_COLORS[edgeType] ?? '#6b7280'
                const edgesIn  = inEdges.filter((e) => e.type === edgeType)
                const edgesOut = outEdges.filter((e) => e.type === edgeType)
                const nodesIn  = edgesIn.map((e) => nodeMap.get(String(e.source))).filter(Boolean).slice(0, 8) as GraphNode[]
                const nodesOut = edgesOut.map((e) => nodeMap.get(String(e.target))).filter(Boolean).slice(0, 8) as GraphNode[]
                return (
                  <div key={edgeType} className="rounded-xl border border-zinc-800/60 overflow-hidden">
                    {/* Edge type header */}
                    <div className="flex items-center gap-2 px-3 py-2 bg-zinc-900/60">
                      <span className="w-5 h-0.5 rounded-full flex-shrink-0" style={{ backgroundColor: edgeColor }} />
                      <span className="text-[11px] font-bold text-zinc-400 uppercase tracking-wider">{edgeType}</span>
                      <span className="ml-auto text-[10px] font-mono text-zinc-600">
                        {edgesIn.length + edgesOut.length} edges
                      </span>
                    </div>
                    <div className="px-3 py-2.5 space-y-3">
                      <NodeConnList
                        label={cfg.inLabel}
                        edgeColor={edgeColor}
                        nodes={nodesIn}
                        overflow={Math.max(0, edgesIn.length - 8)}
                        onJump={onJump}
                      />
                      <NodeConnList
                        label={cfg.outLabel}
                        edgeColor={edgeColor}
                        nodes={nodesOut}
                        overflow={Math.max(0, edgesOut.length - 8)}
                        onJump={onJump}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function Graph() {
  const { activeRepo } = useRepo()
  const [layerFilter, setLayerFilter] = useState<LayerFilter>('all')
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchOpen, setSearchOpen] = useState(false)
  const searchInputRef = useRef<HTMLInputElement>(null)
  const jumpToNodeRef = useRef<KnowledgeGraphControls | null>(null)
  // Tracks the latest request so stale responses are ignored
  const reqId = useRef(0)

  useEffect(() => {
    if (!activeRepo) return
    const id = ++reqId.current
    setLoading(true)
    setError(null)
    setSelectedNode(null)
    const layer = layerFilter !== 'all' ? layerFilter : undefined
    getGraph(activeRepo, layer, 5000)
      .then((data) => { if (id === reqId.current) setGraphData(data) })
      .catch((e: Error) => { if (id === reqId.current) setError(e.message) })
      .finally(() => { if (id === reqId.current) setLoading(false) })
  }, [activeRepo, layerFilter])

  const handleJump = useCallback((node: GraphNode) => {
    setSelectedNode(node)
    jumpToNodeRef.current?.jumpToNode(node)
  }, [])

  // Search results — filter nodes by name/id
  const searchResults = useMemo(() => {
    const q = searchQuery.trim().toLowerCase()
    if (!q || !graphData) return []
    return graphData.nodes
      .filter((n) => String(n.name ?? n.id).toLowerCase().includes(q))
      .slice(0, 10)
  }, [searchQuery, graphData])

  const handleSearchSelect = useCallback((node: GraphNode) => {
    setSearchQuery('')
    setSearchOpen(false)
    setSelectedNode(node)
    jumpToNodeRef.current?.jumpToNode(node)
  }, [])

  // Open search with Ctrl/Cmd+K
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setSearchOpen((v) => !v)
        setTimeout(() => searchInputRef.current?.focus(), 50)
      }
      if (e.key === 'Escape') setSearchOpen(false)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  if (!activeRepo) {
    return (
      <div className="flex-1 flex items-center justify-center h-full">
        <div className="text-center">
          <GitGraph className="w-12 h-12 text-zinc-800 mx-auto mb-4" />
          <p className="text-zinc-400 font-medium">No repository selected</p>
          <p className="text-zinc-600 text-sm mt-1">Select or analyze a repository to view its graph</p>
        </div>
      </div>
    )
  }

  // Edge type breakdown for header
  const edgeTypeCounts = graphData?.edges.reduce((acc, e) => {
    acc[e.type] = (acc[e.type] ?? 0) + 1
    return acc
  }, {} as Record<string, number>) ?? {}

  return (
    <div className="flex flex-col h-screen bg-zinc-950">
      {/* Top bar */}
      <div className="flex-shrink-0 flex items-center justify-between px-5 py-2.5 border-b border-zinc-800/60 bg-zinc-950 gap-3">
        <div className="flex items-center gap-3 min-w-0">
          <div className="flex items-center gap-2 flex-shrink-0">
            <Network className="w-4 h-4 text-purple-400" />
            <h1 className="text-sm font-semibold text-zinc-200">
              Knowledge Graph
              <span className="hidden sm:inline text-zinc-600 font-normal font-mono ml-1.5">— {activeRepo}</span>
            </h1>
          </div>

          {/* Layer tabs */}
          <div className="flex items-center gap-0.5 bg-zinc-900 border border-zinc-800 rounded-lg p-0.5 flex-shrink-0">
            {LAYER_OPTIONS.map((opt) => (
              <button key={opt.id} onClick={() => setLayerFilter(opt.id)}
                className={cn('px-3 py-1.5 rounded-md text-xs font-medium transition-all',
                  layerFilter === opt.id ? 'bg-zinc-700 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300')}>
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2 flex-shrink-0">
          {/* Node type breakdown dots */}
          {graphData && (
            <div className="hidden md:flex items-center gap-1.5 mr-1">
              {Object.entries(
                graphData.nodes.reduce((acc, n) => { acc[n.type] = (acc[n.type] ?? 0) + 1; return acc }, {} as Record<string, number>)
              ).slice(0, 5).map(([type, count]) => (
                <span key={type} className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800/60">
                  <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: NODE_COLORS[type] ?? '#94a3b8' }} />
                  <span className="text-[10px] font-mono text-zinc-500">{count}</span>
                </span>
              ))}
            </div>
          )}

          {/* Search button */}
          {graphData && (
            <button
              onClick={() => { setSearchOpen((v) => !v); setTimeout(() => searchInputRef.current?.focus(), 50) }}
              className={cn(
                'flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs border transition-all',
                searchOpen
                  ? 'bg-zinc-800 border-zinc-600 text-zinc-200'
                  : 'bg-zinc-900 border-zinc-800 text-zinc-500 hover:text-zinc-300 hover:border-zinc-700'
              )}
              title="Search nodes (⌘K)"
            >
              <Search className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Search nodes</span>
              <kbd className="hidden sm:inline text-[10px] text-zinc-600 bg-zinc-800 px-1 rounded">⌘K</kbd>
            </button>
          )}

          {/* Stats chips */}
          {graphData && (
            <div className="hidden lg:flex items-center gap-2">
              {[
                { label: `${formatNumber(graphData.nodes.length)} nodes`, color: '#a1a1aa' },
                ...Object.entries(edgeTypeCounts).slice(0, 2).map(([type, count]) => ({
                  label: `${formatNumber(count)} ${type.toLowerCase()}`,
                  color: EDGE_COLORS[type] ?? '#52525b',
                })),
              ].map(({ label, color }) => (
                <span key={label} className="px-2 py-0.5 rounded-md text-[11px] font-mono border border-zinc-800 text-zinc-500"
                  style={{ borderColor: color + '40' }}>
                  {label}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Search dropdown */}
      {searchOpen && graphData && (
        <div className="flex-shrink-0 px-4 pt-2 pb-1 border-b border-zinc-800/60 bg-zinc-950 relative z-20">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-zinc-500 pointer-events-none" />
            <input
              ref={searchInputRef}
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search nodes by name…"
              className="w-full bg-zinc-900 border border-zinc-700/60 rounded-lg pl-9 pr-3 py-2 text-sm text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-zinc-500 transition-colors"
            />
          </div>
          {searchResults.length > 0 && (
            <div className="absolute left-4 right-4 top-full mt-1 bg-zinc-900 border border-zinc-700/60 rounded-xl shadow-2xl overflow-hidden z-30">
              {searchResults.map((node) => (
                <button
                  key={node.id}
                  onClick={() => handleSearchSelect(node)}
                  className="w-full flex items-center gap-3 px-3 py-2.5 hover:bg-zinc-800 transition-colors text-left border-b border-zinc-800/40 last:border-0"
                >
                  <span className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{ backgroundColor: NODE_COLORS[node.type] ?? '#6b7280' }} />
                  <span className="text-xs font-mono text-zinc-300 flex-1 truncate">
                    {String(node.name ?? node.id).split('::').pop()}
                  </span>
                  <span className="text-[10px] text-zinc-600 flex-shrink-0">{node.type}</span>
                  {node.file && (
                    <span className="text-[10px] text-zinc-700 font-mono flex-shrink-0 hidden sm:inline truncate max-w-32">
                      {node.file.split('/').slice(-2).join('/')}
                    </span>
                  )}
                </button>
              ))}
            </div>
          )}
          {searchQuery.trim() && searchResults.length === 0 && (
            <p className="text-xs text-zinc-600 py-1.5 px-1">No nodes match "{searchQuery}"</p>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-center gap-3 mx-4 mt-3 p-3 rounded-xl bg-red-950/30 border border-red-900/40 text-red-400 text-sm flex-shrink-0">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />{error}
        </div>
      )}

      {/* Graph area */}
      <div className="relative flex-1 overflow-hidden p-3">
        <KnowledgeGraph
          nodes={graphData?.nodes ?? []}
          edges={graphData?.edges ?? []}
          onNodeClick={setSelectedNode}
          onMount={(controls) => { jumpToNodeRef.current = controls }}
          loading={loading}
        />

        {/* Node detail panel — floats over the graph */}
        {selectedNode && (
          <div className="absolute top-3 right-3 bottom-3 w-80 z-20 pointer-events-auto">
            <NodeDetail
              node={selectedNode}
              allNodes={graphData?.nodes ?? []}
              allEdges={graphData?.edges ?? []}
              onClose={() => setSelectedNode(null)}
              onJump={handleJump}
            />
          </div>
        )}
      </div>
    </div>
  )
}
