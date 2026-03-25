import { useRef, useCallback, useState, useEffect, useMemo } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { Loader2, Maximize2, ZoomIn, ZoomOut, Layers } from 'lucide-react'
import { cn } from '../lib/utils'
import type { GraphNode, GraphEdge } from '../lib/api'

export interface KnowledgeGraphControls {
  jumpToNode: (node: GraphNode) => void
}

interface KnowledgeGraphProps {
  nodes: GraphNode[]
  edges: GraphEdge[]
  onNodeClick?: (node: GraphNode | null) => void
  onMount?: (controls: KnowledgeGraphControls) => void
  loading?: boolean
}

export const NODE_COLORS: Record<string, string> = {
  File: '#3b82f6',
  Class: '#22c55e',
  Function: '#f97316',
  BusinessRule: '#a855f7',
  DomainConcept: '#ec4899',
  DecisionPoint: '#ef4444',
}

export const EDGE_COLORS: Record<string, string> = {
  CONTAINS: '#52525b',
  CALLS: '#ef4444',
  IMPORTS: '#eab308',
  INHERITS: '#06b6d4',
  RELATED_TO: '#8b5cf6',
  HAS_DECISION: '#ef4444',
  GOVERNED_BY: '#a855f7',
  REPRESENTS: '#ec4899',
}

const NODE_TYPES = Object.keys(NODE_COLORS)
const EDGE_TYPES = Object.keys(EDGE_COLORS)

export default function KnowledgeGraph({ nodes, edges, onNodeClick, onMount, loading = false }: KnowledgeGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const graphRef = useRef<any>(null)
  const initialFitDone = useRef(false)
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })
  const [visibleNodeTypes, setVisibleNodeTypes] = useState<Set<string>>(new Set(NODE_TYPES))
  const [visibleEdgeTypes, setVisibleEdgeTypes] = useState<Set<string>>(new Set(EDGE_TYPES))
  const [legendOpen, setLegendOpen] = useState(true)

  // State for React-driven UI (hint text, etc.)
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)

  // Refs for canvas callbacks — stable references, no rerenders
  const hoveredNodeIdRef = useRef<string | null>(null)
  const selectedNodeIdRef = useRef<string | null>(null)

  // Build neighbor set for highlight-on-hover
  const neighborMap = useMemo(() => {
    const map = new Map<string, Set<string>>()
    edges.forEach((e) => {
      const s = String(e.source), t = String(e.target)
      if (!map.has(s)) map.set(s, new Set())
      if (!map.has(t)) map.set(t, new Set())
      map.get(s)!.add(t)
      map.get(t)!.add(s)
    })
    return map
  }, [edges])
  const neighborMapRef = useRef(neighborMap)
  useEffect(() => { neighborMapRef.current = neighborMap }, [neighborMap])

  // Reset fit flag when graph data changes
  useEffect(() => { initialFitDone.current = false }, [nodes])

  // Expose jump-to-node to parent
  useEffect(() => {
    if (!onMount) return
    onMount({
      jumpToNode: (node: GraphNode) => {
        const id = node.id
        selectedNodeIdRef.current = id
        setSelectedNodeId(id)
        const n = node as GraphNode & { x?: number; y?: number }
        if (n.x !== undefined && n.y !== undefined) {
          graphRef.current?.centerAt(n.x, n.y, 600)
          graphRef.current?.zoom(4, 600)
        }
      },
    })
  }, [onMount])

  useEffect(() => {
    if (!containerRef.current) return
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect
      if (width > 0 && height > 0) setDimensions({ width, height })
    })
    observer.observe(containerRef.current)
    const { width, height } = containerRef.current.getBoundingClientRect()
    if (width > 0 && height > 0) setDimensions({ width, height })
    return () => observer.disconnect()
  }, [])

  const filteredNodes = useMemo(
    () => nodes.filter((n) => visibleNodeTypes.has(n.type)),
    [nodes, visibleNodeTypes]
  )
  const filteredNodeIds = useMemo(
    () => new Set(filteredNodes.map((n) => n.id)),
    [filteredNodes]
  )
  const filteredEdges = useMemo(
    () => edges.filter(
      (e) => visibleEdgeTypes.has(e.type) &&
        filteredNodeIds.has(String(e.source)) &&
        filteredNodeIds.has(String(e.target))
    ),
    [edges, visibleEdgeTypes, filteredNodeIds]
  )

  const graphData = useMemo(() => ({
    nodes: filteredNodes.map((n) => ({
      ...n,
      id: n.id,
      val: n.pagerank !== undefined ? Math.max(0.5, n.pagerank * 60) : 1,
    })),
    links: filteredEdges.map((e) => ({
      ...e,
      source: String(e.source),
      target: String(e.target),
    })),
  }), [filteredNodes, filteredEdges]) // eslint-disable-line react-hooks/exhaustive-deps

  // Apply compact D3 forces whenever graph data changes
  useEffect(() => {
    if (!graphRef.current) return
    graphRef.current.d3Force('charge')?.strength(-20).distanceMax(120)
    graphRef.current.d3Force('link')?.distance(25).strength(0.8)
    graphRef.current.d3ReheatSimulation()
  }, [graphData])

  // ─── Node rendering — reads from refs, EMPTY deps = stable reference ───────

  const nodeCanvasObject = useCallback(
    (node: Record<string, unknown>, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const n = node as GraphNode & { x: number; y: number; val: number }
      const nodeId = String(n.id)
      const hoveredId = hoveredNodeIdRef.current
      const selectedId = selectedNodeIdRef.current
      const nbrs = neighborMapRef.current

      const isHovered = hoveredId === nodeId
      const isSelected = selectedId === nodeId
      const isNeighbor = hoveredId
        ? nbrs.get(hoveredId)?.has(nodeId) ?? false
        : selectedId
          ? nbrs.get(selectedId)?.has(nodeId) ?? false
          : false
      const hasFocus = hoveredId !== null || selectedId !== null
      const isDimmed = hasFocus && !isHovered && !isSelected && !isNeighbor

      const r = Math.sqrt(n.val ?? 1) * 3.8 + 2
      const color = NODE_COLORS[n.type] ?? '#94a3b8'

      if (isHovered || isSelected) {
        ctx.shadowBlur = isSelected ? 24 : 16
        ctx.shadowColor = color
      }

      ctx.beginPath()
      ctx.arc(n.x, n.y, r, 0, 2 * Math.PI)
      ctx.fillStyle = isDimmed
        ? color + '30'
        : isHovered || isSelected
          ? color
          : isNeighbor
            ? color + 'dd'
            : color + 'bb'
      ctx.fill()

      if (isSelected) {
        ctx.strokeStyle = '#ffffff'
        ctx.lineWidth = 2 / globalScale
        ctx.stroke()
      } else if (isHovered) {
        ctx.strokeStyle = color
        ctx.lineWidth = 1.5 / globalScale
        ctx.stroke()
      }

      ctx.shadowBlur = 0

      const showLabel = isHovered || isSelected || isNeighbor || (globalScale > 1.8 && r > 4)
      if (showLabel && !isDimmed) {
        const raw = String(n.name ?? n.id)
        const label = raw.includes('::') ? raw.split('::').pop()! : raw.split('/').pop()!
        const fontSize = Math.max(9, Math.min(14, r * 1.5)) / globalScale
        ctx.font = `${isSelected || isHovered ? 'bold ' : ''}${fontSize}px Inter, system-ui, sans-serif`
        ctx.textAlign = 'center'
        ctx.textBaseline = 'top'

        const text = label.slice(0, 28)
        const textW = ctx.measureText(text).width
        const pad = 2 / globalScale
        const bx = n.x - textW / 2 - pad
        const by = n.y + r + 2 / globalScale
        ctx.fillStyle = 'rgba(0,0,0,0.65)'
        ctx.beginPath()
        ctx.roundRect?.(bx, by, textW + pad * 2, fontSize + pad * 2, 2 / globalScale)
        ctx.fill()

        ctx.fillStyle = isHovered || isSelected ? '#ffffff' : isNeighbor ? '#e4e4e7' : 'rgba(228,228,231,0.75)'
        ctx.fillText(text, n.x, by + pad)
      }
    },
    [] // stable — reads refs, never recreated
  )

  const nodePointerAreaPaint = useCallback(
    (node: Record<string, unknown>, color: string, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const n = node as GraphNode & { x: number; y: number; val: number }
      const r = Math.sqrt(n.val ?? 1) * 3.8 + 2
      const hitR = Math.max(r + 6 / globalScale, 10 / globalScale)
      ctx.beginPath()
      ctx.arc(n.x, n.y, hitR, 0, 2 * Math.PI)
      ctx.fillStyle = color
      ctx.fill()
    },
    []
  )

  // ─── Edge rendering — reads from refs, EMPTY deps ──────────────────────────

  const linkCanvasObject = useCallback(
    (link: Record<string, unknown>, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const l = link as unknown as GraphEdge & {
        source: { x: number; y: number; id: string }
        target: { x: number; y: number; id: string }
      }
      if (!l.source?.x || !l.target?.x) return

      const srcId = String(l.source.id ?? l.source)
      const tgtId = String(l.target.id ?? l.target)
      const color = EDGE_COLORS[l.type] ?? '#52525b'

      const hoveredId = hoveredNodeIdRef.current
      const selectedId = selectedNodeIdRef.current

      const isActive = hoveredId
        ? srcId === hoveredId || tgtId === hoveredId
        : selectedId
          ? srcId === selectedId || tgtId === selectedId
          : false

      const hasFocus = hoveredId !== null || selectedId !== null
      const isDimmed = hasFocus && !isActive
      if (isDimmed) {
        ctx.beginPath()
        ctx.moveTo(l.source.x, l.source.y)
        ctx.lineTo(l.target.x, l.target.y)
        ctx.strokeStyle = color + '20'
        ctx.lineWidth = Math.max(0.3 / globalScale, 0.5)
        ctx.stroke()
        return
      }

      const alpha = isActive ? 'ee' : 'bb'
      const width = isActive ? Math.max(2.5 / globalScale, 2.0) : Math.max(1.0 / globalScale, 1.5)

      if (isActive) { ctx.shadowBlur = 4; ctx.shadowColor = color }

      ctx.beginPath()
      ctx.moveTo(l.source.x, l.source.y)
      ctx.lineTo(l.target.x, l.target.y)
      ctx.strokeStyle = color + alpha
      ctx.lineWidth = width
      ctx.stroke()
      ctx.shadowBlur = 0

      if (isActive && globalScale > 0.5) {
        const dx = l.target.x - l.source.x
        const dy = l.target.y - l.source.y
        const len = Math.sqrt(dx * dx + dy * dy)
        if (len < 1) return
        const ux = dx / len, uy = dy / len
        const arrowSize = 4 / globalScale
        const ax = l.target.x - ux * arrowSize * 3
        const ay = l.target.y - uy * arrowSize * 3
        ctx.beginPath()
        ctx.moveTo(ax + uy * arrowSize, ay - ux * arrowSize)
        ctx.lineTo(l.target.x, l.target.y)
        ctx.lineTo(ax - uy * arrowSize, ay + ux * arrowSize)
        ctx.strokeStyle = color + 'dd'
        ctx.lineWidth = 1 / globalScale
        ctx.stroke()
      }
    },
    [] // stable — reads refs, never recreated
  )

  // ─── Interactions ──────────────────────────────────────────────────────────

  const handleNodeClick = useCallback((node: Record<string, unknown>) => {
    const n = node as GraphNode
    const id = String(n.id)
    const next = selectedNodeIdRef.current === id ? null : id
    selectedNodeIdRef.current = next
    setSelectedNodeId(next)
    if (next !== null) onNodeClick?.(n)
    else onNodeClick?.(null)
  }, [onNodeClick])

  const handleNodeHover = useCallback((node: Record<string, unknown> | null) => {
    const id = node ? String((node as GraphNode).id) : null
    hoveredNodeIdRef.current = id
    setHoveredNodeId(id)
    document.body.style.cursor = node ? 'pointer' : 'default'
  }, [])

  // Reset cursor when component unmounts to avoid a stuck pointer cursor
  useEffect(() => () => { document.body.style.cursor = 'default' }, [])

  const handleBackgroundClick = useCallback(() => {
    selectedNodeIdRef.current = null
    setSelectedNodeId(null)
  }, [])

  const handleFitView = () => graphRef.current?.zoomToFit(500, 60)
  const handleZoomIn = () => { const c = graphRef.current?.zoom() ?? 1; graphRef.current?.zoom(c * 1.4, 300) }
  const handleZoomOut = () => { const c = graphRef.current?.zoom() ?? 1; graphRef.current?.zoom(c / 1.4, 300) }

  const toggleNodeType = (type: string) =>
    setVisibleNodeTypes((prev) => { const n = new Set(prev); n.has(type) ? n.delete(type) : n.add(type); return n })
  const toggleEdgeType = (type: string) =>
    setVisibleEdgeTypes((prev) => { const n = new Set(prev); n.has(type) ? n.delete(type) : n.add(type); return n })

  return (
    <div ref={containerRef} className="relative w-full h-full bg-[#080810] rounded-2xl overflow-hidden border border-zinc-800/40">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#080810]/90 z-30">
          <div className="flex flex-col items-center gap-3">
            <Loader2 className="w-8 h-8 text-blue-400 animate-spin" />
            <p className="text-sm text-zinc-400">Building knowledge graph...</p>
            <p className="text-xs text-zinc-600">Parsing {nodes.length > 0 ? nodes.length : '...'} nodes</p>
          </div>
        </div>
      )}

      {!loading && nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center z-20">
          <div className="text-center">
            <div className="w-16 h-16 rounded-2xl bg-zinc-900 border border-zinc-800 flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-zinc-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18" />
              </svg>
            </div>
            <p className="text-zinc-400 text-sm font-medium">No graph data available</p>
            <p className="text-zinc-600 text-xs mt-1">Analyze a repository to build its knowledge graph</p>
          </div>
        </div>
      )}

      {/* Zoom controls */}
      {nodes.length > 0 && (
        <div className="absolute top-4 left-4 z-10 flex flex-col gap-1">
          <button onClick={handleFitView} title="Fit all / reset zoom"
            className="w-8 h-8 rounded-lg bg-zinc-900/95 border border-zinc-700/50 flex items-center justify-center hover:bg-zinc-800 hover:border-zinc-600 transition-all backdrop-blur-sm group">
            <Maximize2 className="w-3.5 h-3.5 text-zinc-500 group-hover:text-zinc-200 transition-colors" />
          </button>
          <button onClick={handleZoomIn} title="Zoom in"
            className="w-8 h-8 rounded-lg bg-zinc-900/95 border border-zinc-700/50 flex items-center justify-center hover:bg-zinc-800 hover:border-zinc-600 transition-all backdrop-blur-sm group">
            <ZoomIn className="w-3.5 h-3.5 text-zinc-500 group-hover:text-zinc-200 transition-colors" />
          </button>
          <button onClick={handleZoomOut} title="Zoom out"
            className="w-8 h-8 rounded-lg bg-zinc-900/95 border border-zinc-700/50 flex items-center justify-center hover:bg-zinc-800 hover:border-zinc-600 transition-all backdrop-blur-sm group">
            <ZoomOut className="w-3.5 h-3.5 text-zinc-500 group-hover:text-zinc-200 transition-colors" />
          </button>
        </div>
      )}

      {/* Filter panel — collapsible, bottom-right */}
      <div className="absolute bottom-14 right-4 z-10">
        <button
          onClick={() => setLegendOpen((v) => !v)}
          className="ml-auto flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-zinc-900/95 border border-zinc-700/50 backdrop-blur-sm hover:bg-zinc-800 transition-all mb-1 group"
          title={legendOpen ? 'Hide legend' : 'Show legend'}
        >
          <Layers className="w-3.5 h-3.5 text-zinc-500 group-hover:text-zinc-300 transition-colors" />
          <span className="text-[11px] text-zinc-500 group-hover:text-zinc-300 transition-colors">Legend</span>
        </button>
        {legendOpen && (
          <div className="bg-zinc-900/95 backdrop-blur-sm border border-zinc-700/50 rounded-xl p-3 w-40 shadow-2xl">
            <p className="text-[10px] font-bold text-zinc-600 uppercase tracking-widest mb-2">Node Types</p>
            <div className="space-y-1.5 mb-3">
              {NODE_TYPES.map((type) => (
                <label key={type} className="flex items-center gap-2 cursor-pointer group select-none">
                  <input type="checkbox" checked={visibleNodeTypes.has(type)} onChange={() => toggleNodeType(type)} className="sr-only" />
                  <div className={cn('w-3 h-3 rounded-sm flex items-center justify-center transition-all flex-shrink-0 border',
                    visibleNodeTypes.has(type) ? 'border-transparent' : 'border-zinc-600')}
                    style={visibleNodeTypes.has(type) ? { backgroundColor: NODE_COLORS[type] } : {}}>
                    {visibleNodeTypes.has(type) && (
                      <svg className="w-2 h-2 text-white" viewBox="0 0 12 12" fill="none">
                        <path d="M2 6l3 3 5-5" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    )}
                  </div>
                  <div className="flex items-center gap-1.5 flex-1">
                    <span className="w-2 h-2 rounded-full flex-shrink-0 opacity-70" style={{ backgroundColor: NODE_COLORS[type] }} />
                    <span className="text-xs text-zinc-400 group-hover:text-zinc-200 transition-colors">{type}</span>
                  </div>
                </label>
              ))}
            </div>
            <p className="text-[10px] font-bold text-zinc-600 uppercase tracking-widest mb-2">Edge Types</p>
            <div className="space-y-1.5">
              {EDGE_TYPES.map((type) => (
                <label key={type} className="flex items-center gap-2 cursor-pointer group select-none">
                  <input type="checkbox" checked={visibleEdgeTypes.has(type)} onChange={() => toggleEdgeType(type)} className="sr-only" />
                  <div className={cn('w-5 h-1 rounded-full transition-all flex-shrink-0',
                    visibleEdgeTypes.has(type) ? 'opacity-100' : 'opacity-20 bg-zinc-600')}
                    style={visibleEdgeTypes.has(type) ? { backgroundColor: EDGE_COLORS[type] } : {}} />
                  <span className="text-xs text-zinc-400 group-hover:text-zinc-200 transition-colors">{type}</span>
                </label>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Stats badge */}
      {nodes.length > 0 && (
        <div className="absolute bottom-4 right-4 z-10">
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-zinc-900/95 border border-zinc-700/50 backdrop-blur-sm">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs font-mono text-zinc-400">
              {filteredNodes.length} nodes · {filteredEdges.length} edges
            </span>
          </div>
        </div>
      )}

      {/* Hint when nothing focused */}
      {nodes.length > 0 && !hoveredNodeId && !selectedNodeId && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 pointer-events-none">
          <span className="text-[11px] text-zinc-600 bg-zinc-900/80 px-3 py-1 rounded-full border border-zinc-800 backdrop-blur-sm">
            Click a node to inspect · Scroll to zoom · Drag to pan
          </span>
        </div>
      )}

      {/* Graph */}
      {!loading && nodes.length > 0 && (
        <ForceGraph2D
          ref={graphRef}
          graphData={graphData}
          width={dimensions.width}
          height={dimensions.height}
          backgroundColor="#080810"
          nodeCanvasObject={nodeCanvasObject as never}
          nodeCanvasObjectMode={() => 'replace'}
          nodePointerAreaPaint={nodePointerAreaPaint as never}
          linkCanvasObject={linkCanvasObject as never}
          linkCanvasObjectMode={() => 'replace'}
          onNodeClick={handleNodeClick as never}
          onNodeHover={handleNodeHover as never}
          onBackgroundClick={handleBackgroundClick}
          nodeLabel={() => ''}
          enableZoomInteraction
          enablePanInteraction
          minZoom={0.05}
          maxZoom={12}
          cooldownTicks={150}
          d3AlphaDecay={0.028}
          d3VelocityDecay={0.4}
          warmupTicks={30}
          onEngineStop={() => {
            if (!initialFitDone.current) {
              initialFitDone.current = true
              graphRef.current?.zoomToFit(400, 80)
            }
          }}
        />
      )}
    </div>
  )
}
