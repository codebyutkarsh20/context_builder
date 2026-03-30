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
  HAS_DECISION: '#f97316',
  GOVERNED_BY: '#a855f7',
  REPRESENTS: '#ec4899',
}

// Edge types that have direction (get arrow heads)
const DIRECTED_EDGE_TYPES = new Set(['CALLS', 'IMPORTS', 'INHERITS', 'GOVERNED_BY', 'REPRESENTS', 'HAS_DECISION'])

const NODE_TYPES = Object.keys(NODE_COLORS)
const EDGE_TYPES = Object.keys(EDGE_COLORS)

// Node radius helper — consistent formula used in both nodeCanvasObject and linkCanvasObject
function nodeRadius(val: number | undefined): number {
  return Math.sqrt(val ?? 1) * 4.5 + 4
}

export default function KnowledgeGraph({ nodes, edges, onNodeClick, onMount, loading = false }: KnowledgeGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const graphRef = useRef<any>(null)
  const initialFitDone = useRef(false)
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })
  const [visibleNodeTypes, setVisibleNodeTypes] = useState<Set<string>>(new Set(NODE_TYPES))
  const [visibleEdgeTypes, setVisibleEdgeTypes] = useState<Set<string>>(new Set(EDGE_TYPES))
  const [legendOpen, setLegendOpen] = useState(true)

  // State for React-driven UI
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [hoveredNodeData, setHoveredNodeData] = useState<(GraphNode & { x: number; y: number }) | null>(null)
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null)

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
      val: n.pagerank !== undefined ? Math.max(0.8, n.pagerank * 80) : 1,
    })),
    links: filteredEdges.map((e) => ({
      ...e,
      source: String(e.source),
      target: String(e.target),
    })),
  }), [filteredNodes, filteredEdges]) // eslint-disable-line react-hooks/exhaustive-deps

  // Apply improved D3 forces whenever graph data changes
  useEffect(() => {
    if (!graphRef.current) return
    // Strong repulsion + longer links to spread nodes out and prevent overlap
    graphRef.current.d3Force('charge')?.strength(-250).distanceMax(500)
    graphRef.current.d3Force('link')?.distance(70).strength(0.3)
    graphRef.current.d3ReheatSimulation()
  }, [graphData])

  // ─── Node rendering ────────────────────────────────────────────────────────

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

      const r = nodeRadius(n.val)
      const color = NODE_COLORS[n.type] ?? '#94a3b8'

      // Glow for selected/hovered
      if (isHovered || isSelected) {
        ctx.shadowBlur = isSelected ? 28 : 18
        ctx.shadowColor = color
      }

      // Node circle
      ctx.beginPath()
      ctx.arc(n.x, n.y, r, 0, 2 * Math.PI)
      ctx.fillStyle = isDimmed
        ? color + '28'
        : isHovered || isSelected
          ? color
          : isNeighbor
            ? color + 'e0'
            : color + 'cc'
      ctx.fill()

      // Border
      if (isSelected) {
        ctx.strokeStyle = '#ffffff'
        ctx.lineWidth = 2.5 / globalScale
        ctx.stroke()
      } else if (isHovered) {
        ctx.strokeStyle = '#ffffffcc'
        ctx.lineWidth = 1.5 / globalScale
        ctx.stroke()
      } else if (isNeighbor) {
        ctx.strokeStyle = color + '80'
        ctx.lineWidth = 1 / globalScale
        ctx.stroke()
      }

      ctx.shadowBlur = 0

      // Type ring indicator (small colored ring segment for type identification)
      if (!isDimmed && r > 5) {
        ctx.beginPath()
        ctx.arc(n.x, n.y, r + 1.5 / globalScale, -Math.PI / 2, Math.PI * 0.5)
        ctx.strokeStyle = color + (isSelected || isHovered ? 'ff' : '60')
        ctx.lineWidth = 1.5 / globalScale
        ctx.stroke()
      }

      // Label — show at lower zoom threshold and always for active nodes
      const showLabel = isHovered || isSelected || isNeighbor || (globalScale > 1.2 && r > 5)
      if (showLabel && !isDimmed) {
        const raw = String(n.name ?? n.id)
        const label = raw.includes('::') ? raw.split('::').pop()! : raw.split('/').pop()!
        const fontSize = Math.max(9, Math.min(13, r * 1.4)) / globalScale
        ctx.font = `${isSelected || isHovered ? '600 ' : ''}${fontSize}px Inter, system-ui, sans-serif`
        ctx.textAlign = 'center'
        ctx.textBaseline = 'top'

        const text = label.slice(0, 30)
        const textW = ctx.measureText(text).width
        const pad = 2.5 / globalScale
        const bx = n.x - textW / 2 - pad
        const by = n.y + r + 3 / globalScale

        // Label background
        ctx.fillStyle = 'rgba(0,0,0,0.72)'
        ctx.beginPath()
        if (ctx.roundRect) {
          ctx.roundRect(bx, by, textW + pad * 2, fontSize + pad * 2, 3 / globalScale)
        }
        ctx.fill()

        ctx.fillStyle = isHovered || isSelected ? '#ffffff' : isNeighbor ? '#e4e4e7' : 'rgba(212,212,216,0.85)'
        ctx.fillText(text, n.x, by + pad)

        // Show type badge below name for hovered/selected
        if ((isHovered || isSelected) && globalScale > 0.6) {
          const typeText = n.type
          const typeFontSize = Math.max(7, fontSize * 0.8)
          ctx.font = `bold ${typeFontSize}px Inter, system-ui, sans-serif`
          const typeW = ctx.measureText(typeText).width
          const tyBy = by + fontSize + pad * 2 + 2 / globalScale
          ctx.fillStyle = 'rgba(0,0,0,0.65)'
          ctx.beginPath()
          if (ctx.roundRect) {
            ctx.roundRect(n.x - typeW / 2 - pad, tyBy, typeW + pad * 2, typeFontSize + pad * 2, 2 / globalScale)
          }
          ctx.fill()
          ctx.fillStyle = color + 'ee'
          ctx.fillText(typeText, n.x, tyBy + pad)
        }
      }
    },
    [] // stable — reads refs
  )

  const nodePointerAreaPaint = useCallback(
    (node: Record<string, unknown>, color: string, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const n = node as GraphNode & { x: number; y: number; val: number }
      const r = nodeRadius(n.val)
      const hitR = Math.max(r + 8 / globalScale, 12 / globalScale)
      ctx.beginPath()
      ctx.arc(n.x, n.y, hitR, 0, 2 * Math.PI)
      ctx.fillStyle = color
      ctx.fill()
    },
    []
  )

  // ─── Edge rendering ─────────────────────────────────────────────────────────

  const linkCanvasObject = useCallback(
    (link: Record<string, unknown>, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const l = link as unknown as GraphEdge & {
        source: { x: number; y: number; id: string; val?: number }
        target: { x: number; y: number; id: string; val?: number }
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

      const alpha = isDimmed ? '18' : isActive ? 'ee' : '88'
      const width = isDimmed
        ? Math.max(0.3 / globalScale, 0.3)
        : isActive
          ? Math.max(2.5 / globalScale, 1.8)
          : Math.max(1.2 / globalScale, 1.0)

      if (isActive) { ctx.shadowBlur = 6; ctx.shadowColor = color }

      // Edge line
      ctx.beginPath()
      ctx.moveTo(l.source.x, l.source.y)
      ctx.lineTo(l.target.x, l.target.y)
      ctx.strokeStyle = color + alpha
      ctx.lineWidth = width
      ctx.stroke()
      ctx.shadowBlur = 0

      // Arrow head — show for directed edges at sufficient zoom
      const isDirected = DIRECTED_EDGE_TYPES.has(l.type)
      if (isDirected && globalScale > 0.25 && !isDimmed) {
        const dx = l.target.x - l.source.x
        const dy = l.target.y - l.source.y
        const len = Math.sqrt(dx * dx + dy * dy)
        if (len < 2) return

        const ux = dx / len, uy = dy / len
        const tgtR = nodeRadius(l.target.val)
        const arrowSize = isActive ? 6 / globalScale : 4 / globalScale

        // Place arrow tip just at the node edge
        const endX = l.target.x - ux * (tgtR + 1 / globalScale)
        const endY = l.target.y - uy * (tgtR + 1 / globalScale)
        const ax = endX - ux * arrowSize * 2.2
        const ay = endY - uy * arrowSize * 2.2

        ctx.beginPath()
        ctx.moveTo(ax + uy * arrowSize, ay - ux * arrowSize)
        ctx.lineTo(endX, endY)
        ctx.lineTo(ax - uy * arrowSize, ay + ux * arrowSize)
        ctx.closePath()
        ctx.fillStyle = color + (isDimmed ? '18' : isActive ? 'ee' : '88')
        ctx.fill()
      }

      // Edge type label — show on active edges at medium zoom
      if (isActive && globalScale > 0.7) {
        const midX = (l.source.x + l.target.x) / 2
        const midY = (l.source.y + l.target.y) / 2
        const fontSize = Math.max(7, 9 / globalScale)
        ctx.font = `bold ${fontSize}px Inter, system-ui, sans-serif`
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'

        const text = l.type.replace(/_/g, ' ')
        const textW = ctx.measureText(text).width
        const pad = 3 / globalScale

        ctx.fillStyle = 'rgba(0,0,0,0.82)'
        ctx.beginPath()
        if (ctx.roundRect) {
          ctx.roundRect(midX - textW / 2 - pad, midY - fontSize / 2 - pad, textW + pad * 2, fontSize + pad * 2, 3 / globalScale)
        }
        ctx.fill()

        ctx.fillStyle = color + 'f0'
        ctx.fillText(text, midX, midY)
      }
    },
    [] // stable — reads refs
  )

  // ─── Interactions ──────────────────────────────────────────────────────────

  const handleNodeClick = useCallback((node: Record<string, unknown>) => {
    const n = node as GraphNode
    const id = String(n.id)
    const next = selectedNodeIdRef.current === id ? null : id
    selectedNodeIdRef.current = next
    setSelectedNodeId(next)
    setHoveredNodeData(null)
    setTooltipPos(null)
    if (next !== null) onNodeClick?.(n)
    else onNodeClick?.(null)
  }, [onNodeClick])

  const handleNodeHover = useCallback((node: Record<string, unknown> | null) => {
    const n = node as (GraphNode & { x: number; y: number }) | null
    const id = n ? String(n.id) : null
    hoveredNodeIdRef.current = id
    setHoveredNodeId(id)
    document.body.style.cursor = node ? 'pointer' : 'default'

    if (n && graphRef.current && n.x !== undefined) {
      const screenPos = graphRef.current.graph2ScreenCoords(n.x, n.y)
      setTooltipPos(screenPos)
      setHoveredNodeData(n)
    } else {
      setTooltipPos(null)
      setHoveredNodeData(null)
    }
  }, [])

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

      {/* Hover tooltip — only when not selected */}
      {hoveredNodeData && tooltipPos && !selectedNodeId && (
        <div
          className="absolute z-20 pointer-events-none"
          style={{
            left: Math.min(tooltipPos.x + 18, dimensions.width - 220),
            top: Math.max(tooltipPos.y - 20, 8),
          }}
        >
          <div className="bg-zinc-900/98 border border-zinc-700/70 rounded-xl p-3 shadow-2xl backdrop-blur-sm w-52">
            <div className="flex items-center gap-2 mb-1.5">
              <span className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ backgroundColor: NODE_COLORS[hoveredNodeData.type] ?? '#94a3b8' }} />
              <span className="text-[10px] text-zinc-500 uppercase tracking-wider font-bold">
                {hoveredNodeData.type}
              </span>
            </div>
            <p className="text-xs font-mono text-zinc-100 font-semibold break-all leading-snug">
              {String(hoveredNodeData.name ?? hoveredNodeData.id).split('::').pop() ?? hoveredNodeData.id}
            </p>
            {String(hoveredNodeData.name ?? '').includes('::') && (
              <p className="text-[10px] text-zinc-600 font-mono mt-0.5 break-all leading-snug">
                {String(hoveredNodeData.name ?? hoveredNodeData.id)}
              </p>
            )}
            {hoveredNodeData.summary && (
              <p className="text-[11px] text-zinc-400 mt-2 leading-relaxed line-clamp-3">
                {hoveredNodeData.summary}
              </p>
            )}
            {hoveredNodeData.pagerank != null && (
              <div className="flex items-center gap-2 mt-2">
                <div className="flex-1 h-1 bg-zinc-800 rounded-full overflow-hidden">
                  <div className="h-full rounded-full transition-all"
                    style={{
                      width: `${Math.min(100, hoveredNodeData.pagerank * 200)}%`,
                      backgroundColor: NODE_COLORS[hoveredNodeData.type] ?? '#94a3b8',
                    }} />
                </div>
                <span className="text-[10px] font-mono text-zinc-600 flex-shrink-0">
                  PR {hoveredNodeData.pagerank.toFixed(4)}
                </span>
              </div>
            )}
            {hoveredNodeData.file && (
              <p className="text-[10px] text-zinc-600 font-mono mt-1.5 truncate">
                {hoveredNodeData.file.split('/').slice(-2).join('/')}
              </p>
            )}
            <p className="text-[10px] text-zinc-700 mt-2">Click to inspect details →</p>
          </div>
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
          <div className="bg-zinc-900/95 backdrop-blur-sm border border-zinc-700/50 rounded-xl p-3 w-44 shadow-2xl">
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
                  <div className="flex items-center gap-1.5 flex-1 min-w-0">
                    <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: NODE_COLORS[type] }} />
                    <span className="text-xs text-zinc-400 group-hover:text-zinc-200 transition-colors truncate">{type}</span>
                  </div>
                </label>
              ))}
            </div>
            <p className="text-[10px] font-bold text-zinc-600 uppercase tracking-widest mb-2">Edge Types</p>
            <div className="space-y-1.5">
              {EDGE_TYPES.map((type) => (
                <label key={type} className="flex items-center gap-2 cursor-pointer group select-none">
                  <input type="checkbox" checked={visibleEdgeTypes.has(type)} onChange={() => toggleEdgeType(type)} className="sr-only" />
                  <div className="flex items-center gap-1.5 flex-1 min-w-0">
                    <div className={cn('w-5 h-1 rounded-full transition-all flex-shrink-0',
                      visibleEdgeTypes.has(type) ? 'opacity-100' : 'opacity-20 bg-zinc-600')}
                      style={visibleEdgeTypes.has(type) ? { backgroundColor: EDGE_COLORS[type] } : {}} />
                    <span className="text-xs text-zinc-400 group-hover:text-zinc-200 transition-colors truncate">{type}</span>
                  </div>
                  {DIRECTED_EDGE_TYPES.has(type) && (
                    <span className="text-[9px] text-zinc-700 flex-shrink-0">→</span>
                  )}
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
            Hover to preview · Click to inspect · Scroll to zoom · Drag to pan
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
          maxZoom={16}
          cooldownTicks={200}
          d3AlphaDecay={0.022}
          d3VelocityDecay={0.38}
          warmupTicks={40}
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
