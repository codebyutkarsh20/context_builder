import { useState } from 'react'
import { X, FileCode, Loader2, AlertCircle, Flame } from 'lucide-react'
import { cn, truncate } from '../lib/utils'
import type { Hotspot, GraphNode } from '../lib/api'

interface HotspotsProps {
  hotspots: Hotspot[]
  loading?: boolean
  error?: string | null
}

const TYPE_COLORS: Record<GraphNode['type'], string> = {
  File: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
  Class: 'bg-green-500/20 text-green-300 border-green-500/30',
  Function: 'bg-orange-500/20 text-orange-300 border-orange-500/30',
  BusinessRule: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
  DomainConcept: 'bg-pink-500/20 text-pink-300 border-pink-500/30',
  DecisionPoint: 'bg-red-500/20 text-red-300 border-red-500/30',
}

function TypeBadge({ type }: { type: GraphNode['type'] }) {
  return (
    <span
      className={cn(
        'inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium border',
        TYPE_COLORS[type] ?? 'bg-zinc-700 text-zinc-300 border-zinc-600'
      )}
    >
      {type}
    </span>
  )
}

interface DetailPanelProps {
  hotspot: Hotspot
  onClose: () => void
}

function DetailPanel({ hotspot, onClose }: DetailPanelProps) {
  return (
    <div className="fixed inset-y-0 right-0 w-80 bg-zinc-900 border-l border-zinc-700 shadow-2xl z-50 flex flex-col">
      <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-700/60">
        <h3 className="text-sm font-semibold text-zinc-100">Node Detail</h3>
        <button
          onClick={onClose}
          className="w-7 h-7 rounded-md hover:bg-zinc-700 flex items-center justify-center transition-colors"
        >
          <X className="w-4 h-4 text-zinc-400" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-5 space-y-5">
        <div>
          <div className="flex items-center gap-2 mb-3">
            <TypeBadge type={hotspot.type} />
            <span className="text-xs text-zinc-500">Rank #{hotspot.rank}</span>
          </div>
          <h4 className="text-base font-semibold text-zinc-100 font-mono break-all">
            {hotspot.name}
          </h4>
        </div>

        {hotspot.file && (
          <div>
            <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1">File</p>
            <div className="flex items-center gap-2 p-3 rounded-lg bg-zinc-800 border border-zinc-700/50">
              <FileCode className="w-4 h-4 text-zinc-500 flex-shrink-0" />
              <span className="text-xs text-zinc-300 font-mono break-all">{hotspot.file}</span>
            </div>
          </div>
        )}

        <div>
          <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2">PageRank Score</p>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-2xl font-bold text-zinc-100 font-mono">
                {(hotspot.pagerank ?? 0).toFixed(4)}
              </span>
              <span className="text-xs text-zinc-500">
                {((hotspot.pagerank ?? 0) * 100).toFixed(2)}%
              </span>
            </div>
            <div className="h-2 bg-zinc-700 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-blue-500 to-blue-400"
                style={{ width: `${Math.min(100, (hotspot.pagerank ?? 0) * 100)}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function Hotspots({ hotspots, loading, error }: HotspotsProps) {
  const [selected, setSelected] = useState<Hotspot | null>(null)

  if (loading) {
    return (
      <div className="flex items-center justify-center h-32">
        <div className="flex items-center gap-3">
          <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
          <p className="text-sm text-zinc-400">Loading hotspots...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center gap-3 p-4 rounded-xl bg-red-950/30 border border-red-800/40 text-red-400">
        <AlertCircle className="w-5 h-5 flex-shrink-0" />
        <p className="text-sm">{error}</p>
      </div>
    )
  }

  if (hotspots.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-32 gap-3">
        <Flame className="w-8 h-8 text-zinc-700" />
        <p className="text-sm text-zinc-500">No hotspot data available</p>
      </div>
    )
  }

  const maxPagerank = Math.max(...hotspots.map((h) => h.pagerank ?? 0), 0.001)

  return (
    <>
      <div className="overflow-x-auto rounded-xl border border-zinc-700/50">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-700/50 bg-zinc-800/40">
              <th className="text-left px-4 py-3 text-xs font-semibold text-zinc-400 uppercase tracking-wider w-12">
                #
              </th>
              <th className="text-left px-4 py-3 text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                Symbol
              </th>
              <th className="text-left px-4 py-3 text-xs font-semibold text-zinc-400 uppercase tracking-wider w-32">
                Type
              </th>
              <th className="text-left px-4 py-3 text-xs font-semibold text-zinc-400 uppercase tracking-wider hidden md:table-cell">
                File
              </th>
              <th className="text-left px-4 py-3 text-xs font-semibold text-zinc-400 uppercase tracking-wider w-40">
                PageRank
              </th>
            </tr>
          </thead>
          <tbody>
            {hotspots.map((h, idx) => (
              <tr
                key={h.id}
                onClick={() => setSelected(h)}
                className={cn(
                  'border-b border-zinc-800/50 cursor-pointer transition-colors hover:bg-zinc-800/40',
                  selected?.id === h.id && 'bg-zinc-800/60'
                )}
              >
                <td className="px-4 py-3 text-zinc-500 font-mono text-xs">
                  {idx === 0 ? (
                    <span className="text-yellow-400 font-bold">#1</span>
                  ) : (
                    `#${h.rank}`
                  )}
                </td>
                <td className="px-4 py-3">
                  <span className="text-zinc-200 font-mono text-xs font-medium">
                    {truncate(h.name, 40)}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <TypeBadge type={h.type} />
                </td>
                <td className="px-4 py-3 hidden md:table-cell">
                  <span className="text-zinc-400 text-xs font-mono">
                    {h.file ? truncate(h.file, 35) : '—'}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <div className="flex-1 h-1.5 bg-zinc-700 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-blue-600 to-blue-400"
                        style={{ width: `${((h.pagerank ?? 0) / maxPagerank) * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-zinc-400 font-mono w-14 text-right">
                      {(h.pagerank ?? 0).toFixed(4)}
                    </span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selected && (
        <>
          <div
            className="fixed inset-0 bg-black/30 z-40"
            onClick={() => setSelected(null)}
          />
          <DetailPanel hotspot={selected} onClose={() => setSelected(null)} />
        </>
      )}
    </>
  )
}
