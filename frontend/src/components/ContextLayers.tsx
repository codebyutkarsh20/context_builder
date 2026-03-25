import { cn, formatTokens, formatNumber } from '../lib/utils'
import type { ContextLayersResponse } from '../lib/api'
import { Loader2, AlertCircle, Layers } from 'lucide-react'

interface ContextLayersProps {
  data: ContextLayersResponse | null
  loading?: boolean
  error?: string | null
}

const LAYER_ICONS = ['①', '②', '③', '④', '⑤', '⑥']

function completenessColor(pct: number): string {
  if (pct >= 80) return 'bg-green-500'
  if (pct >= 50) return 'bg-yellow-500'
  return 'bg-red-500'
}

function completenessText(pct: number): string {
  if (pct >= 80) return 'text-green-400'
  if (pct >= 50) return 'text-yellow-400'
  return 'text-red-400'
}

export default function ContextLayers({ data, loading, error }: ContextLayersProps) {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-48">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="w-6 h-6 text-blue-400 animate-spin" />
          <p className="text-sm text-zinc-400">Loading context layers...</p>
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

  if (!data) {
    return (
      <div className="flex flex-col items-center justify-center h-48 gap-3">
        <Layers className="w-10 h-10 text-zinc-700" />
        <p className="text-sm text-zinc-500">No context layers available</p>
      </div>
    )
  }

  const budgetUsedPct = Math.min(100, (data.total_tokens / data.token_budget) * 100)

  return (
    <div className="space-y-5">
      {/* Token budget bar */}
      <div className="p-4 rounded-xl bg-zinc-800/50 border border-zinc-700/50">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-zinc-300">Total Token Budget</span>
          <span className="text-sm text-zinc-400">
            <span className="text-zinc-100 font-mono">{formatTokens(data.total_tokens)}</span>
            {' / '}
            <span className="text-zinc-500 font-mono">{formatTokens(data.token_budget)}</span>
          </span>
        </div>
        <div className="h-2 bg-zinc-700 rounded-full overflow-hidden">
          <div
            className={cn(
              'h-full rounded-full transition-all',
              budgetUsedPct >= 90 ? 'bg-red-500' : budgetUsedPct >= 70 ? 'bg-yellow-500' : 'bg-blue-500'
            )}
            style={{ width: `${budgetUsedPct}%` }}
          />
        </div>
        <p className="text-xs text-zinc-500 mt-1">{budgetUsedPct.toFixed(1)}% of budget used</p>
      </div>

      {/* Layer cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {data.layers.map((layer, idx) => {
          const pct = Math.round(layer.completeness * 100)
          return (
            <div
              key={layer.layer}
              className="p-4 rounded-xl bg-zinc-800/40 border border-zinc-700/50 hover:border-zinc-600/60 transition-colors"
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-lg leading-none text-zinc-500">{LAYER_ICONS[idx] ?? `L${layer.layer}`}</span>
                  <div>
                    <h3 className="text-sm font-semibold text-zinc-200 leading-tight">{layer.name}</h3>
                    <span className="text-xs text-zinc-500">Layer {layer.layer}</span>
                  </div>
                </div>
                <span className={cn('text-xs font-semibold', completenessText(pct))}>
                  {pct}%
                </span>
              </div>

              <p className="text-xs text-zinc-400 mb-3 leading-relaxed line-clamp-2">
                {layer.description}
              </p>

              <div className="flex items-center justify-between text-xs text-zinc-500 mb-2">
                <span>{formatNumber(layer.node_count)} nodes</span>
                <span className="font-mono">{formatTokens(layer.token_estimate)}</span>
              </div>

              {/* Completeness bar */}
              <div className="h-1.5 bg-zinc-700 rounded-full overflow-hidden">
                <div
                  className={cn('h-full rounded-full transition-all', completenessColor(pct))}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
