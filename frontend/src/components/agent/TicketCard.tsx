import { Play, Loader2 } from 'lucide-react'
import { cn } from '../../lib/utils'
import type { AgentTicket } from '../../lib/api'

export function TicketCard({ ticket, onRun, isRunning, isDisabled }: {
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
