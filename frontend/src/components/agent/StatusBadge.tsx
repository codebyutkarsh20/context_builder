import { Loader2, CheckCircle2, XCircle, AlertTriangle } from 'lucide-react'

export function StatusBadge({ status, isRunning }: { status: string; isRunning: boolean }) {
  if (isRunning) {
    return (
      <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-500/20 text-blue-400 text-xs font-medium border border-blue-500/30">
        <Loader2 className="w-3 h-3 animate-spin" />
        Running...
      </div>
    )
  }
  if (status === 'done') {
    return (
      <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-green-500/20 text-green-400 text-xs font-medium border border-green-500/30">
        <CheckCircle2 className="w-3 h-3" />
        Complete
      </div>
    )
  }
  if (status === 'failed') {
    return (
      <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-500/20 text-red-400 text-xs font-medium border border-red-500/30">
        <XCircle className="w-3 h-3" />
        Failed
      </div>
    )
  }
  if (status === 'escalated') {
    return (
      <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-yellow-500/20 text-yellow-400 text-xs font-medium border border-yellow-500/30">
        <AlertTriangle className="w-3 h-3" />
        Escalated
      </div>
    )
  }
  return null
}
