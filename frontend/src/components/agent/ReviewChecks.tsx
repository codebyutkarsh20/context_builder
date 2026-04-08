import { CheckCircle2, XCircle, AlertTriangle } from 'lucide-react'

export function ReviewChecks({ checks }: { checks: { name: string; status: string; comment: string }[] }) {
  if (!checks?.length) return null
  const statusIcon = (s: string) =>
    s === 'PASS' ? <CheckCircle2 className="w-3.5 h-3.5 text-green-400" /> :
    s === 'FAIL' ? <XCircle className="w-3.5 h-3.5 text-red-400" /> :
    <AlertTriangle className="w-3.5 h-3.5 text-yellow-400" />

  return (
    <div className="space-y-1.5">
      {checks.map((check) => (
        <div key={check.name} className="flex items-start gap-2 px-3 py-2 rounded-lg bg-zinc-900/60 border border-zinc-800/40">
          {statusIcon(check.status)}
          <div className="flex-1 min-w-0">
            <span className="text-xs font-bold text-zinc-400">{check.name}</span>
            <p className="text-xs text-zinc-500 mt-0.5">{check.comment}</p>
          </div>
        </div>
      ))}
    </div>
  )
}
