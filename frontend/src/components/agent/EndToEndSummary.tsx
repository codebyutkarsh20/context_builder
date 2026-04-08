import {
  Loader2, CheckCircle2, XCircle, AlertTriangle,
  FileCode, Wrench, Eye, GitPullRequest, Target, TestTube,
} from 'lucide-react'
import { cn } from '../../lib/utils'
import type { AgentJobStatus } from '../../lib/api'
import { ReviewChecks } from './ReviewChecks'
import { PatchCard } from './PatchCard'

function getOutcomeDisplay(activeJob: AgentJobStatus) {
  const result = activeJob.result
  const status = activeJob.status
  const reviewVerdict = result?.review?.verdict

  if (status === 'escalated') {
    return { label: 'Escalated to Human', icon: AlertTriangle, iconColor: 'text-yellow-400', bg: 'bg-yellow-950/30 border-yellow-800/40' }
  }
  if (status === 'failed') {
    return { label: 'Pipeline Failed', icon: XCircle, iconColor: 'text-red-400', bg: 'bg-red-950/30 border-red-800/40' }
  }
  if (status === 'done' && reviewVerdict === 'APPROVE') {
    return { label: 'Fix Approved', icon: CheckCircle2, iconColor: 'text-green-400', bg: 'bg-green-950/30 border-green-800/40' }
  }
  if (status === 'done' && (reviewVerdict === 'ESCALATE' || reviewVerdict === 'CHANGES_REQUESTED')) {
    return { label: reviewVerdict === 'ESCALATE' ? 'Escalated to Human' : 'Changes Requested', icon: AlertTriangle, iconColor: 'text-yellow-400', bg: 'bg-yellow-950/30 border-yellow-800/40' }
  }
  if (status === 'done') {
    return { label: 'Completed', icon: CheckCircle2, iconColor: 'text-green-400', bg: 'bg-green-950/30 border-green-800/40' }
  }
  return { label: 'Running...', icon: Loader2, iconColor: 'text-blue-400', bg: 'bg-zinc-900 border-zinc-800' }
}

export function EndToEndSummary({ job }: { job: AgentJobStatus }) {
  const result = job.result
  if (!result) return null

  const outcome = getOutcomeDisplay(job)
  const OutcomeIcon = outcome.icon
  const patches = result.repair?.patches ?? []
  const checks = result.review?.checks ?? []
  const passCount = checks.filter(c => c.status === 'PASS').length

  return (
    <div className="space-y-4">
      {/* Status */}
      <div className={cn('p-4 rounded-xl border flex items-center justify-between', outcome.bg)}>
        <div className="flex items-center gap-3">
          <OutcomeIcon className={cn('w-6 h-6', outcome.iconColor, outcome.label === 'Running...' && 'animate-spin')} />
          <div>
            <span className="text-base font-bold text-zinc-100">{outcome.label}</span>
            {job.error && <p className="text-xs text-red-400 mt-0.5 max-w-2xl">{job.error}</p>}
          </div>
        </div>
      </div>

      {/* What was found */}
      {result.localization && (
        <div className="p-5 rounded-xl bg-zinc-900 border border-zinc-800 space-y-4">
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <Target className="w-4 h-4 text-amber-400" />
                <h3 className="text-sm font-bold text-zinc-200">What was found</h3>
              </div>
              <p className="text-xs text-zinc-400 leading-relaxed">{result.localization.root_cause_hypothesis}</p>
            </div>
            {result.localization.confidence > 0 && (
              <span className="px-3 py-1 rounded-full bg-amber-500/10 text-amber-400 text-xs font-bold border border-amber-500/20 flex-shrink-0">
                {Math.round(result.localization.confidence * 100)}% confident
              </span>
            )}
          </div>

          {result.localization.fault_files?.length > 0 && (
            <div>
              <p className="text-[10px] font-bold text-zinc-600 uppercase mb-2">Fault Files</p>
              <div className="flex flex-wrap gap-2">
                {result.localization.fault_files.map((f: string) => (
                  <span key={f} className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-zinc-800 border border-zinc-700/50 text-xs text-zinc-300 font-mono">
                    <FileCode className="w-3 h-3 text-amber-400" />{f}
                  </span>
                ))}
              </div>
            </div>
          )}

          {result.localization.fault_functions?.length > 0 && (
            <div>
              <p className="text-[10px] font-bold text-zinc-600 uppercase mb-2">Fault Functions</p>
              <div className="flex flex-wrap gap-2">
                {result.localization.fault_functions.map((f: string) => (
                  <span key={f} className="px-2.5 py-1 rounded-lg bg-orange-500/10 border border-orange-500/20 text-xs text-orange-300 font-mono">{f}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* What was fixed */}
      {patches.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Wrench className="w-4 h-4 text-orange-400" />
            <h3 className="text-sm font-bold text-zinc-200">Code changes ({patches.length} files)</h3>
          </div>
          {result.repair?.explanation && (
            <p className="text-xs text-zinc-400 px-3">{result.repair.explanation}</p>
          )}
          <div className="space-y-2">
            {patches.map((p) => (
              <PatchCard key={p.file_path || p.explanation} patch={p} />
            ))}
          </div>
        </div>
      )}

      {/* Test results */}
      {(result.test_result || checks.length > 0) && (
        <div className="space-y-3">
          {result.test_result && (
            <div className={cn('p-4 rounded-xl border',
              result.test_result.includes('passed') ? 'bg-green-950/20 border-green-800/30' : 'bg-red-950/20 border-red-800/30'
            )}>
              <div className="flex items-center gap-2 mb-2">
                <TestTube className={cn('w-4 h-4', result.test_result.includes('passed') ? 'text-green-400' : 'text-red-400')} />
                <h3 className="text-sm font-bold text-zinc-200">Test execution</h3>
              </div>
              <pre className="text-[11px] font-mono text-zinc-300 whitespace-pre-wrap bg-black/20 rounded p-2 max-h-40 overflow-y-auto">{result.test_result.slice(0, 500)}</pre>
            </div>
          )}

          {checks.length > 0 && (
            <div className="p-4 rounded-xl bg-zinc-900 border border-zinc-800">
              <div className="flex items-center gap-2 mb-3">
                <Eye className="w-4 h-4 text-cyan-400" />
                <h3 className="text-sm font-bold text-zinc-200">Review checks</h3>
                <span className="ml-auto text-xs font-mono text-emerald-400">{passCount}/{checks.length} passed</span>
              </div>
              <ReviewChecks checks={checks} />
              {result.review?.feedback && (
                <p className="text-xs text-zinc-400 mt-3 pt-3 border-t border-zinc-700">{result.review.feedback}</p>
              )}
            </div>
          )}
        </div>
      )}

      {/* PR link */}
      {result.pr_url && (
        <div className="p-4 rounded-xl bg-green-950/20 border border-green-800/30 flex items-center gap-3">
          <GitPullRequest className="w-5 h-5 text-green-400 flex-shrink-0" />
          <a href={result.pr_url} target="_blank" rel="noopener noreferrer" className="text-sm text-green-400 hover:underline flex-1">
            {result.pr_url}
          </a>
        </div>
      )}
    </div>
  )
}
