import { useState } from 'react'
import { FileCode, ChevronDown, ChevronRight } from 'lucide-react'
import { CodeDiff } from './CodeDiff'

export function PatchCard({ patch }: { patch: { file_path: string; original_code: string; patched_code: string; explanation: string } }) {
  const [showDiff, setShowDiff] = useState(false)
  return (
    <div className="rounded-lg bg-zinc-800/60 border border-zinc-700/30 overflow-hidden">
      <div
        className="flex items-center justify-between px-3 py-2 cursor-pointer hover:bg-zinc-800/80 transition-colors"
        onClick={() => setShowDiff(!showDiff)}
      >
        <div className="flex items-center gap-2 min-w-0">
          <FileCode className="w-3.5 h-3.5 text-orange-400 flex-shrink-0" />
          <span className="text-xs font-mono text-zinc-300 truncate">{patch.file_path}</span>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="text-[10px] text-zinc-600">
            {patch.original_code ? `${patch.original_code.split('\n').length} lines` : 'new file'}
          </span>
          {showDiff ? <ChevronDown className="w-3 h-3 text-zinc-600" /> : <ChevronRight className="w-3 h-3 text-zinc-600" />}
        </div>
      </div>
      {patch.explanation && (
        <p className="px-3 pb-2 text-xs text-zinc-500">{patch.explanation}</p>
      )}
      {showDiff && patch.original_code && patch.patched_code && (
        <CodeDiff original={patch.original_code} patched={patch.patched_code} filePath={patch.file_path} />
      )}
      {showDiff && !patch.original_code && patch.patched_code && (
        <pre className="px-3 py-2 text-[11px] font-mono text-green-400 bg-green-950/20 max-h-60 overflow-y-auto whitespace-pre">
          {patch.patched_code.slice(0, 3000)}
        </pre>
      )}
    </div>
  )
}
