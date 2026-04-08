import { useState } from 'react'
import { FileCode } from 'lucide-react'
import { cn } from '../../lib/utils'

export function CodeDiff({ original, patched, filePath }: { original: string; patched: string; filePath: string }) {
  const [view, setView] = useState<'diff' | 'original' | 'patched'>('diff')

  const origLines = original.split('\n')
  const patchedLines = patched.split('\n')

  const maxLen = Math.max(origLines.length, patchedLines.length)
  const diffLines: { type: 'same' | 'removed' | 'added' | 'changed'; orig: string; patched: string; lineNum: number }[] = []

  for (let i = 0; i < maxLen; i++) {
    const o = origLines[i] ?? ''
    const p = patchedLines[i] ?? ''
    if (o === p) {
      diffLines.push({ type: 'same', orig: o, patched: p, lineNum: i + 1 })
    } else if (i >= origLines.length) {
      diffLines.push({ type: 'added', orig: '', patched: p, lineNum: i + 1 })
    } else if (i >= patchedLines.length) {
      diffLines.push({ type: 'removed', orig: o, patched: '', lineNum: i + 1 })
    } else {
      diffLines.push({ type: 'changed', orig: o, patched: p, lineNum: i + 1 })
    }
  }

  return (
    <div className="rounded-lg bg-zinc-950 border border-zinc-800 overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b border-zinc-800 bg-zinc-900/50">
        <span className="text-xs font-mono text-zinc-400 flex items-center gap-1.5">
          <FileCode className="w-3 h-3 text-orange-400" />
          {filePath}
        </span>
        <div className="flex gap-1">
          {(['diff', 'original', 'patched'] as const).map(v => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={cn('px-2 py-0.5 rounded text-[10px] font-medium transition-colors',
                view === v ? 'bg-zinc-700 text-zinc-200' : 'text-zinc-600 hover:text-zinc-400'
              )}
            >{v === 'diff' ? 'Diff' : v === 'original' ? 'Before' : 'After'}</button>
          ))}
        </div>
      </div>
      <div className="max-h-80 overflow-y-auto overflow-x-auto">
        {view === 'diff' ? (
          <div className="text-[11px] font-mono leading-5">
            {diffLines.filter(l => l.type !== 'same' || diffLines.filter(d => d.type !== 'same').length < 20).map((line, i) => {
              if (line.type === 'same') return (
                <div key={i} className="px-3 py-0 text-zinc-600 flex">
                  <span className="w-8 text-right text-zinc-700 select-none mr-3 flex-shrink-0">{line.lineNum}</span>
                  <span className="whitespace-pre">{line.orig}</span>
                </div>
              )
              if (line.type === 'removed' || line.type === 'changed') return (
                <div key={`r-${i}`} className="px-3 py-0 bg-red-950/30 text-red-300 flex">
                  <span className="w-8 text-right text-red-800 select-none mr-3 flex-shrink-0">{line.lineNum}</span>
                  <span className="select-none text-red-600 mr-1">-</span>
                  <span className="whitespace-pre">{line.orig}</span>
                </div>
              )
              return null
            })}
            {diffLines.filter(l => l.type !== 'same').map((line, i) => {
              if (line.type === 'added' || line.type === 'changed') return (
                <div key={`a-${i}`} className="px-3 py-0 bg-green-950/30 text-green-300 flex">
                  <span className="w-8 text-right text-green-800 select-none mr-3 flex-shrink-0">{line.lineNum}</span>
                  <span className="select-none text-green-600 mr-1">+</span>
                  <span className="whitespace-pre">{line.patched}</span>
                </div>
              )
              return null
            })}
          </div>
        ) : (
          <pre className="px-3 py-2 text-[11px] font-mono text-zinc-400 whitespace-pre overflow-x-auto">
            {view === 'original' ? original : patched}
          </pre>
        )}
      </div>
    </div>
  )
}
