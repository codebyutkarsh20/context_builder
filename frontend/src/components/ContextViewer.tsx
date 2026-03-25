import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Copy, Download, Check, Loader2, AlertCircle, FileText } from 'lucide-react'
import { cn } from '../lib/utils'

interface ContextViewerProps {
  content?: string
  loading?: boolean
  error?: string | null
  title?: string
  filename?: string
}

export default function ContextViewer({
  content,
  loading,
  error,
  title,
  filename = 'context.md',
}: ContextViewerProps) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    if (!content) return
    await navigator.clipboard.writeText(content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleDownload = () => {
    if (!content) return
    const blob = new Blob([content], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-48">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="w-6 h-6 text-blue-400 animate-spin" />
          <p className="text-sm text-zinc-400">Loading context...</p>
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

  if (!content) {
    return (
      <div className="flex flex-col items-center justify-center h-48 gap-3">
        <FileText className="w-10 h-10 text-zinc-700" />
        <p className="text-sm text-zinc-500">No context available</p>
        <p className="text-xs text-zinc-600">Analyze a repository to generate context</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-700/50 flex-shrink-0">
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-zinc-500" />
          <span className="text-sm font-medium text-zinc-300">{title ?? 'Context'}</span>
          <span className="text-xs text-zinc-500 font-mono">
            {(content.length / 1000).toFixed(1)}K chars
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleCopy}
            className={cn(
              'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all',
              copied
                ? 'bg-green-900/40 text-green-300 border border-green-700/40'
                : 'bg-zinc-800 text-zinc-300 border border-zinc-700 hover:bg-zinc-700 hover:text-zinc-100'
            )}
          >
            {copied ? (
              <>
                <Check className="w-3 h-3" />
                Copied!
              </>
            ) : (
              <>
                <Copy className="w-3 h-3" />
                Copy
              </>
            )}
          </button>
          <button
            onClick={handleDownload}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-zinc-800 text-zinc-300 border border-zinc-700 hover:bg-zinc-700 hover:text-zinc-100 transition-all"
          >
            <Download className="w-3 h-3" />
            Download
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="prose-dark max-w-none">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
        </div>
      </div>
    </div>
  )
}
