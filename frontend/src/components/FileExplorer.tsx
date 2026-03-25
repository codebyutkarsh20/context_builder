import { useState } from 'react'
import {
  ChevronRight,
  ChevronDown,
  FileCode,
  FileText,
  File,
  Folder,
  FolderOpen,
  Loader2,
  AlertCircle,
} from 'lucide-react'
import { cn } from '../lib/utils'
import type { GraphNode } from '../lib/api'

// ─── Types ────────────────────────────────────────────────────────────────────

export interface FileTreeNode {
  name: string
  path: string
  type: 'file' | 'directory'
  language?: string
  pagerank?: number
  children?: FileTreeNode[]
  nodeDetails?: {
    classes: string[]
    functions: string[]
  }
}

interface FileExplorerProps {
  tree: FileTreeNode[]
  nodes?: GraphNode[]
  loading?: boolean
  error?: string | null
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

const LANGUAGE_ICONS: Record<string, string> = {
  python: '🐍',
  typescript: '📘',
  javascript: '📒',
  java: '☕',
  go: '🐹',
  rust: '🦀',
  cpp: '⚙️',
  c: '⚙️',
  ruby: '💎',
  php: '🐘',
  swift: '🍎',
  kotlin: '🟣',
  scala: '🔴',
  html: '🌐',
  css: '🎨',
  json: '📋',
  yaml: '📄',
  markdown: '📝',
}

function getFileIcon(name: string, language?: string) {
  if (language && LANGUAGE_ICONS[language.toLowerCase()]) {
    return (
      <span className="text-sm leading-none">{LANGUAGE_ICONS[language.toLowerCase()]}</span>
    )
  }

  const ext = name.split('.').pop()?.toLowerCase()
  if (!ext) return <File className="w-3.5 h-3.5 text-zinc-500" />

  if (['ts', 'tsx'].includes(ext)) return <FileCode className="w-3.5 h-3.5 text-blue-400" />
  if (['js', 'jsx', 'mjs'].includes(ext)) return <FileCode className="w-3.5 h-3.5 text-yellow-400" />
  if (['py'].includes(ext)) return <FileCode className="w-3.5 h-3.5 text-green-400" />
  if (['md', 'mdx'].includes(ext)) return <FileText className="w-3.5 h-3.5 text-zinc-400" />
  if (['json', 'yaml', 'yml', 'toml'].includes(ext)) return <FileText className="w-3.5 h-3.5 text-orange-400" />

  return <FileCode className="w-3.5 h-3.5 text-zinc-500" />
}

function pagerankColor(pr?: number): string {
  if (!pr) return 'bg-zinc-700 text-zinc-400'
  if (pr > 0.01) return 'bg-red-900/60 text-red-300'
  if (pr > 0.005) return 'bg-orange-900/60 text-orange-300'
  if (pr > 0.001) return 'bg-yellow-900/60 text-yellow-300'
  return 'bg-zinc-700 text-zinc-400'
}

// ─── File Detail Panel ────────────────────────────────────────────────────────

interface FileDetailPanelProps {
  node: FileTreeNode
  onClose: () => void
}

function FileDetailPanel({ node, onClose }: FileDetailPanelProps) {
  return (
    <div className="mt-1 ml-6 rounded-lg bg-zinc-800/60 border border-zinc-700/50 p-3 text-xs space-y-2">
      <div className="flex items-center justify-between">
        <span className="font-medium text-zinc-300">{node.name}</span>
        <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300 text-xs px-1">
          ✕
        </button>
      </div>

      {node.pagerank !== undefined && (
        <div className="flex items-center gap-2">
          <span className="text-zinc-500">PageRank:</span>
          <span className="font-mono text-zinc-300">{node.pagerank.toFixed(5)}</span>
        </div>
      )}

      {node.nodeDetails && node.nodeDetails.classes.length > 0 && (
        <div>
          <p className="text-zinc-500 mb-1">Classes ({node.nodeDetails.classes.length})</p>
          <ul className="space-y-0.5">
            {node.nodeDetails.classes.map((cls) => (
              <li key={cls} className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-green-500 flex-shrink-0" />
                <span className="font-mono text-green-300">{cls}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {node.nodeDetails && node.nodeDetails.functions.length > 0 && (
        <div>
          <p className="text-zinc-500 mb-1">
            Functions ({node.nodeDetails.functions.length})
          </p>
          <ul className="space-y-0.5 max-h-32 overflow-y-auto">
            {node.nodeDetails.functions.map((fn) => (
              <li key={fn} className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-sm bg-orange-500 flex-shrink-0" />
                <span className="font-mono text-orange-300">{fn}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

// ─── Tree Node ────────────────────────────────────────────────────────────────

interface TreeNodeProps {
  node: FileTreeNode
  depth: number
}

function TreeNode({ node, depth }: TreeNodeProps) {
  const [expanded, setExpanded] = useState(depth < 2)
  const [showDetail, setShowDetail] = useState(false)

  const isDir = node.type === 'directory'

  return (
    <div>
      <div
        className={cn(
          'flex items-center gap-1.5 py-1 px-2 rounded-md cursor-pointer hover:bg-zinc-800/60 transition-colors group',
        )}
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
        onClick={() => {
          if (isDir) setExpanded((e) => !e)
          else setShowDetail((s) => !s)
        }}
      >
        {/* Expand/Collapse chevron */}
        <span className="w-3 h-3 flex items-center justify-center flex-shrink-0">
          {isDir ? (
            expanded ? (
              <ChevronDown className="w-3 h-3 text-zinc-500" />
            ) : (
              <ChevronRight className="w-3 h-3 text-zinc-500" />
            )
          ) : null}
        </span>

        {/* Icon */}
        <span className="flex-shrink-0">
          {isDir ? (
            expanded ? (
              <FolderOpen className="w-3.5 h-3.5 text-yellow-400/70" />
            ) : (
              <Folder className="w-3.5 h-3.5 text-yellow-400/70" />
            )
          ) : (
            getFileIcon(node.name, node.language)
          )}
        </span>

        {/* Name */}
        <span
          className={cn(
            'text-xs flex-1 truncate min-w-0',
            isDir ? 'text-zinc-300 font-medium' : 'text-zinc-400 group-hover:text-zinc-200'
          )}
        >
          {node.name}
        </span>

        {/* PageRank badge */}
        {!isDir && node.pagerank !== undefined && node.pagerank > 0 && (
          <span
            className={cn(
              'text-[10px] font-mono px-1.5 py-0.5 rounded flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity',
              pagerankColor(node.pagerank)
            )}
          >
            {node.pagerank.toFixed(3)}
          </span>
        )}
      </div>

      {/* File detail panel */}
      {!isDir && showDetail && node.nodeDetails && (
        <div style={{ paddingLeft: `${depth * 12 + 8}px` }}>
          <FileDetailPanel node={node} onClose={() => setShowDetail(false)} />
        </div>
      )}

      {/* Children */}
      {isDir && expanded && node.children && (
        <div>
          {node.children.map((child) => (
            <TreeNode key={child.path} node={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function FileExplorer({ tree, loading, error }: FileExplorerProps) {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-32">
        <div className="flex items-center gap-3">
          <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
          <p className="text-sm text-zinc-400">Loading file tree...</p>
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

  if (tree.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-32 gap-3">
        <Folder className="w-8 h-8 text-zinc-700" />
        <p className="text-sm text-zinc-500">No files to display</p>
      </div>
    )
  }

  return (
    <div className="space-y-0.5">
      {tree.map((node) => (
        <TreeNode key={node.path} node={node} depth={0} />
      ))}
    </div>
  )
}
