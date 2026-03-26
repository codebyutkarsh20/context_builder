import { useState, useEffect, useRef, useCallback } from 'react'
import { Search as SearchIcon, FileCode, AlertCircle, Loader2 } from 'lucide-react'
import { cn } from '../lib/utils'
import { search } from '../lib/api'
import type { SearchResult, GraphNode } from '../lib/api'
import { useRepo } from '../lib/RepoContext'

// ─── Constants ────────────────────────────────────────────────────────────────

const TYPE_COLORS: Record<GraphNode['type'], string> = {
  File: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
  Class: 'bg-green-500/20 text-green-300 border-green-500/30',
  Function: 'bg-orange-500/20 text-orange-300 border-orange-500/30',
  BusinessRule: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
  DomainConcept: 'bg-pink-500/20 text-pink-300 border-pink-500/30',
  DecisionPoint: 'bg-red-500/20 text-red-300 border-red-500/30',
}

// ─── Sub-components ───────────────────────────────────────────────────────────

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

function EmptyState({ query }: { query: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 px-4">
      <div className="w-20 h-20 rounded-2xl bg-zinc-800/60 flex items-center justify-center mb-5">
        <SearchIcon className="w-10 h-10 text-zinc-600" />
      </div>
      {query ? (
        <>
          <p className="text-zinc-300 font-medium text-lg mb-1">No results found</p>
          <p className="text-zinc-500 text-sm text-center max-w-xs">
            No matches for <span className="font-mono text-zinc-400">"{query}"</span>. Try a different search term or check the spelling.
          </p>
        </>
      ) : (
        <>
          <p className="text-zinc-300 font-medium text-lg mb-1">Search your codebase</p>
          <p className="text-zinc-500 text-sm text-center max-w-xs">
            Type a function name, class, file, or concept to find relevant code elements.
          </p>
          <div className="mt-6 flex flex-wrap gap-2 justify-center">
            {['authentication', 'database', 'api endpoint', 'config', 'utils'].map((hint) => (
              <span
                key={hint}
                className="px-3 py-1 rounded-full bg-zinc-800 text-zinc-400 text-xs border border-zinc-700 cursor-default"
              >
                {hint}
              </span>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

interface ResultCardProps {
  result: SearchResult
  query: string
}

function highlightMatch(text: string, query: string): React.ReactNode {
  if (!query || !text) return text
  const idx = text.toLowerCase().indexOf(query.toLowerCase())
  if (idx === -1) return text
  return (
    <>
      {text.slice(0, idx)}
      <mark className="bg-yellow-500/30 text-yellow-200 rounded-sm px-0.5">{text.slice(idx, idx + query.length)}</mark>
      {text.slice(idx + query.length)}
    </>
  )
}

function ResultCard({ result, query }: ResultCardProps) {
  return (
    <div className="p-4 rounded-xl bg-zinc-800/40 border border-zinc-700/50 hover:border-zinc-600/60 hover:bg-zinc-800/60 transition-all cursor-pointer group">
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex items-center gap-2 min-w-0">
          <TypeBadge type={result.type} />
          <h3 className="text-sm font-semibold text-zinc-100 font-mono truncate group-hover:text-white transition-colors">
            {highlightMatch(result.name, query)}
          </h3>
        </div>
        <span className="text-xs text-zinc-500 flex-shrink-0 font-mono">
          {Math.min(Math.round(result.score * 100), 100)}% match
        </span>
      </div>

      {result.file && (
        <div className="flex items-center gap-1.5 mb-2">
          <FileCode className="w-3 h-3 text-zinc-600 flex-shrink-0" />
          <span className="text-xs text-zinc-400 font-mono truncate">
            {highlightMatch(result.file, query)}
          </span>
        </div>
      )}

      {result.snippet && (
        <div className="mt-2 p-2 rounded-md bg-zinc-900/60 border border-zinc-700/30">
          <p className="text-xs text-zinc-400 leading-relaxed font-mono line-clamp-3">
            {highlightMatch(result.snippet, query)}
          </p>
        </div>
      )}
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function Search() {
  const { activeRepo } = useRepo()
  const [query, setQuery] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [searched, setSearched] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  // Debounce query
  useEffect(() => {
    const t = setTimeout(() => setDebouncedQuery(query), 300)
    return () => clearTimeout(t)
  }, [query])

  // Execute search
  const executeSearch = useCallback(
    async (q: string) => {
      if (!activeRepo || !q.trim()) {
        setResults([])
        setError(null)
        setSearched(false)
        return
      }
      setLoading(true)
      setError(null)
      try {
        const data = await search(activeRepo, q.trim())
        setResults(data)
        setSearched(true)
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Search failed')
        setResults([])
      } finally {
        setLoading(false)
      }
    },
    [activeRepo]
  )

  useEffect(() => {
    executeSearch(debouncedQuery)
  }, [debouncedQuery, executeSearch])

  // Type filter state
  const [typeFilter, setTypeFilter] = useState<Set<string>>(
    new Set(['File', 'Class', 'Function', 'BusinessRule', 'DomainConcept'])
  )

  const toggleType = (type: string) => {
    setTypeFilter((prev) => {
      const next = new Set(prev)
      if (next.has(type)) next.delete(type)
      else next.add(type)
      return next
    })
  }

  const filteredResults = results.filter((r) => typeFilter.has(r.type))

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      {/* Search header */}
      <div className="flex-shrink-0 px-6 py-5 border-b border-zinc-700/50 bg-zinc-900">
        <h1 className="text-lg font-bold text-zinc-100 mb-4 flex items-center gap-2">
          <SearchIcon className="w-5 h-5 text-blue-400" />
          Search
          {activeRepo && (
            <span className="text-zinc-500 font-normal font-mono text-sm">— {activeRepo}</span>
          )}
        </h1>

        {/* Search input */}
        <div className="relative max-w-2xl">
          <SearchIcon className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500 pointer-events-none" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={
              activeRepo
                ? `Search in ${activeRepo}...`
                : 'Select a repository first...'
            }
            disabled={!activeRepo}
            className={cn(
              'w-full pl-10 pr-12 py-3 rounded-xl bg-zinc-800 border text-zinc-200 placeholder-zinc-500 text-sm focus:outline-none transition-all',
              !activeRepo
                ? 'border-zinc-700 opacity-50 cursor-not-allowed'
                : 'border-zinc-700 focus:border-blue-500 focus:ring-1 focus:ring-blue-500/20'
            )}
          />
          {loading && (
            <div className="absolute right-3.5 top-1/2 -translate-y-1/2">
              <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
            </div>
          )}
          {!loading && query && (
            <button
              onClick={() => setQuery('')}
              className="absolute right-3.5 top-1/2 -translate-y-1/2 w-5 h-5 rounded flex items-center justify-center text-zinc-500 hover:text-zinc-300 hover:bg-zinc-700 transition-colors"
            >
              ×
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-hidden flex">
        {/* Results */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {/* Error */}
          {error && (
            <div className="flex items-center gap-3 mb-4 p-4 rounded-xl bg-red-950/30 border border-red-800/40 text-red-400">
              <AlertCircle className="w-5 h-5 flex-shrink-0" />
              <p className="text-sm">{error}</p>
            </div>
          )}

          {/* Results count */}
          {searched && !loading && (
            <p className="text-xs text-zinc-500 mb-4">
              {filteredResults.length === 0
                ? 'No results'
                : `${filteredResults.length} result${filteredResults.length !== 1 ? 's' : ''}`}
              {query && (
                <> for <span className="font-mono text-zinc-400">"{query}"</span></>
              )}
            </p>
          )}

          {/* Results list or empty state */}
          {!loading && filteredResults.length === 0 ? (
            <EmptyState query={searched ? query : ''} />
          ) : (
            <div className="space-y-3">
              {filteredResults.map((result) => (
                <ResultCard key={result.id} result={result} query={query} />
              ))}
            </div>
          )}
        </div>

        {/* Sidebar: filters */}
        <div className="w-52 flex-shrink-0 border-l border-zinc-700/50 p-4 overflow-y-auto">
          <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">
            Filter by Type
          </h3>
          <div className="space-y-2">
            {(['File', 'Class', 'Function', 'BusinessRule', 'DomainConcept'] as const).map((type) => {
              const count = results.filter((r) => r.type === type).length
              return (
                <label key={type} className="flex items-center gap-2 cursor-pointer group">
                  <input
                    type="checkbox"
                    checked={typeFilter.has(type)}
                    onChange={() => toggleType(type)}
                    className="w-3.5 h-3.5 rounded border-zinc-600 bg-zinc-800 text-blue-500 focus:ring-0 focus:ring-offset-0"
                  />
                  <span className="text-sm text-zinc-300 group-hover:text-zinc-100 transition-colors flex-1">
                    {type}
                  </span>
                  {searched && (
                    <span className="text-xs text-zinc-600 font-mono">{count}</span>
                  )}
                </label>
              )
            })}
          </div>

          {searched && results.length > 0 && (
            <div className="mt-6 pt-4 border-t border-zinc-800">
              <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">
                Score Range
              </h3>
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs text-zinc-500">
                  <span>Min</span>
                  <span className="font-mono">
                    {(Math.min(...results.map((r) => r.score)) * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex justify-between text-xs text-zinc-500">
                  <span>Max</span>
                  <span className="font-mono">
                    {(Math.max(...results.map((r) => r.score)) * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex justify-between text-xs text-zinc-500">
                  <span>Avg</span>
                  <span className="font-mono">
                    {(
                      (results.reduce((sum, r) => sum + r.score, 0) / results.length) * 100
                    ).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
