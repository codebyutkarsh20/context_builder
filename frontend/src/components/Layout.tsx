import { ReactNode, useState } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import {
  LayoutDashboard, BookOpen,
  ChevronDown, Check, Plus, Loader2, Cpu, Trash2,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { useRepo } from '../lib/RepoContext'
import { deleteRepo } from '../lib/api'

interface LayoutProps {
  children: ReactNode
}

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Overview', color: 'text-blue-400' },
  { to: '/agent', icon: Cpu, label: 'Agent', color: 'text-rose-400' },
  { to: '/knowledge', icon: BookOpen, label: 'Knowledge', color: 'text-purple-400' },
]

export default function Layout({ children }: LayoutProps) {
  const { repos, activeRepo, setActiveRepo, loading, refresh } = useRepo()
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [deleting, setDeleting] = useState<string | null>(null)
  const navigate = useNavigate()

  const handleDelete = async (repoName: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (!confirm(`Delete all data for "${repoName}"? This cannot be undone.`)) return
    setDeleting(repoName)
    try {
      await deleteRepo(repoName)
      if (activeRepo === repoName) setActiveRepo('')
      refresh()
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to delete repo')
    } finally {
      setDeleting(null)
    }
  }

  return (
    <div className="flex h-screen bg-zinc-950 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-60 flex-shrink-0 flex flex-col bg-zinc-950 border-r border-zinc-800/80">
        {/* Logo */}
        <div className="flex items-center gap-3 px-4 py-5">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-blue-500 to-blue-700 flex items-center justify-center flex-shrink-0 shadow-lg shadow-blue-500/20">
            <Cpu className="w-4 h-4 text-white" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-zinc-100 leading-tight">Context Builder</h1>
            <p className="text-[11px] text-zinc-600 leading-tight">Code Intelligence</p>
          </div>
        </div>

        {/* Repo Selector */}
        <div className="px-3 mb-2">
          <p className="text-[10px] font-bold text-zinc-600 uppercase tracking-widest mb-1.5 px-1">
            Active Repository
          </p>
          <div className="relative">
            <button
              onClick={() => setDropdownOpen((o) => !o)}
              className="w-full flex items-center justify-between gap-2 px-3 py-2 rounded-xl bg-zinc-900 border border-zinc-800 hover:border-zinc-700 transition-all text-sm text-zinc-200"
            >
              {loading ? (
                <span className="flex items-center gap-2 text-zinc-500">
                  <Loader2 className="w-3 h-3 animate-spin" />Loading...
                </span>
              ) : activeRepo ? (
                <div className="flex items-center gap-2 min-w-0">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 flex-shrink-0" />
                  <span className="truncate font-medium text-zinc-200 font-mono text-xs">{activeRepo}</span>
                </div>
              ) : (
                <span className="text-zinc-600 text-xs">No repo selected</span>
              )}
              <ChevronDown className={cn('w-3.5 h-3.5 text-zinc-500 flex-shrink-0 transition-transform', dropdownOpen && 'rotate-180')} />
            </button>

            {dropdownOpen && (
              <div className="absolute top-full left-0 right-0 mt-1.5 bg-zinc-900 border border-zinc-800 rounded-xl shadow-2xl z-50 overflow-hidden">
                {repos.length === 0 ? (
                  <div className="px-3 py-3 text-xs text-zinc-600 text-center">No repositories yet</div>
                ) : (
                  <ul className="py-1 max-h-48 overflow-y-auto">
                    {repos.map((repo) => (
                      <li key={repo.name}>
                        <div className="flex items-center group">
                          <button
                            onClick={() => { setActiveRepo(repo.name); setDropdownOpen(false) }}
                            className="flex-1 flex items-center justify-between px-3 py-2 text-sm hover:bg-zinc-800 transition-colors min-w-0"
                          >
                            <div className="flex items-center gap-2 min-w-0">
                              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 flex-shrink-0" />
                              <span className="truncate text-zinc-300 font-mono text-xs">{repo.name}</span>
                            </div>
                            {activeRepo === repo.name && <Check className="w-3.5 h-3.5 text-blue-400 flex-shrink-0" />}
                          </button>
                          <button
                            onClick={(e) => handleDelete(repo.name, e)}
                            disabled={deleting === repo.name}
                            className="px-2 py-2 opacity-0 group-hover:opacity-100 text-zinc-600 hover:text-red-400 transition-all flex-shrink-0"
                            title={`Delete ${repo.name}`}
                          >
                            {deleting === repo.name
                              ? <Loader2 className="w-3 h-3 animate-spin" />
                              : <Trash2 className="w-3 h-3" />}
                          </button>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
                <div className="border-t border-zinc-800 p-1">
                  <button
                    onClick={() => { setDropdownOpen(false); navigate('/') }}
                    className="w-full flex items-center gap-2 px-3 py-2 text-xs text-blue-400 hover:bg-zinc-800 rounded-lg transition-colors font-medium"
                  >
                    <Plus className="w-3.5 h-3.5" />Analyze new repo
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-2">
          <p className="text-[10px] font-bold text-zinc-600 uppercase tracking-widest mb-1.5 px-1">Navigation</p>
          <ul className="space-y-0.5">
            {navItems.map(({ to, icon: Icon, label, color }) => (
              <li key={to}>
                <NavLink
                  to={to}
                  end={to === '/'}
                  className={({ isActive }) =>
                    cn('flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all',
                      isActive ? 'bg-zinc-800/80 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900')
                  }
                >
                  {({ isActive }) => (
                    <>
                      <Icon className={cn('w-4 h-4 flex-shrink-0 transition-colors', isActive ? color : 'text-zinc-600')} />
                      {label}
                    </>
                  )}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-zinc-800/60">
          <p className="text-[10px] text-zinc-700 font-mono">v0.1.0 · AI Deploy Agent</p>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 bg-zinc-950 overflow-hidden flex flex-col">
        {dropdownOpen && <div className="fixed inset-0 z-40" onClick={() => setDropdownOpen(false)} />}
        <div className="flex-1 overflow-hidden">
          {children}
        </div>
      </main>
    </div>
  )
}
