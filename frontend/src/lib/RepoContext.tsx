import { createContext, useContext, useState, useEffect, useMemo, ReactNode } from 'react'
import { listRepos, type Repo } from './api'

interface RepoContextValue {
  repos: Repo[]
  activeRepo: string
  activeRepoData: Repo | null
  setActiveRepo: (repo: string) => void
  loading: boolean
  error: string | null
  refresh: () => void
}

const RepoContext = createContext<RepoContextValue>({
  repos: [],
  activeRepo: '',
  activeRepoData: null,
  setActiveRepo: () => {},
  loading: false,
  error: null,
  refresh: () => {},
})

const REPO_STORAGE_KEY = 'active_repo'

export function RepoProvider({ children }: { children: ReactNode }) {
  const [repos, setRepos] = useState<Repo[]>([])
  const [activeRepo, setActiveRepoRaw] = useState<string>(() => {
    try { return localStorage.getItem(REPO_STORAGE_KEY) || '' } catch { return '' }
  })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const setActiveRepo = (repo: string) => {
    setActiveRepoRaw(repo)
    try { localStorage.setItem(REPO_STORAGE_KEY, repo) } catch {}
  }

  const refresh = () => {
    setLoading(true)
    setError(null)
    listRepos()
      .then((data) => {
        setRepos(data)
        // Only default to first repo if nothing is saved or saved repo no longer exists
        if ((!activeRepo || !data.find(r => r.name === activeRepo)) && data.length > 0) {
          setActiveRepo(data[0].name)
        }
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    refresh()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const activeRepoData = useMemo(
    () => repos.find(r => r.name === activeRepo) ?? null,
    [repos, activeRepo],
  )

  const value = useMemo(() => ({
    repos, activeRepo, activeRepoData, setActiveRepo, loading, error, refresh,
  }), [repos, activeRepo, activeRepoData, loading, error]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <RepoContext.Provider value={value}>
      {children}
    </RepoContext.Provider>
  )
}

export function useRepo() {
  return useContext(RepoContext)
}
