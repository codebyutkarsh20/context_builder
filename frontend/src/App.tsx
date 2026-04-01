import { Routes, Route, Navigate } from 'react-router-dom'
import { RepoProvider } from './lib/RepoContext'
import Layout from './components/Layout'
import { ErrorBoundary } from './components/ErrorBoundary'
import Overview from './pages/Overview'
import AgentPage from './pages/Agent'
import KnowledgePage from './pages/Knowledge'

export default function App() {
  return (
    <RepoProvider>
      <Layout>
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/agent" element={<AgentPage />} />
            <Route path="/knowledge" element={<KnowledgePage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </ErrorBoundary>
      </Layout>
    </RepoProvider>
  )
}
