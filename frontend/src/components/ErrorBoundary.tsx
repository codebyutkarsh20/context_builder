import { Component, type ReactNode, type ErrorInfo } from 'react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[ErrorBoundary]', error, info.componentStack)
  }

  render() {
    if (this.state.error) {
      return this.props.fallback ?? (
        <div className="flex flex-col items-center justify-center h-full gap-3 text-center px-6">
          <div className="w-12 h-12 rounded-xl bg-red-950/40 border border-red-900/40 flex items-center justify-center">
            <svg className="w-6 h-6 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
            </svg>
          </div>
          <p className="text-sm font-medium text-red-400">Something went wrong</p>
          <p className="text-xs text-zinc-500 max-w-sm">{this.state.error.message}</p>
          <button
            onClick={() => this.setState({ error: null })}
            className="mt-1 px-3 py-1.5 rounded-lg text-xs bg-zinc-900 border border-zinc-700 text-zinc-400 hover:text-zinc-200 hover:border-zinc-600 transition-all"
          >
            Try again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
