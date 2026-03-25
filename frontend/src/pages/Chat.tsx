import { useState, useEffect, useRef } from 'react'
import {
  MessageCircle, Send, Loader2, AlertCircle, Trash2, Bot, User, Zap,
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { cn } from '../lib/utils'
import { chatAsk } from '../lib/api'
import type { ChatMessage } from '../lib/api'
import { useRepo } from '../lib/RepoContext'

// ─── Suggested questions ─────────────────────────────────────────────────────

const SUGGESTIONS = [
  'What is the overall architecture of this project?',
  'What are the main business rules?',
  'How does authentication work?',
  'What are the key API endpoints?',
  'What happens when a request fails?',
]

// ─── Sub-components ──────────────────────────────────────────────────────────

function EmptyState({ onSuggest }: { onSuggest: (q: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full px-4">
      <div className="w-16 h-16 rounded-2xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center mb-5">
        <Bot className="w-8 h-8 text-cyan-400" />
      </div>
      <h2 className="text-lg font-semibold text-zinc-200 mb-1">Ask about this repo</h2>
      <p className="text-sm text-zinc-500 text-center max-w-md mb-6">
        Ask questions about the codebase. Answers come exclusively from the
        generated context document — testing how well the context captures
        business logic and architecture.
      </p>
      <div className="flex flex-wrap gap-2 justify-center max-w-lg">
        {SUGGESTIONS.map((q) => (
          <button
            key={q}
            onClick={() => onSuggest(q)}
            className="px-3 py-1.5 rounded-lg bg-zinc-800 text-zinc-400 text-xs border border-zinc-700 hover:border-cyan-500/40 hover:text-cyan-300 transition-all"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  )
}

function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === 'user'
  return (
    <div className={cn('flex gap-3 max-w-4xl', isUser ? 'ml-auto flex-row-reverse' : 'mr-auto')}>
      <div
        className={cn(
          'w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5',
          isUser ? 'bg-blue-500/20' : 'bg-cyan-500/20'
        )}
      >
        {isUser ? (
          <User className="w-3.5 h-3.5 text-blue-400" />
        ) : (
          <Bot className="w-3.5 h-3.5 text-cyan-400" />
        )}
      </div>
      <div
        className={cn(
          'rounded-2xl px-4 py-3 text-sm leading-relaxed max-w-[85%]',
          isUser
            ? 'bg-blue-500/10 border border-blue-500/20 text-zinc-200'
            : 'bg-zinc-800/60 border border-zinc-700/50 text-zinc-300'
        )}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{msg.content}</p>
        ) : (
          <div className="prose prose-sm prose-invert max-w-none prose-p:my-1.5 prose-headings:mt-3 prose-headings:mb-1.5 prose-pre:bg-zinc-900 prose-pre:border prose-pre:border-zinc-700 prose-code:text-cyan-300 prose-code:before:content-none prose-code:after:content-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  )
}

function ThinkingBubble() {
  return (
    <div className="flex gap-3 max-w-4xl mr-auto">
      <div className="w-7 h-7 rounded-lg bg-cyan-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
        <Bot className="w-3.5 h-3.5 text-cyan-400" />
      </div>
      <div className="rounded-2xl px-4 py-3 bg-zinc-800/60 border border-zinc-700/50">
        <div className="flex items-center gap-2 text-zinc-500 text-sm">
          <Loader2 className="w-3.5 h-3.5 animate-spin" />
          Thinking...
        </div>
      </div>
    </div>
  )
}

// ─── Main Component ──────────────────────────────────────────────────────────

export default function Chat() {
  const { activeRepo } = useRepo()
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastUsage, setLastUsage] = useState<{ input_tokens: number; output_tokens: number } | null>(null)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Clear conversation when repo changes
  useEffect(() => {
    setMessages([])
    setError(null)
    setLastUsage(null)
  }, [activeRepo])

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 160) + 'px'
    }
  }, [input])

  const handleSend = async (question?: string) => {
    const q = (question ?? input).trim()
    if (!q || !activeRepo || loading) return

    setInput('')
    setError(null)

    const userMsg: ChatMessage = { role: 'user', content: q }
    const updatedMessages = [...messages, userMsg]
    setMessages(updatedMessages)
    setLoading(true)

    try {
      // Send history (all previous messages) + new question
      const res = await chatAsk(activeRepo, q, messages)
      const assistantMsg: ChatMessage = { role: 'assistant', content: res.answer }
      setMessages([...updatedMessages, assistantMsg])
      setLastUsage(res.usage)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to get response')
    } finally {
      setLoading(false)
      textareaRef.current?.focus()
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const clearConversation = () => {
    setMessages([])
    setError(null)
    setLastUsage(null)
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex-shrink-0 px-6 py-4 border-b border-zinc-700/50 bg-zinc-900 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-cyan-500/20 flex items-center justify-center">
            <MessageCircle className="w-4 h-4 text-cyan-400" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-zinc-100 flex items-center gap-2">
              Ask
              {activeRepo && (
                <span className="text-zinc-500 font-normal font-mono text-xs">— {activeRepo}</span>
              )}
            </h1>
            <p className="text-[11px] text-zinc-600">Answers from context document only</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {lastUsage && (
            <div className="flex items-center gap-1.5 text-[10px] text-zinc-600 font-mono">
              <Zap className="w-3 h-3" />
              {lastUsage.input_tokens.toLocaleString()}in / {lastUsage.output_tokens.toLocaleString()}out
            </div>
          )}
          {messages.length > 0 && (
            <button
              onClick={clearConversation}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800 transition-all"
            >
              <Trash2 className="w-3 h-3" />
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {messages.length === 0 && !loading ? (
          <EmptyState onSuggest={(q) => handleSend(q)} />
        ) : (
          <div className="space-y-4">
            {messages.map((msg, i) => (
              <MessageBubble key={i} msg={msg} />
            ))}
            {loading && <ThinkingBubble />}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Error banner */}
      {error && (
        <div className="flex-shrink-0 mx-6 mb-2 flex items-center gap-2 p-3 rounded-xl bg-red-950/30 border border-red-800/40 text-red-400 text-sm">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          {error}
        </div>
      )}

      {/* Input area */}
      <div className="flex-shrink-0 px-6 py-4 border-t border-zinc-700/50 bg-zinc-900">
        <div className="max-w-4xl mx-auto flex gap-3 items-end">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              activeRepo
                ? 'Ask a question about the codebase...'
                : 'Select a repository first...'
            }
            disabled={!activeRepo || loading}
            rows={1}
            className={cn(
              'flex-1 px-4 py-3 rounded-xl bg-zinc-800 border text-zinc-200 placeholder-zinc-600 text-sm focus:outline-none transition-all resize-none',
              !activeRepo
                ? 'border-zinc-700 opacity-50 cursor-not-allowed'
                : 'border-zinc-700 focus:border-cyan-500/70 focus:ring-1 focus:ring-cyan-500/20'
            )}
          />
          <button
            onClick={() => handleSend()}
            disabled={!input.trim() || !activeRepo || loading}
            className={cn(
              'flex items-center justify-center w-10 h-10 rounded-xl transition-all flex-shrink-0',
              !input.trim() || !activeRepo || loading
                ? 'bg-zinc-800 text-zinc-600 cursor-not-allowed'
                : 'bg-cyan-600 hover:bg-cyan-500 text-white shadow-lg shadow-cyan-500/20'
            )}
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>
        <p className="text-[10px] text-zinc-700 text-center mt-2">
          Shift+Enter for new line · Answers sourced exclusively from generated context
        </p>
      </div>
    </div>
  )
}
