import { useState, useEffect } from 'react'
import { Layers, AlertCircle } from 'lucide-react'
import { cn } from '../lib/utils'
import ContextLayers from '../components/ContextLayers'
import ContextViewer from '../components/ContextViewer'
import Hotspots from '../components/Hotspots'
import { getContextLayers, getContextFull, getHotspots } from '../lib/api'
import type { ContextLayersResponse, Hotspot } from '../lib/api'
import { useRepo } from '../lib/RepoContext'

type ContentTab = 'full' | 'summary'

export default function Context() {
  const { activeRepo } = useRepo()

  const [contextLayers, setContextLayers] = useState<ContextLayersResponse | null>(null)
  const [layersLoading, setLayersLoading] = useState(false)
  const [layersError, setLayersError] = useState<string | null>(null)

  const [fullContent, setFullContent] = useState<string | undefined>(undefined)
  const [fullLoading, setFullLoading] = useState(false)
  const [fullError, setFullError] = useState<string | null>(null)

  const [hotspots, setHotspots] = useState<Hotspot[]>([])
  const [hotspotsLoading, setHotspotsLoading] = useState(false)
  const [hotspotsError, setHotspotsError] = useState<string | null>(null)

  const [activeTab, setActiveTab] = useState<ContentTab>('full')

  useEffect(() => {
    if (!activeRepo) return

    setLayersLoading(true)
    setLayersError(null)
    getContextLayers(activeRepo)
      .then(setContextLayers)
      .catch((e: Error) => setLayersError(e.message))
      .finally(() => setLayersLoading(false))

    setFullLoading(true)
    setFullError(null)
    getContextFull(activeRepo)
      .then((r) => setFullContent(r.content))
      .catch((e: Error) => setFullError(e.message))
      .finally(() => setFullLoading(false))

    setHotspotsLoading(true)
    setHotspotsError(null)
    getHotspots(activeRepo, 20)
      .then(setHotspots)
      .catch((e: Error) => setHotspotsError(e.message))
      .finally(() => setHotspotsLoading(false))
  }, [activeRepo])

  if (!activeRepo) {
    return (
      <div className="flex items-center justify-center h-full p-12">
        <div className="text-center">
          <Layers className="w-12 h-12 text-zinc-700 mx-auto mb-4" />
          <p className="text-zinc-400">No repository selected</p>
          <p className="text-zinc-600 text-sm mt-1">Select or analyze a repository to view context layers</p>
        </div>
      </div>
    )
  }

  // Extract summary from full context (first 20% or until second heading)
  const summaryContent = (() => {
    if (!fullContent) return undefined
    const lines = fullContent.split('\n')
    let headingCount = 0
    const summaryLines: string[] = []
    for (const line of lines) {
      if (line.startsWith('## ') || line.startsWith('# ')) {
        headingCount++
        if (headingCount > 2) break
      }
      summaryLines.push(line)
      if (summaryLines.length > 80) break
    }
    return summaryLines.join('\n')
  })()

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      {/* Header */}
      <div className="flex-shrink-0 px-6 py-4 border-b border-zinc-700/50">
        <h1 className="text-lg font-bold text-zinc-100 flex items-center gap-2">
          <Layers className="w-5 h-5 text-blue-400" />
          Context Layers
          <span className="text-zinc-500 font-normal font-mono text-sm">— {activeRepo}</span>
        </h1>
      </div>

      <div className="flex-1 overflow-hidden flex flex-col lg:flex-row gap-0">
        {/* Left: Context Layers */}
        <div className="lg:w-[380px] flex-shrink-0 overflow-y-auto p-5 border-b lg:border-b-0 lg:border-r border-zinc-700/50">
          <h2 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-4">
            Layer Overview
          </h2>
          <ContextLayers
            data={contextLayers}
            loading={layersLoading}
            error={layersError}
          />
        </div>

        {/* Right: Context viewer + hotspots */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Tabs */}
          <div className="flex-shrink-0 flex items-center gap-1 px-5 pt-4 pb-0 border-b border-zinc-700/50">
            <div className="flex items-center gap-0.5 bg-zinc-800/60 rounded-lg p-1">
              {(['full', 'summary'] as ContentTab[]).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={cn(
                    'px-4 py-2 rounded-md text-xs font-semibold transition-all capitalize',
                    activeTab === tab
                      ? 'bg-zinc-700 text-zinc-100'
                      : 'text-zinc-400 hover:text-zinc-200'
                  )}
                >
                  {tab === 'full' ? 'Full Context' : 'Summary'}
                </button>
              ))}
            </div>
          </div>

          {/* Viewer */}
          <div className="flex-1 overflow-hidden">
            {(fullError || layersError) && (
              <div className="m-4 flex items-center gap-3 p-4 rounded-xl bg-red-950/30 border border-red-800/40 text-red-400">
                <AlertCircle className="w-5 h-5 flex-shrink-0" />
                <p className="text-sm">{fullError ?? layersError}</p>
              </div>
            )}
            <ContextViewer
              content={activeTab === 'full' ? fullContent : summaryContent}
              loading={fullLoading}
              error={!fullError ? null : fullError}
              title={activeTab === 'full' ? 'Full Context' : 'Summary'}
              filename={activeTab === 'full' ? `${activeRepo}-context.md` : `${activeRepo}-summary.md`}
            />
          </div>
        </div>
      </div>

      {/* Hotspots section */}
      <div className="flex-shrink-0 border-t border-zinc-700/50 p-5 max-h-72 overflow-y-auto">
        <h2 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-4">
          Top Hotspots
        </h2>
        <Hotspots
          hotspots={hotspots}
          loading={hotspotsLoading}
          error={hotspotsError}
        />
      </div>
    </div>
  )
}
