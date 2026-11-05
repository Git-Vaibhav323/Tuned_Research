import { useEffect, useState } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { AlertTriangle, X } from 'lucide-react'
import Sidebar from './components/Sidebar'
import Chat      from './pages/Chat'
import Analyse   from './pages/Analyse'
import Compare   from './pages/Compare'
import Gaps      from './pages/Gaps'
import Sources   from './pages/Sources'
import Browse    from './pages/Browse'
import Stats     from './pages/Stats'
import IndexMgmt from './pages/IndexMgmt'
import { getHealth } from './api'

export default function App() {
  const [health,          setHealth]          = useState<any>(null)
  const [bannerDismissed, setBannerDismissed] = useState(false)

  useEffect(() => {
    getHealth()
      .then(d => { setHealth(d); setBannerDismissed(false) })
      .catch(() => setHealth({ ollama: false, n_chunks: 0, n_docs: 0 }))

    const t = setInterval(() => {
      getHealth()
        .then(d => {
          // If Ollama comes back online after being offline, re-show the banner
          setHealth((prev: any) => {
            if (prev && !prev.ollama && d.ollama) setBannerDismissed(false)
            return d
          })
        })
        .catch(() => {})
    }, 15000)
    return () => clearInterval(t)
  }, [])

  // Show banner when Ollama is confirmed offline (health loaded but ollama=false)
  const showOllamaBanner = health !== null && !health.ollama && !bannerDismissed

  return (
    <BrowserRouter>
      <div className="flex flex-col h-screen overflow-hidden">

        {/* ── Ollama offline banner ─────────────────────────────────────────── */}
        {showOllamaBanner && (
          <div className="shrink-0 flex items-center gap-3 bg-red-950 border-b border-red-800
                          px-4 py-2.5 text-sm text-red-300 z-50">
            <AlertTriangle size={15} className="text-red-400 shrink-0" />
            <span className="flex-1">
              <strong className="text-red-200">Ollama is unreachable.</strong>
              {' '}RAG chat and generation features are disabled.
              Start Ollama with{' '}
              <code className="bg-red-900/60 px-1.5 py-0.5 rounded text-xs font-mono text-red-200">
                ollama serve
              </code>
              {' '}then reload.
            </span>
            <button
              onClick={() => setBannerDismissed(true)}
              className="shrink-0 text-red-500 hover:text-red-300 transition-colors p-0.5 rounded"
              aria-label="Dismiss"
            >
              <X size={14} />
            </button>
          </div>
        )}

        <div className="flex flex-1 overflow-hidden">
          <Sidebar health={health} />
          <main className="flex-1 overflow-y-auto">
            <Routes>
              <Route path="/"        element={<Chat   ollama={!!health?.ollama} />} />
              <Route path="/analyse" element={<Analyse />} />
              <Route path="/compare" element={<Compare />} />
              <Route path="/gaps"    element={<Gaps />} />
              <Route path="/sources" element={<Sources />} />
              <Route path="/browse"  element={<Browse />} />
              <Route path="/stats"   element={<Stats />} />
              <Route path="/index"   element={<IndexMgmt />} />
            </Routes>
          </main>
        </div>

      </div>
    </BrowserRouter>
  )
}
