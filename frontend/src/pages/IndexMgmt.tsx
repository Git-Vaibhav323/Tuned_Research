import { useState, useEffect } from 'react'
import { getStats, buildIndex, resetIndex } from '../api'
import PageHeader from '../components/PageHeader'

export default function IndexMgmt() {
  const [stats,    setStats]    = useState<any>(null)
  const [limit,    setLimit]    = useState(0)
  const [building, setBuilding] = useState(false)
  const [result,   setResult]   = useState<any>(null)
  const [confirm,  setConfirm]  = useState(false)
  const [resetting,setResetting]= useState(false)
  const [msg,      setMsg]      = useState('')

  const load = () => getStats().then(setStats)
  useEffect(() => { load() }, [])

  const build = async () => {
    setBuilding(true); setResult(null); setMsg('')
    const res = await buildIndex(limit)
    setResult(res)
    setBuilding(false)
    load()
  }

  const reset = async () => {
    if (!confirm) { setConfirm(true); return }
    setResetting(true)
    await resetIndex()
    setMsg('Index cleared.')
    setConfirm(false)
    setResetting(false)
    load()
  }

  const idx = stats?.index

  return (
    <div className="p-6 max-w-3xl">
      <PageHeader icon="⚙️" title="Index Management"
        sub="Build and manage the RAG vector index — idempotent, already-indexed chunks are skipped" />

      {/* Live stats */}
      {idx && (
        <div className="grid grid-cols-3 gap-4 mb-6">
          {[
            ['Total chunks',  idx.total_chunks],
            ['Unique papers', idx.unique_docs],
            ['Ollama',        stats.ollama ? '🟢 Ready' : '🔴 Offline'],
          ].map(([lbl, val]) => (
            <div key={lbl as string} className="card text-center py-5">
              <div className="text-2xl font-bold text-blue-400">{val}</div>
              <div className="text-xs text-gray-500 mt-1">{lbl}</div>
            </div>
          ))}
        </div>
      )}

      {/* Build */}
      <div className="card mb-5">
        <h2 className="font-semibold text-white mb-1">Index OpenAlex Corpus</h2>
        <p className="text-xs text-gray-500 mb-4">
          Source: <code className="text-blue-400">data/final/final_dataset.csv</code> ·
          Safe to run multiple times — duplicates are detected by SHA-1 chunk ID
        </p>
        <div className="flex gap-3 items-end mb-4">
          <div>
            <label className="text-xs text-gray-400 block mb-1">Papers to index (0 = all 2,000)</label>
            <input type="number" min={0} max={2000} step={100} value={limit}
                   onChange={e => setLimit(Number(e.target.value))}
                   className="input w-40" />
          </div>
          <button onClick={build} disabled={building}
                  className="btn-primary flex items-center gap-2 h-9">
            {building ? (
              <><span className="animate-spin">⟳</span> Indexing…</>
            ) : '▶ Start indexing'}
          </button>
        </div>

        {result && (
          <div className="bg-emerald-900/20 border border-emerald-800 rounded-lg px-4 py-3 text-sm text-emerald-400">
            ✅ {result.docs_loaded} papers · {result.chunks_created} chunks created ·
            {result.chunks_added} new · {result.chunks_skipped} already existed
          </div>
        )}
      </div>

      {/* Reset */}
      <div className="card border-red-900/40">
        <h2 className="font-semibold text-red-400 mb-1">⚠️ Reset Index</h2>
        <p className="text-xs text-gray-500 mb-4">
          Permanently deletes all indexed chunks. You will need to re-index afterwards.
        </p>
        {msg && (
          <div className="bg-amber-900/30 border border-amber-800 rounded-lg px-3 py-2 text-sm text-amber-400 mb-3">
            {msg}
          </div>
        )}
        {confirm && (
          <p className="text-sm text-red-400 mb-3">
            ⚠️ Are you sure? This cannot be undone. Click again to confirm.
          </p>
        )}
        <button onClick={reset} disabled={resetting} className="btn-danger">
          {resetting ? 'Deleting…' : confirm ? '⚠️ Confirm delete' : '🗑️ Delete entire index'}
        </button>
        {confirm && (
          <button onClick={() => setConfirm(false)} className="btn-ghost ml-2">Cancel</button>
        )}
      </div>

      {/* Config info */}
      <div className="card mt-5">
        <h2 className="font-semibold text-white mb-3">Index Configuration</h2>
        <div className="font-mono text-xs text-gray-400 space-y-1">
          {[
            ['Embedding model', 'all-MiniLM-L6-v2  (384-dim, ~22 MB, fully local)'],
            ['Vector store',    'ChromaDB 1.5.9  (L2 distance, persistent)'],
            ['Collection',      'researchpilot_papers'],
            ['Store path',      'data/rag_vector_store/'],
            ['Chunk size',      '700 characters'],
            ['Chunk overlap',   '100 characters'],
            ['Min chunk',       '80 characters'],
            ['Deduplication',   'SHA-1 chunk_id  (idempotent)'],
          ].map(([k, v]) => (
            <div key={k} className="flex gap-2">
              <span className="text-gray-600 w-36 shrink-0">{k}:</span>
              <span className="text-blue-400">{v}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
