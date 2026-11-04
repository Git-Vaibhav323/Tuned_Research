import { useState } from 'react'
import { Search } from 'lucide-react'
import { getSources, type Hit } from '../api'
import PageHeader from '../components/PageHeader'

export default function Sources() {
  const [query,    setQuery]    = useState('')
  const [topK,     setTopK]     = useState(5)
  const [maxDist,  setMaxDist]  = useState(1.6)
  const [hits,     setHits]     = useState<Hit[]>([])
  const [loading,  setLoading]  = useState(false)
  const [searched, setSearched] = useState(false)

  const run = async () => {
    if (!query.trim()) return
    setLoading(true)
    const res = await getSources(query, topK, maxDist)
    setHits(res.hits)
    setSearched(true)
    setLoading(false)
  }

  return (
    <div className="p-6 max-w-4xl">
      <PageHeader icon="🔎" title="Source Inspector"
        sub="See exactly which passages are retrieved for any query — with relevance scores and raw text" />

      <div className="card mb-5">
        <div className="flex gap-2 mb-3">
          <input className="input flex-1" placeholder="e.g. attention mechanism in vision transformers"
                 value={query} onChange={e => setQuery(e.target.value)}
                 onKeyDown={e => e.key === 'Enter' && run()} />
          <button onClick={run} disabled={loading || !query.trim()}
                  className="btn-primary flex items-center gap-1.5">
            <Search size={14} /> {loading ? '…' : 'Inspect'}
          </button>
        </div>
        <div className="flex gap-4 text-xs text-gray-400 items-center">
          <label className="flex items-center gap-2">
            Top-k
            <select className="input w-20 py-1" value={topK}
                    onChange={e => setTopK(Number(e.target.value))}>
              {[1,3,5,8,10,15].map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
          <label className="flex items-center gap-2">
            Max L2 dist
            <input type="range" min="0.5" max="2.0" step="0.1" value={maxDist}
                   onChange={e => setMaxDist(Number(e.target.value))}
                   className="w-24 accent-blue-500" />
            <span className="w-8 text-right">{maxDist.toFixed(1)}</span>
          </label>
        </div>
      </div>

      {searched && hits.length === 0 && (
        <div className="card text-center text-gray-500 text-sm py-8">
          No passages matched. Try increasing Max distance or changing the query.
        </div>
      )}

      {hits.length > 0 && (
        <>
          {/* Summary table */}
          <div className="card mb-4 overflow-x-auto">
            <h2 className="text-sm font-semibold text-white mb-3">
              {hits.length} passage{hits.length !== 1 ? 's' : ''} retrieved
            </h2>
            <table className="w-full text-xs">
              <thead>
                <tr className="text-gray-500 border-b border-gray-800">
                  <th className="text-left pb-2 pr-4">#</th>
                  <th className="text-left pb-2 pr-4">Title</th>
                  <th className="text-left pb-2 pr-4">Year</th>
                  <th className="text-left pb-2 pr-4">Score</th>
                  <th className="text-left pb-2 pr-4">Distance</th>
                  <th className="text-left pb-2">Type</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                {hits.map((h, i) => (
                  <tr key={i} className="text-gray-300">
                    <td className="py-2 pr-4 text-gray-500">{i+1}</td>
                    <td className="py-2 pr-4 max-w-xs truncate">{h.title?.slice(0, 55)}</td>
                    <td className="py-2 pr-4">{h.year}</td>
                    <td className="py-2 pr-4">
                      <span className={`font-semibold ${
                        h.relevance_score >= 0.4 ? 'text-emerald-400'
                          : h.relevance_score >= 0.2 ? 'text-amber-400'
                          : 'text-gray-500'}`}>
                        {h.relevance_score.toFixed(3)}
                      </span>
                    </td>
                    <td className="py-2 pr-4 text-gray-500">{h.distance.toFixed(3)}</td>
                    <td className="py-2 text-gray-500">
                      {h.is_fulltext ? 'Full text' : 'Abstract'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Full passages */}
          <div className="space-y-3">
            {hits.map((h, i) => (
              <div key={i} className="source-card">
                <div className="flex items-start justify-between gap-3 mb-2">
                  <div>
                    <span className="text-blue-400 font-semibold text-xs">[{i+1}]</span>
                    <span className="text-white text-xs font-medium ml-1.5">{h.title}</span>
                  </div>
                  <span className={`text-xs font-bold shrink-0 px-2 py-0.5 rounded-full ${
                    h.relevance_score >= 0.4 ? 'bg-emerald-900/50 text-emerald-400'
                      : h.relevance_score >= 0.2 ? 'bg-amber-900/50 text-amber-400'
                      : 'bg-gray-800 text-gray-400'}`}>
                    {h.relevance_score.toFixed(3)}
                  </span>
                </div>
                <p className="text-gray-500 text-xs mb-2">
                  {h.year} · L2 {h.distance.toFixed(3)} · {h.is_fulltext ? 'Full text' : 'Abstract'}
                </p>
                <p className="text-gray-400 text-xs leading-relaxed">{h.text}</p>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
