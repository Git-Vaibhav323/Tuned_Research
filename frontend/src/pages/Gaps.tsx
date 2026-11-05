import { useState } from 'react'
import { Lightbulb } from 'lucide-react'
import { streamGaps, getSources, type Citation, type Hit } from '../api'
import CitationList from '../components/CitationList'
import PageHeader from '../components/PageHeader'

const QUICK = [
  'federated learning privacy preservation',
  'LLMs in clinical healthcare',
  'vision transformers efficiency',
  'continual learning catastrophic forgetting',
  'self-supervised learning for medical imaging',
  'explainability in deep neural networks',
]

export default function Gaps() {
  const [topic,   setTopic]   = useState('')
  const [topK,    setTopK]    = useState(6)
  const [answer,  setAnswer]  = useState('')
  const [cits,    setCits]    = useState<Citation[]>([])
  const [hits,    setHits]    = useState<Hit[]>([])
  const [loading, setLoading] = useState(false)

  const run = async (t: string) => {
    if (!t.trim()) return
    setLoading(true); setAnswer(''); setCits([]); setHits([])
    let full = ''

    // Run streaming and source fetch independently — don't use Promise.all
    // because streamGaps returns void (the generator runs async internally)
    getSources(t, topK).then(res => setHits(res.hits)).catch(() => {})

    await streamGaps(
      t, topK,
      tok => { full += tok; setAnswer(full) },
      ci  => { setCits(ci) },
      ()  => { setLoading(false) },
    )
  }

  return (
    <div className="p-6 max-w-4xl">
      <PageHeader icon="💡" title="Research Gaps"
        sub="Find candidate gaps based on retrieved literature — always verify novelty against broader sources" />

      <div className="card mb-5">
        <div className="flex gap-2 mb-4">
          <input className="input flex-1" placeholder="e.g. transformer models for medical image segmentation"
                 value={topic} onChange={e => setTopic(e.target.value)}
                 onKeyDown={e => e.key === 'Enter' && run(topic)} />
          <select className="input w-28" value={topK}
                  onChange={e => setTopK(Number(e.target.value))}>
            {[3,4,5,6,8,10,12].map(n => <option key={n} value={n}>top-{n}</option>)}
          </select>
          <button onClick={() => run(topic)} disabled={loading || !topic.trim()}
                  className="btn-primary flex items-center gap-1.5">
            <Lightbulb size={14} /> {loading ? '…' : 'Find gaps'}
          </button>
        </div>

        <p className="text-xs text-gray-500 mb-3">Quick topics:</p>
        <div className="flex flex-wrap gap-1.5">
          {QUICK.map(q => (
            <button key={q} onClick={() => { setTopic(q); run(q) }}
                    className="text-xs px-2.5 py-1.5 bg-gray-800 hover:bg-gray-700
                               border border-gray-700 rounded-lg text-gray-400 hover:text-gray-200 transition-colors">
              {q}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-amber-900/20 border border-amber-800/40 rounded-lg px-4 py-2.5 mb-5">
        <p className="text-xs text-amber-400">
          ℹ️ Suggestions are based on retrieved passages only — not the full literature.
          Treat as starting points for your own review.
        </p>
      </div>

      {(answer || loading) && (
        <div className="card mb-5">
          <h2 className="text-sm font-semibold text-white mb-3">Suggested Gaps</h2>
          <div className={`text-sm text-gray-200 whitespace-pre-wrap leading-relaxed
                          ${loading && !answer ? 'cursor-blink' : ''}`}>
            {answer}
          </div>
          {!loading && <CitationList citations={cits} />}
        </div>
      )}

      {hits.length > 0 && (
        <div className="card">
          <h2 className="text-sm font-semibold text-white mb-3">Evidence used ({hits.length})</h2>
          <div className="space-y-2">
            {hits.map((h, i) => (
              <div key={i} className="source-card">
                <div className="flex items-center justify-between">
                  <span className="font-medium text-white text-xs">{h.title?.slice(0, 80)}</span>
                  <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${
                    h.relevance_score >= 0.4 ? 'bg-emerald-900/50 text-emerald-400'
                      : 'bg-gray-800 text-gray-400'}`}>
                    {h.relevance_score.toFixed(2)}
                  </span>
                </div>
                <p className="text-gray-500 text-xs mt-1">{h.year} · dist {h.distance.toFixed(3)}</p>
                <p className="text-gray-400 text-xs mt-2 line-clamp-2">{h.text}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
