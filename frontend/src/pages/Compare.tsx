import { useState } from 'react'
import { Plus, X, Scale } from 'lucide-react'
import { searchPapers, streamCompare, type Hit, type Citation } from '../api'
import CitationList from '../components/CitationList'
import PageHeader from '../components/PageHeader'

const DIMS = ['Research objective','Methodology','Dataset','Evaluation metrics',
              'Key findings','Limitations','Future directions']

export default function Compare() {
  const [query,    setQuery]    = useState('')
  const [results,  setResults]  = useState<Hit[]>([])
  const [selected, setSelected] = useState<Hit[]>([])
  const [dims,     setDims]     = useState(DIMS.slice(0, 4))
  const [answer,   setAnswer]   = useState('')
  const [cits,     setCits]     = useState<Citation[]>([])
  const [loading,  setLoading]  = useState(false)
  const [searching,setSearching]= useState(false)

  const search = async () => {
    if (!query.trim()) return
    setSearching(true)
    const res = await searchPapers(query, 8)
    setResults(res.hits)
    setSearching(false)
  }

  const add = (h: Hit) => {
    if (selected.find(s => s.doc_id === h.doc_id) || selected.length >= 4) return
    setSelected(s => [...s, h])
  }

  const remove = (id: string) => setSelected(s => s.filter(x => x.doc_id !== id))

  const compare = async () => {
    if (selected.length < 2) return
    setLoading(true); setAnswer(''); setCits([])
    let full = ''
    await streamCompare(
      selected.map(s => s.doc_id),
      selected.map(s => s.title),
      dims,
      tok => { full += tok; setAnswer(full) },
      ci  => { setCits(ci) },
      ()  => { setLoading(false) },
    )
  }

  const toggleDim = (d: string) =>
    setDims(prev => prev.includes(d) ? prev.filter(x => x !== d) : [...prev, d])

  return (
    <div className="p-6 max-w-5xl">
      <PageHeader icon="⚖️" title="Compare Papers"
        sub="Select 2–4 indexed papers and compare them side-by-side" />

      <div className="grid grid-cols-2 gap-5 mb-5">
        {/* Search */}
        <div className="card">
          <h2 className="text-sm font-semibold text-white mb-3">Search papers</h2>
          <div className="flex gap-2 mb-3">
            <input className="input flex-1" placeholder="e.g. transformer object detection"
                   value={query} onChange={e => setQuery(e.target.value)}
                   onKeyDown={e => e.key === 'Enter' && search()} />
            <button onClick={search} disabled={searching} className="btn-primary">
              {searching ? '…' : 'Search'}
            </button>
          </div>
          <div className="space-y-2 max-h-80 overflow-y-auto">
            {results.map(h => (
              <div key={h.chunk_id}
                   className="flex items-start gap-2 bg-gray-950 rounded-lg p-3 border border-gray-800">
                <div className="flex-1 min-w-0">
                  <div className="text-xs text-white font-medium truncate">{h.title}</div>
                  <div className="text-xs text-gray-500 mt-0.5">{h.year}</div>
                </div>
                <button
                  onClick={() => add(h)}
                  disabled={!!selected.find(s => s.doc_id === h.doc_id) || selected.length >= 4}
                  className="btn-secondary px-2 py-1 text-xs shrink-0">
                  <Plus size={12} />
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Selection */}
        <div className="card">
          <h2 className="text-sm font-semibold text-white mb-3">
            Selected ({selected.length}/4)
          </h2>
          {selected.length === 0 && (
            <p className="text-xs text-gray-500">Search and add 2–4 papers</p>
          )}
          <div className="space-y-2 mb-4">
            {selected.map(s => (
              <div key={s.doc_id}
                   className="flex items-center gap-2 bg-blue-600/10 border border-blue-600/20
                              rounded-lg p-2.5">
                <div className="flex-1 min-w-0">
                  <div className="text-xs text-blue-300 font-medium truncate">{s.title}</div>
                  <div className="text-xs text-gray-500">{s.year}</div>
                </div>
                <button onClick={() => remove(s.doc_id)} className="text-gray-500 hover:text-red-400">
                  <X size={13} />
                </button>
              </div>
            ))}
          </div>

          <div className="mb-4">
            <p className="text-xs text-gray-400 mb-2 font-medium">Compare on:</p>
            <div className="flex flex-wrap gap-1.5">
              {DIMS.map(d => (
                <button key={d} onClick={() => toggleDim(d)}
                  className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
                    dims.includes(d)
                      ? 'bg-blue-600/20 border-blue-600/40 text-blue-400'
                      : 'border-gray-700 text-gray-500 hover:border-gray-600'}`}>
                  {d}
                </button>
              ))}
            </div>
          </div>

          <button onClick={compare} disabled={selected.length < 2 || loading}
                  className="btn-primary w-full flex items-center justify-center gap-2">
            <Scale size={14} /> {loading ? 'Comparing…' : 'Compare'}
          </button>
        </div>
      </div>

      {/* Result */}
      {(answer || loading) && (
        <div className="card">
          <h2 className="text-sm font-semibold text-white mb-3">Comparison</h2>
          <div className={`text-sm text-gray-200 whitespace-pre-wrap leading-relaxed
                          ${loading && !answer ? 'cursor-blink' : ''}`}>
            {answer}
          </div>
          {!loading && <CitationList citations={cits} />}
        </div>
      )}
    </div>
  )
}
