import { useState, useEffect } from 'react'
import { Search, BookOpen } from 'lucide-react'
import { getPapers, type Paper } from '../api'
import OaBadge from '../components/OaBadge'
import PageHeader from '../components/PageHeader'

export default function Browse() {
  const [q,       setQ]       = useState('')
  const [oa,      setOa]      = useState('')
  const [year,    setYear]    = useState('')
  const [papers,  setPapers]  = useState<Paper[]>([])
  const [total,   setTotal]   = useState(0)
  const [loading, setLoading] = useState(false)

  const load = async () => {
    setLoading(true)
    const res = await getPapers(q, oa, year, 24)
    setPapers(res.papers); setTotal(res.total)
    setLoading(false)
  }

  useEffect(() => { load() }, [])

  return (
    <div className="p-6">
      <PageHeader icon="📚" title="Browse Papers"
        sub="Explore 2,000 AI/ML papers from the OpenAlex corpus" />

      {/* Filters */}
      <div className="flex gap-2 mb-5 flex-wrap">
        <div className="relative flex-1 min-w-48">
          <Search size={14} className="absolute left-3 top-2.5 text-gray-500" />
          <input className="input pl-8" placeholder="Search titles and abstracts…"
                 value={q} onChange={e => setQ(e.target.value)}
                 onKeyDown={e => e.key === 'Enter' && load()} />
        </div>
        <select className="input w-40" value={oa} onChange={e => setOa(e.target.value)}>
          <option value="">All categories</option>
          <option value="fully_open">Fully Open</option>
          <option value="partially_open">Partially Open</option>
          <option value="closed">Closed</option>
        </select>
        <select className="input w-28" value={year} onChange={e => setYear(e.target.value)}>
          <option value="">All years</option>
          <option value="2025">2025</option>
          <option value="2024">2024</option>
          <option value="2023">2023</option>
          <option value="2022">2022</option>
        </select>
        <button onClick={load} disabled={loading} className="btn-primary">
          {loading ? '…' : 'Search'}
        </button>
      </div>

      {total > 0 && (
        <p className="text-xs text-gray-500 mb-4">
          Showing {papers.length} of {total} papers
          {q ? ` matching "${q}"` : ' (random sample)'}
        </p>
      )}

      {loading && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {Array.from({length: 6}).map((_, i) => (
            <div key={i} className="card animate-pulse">
              <div className="h-4 bg-gray-800 rounded mb-3 w-3/4" />
              <div className="h-3 bg-gray-800 rounded mb-2 w-1/3" />
              <div className="h-3 bg-gray-800 rounded w-full" />
              <div className="h-3 bg-gray-800 rounded w-5/6 mt-1" />
            </div>
          ))}
        </div>
      )}

      {!loading && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {papers.map((p, i) => (
            <PaperCard key={i} paper={p} />
          ))}
        </div>
      )}

      {!loading && papers.length === 0 && (
        <div className="card text-center text-gray-500 text-sm py-12">
          <BookOpen size={32} className="mx-auto mb-3 opacity-30" />
          No papers found. Try different search terms.
        </div>
      )}
    </div>
  )
}

function PaperCard({ paper: p }: { paper: Paper }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div className="card hover:border-gray-700 transition-colors">
      <h3 className="text-sm font-semibold text-white leading-snug mb-2 line-clamp-2">
        {p.title}
      </h3>
      <div className="flex items-center gap-2 flex-wrap mb-3">
        <span className="text-xs text-gray-500">📅 {p.publication_year}</span>
        <span className="text-xs text-gray-500">· 📊 {p.cited_by_count} cit.</span>
        <OaBadge status={p.oa_status} />
      </div>
      <p className={`text-xs text-gray-400 leading-relaxed ${expanded ? '' : 'line-clamp-3'}`}>
        {p.abstract || 'No abstract available.'}
      </p>
      {p.abstract && p.abstract.length > 200 && (
        <button onClick={() => setExpanded(e => !e)}
                className="text-xs text-blue-400 hover:text-blue-300 mt-1">
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
    </div>
  )
}
