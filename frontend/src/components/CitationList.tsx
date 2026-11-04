import { useState } from 'react'
import { ChevronDown, ChevronUp, ExternalLink } from 'lucide-react'
import type { Citation } from '../api'

export default function CitationList({ citations }: { citations: Citation[] }) {
  const [open, setOpen] = useState(false)
  if (!citations.length) return null
  return (
    <div className="mt-3">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1.5 text-xs text-blue-400 hover:text-blue-300 transition-colors"
      >
        {open ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        {citations.length} source{citations.length !== 1 ? 's' : ''}
      </button>
      {open && (
        <div className="mt-2 space-y-2">
          {citations.map((c, i) => (
            <div key={i} className="source-card">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <span className="text-blue-400 font-semibold">{c.label}</span>
                  <span className="text-gray-200 ml-1">{c.title}</span>
                </div>
                <div className="flex gap-2 shrink-0">
                  {c.doi_url && (
                    <a href={c.doi_url} target="_blank" rel="noreferrer"
                       className="text-blue-400 hover:text-blue-300">
                      <ExternalLink size={13} />
                    </a>
                  )}
                </div>
              </div>
              <div className="text-gray-500 text-xs mt-1">
                {c.year} · {c.evidence_type} · {c.source}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
