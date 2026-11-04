import { useState } from 'react'
import { Upload, FileText, Zap } from 'lucide-react'
import { streamChat, uploadPdf, uploadCsv, type Citation, type ChatMessage } from '../api'
import CitationList from '../components/CitationList'
import PageHeader from '../components/PageHeader'

const DEMO = {
  name: 'Attention Is All You Need (2017)',
  text: 'Title: Attention Is All You Need\n\nAbstract: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.',
}

export default function Analyse() {
  const [tab,    setTab]    = useState<'pdf'|'text'|'csv'>('pdf')
  const [text,   setText]   = useState('')
  const [name,   setName]   = useState('')
  const [docId,  setDocId]  = useState('')
  const [status, setStatus] = useState('')

  const [question, setQuestion] = useState('')
  const [answer,   setAnswer]   = useState('')
  const [cits,     setCits]     = useState<Citation[]>([])
  const [loading,  setLoading]  = useState(false)
  const [history,  setHistory]  = useState<ChatMessage[]>([])

  const handlePdf = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]; if (!f) return
    setStatus('Uploading…')
    const res = await uploadPdf(f)
    if (res.status === 'ok') {
      setDocId(res.doc_id || '')
      setName(f.name)
      setText('[PDF indexed — use Ask Questions tab]')
      setStatus(`✅ ${res.message}`)
    } else {
      setStatus(`❌ ${res.message}`)
    }
  }

  const handleCsv = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]; if (!f) return
    setStatus('Indexing CSV…')
    const res = await uploadCsv(f)
    setStatus(res.status === 'ok' ? `✅ ${res.message}` : `❌ ${res.message}`)
  }

  const ask = async () => {
    if (!question.trim() || loading) return
    setLoading(true)
    const q = question.trim()
    setQuestion('')
    let full = ''
    setAnswer('')
    const ctx = text && text !== '[PDF indexed — use Ask Questions tab]'
      ? `Context:\n${text.slice(0, 2000)}\n\nQuestion: ${q}`
      : q
    await streamChat({ question: ctx, history },
      tok => { full += tok; setAnswer(full) },
      ci  => { setCits(ci) },
      ()  => {
        setHistory(h => [...h,
          { role: 'user',      content: q    },
          { role: 'assistant', content: full },
        ])
        setLoading(false)
      },
    )
  }

  const quickPrompts = [
    ['📋 Summarise',   'Summarise this paper clearly.'],
    ['🔍 Methodology', 'What methodology does this paper use?'],
    ['⚠️ Limitations', 'What are the main limitations?'],
    ['💡 Findings',    'What are the key findings and contributions?'],
  ]

  return (
    <div className="p-6 max-w-4xl">
      <PageHeader icon="📄" title="Analyse a Paper"
        sub="Upload a PDF, paste text, or upload CSV — then ask questions about it" />

      {/* Input tabs */}
      <div className="card mb-5">
        <div className="flex gap-1 mb-4 border-b border-gray-800 pb-3">
          {(['pdf','text','csv'] as const).map(t => (
            <button key={t} onClick={() => setTab(t)}
              className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                tab === t ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-gray-200'}`}>
              {t === 'pdf' ? '📎 PDF' : t === 'text' ? '📝 Text' : '📊 CSV'}
            </button>
          ))}
        </div>

        {tab === 'pdf' && (
          <label className="flex flex-col items-center gap-3 border-2 border-dashed
                             border-gray-700 hover:border-blue-500 rounded-xl p-8 cursor-pointer
                             transition-colors">
            <Upload size={28} className="text-gray-500" />
            <span className="text-sm text-gray-400">Click to upload a PDF research paper</span>
            <input type="file" accept=".pdf" className="hidden" onChange={handlePdf} />
          </label>
        )}

        {tab === 'text' && (
          <div className="space-y-3">
            <input className="input" placeholder="Paper name…" value={name}
                   onChange={e => setName(e.target.value)} />
            <textarea className="input h-40 resize-none" placeholder="Paste title + abstract…"
                      value={text} onChange={e => setText(e.target.value)} />
          </div>
        )}

        {tab === 'csv' && (
          <label className="flex flex-col items-center gap-3 border-2 border-dashed
                             border-gray-700 hover:border-blue-500 rounded-xl p-8 cursor-pointer
                             transition-colors">
            <FileText size={28} className="text-gray-500" />
            <span className="text-sm text-gray-400">Upload CSV with paper metadata to index</span>
            <input type="file" accept=".csv" className="hidden" onChange={handleCsv} />
          </label>
        )}

        {status && (
          <div className={`mt-3 text-sm px-3 py-2 rounded-lg ${
            status.startsWith('✅') ? 'bg-emerald-900/30 text-emerald-400'
              : status.startsWith('❌') ? 'bg-red-900/30 text-red-400'
              : 'bg-gray-800 text-gray-400'}`}>
            {status}
          </div>
        )}

        <div className="flex gap-2 mt-4">
          <button onClick={() => { setText(DEMO.text); setName(DEMO.name) }}
                  className="btn-ghost flex items-center gap-1.5">
            <Zap size={13} /> Load demo paper
          </button>
          {text && (
            <button onClick={() => { setText(''); setName(''); setDocId(''); setStatus('') }}
                    className="btn-ghost text-red-400">
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Ask questions */}
      {(text || docId) && (
        <div className="card">
          <h2 className="font-semibold text-white mb-3 text-sm">Ask about this paper</h2>
          <div className="grid grid-cols-2 gap-2 mb-4">
            {quickPrompts.map(([label, prompt]) => (
              <button key={label} onClick={() => { setQuestion(prompt); }}
                      className="btn-secondary text-xs text-left">{label}</button>
            ))}
          </div>
          <div className="flex gap-2 mb-4">
            <input className="input flex-1" placeholder="Ask a question…"
                   value={question} onChange={e => setQuestion(e.target.value)}
                   onKeyDown={e => e.key === 'Enter' && ask()} />
            <button onClick={ask} disabled={loading || !question.trim()} className="btn-primary">
              {loading ? '…' : 'Ask'}
            </button>
          </div>
          {(answer || loading) && (
            <div className={`text-sm text-gray-200 whitespace-pre-wrap leading-relaxed
                            bg-gray-950 rounded-lg p-4 ${loading && !answer ? 'cursor-blink' : ''}`}>
              {answer}
              {!loading && <CitationList citations={cits} />}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
