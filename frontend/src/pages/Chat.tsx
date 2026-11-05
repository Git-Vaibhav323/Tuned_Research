import { useState, useRef, useEffect } from 'react'
import { Send, Trash2, Bot, User, Download, AlertTriangle } from 'lucide-react'
import { streamChat, type ChatMessage, type Citation } from '../api'
import CitationList from '../components/CitationList'

interface Message {
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  streaming?: boolean
}

interface Props {
  ollama?: boolean
}

const SUGGESTIONS = [
  'What transformer architectures are used for object detection?',
  'What are the main limitations of current deep learning?',
  'Which papers discuss federated learning?',
  'How is ChatGPT used in medical education?',
  'What evaluation metrics are used for image classification?',
  'What are recent advances in protein structure prediction?',
]

// ── Helpers ───────────────────────────────────────────────────────────────────

function buildMarkdown(messages: Message[]): string {
  const lines: string[] = [
    '# ResearchPilot — Chat Export',
    '',
    `*Exported ${new Date().toLocaleString()}*`,
    '',
    '---',
    '',
  ]

  messages.forEach(m => {
    if (m.streaming) return
    if (m.role === 'user') {
      lines.push(`**You:** ${m.content}`, '')
    } else {
      lines.push('**ResearchPilot:**', '', m.content, '')
      if (m.citations?.length) {
        lines.push('**Sources:**', '')
        m.citations.forEach(c => {
          const doi = c.doi_url ? ` · [DOI](${c.doi_url})` : ''
          lines.push(`- ${c.label} ${c.title} (${c.year ?? ''})${doi}`)
        })
        lines.push('')
      }
      lines.push('---', '')
    }
  })

  return lines.join('\n')
}

function downloadMarkdown(messages: Message[]): void {
  const md   = buildMarkdown(messages)
  const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href     = url
  a.download = `researchpilot-chat-${Date.now()}.md`
  a.click()
  URL.revokeObjectURL(url)
}

// ── Per-message export button ─────────────────────────────────────────────────

function ExportMsgButton({ message, index }: { message: Message; index: number }) {
  const download = () => {
    const md   = `# ResearchPilot — Answer\n\n*Exported ${new Date().toLocaleString()}*\n\n---\n\n${message.content}\n`
    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href     = url
    a.download = `answer-${index + 1}.md`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <button
      onClick={download}
      title="Export this answer as Markdown"
      className="opacity-0 group-hover:opacity-100 transition-opacity
                 p-1 rounded text-gray-500 hover:text-gray-300 hover:bg-gray-800"
    >
      <Download size={12} />
    </button>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export default function Chat({ ollama = true }: Props) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input,    setInput]    = useState('')
  const [loading,  setLoading]  = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = async (question: string) => {
    if (!question.trim() || loading) return
    const q = question.trim()
    setInput('')
    setLoading(true)

    // Snapshot history BEFORE adding the new messages
    const history: ChatMessage[] = messages
      .filter(m => !m.streaming)
      .map(m => ({ role: m.role, content: m.content }))

    setMessages(prev => [
      ...prev,
      { role: 'user',      content: q },
      { role: 'assistant', content: '', streaming: true },
    ])

    let full = ''
    let cits: Citation[] = []

    await streamChat(
      { question: q, history },
      tok => {
        full += tok
        setMessages(prev => {
          const next = [...prev]
          next[next.length - 1] = { role: 'assistant', content: full, streaming: true }
          return next
        })
      },
      c => { cits = c },
      () => {
        setMessages(prev => {
          const next = [...prev]
          next[next.length - 1] = {
            role:      'assistant',
            content:   full || '*No response received — check Ollama is running.*',
            citations: cits,
            streaming: false,
          }
          return next
        })
        setLoading(false)
      },
    )
  }

  const completedMessages = messages.filter(m => !m.streaming)

  return (
    <div className="flex flex-col h-full">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div className="p-5 border-b border-gray-800 flex items-center justify-between shrink-0">
        <div>
          <h1 className="font-bold text-white flex items-center gap-2">
            <Bot size={18} className="text-blue-400" /> RAG Research Chat
          </h1>
          <p className="text-xs text-gray-500 mt-0.5">
            Every answer is grounded in retrieved evidence with traceable citations
          </p>
        </div>
        <div className="flex items-center gap-2">
          {completedMessages.length > 0 && (
            <button
              onClick={() => downloadMarkdown(completedMessages)}
              className="btn-ghost flex items-center gap-1.5 text-xs"
              title="Export full conversation as Markdown"
            >
              <Download size={13} /> Export chat
            </button>
          )}
          {messages.length > 0 && (
            <button onClick={() => setMessages([])} className="btn-ghost flex items-center gap-1.5">
              <Trash2 size={14} /> Clear
            </button>
          )}
        </div>
      </div>

      {/* ── Ollama inline warning (when banner is dismissed but still offline) */}
      {!ollama && (
        <div className="shrink-0 mx-5 mt-4 flex items-center gap-2.5 bg-red-950/60
                        border border-red-800/60 rounded-lg px-4 py-3 text-sm text-red-300">
          <AlertTriangle size={15} className="text-red-400 shrink-0" />
          <span>
            Ollama is offline — answers cannot be generated.
            Run <code className="bg-red-900/50 px-1.5 py-0.5 rounded text-xs font-mono">
              ollama serve
            </code> then reload.
          </span>
        </div>
      )}

      {/* ── Messages ───────────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto p-5 space-y-4">

        {/* Empty state */}
        {messages.length === 0 && (
          <div className="max-w-2xl mx-auto">
            <p className="text-gray-400 text-sm text-center mb-5">
              Ask anything about the 2,000 indexed AI/ML papers
            </p>
            <div className="grid grid-cols-2 gap-2">
              {SUGGESTIONS.map((s, i) => (
                <button
                  key={i}
                  onClick={() => send(s)}
                  disabled={!ollama}
                  className="text-left text-xs text-gray-400 hover:text-gray-200
                             bg-gray-900 hover:bg-gray-800 border border-gray-800
                             hover:border-gray-700 rounded-lg p-3 transition-all
                             disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  💬 {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Conversation */}
        {messages.map((m, i) => (
          <div
            key={i}
            className={`flex gap-3 ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {/* Bot avatar */}
            {m.role === 'assistant' && (
              <div className="w-7 h-7 rounded-full bg-blue-600/20 border border-blue-600/30
                              flex items-center justify-center shrink-0 mt-1">
                <Bot size={13} className="text-blue-400" />
              </div>
            )}

            {/* Bubble — group class enables hover-reveal of export button */}
            <div className={`group max-w-2xl rounded-xl px-4 py-3 text-sm relative ${
              m.role === 'user'
                ? 'bg-blue-600/20 border border-blue-600/30 text-blue-100'
                : 'bg-gray-900 border border-gray-800 text-gray-200'
            }`}>
              <div className={`whitespace-pre-wrap leading-relaxed ${m.streaming ? 'cursor-blink' : ''}`}>
                {m.content}
              </div>

              {/* Citations */}
              {!m.streaming && m.citations && <CitationList citations={m.citations} />}

              {/* Per-message export — only for completed assistant messages */}
              {!m.streaming && m.role === 'assistant' && m.content && (
                <div className="flex justify-end mt-2">
                  <ExportMsgButton message={m} index={i} />
                </div>
              )}
            </div>

            {/* User avatar */}
            {m.role === 'user' && (
              <div className="w-7 h-7 rounded-full bg-gray-700 flex items-center
                              justify-center shrink-0 mt-1">
                <User size={13} className="text-gray-300" />
              </div>
            )}
          </div>
        ))}

        <div ref={bottomRef} />
      </div>

      {/* ── Input ──────────────────────────────────────────────────────────── */}
      <div className="p-4 border-t border-gray-800 shrink-0">
        <div className="flex gap-2 max-w-3xl mx-auto">
          <input
            className="input flex-1"
            placeholder={ollama ? 'Ask a research question…' : 'Ollama offline — start ollama serve first'}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send(input)}
            disabled={loading || !ollama}
          />
          <button
            onClick={() => send(input)}
            disabled={loading || !input.trim() || !ollama}
            className="btn-primary flex items-center gap-1.5 px-4"
          >
            <Send size={14} /> Send
          </button>
        </div>
      </div>

    </div>
  )
}
