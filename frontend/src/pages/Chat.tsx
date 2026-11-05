import { useState, useRef, useEffect } from 'react'
import { Send, Trash2, Bot, User } from 'lucide-react'
import { streamChat, type ChatMessage, type Citation } from '../api'
import CitationList from '../components/CitationList'

interface Message {
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  streaming?: boolean
}

const SUGGESTIONS = [
  'What transformer architectures are used for object detection?',
  'What are the main limitations of current deep learning?',
  'Which papers discuss federated learning?',
  'How is ChatGPT used in medical education?',
  'What evaluation metrics are used for image classification?',
  'What are recent advances in protein structure prediction?',
]

export default function Chat() {
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

    // Snapshot history BEFORE adding the new user message
    const history: ChatMessage[] = messages
      .filter(m => !m.streaming)
      .map(m => ({ role: m.role, content: m.content }))

    setMessages(prev => [
      ...prev,
      { role: 'user', content: q },
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
            role: 'assistant',
            content: full || '*No response received — check Ollama is running.*',
            citations: cits,
            streaming: false,
          }
          return next
        })
        setLoading(false)
      },
    )
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-5 border-b border-gray-800 flex items-center justify-between">
        <div>
          <h1 className="font-bold text-white flex items-center gap-2">
            <Bot size={18} className="text-blue-400" /> RAG Research Chat
          </h1>
          <p className="text-xs text-gray-500 mt-0.5">
            Every answer is grounded in retrieved evidence with traceable citations
          </p>
        </div>
        {messages.length > 0 && (
          <button onClick={() => setMessages([])} className="btn-ghost flex items-center gap-1.5">
            <Trash2 size={14} /> Clear
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-5 space-y-4">
        {messages.length === 0 && (
          <div className="max-w-2xl mx-auto">
            <p className="text-gray-400 text-sm text-center mb-5">
              Ask anything about the 2,000 indexed AI/ML papers
            </p>
            <div className="grid grid-cols-2 gap-2">
              {SUGGESTIONS.map((s, i) => (
                <button key={i} onClick={() => send(s)}
                  className="text-left text-xs text-gray-400 hover:text-gray-200
                             bg-gray-900 hover:bg-gray-800 border border-gray-800
                             hover:border-gray-700 rounded-lg p-3 transition-all">
                  💬 {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`flex gap-3 ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {m.role === 'assistant' && (
              <div className="w-7 h-7 rounded-full bg-blue-600/20 border border-blue-600/30
                              flex items-center justify-center shrink-0 mt-1">
                <Bot size={13} className="text-blue-400" />
              </div>
            )}
            <div className={`max-w-2xl rounded-xl px-4 py-3 text-sm ${
              m.role === 'user'
                ? 'bg-blue-600/20 border border-blue-600/30 text-blue-100'
                : 'bg-gray-900 border border-gray-800 text-gray-200'
            }`}>
              <div className={`whitespace-pre-wrap leading-relaxed ${m.streaming ? 'cursor-blink' : ''}`}>
                {m.content}
              </div>
              {!m.streaming && m.citations && <CitationList citations={m.citations} />}
            </div>
            {m.role === 'user' && (
              <div className="w-7 h-7 rounded-full bg-gray-700 flex items-center justify-center shrink-0 mt-1">
                <User size={13} className="text-gray-300" />
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-gray-800">
        <div className="flex gap-2 max-w-3xl mx-auto">
          <input
            className="input flex-1"
            placeholder="Ask a research question…"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send(input)}
            disabled={loading}
          />
          <button onClick={() => send(input)} disabled={loading || !input.trim()}
                  className="btn-primary flex items-center gap-1.5 px-4">
            <Send size={14} /> Send
          </button>
        </div>
      </div>
    </div>
  )
}
