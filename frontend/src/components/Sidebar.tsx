import { NavLink } from 'react-router-dom'
import {
  MessageSquare, FileText, Scale, Lightbulb,
  Search, BookOpen, BarChart2, Settings, Microscope
} from 'lucide-react'

const nav = [
  { to: '/',        icon: MessageSquare, label: 'RAG Chat' },
  { to: '/analyse', icon: FileText,      label: 'Analyse Paper' },
  { to: '/compare', icon: Scale,         label: 'Compare' },
  { to: '/gaps',    icon: Lightbulb,     label: 'Research Gaps' },
  { to: '/sources', icon: Search,        label: 'Source Inspector' },
  { to: '/browse',  icon: BookOpen,      label: 'Browse Papers' },
  { to: '/stats',   icon: BarChart2,     label: 'Project Stats' },
  { to: '/index',   icon: Settings,      label: 'Index Mgmt' },
]

interface Props { health: { ollama: boolean; n_chunks: number; n_docs: number } | null }

export default function Sidebar({ health }: Props) {
  return (
    <aside className="w-56 shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col h-screen sticky top-0">
      {/* Logo */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-600 to-purple-600
                          flex items-center justify-center shrink-0">
            <Microscope size={16} className="text-white" />
          </div>
          <div>
            <div className="font-bold text-sm text-white leading-tight">ResearchPilot</div>
            <div className="text-xs text-gray-500">AI Research Assistant</div>
          </div>
        </div>
      </div>

      {/* Status */}
      <div className="px-4 py-3 border-b border-gray-800 space-y-1.5">
        <StatusPill
          label={health?.ollama ? 'Ollama ready' : 'Ollama offline'}
          ok={!!health?.ollama}
        />
        <StatusPill
          label={health?.n_chunks ? `${health.n_docs} papers · ${health.n_chunks} chunks` : 'Index empty'}
          ok={!!health?.n_chunks}
          warn={!health?.n_chunks}
        />
      </div>

      {/* Nav */}
      <nav className="flex-1 overflow-y-auto p-2 space-y-0.5">
        {nav.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-blue-600/20 text-blue-400 border border-blue-600/30'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
              }`
            }
          >
            <Icon size={15} />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-gray-800">
        <div className="text-xs text-gray-600 leading-relaxed">
          <div className="font-medium text-gray-500 mb-1">ResearchPilot v3.0</div>
          all-MiniLM-L6-v2<br />
          ChromaDB 1.5.9<br />
          Qwen2.5-1.5B · Ollama
        </div>
      </div>
    </aside>
  )
}

function StatusPill({ label, ok, warn }: { label: string; ok: boolean; warn?: boolean }) {
  const colour = ok ? 'text-emerald-400' : warn ? 'text-amber-400' : 'text-red-400'
  const dot    = ok ? 'bg-emerald-400'  : warn ? 'bg-amber-400'   : 'bg-red-400'
  return (
    <div className={`flex items-center gap-1.5 text-xs ${colour}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${dot} shrink-0`} />
      <span className="truncate">{label}</span>
    </div>
  )
}
