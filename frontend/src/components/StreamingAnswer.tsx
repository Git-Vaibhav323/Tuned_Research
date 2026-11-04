import CitationList from './CitationList'
import type { Citation } from '../api'

interface Props {
  text: string
  citations: Citation[]
  streaming: boolean
}

export default function StreamingAnswer({ text, citations, streaming }: Props) {
  if (!text && !streaming) return null
  return (
    <div className="mt-4">
      <div className={`prose-dark text-sm leading-relaxed whitespace-pre-wrap ${streaming ? 'cursor-blink' : ''}`}>
        {text || (streaming ? '' : '')}
      </div>
      {!streaming && <CitationList citations={citations} />}
    </div>
  )
}
