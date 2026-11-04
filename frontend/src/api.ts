const BASE = '/api'

export interface ChatMessage { role: 'user' | 'assistant'; content: string }
export interface Citation {
  label: string; title: string; year?: string
  doi_url?: string; openalex_url?: string
  evidence_type?: string; source?: string
}
export interface Hit {
  chunk_id: string; doc_id: string; title: string; year?: string
  text: string; distance: number; relevance_score: number
  is_fulltext?: boolean; oa_category?: string; doi?: string
}
export interface Paper {
  id?: string; title: string; publication_year: string
  abstract: string; cited_by_count: string
  oa_status: string; oa_category: string; doi?: string
}

// ── REST ──────────────────────────────────────────────────────────────────────

export async function getHealth() {
  const r = await fetch(`${BASE}/health`)
  return r.json()
}

export async function getStats() {
  const r = await fetch(`${BASE}/stats`)
  return r.json()
}

export async function getPapers(q = '', oa = '', year = '', limit = 20) {
  const p = new URLSearchParams({ q, oa, year, limit: String(limit) })
  const r = await fetch(`${BASE}/papers?${p}`)
  return r.json() as Promise<{ papers: Paper[]; total: number }>
}

export async function searchPapers(q: string, top_k = 8) {
  const p = new URLSearchParams({ q, top_k: String(top_k) })
  const r = await fetch(`${BASE}/search?${p}`)
  return r.json() as Promise<{ hits: Hit[]; count: number }>
}

export async function getSources(query: string, top_k = 5, max_distance = 1.6) {
  const r = await fetch(`${BASE}/sources`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k, max_distance }),
  })
  return r.json() as Promise<{ hits: Hit[]; count: number }>
}

export async function buildIndex(limit = 0) {
  const r = await fetch(`${BASE}/index/build?limit=${limit}`, { method: 'POST' })
  return r.json()
}

export async function resetIndex() {
  const r = await fetch(`${BASE}/index/reset`, { method: 'POST' })
  return r.json()
}

export async function uploadPdf(file: File) {
  const fd = new FormData(); fd.append('file', file)
  const r = await fetch(`${BASE}/upload/pdf`, { method: 'POST', body: fd })
  return r.json()
}

export async function uploadCsv(file: File) {
  const fd = new FormData(); fd.append('file', file)
  const r = await fetch(`${BASE}/upload/csv`, { method: 'POST', body: fd })
  return r.json()
}

// ── SSE streaming ─────────────────────────────────────────────────────────────

type TokenCb    = (token: string) => void
type CiteCb     = (citations: Citation[]) => void
type DoneCb     = () => void

async function streamPost(
  url: string,
  body: object,
  onToken: TokenCb,
  onCitations: CiteCb,
  onDone: DoneCb,
) {
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!resp.ok || !resp.body) { onDone(); return }

  const reader  = resp.body.getReader()
  const decoder = new TextDecoder()
  let   buf     = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    const lines = buf.split('\n')
    buf = lines.pop() ?? ''
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const raw = line.slice(6).trim()
      if (raw === '[DONE]') { onDone(); return }
      try {
        const obj = JSON.parse(raw)
        if (obj.token     !== undefined) onToken(obj.token.replace(/\\n/g, '\n'))
        if (obj.citations !== undefined) onCitations(obj.citations)
      } catch { /* skip malformed */ }
    }
  }
  onDone()
}

export function streamChat(req: { question: string; history: ChatMessage[]; top_k?: number },
    onToken: TokenCb, onCitations: CiteCb, onDone: DoneCb) {
  return streamPost(`${BASE}/chat`, req, onToken, onCitations, onDone)
}

export function streamGaps(topic: string, top_k: number,
    onToken: TokenCb, onCitations: CiteCb, onDone: DoneCb) {
  return streamPost(`${BASE}/gaps`, { topic, top_k }, onToken, onCitations, onDone)
}

export function streamCompare(doc_ids: string[], titles: string[], dimensions: string[],
    onToken: TokenCb, onCitations: CiteCb, onDone: DoneCb) {
  return streamPost(`${BASE}/compare`, { doc_ids, titles, dimensions }, onToken, onCitations, onDone)
}

export function streamSummarise(doc_id: string, title: string,
    onToken: TokenCb, onCitations: CiteCb, onDone: DoneCb) {
  return streamPost(`${BASE}/summarise`, { doc_id, title }, onToken, onCitations, onDone)
}
