export default function OaBadge({ status }: { status: string }) {
  const s = (status || '').toLowerCase()
  if (['gold','diamond','green'].includes(s))
    return <span className="badge-open">{status}</span>
  if (s === 'closed')
    return <span className="badge-closed">closed</span>
  return <span className="badge-hybrid">{status || '—'}</span>
}
