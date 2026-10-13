/** Past captions, searchable. Clicking a row reopens it on the caption page. */
import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { fetchCaptions, imageUrl } from '../api/client'
import ErrorMessage from '../components/ErrorMessage'

function formatDate(iso) {
  if (!iso) return '—'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return '—'
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function timeOnly(iso) {
  if (!iso) return '—'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return '—'
  return date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
}

export default function HistoryPage() {
  const navigate = useNavigate()
  const [captions, setCaptions] = useState([])
  const [query, setQuery] = useState('')
  const [debounced, setDebounced] = useState('')
  const [settled, setSettled] = useState(null) // { key, error }
  const [error, setError] = useState('')

  // Derived rather than set inside the effect: a request is in flight until one
  // settles for the current key, so there is no cascading-render setState.
  const loading = settled?.key !== debounced

  // Debounce so typing does not fire a request per keystroke.
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(query.trim()), 250)
    return () => clearTimeout(timer)
  }, [query])

  useEffect(() => {
    let cancelled = false
    fetchCaptions({ limit: 100, q: debounced })
      .then((page) => {
        if (cancelled) return
        setCaptions(page.items ?? [])
        setError('')
        setSettled({ key: debounced, error: null })
      })
      .catch((problem) => {
        if (cancelled) return
        const message = problem?.message ?? 'Could not load the history.'
        setError(message)
        setSettled({ key: debounced, error: message })
      })
    return () => {
      cancelled = true
    }
    // `debounced` is the only input; re-running on the raw query would fire a
    // request per keystroke and defeat the debounce.
  }, [debounced])

  const summary = useMemo(
    () => `${captions.length} ${captions.length === 1 ? 'caption' : 'captions'}`,
    [captions.length],
  )

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold tracking-tight">History</h1>
        <p className="mt-1 text-sm text-fg-muted">
          Every caption generated in this session. Select a row to reopen it.
        </p>
      </header>

      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="w-full max-w-sm">
          <label className="label" htmlFor="history-search">
            Search captions
          </label>
          <input
            id="history-search"
            type="search"
            className="field"
            placeholder="e.g. dog"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </div>
        <p className="text-sm text-fg-muted" role="status" aria-live="polite">
          {loading ? 'Loading…' : summary}
        </p>
      </div>

      {error ? <ErrorMessage message={error} /> : null}

      <div className="card overflow-x-auto">
        <table className="w-full border-collapse text-left text-sm">
          <caption className="sr-only">
            Generated captions, newest first. Activate a row to reopen it in the caption view.
          </caption>
          <thead>
            <tr className="border-b border-border bg-canvas">
              <th scope="col" className="px-4 py-3 font-semibold">
                Image
              </th>
              <th scope="col" className="px-4 py-3 font-semibold">
                Caption
              </th>
              <th scope="col" className="px-4 py-3 font-semibold">
                Inference
              </th>
              <th scope="col" className="px-4 py-3 font-semibold">
                Date
              </th>
            </tr>
          </thead>
          <tbody>
            {captions.map((caption) => (
              <tr
                key={caption.id}
                tabIndex={0}
                role="link"
                onClick={() => navigate('/caption', { state: { caption } })}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault()
                    navigate('/caption', { state: { caption } })
                  }
                }}
                className="cursor-pointer border-b border-border last:border-0 hover:bg-accent-soft"
              >
                <td className="px-4 py-3">
                  {caption.image_url ? (
                    <img
                      src={imageUrl(caption.image_url)}
                      alt=""
                      width="64"
                      height="48"
                      loading="lazy"
                      className="h-12 w-16 rounded border border-border object-cover"
                    />
                  ) : (
                    <span className="text-fg-muted" aria-hidden="true">
                      —
                    </span>
                  )}
                </td>
                <td className="px-4 py-3 font-medium">{caption.caption || '—'}</td>
                <td className="px-4 py-3 tabular-nums text-fg-muted">
                  {Math.round(caption.time_ms)} ms
                </td>
                <td className="px-4 py-3 text-fg-muted">
                  <time dateTime={caption.created_at} title={formatDate(caption.created_at)}>
                    {formatDate(caption.created_at)}
                  </time>
                  <span className="block text-xs">{timeOnly(caption.created_at)}</span>
                </td>
              </tr>
            ))}
            {!loading && captions.length === 0 ? (
              <tr>
                <td colSpan={4} className="px-4 py-10 text-center text-fg-muted">
                  {debounced
                    ? `No captions match “${debounced}”.`
                    : 'No captions yet. Upload an image to get started.'}
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  )
}