/** Aggregate numbers from /api/v1/stats. */
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

import { fetchStats } from '../api/client'
import ErrorMessage from '../components/ErrorMessage'

function StatCard({ label, value, hint, to }) {
  const body = (
    <>
      <dt className="text-sm font-medium text-fg-muted">{label}</dt>
      <dd className="mt-2 text-3xl font-bold tabular-nums tracking-tight">{value}</dd>
      {hint ? <p className="mt-1 text-sm text-fg-muted">{hint}</p> : null}
    </>
  )

  if (to) {
    return (
      <Link
        to={to}
        className="card block px-5 py-4 transition-colors hover:border-accent hover:bg-accent-soft"
      >
        <dl>{body}</dl>
      </Link>
    )
  }
  return (
    <div className="card px-5 py-4">
      <dl>{body}</dl>
    </div>
  )
}

export default function DashboardPage() {
  const [stats, setStats] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    fetchStats()
      .then((payload) => {
        if (!cancelled) setStats(payload)
      })
      .catch((problem) => {
        if (!cancelled) setError(problem?.message ?? 'Could not load the statistics.')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const scores = stats?.evaluation?.scores ?? null
  const bleu4 = scores?.Bleu_4
  const hasBleu4 = typeof bleu4 === 'number'

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
        <p className="mt-1 text-sm text-fg-muted">
          What this instance has produced so far.
        </p>
      </header>

      {error ? <ErrorMessage message={error} /> : null}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <StatCard
          label="Total captions"
          value={loading ? '—' : (stats?.total_captions ?? 0).toLocaleString()}
          hint="Rows in the history table"
          to="/history"
        />
        <StatCard
          label="Average inference time"
          value={loading ? '—' : `${Math.round(stats?.avg_generation_time_ms ?? 0)} ms`}
          hint="Mean over every caption"
        />
        <StatCard
          label="Latest BLEU-4"
          // A dash rather than 0 when no evaluation has been recorded: zero is a
          // real score and would read as a measured failure.
          value={loading ? '—' : hasBleu4 ? bleu4.toFixed(4) : '—'}
          hint={
            stats?.evaluation
              ? `${stats.evaluation.split} split · ${stats.evaluation.num_images?.toLocaleString() ?? 0} images`
              : 'No evaluation recorded yet'
          }
          to="/evaluation"
        />
      </div>

      {stats?.evaluation ? (
        <section className="card overflow-x-auto" aria-labelledby="latest-eval">
          <h2 id="latest-eval" className="border-b border-border px-5 py-3 text-sm font-semibold">
            Latest evaluation
          </h2>
          <table className="w-full border-collapse text-left text-sm">
            <caption className="sr-only">
              Scores from the most recent ml/evaluate.py run.
            </caption>
            <thead>
              <tr className="border-b border-border bg-canvas">
                <th scope="col" className="px-5 py-2.5 font-semibold">
                  Metric
                </th>
                <th scope="col" className="px-5 py-2.5 font-semibold">
                  Score
                </th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(stats.evaluation.scores ?? {}).map(([name, value]) => (
                <tr key={name} className="border-b border-border last:border-0">
                  <th scope="row" className="px-5 py-2.5 font-medium">
                    {name}
                  </th>
                  <td className="px-5 py-2.5 tabular-nums">{Number(value).toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="border-t border-border px-5 py-3 text-xs text-fg-muted">
            Tokenised with {stats.evaluation.tokenizer || 'unknown tokenizer'} on the{' '}
            {stats.evaluation.split} split.
          </p>
        </section>
      ) : null}
    </div>
  )
}