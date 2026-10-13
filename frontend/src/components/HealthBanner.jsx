import { useEffect, useState } from 'react'

import { fetchHealth } from '../api/client'

/**
 * Shows whether the backend is reachable and whether the model loaded.
 * A failed model load is a degraded service, not a dead one, so it is called out
 * separately from "no backend at all".
 */
export default function HealthBanner() {
  const [state, setState] = useState({ status: 'checking', modelLoaded: null })

  useEffect(() => {
    let cancelled = false
    fetchHealth()
      .then((payload) => {
        if (cancelled) return
        setState({ status: 'ok', modelLoaded: Boolean(payload.model_loaded) })
      })
      .catch(() => {
        if (!cancelled) setState({ status: 'offline', modelLoaded: null })
      })
    return () => {
      cancelled = true
    }
  }, [])

  if (state.status === 'checking') {
    return (
      <p className="border-b border-border bg-canvas px-4 py-2 text-center text-sm text-fg-muted">
        Checking the API…
      </p>
    )
  }

  if (state.status === 'offline') {
    return (
      <p role="status" className="border-b border-border bg-danger-soft px-4 py-2 text-center text-sm font-medium text-danger">
        Cannot reach the API. Set <code className="font-mono">VITE_API_URL</code> and make sure the
        backend is running.
      </p>
    )
  }

  if (state.modelLoaded === false) {
    return (
      <p role="status" className="border-b border-border bg-danger-soft px-4 py-2 text-center text-sm font-medium text-danger">
        The API is up but the model is not loaded — captions will return 503. Check
        CHECKPOINT_PATH.
      </p>
    )
  }

  return null
}