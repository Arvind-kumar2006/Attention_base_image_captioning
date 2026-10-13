/** Upload an image and read the caption the model produced. */
import { useCallback, useEffect, useState } from 'react'
import { useLocation, useSearchParams } from 'react-router-dom'

import { fetchCaption, imageUrl, uploadImage, validateImageFile } from '../api/client'
import AttentionOverlay from '../components/AttentionOverlay'
import CaptionSkeleton from '../components/CaptionSkeleton'
import CaptionWords from '../components/CaptionWords'
import DropZone from '../components/DropZone'
import ErrorMessage from '../components/ErrorMessage'
import { gridRange } from '../lib/attention'

export default function CaptionPage() {
  const location = useLocation()
  const [searchParams, setSearchParams] = useSearchParams()

  const [result, setResult] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [status, setStatus] = useState('idle') // idle | loading | done | error
  const [error, setError] = useState('')
  const [selectedIndex, setSelectedIndex] = useState(-1)

  // A row clicked in the history table arrives through router state, so no
  // refetch is needed and the record shown is exactly the one that was clicked.
  const handedOver = location.state?.caption

  useEffect(() => {
    if (!handedOver) return
    setResult(handedOver)
    setStatus('done')
    setError('')
    setSelectedIndex(-1)
    setPreviewUrl(null)
  }, [handedOver])

  // /caption?id=<id> also works, for a shared link.
  const captionId = searchParams.get('id')
  useEffect(() => {
    if (!captionId || handedOver || result?.id === captionId) return
    let cancelled = false
    setStatus('loading')
    fetchCaption(captionId)
      .then((caption) => {
        if (cancelled) return
        setResult(caption)
        setStatus('done')
      })
      .catch((problem) => {
        if (cancelled) return
        setError(problem?.message ?? 'Could not load that caption.')
        setStatus('error')
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [captionId])

  const handleFile = useCallback(
    async (file) => {
      // Validated in the browser first, so a bad file never becomes a round trip.
      const problem = validateImageFile(file)
      if (problem) {
        setError(problem)
        setStatus('error')
        return
      }

      setError('')
      setStatus('loading')
      setSelectedIndex(-1)
      if (previewUrl) URL.revokeObjectURL(previewUrl)
      setPreviewUrl(URL.createObjectURL(file))

      // Clear any ?id= left over from a previous view.
      if (searchParams.has('id')) setSearchParams({}, { replace: true })

      try {
        const caption = await uploadImage(file)
        setResult(caption)
        setStatus('done')
      } catch (uploadProblem) {
        setError(uploadProblem?.message ?? 'Something went wrong.')
        setStatus('error')
      }
    },
    [previewUrl, searchParams, setSearchParams],
  )

  const activeGrid =
    result?.attention && selectedIndex >= 0 ? result.attention[selectedIndex] : null
  const range = activeGrid ? gridRange(activeGrid) : null
  const selectedWord = selectedIndex >= 0 ? result?.words?.[selectedIndex] : null
  const source = result?.image_url ? imageUrl(result.image_url) : previewUrl

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold tracking-tight">Caption an image</h1>
        <p className="mt-1 text-sm text-fg-muted">
          The model writes a caption and shows where it looked for each word.
        </p>
      </header>

      <DropZone onFile={handleFile} disabled={status === 'loading'} />

      {error ? <ErrorMessage message={error} onDismiss={() => setError('')} /> : null}

      {status === 'idle' && !result ? (
        <p className="card px-6 py-12 text-center text-lg text-fg-muted">
          Upload an image to generate a caption
        </p>
      ) : null}

      {status === 'loading' ? <CaptionSkeleton /> : null}

      {result && status !== 'loading' ? (
        <section className="grid gap-6 lg:grid-cols-2" aria-label="Generated caption">
          <div>
            <AttentionOverlay
              imageSrc={source}
              grid={activeGrid}
              alt={result.caption ? `Image captioned: ${result.caption}` : 'Uploaded image'}
              active={Boolean(activeGrid)}
            />
            <dl className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-sm">
              <div className="flex gap-2">
                <dt className="text-fg-muted">Inference time</dt>
                <dd className="font-semibold">{Math.round(result.time_ms)} ms</dd>
              </div>
              {range ? (
                <div className="flex gap-2">
                  <dt className="text-fg-muted">Attention range</dt>
                  <dd className="font-semibold">
                    {range.min.toFixed(4)} – {range.max.toFixed(4)}
                  </dd>
                </div>
              ) : null}
            </dl>
            <p className="mt-2 text-sm text-fg-muted">
              Select a word to see the 7×7 attention map the model used for it.
            </p>
          </div>

          <div>
            <h2 className="text-sm font-semibold uppercase tracking-wide text-fg-muted">
              Caption
            </h2>
            <div className="mt-2">
              <CaptionWords
                words={result.words ?? []}
                selectedIndex={selectedIndex}
                onSelect={setSelectedIndex}
              />
            </div>

            {/* Announces the selected word to screen readers. */}
            <p className="sr-only" role="status" aria-live="polite">
              {selectedWord
                ? `Selected word: ${selectedWord}. Attention map shown over the image.`
                : 'No word selected.'}
            </p>

            {result.id ? (
              <p className="mt-4 text-xs text-fg-muted">
                Caption id <code className="font-mono">{result.id}</code>
              </p>
            ) : null}
          </div>
        </section>
      ) : null}
    </div>
  )
}