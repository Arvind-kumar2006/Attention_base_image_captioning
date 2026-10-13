/**
 * Draws a word's 7x7 attention grid over the image on a canvas.
 *
 * The overlay is purely decorative: the words below the image are the accessible
 * representation of the same data, so this is aria-hidden.
 */
import { useEffect, useRef } from 'react'

import { normaliseGrid, renderHeatmap } from '../lib/attention'

export default function AttentionOverlay({ imageSrc, grid, alt, active }) {
  const canvasRef = useRef(null)
  const imageRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !grid || grid.length === 0) return undefined

    const image = new Image()
    image.crossOrigin = 'anonymous'
    imageRef.current = image

    let cancelled = false

    const draw = () => {
      if (cancelled) return
      const width = image.naturalWidth || image.width || 1
      const height = image.naturalHeight || image.height || 1

      canvas.width = width
      canvas.height = height

      const context = canvas.getContext('2d')
      if (!context) return

      // Upscale the 7x7 grid to the image size with smoothing on, so the heatmap
      // is a soft field rather than 49 visible blocks.
      const heatmap = renderHeatmap(normaliseGrid(grid), width, height)
      const scratch = document.createElement('canvas')
      scratch.width = width
      scratch.height = height
      scratch.getContext('2d')?.putImageData(heatmap, 0, 0)

      context.clearRect(0, 0, width, height)
      context.imageSmoothingEnabled = true
      context.imageSmoothingQuality = 'high'
      context.drawImage(scratch, 0, 0, width, height)
    }

    image.onload = draw
    image.onerror = () => {
      // A failed overlay should never take the caption view down with it.
      if (!cancelled) {
        const context = canvas.getContext('2d')
        context?.clearRect(0, 0, canvas.width, canvas.height)
      }
    }
    if (imageSrc) image.src = imageSrc

    return () => {
      cancelled = true
      image.onload = null
      image.onerror = null
    }
  }, [imageSrc, grid])

  return (
    <div className="relative overflow-hidden rounded-xl border border-border bg-black/5">
      <img
        ref={imageRef}
        src={imageSrc}
        alt={alt}
        className="block max-h-[60vh] w-full object-contain"
      />
      <canvas
        ref={canvasRef}
        aria-hidden="true"
        className={`pointer-events-none absolute inset-0 h-full w-full transition-opacity duration-150 ${
          active ? 'opacity-100' : 'opacity-0'
        }`}
      />
    </div>
  )
}