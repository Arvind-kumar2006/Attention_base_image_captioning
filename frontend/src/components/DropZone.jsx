/**
 * Drag-and-drop plus a keyboard-reachable file button.
 *
 * The visible drop zone is a label wrapping a real <input type="file">, so
 * Space/Enter opens the picker and the control is announced as a file input by
 * screen readers rather than as a custom button.
 */
import { useRef, useState } from 'react'

import { ACCEPT, ALLOWED_EXTENSIONS } from '../api/client'

export default function DropZone({ onFile, disabled, hint }) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)

  const handleDrop = (event) => {
    event.preventDefault()
    setDragging(false)
    if (disabled) return
    const file = event.dataTransfer?.files?.[0]
    if (file) onFile(file)
  }

  return (
    <div
      onDragOver={(event) => {
        event.preventDefault()
        if (!disabled) setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={`rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
        dragging ? 'border-accent bg-accent-soft' : 'border-border-strong bg-surface'
      }`}
    >
      <input
        ref={inputRef}
        id="image-upload"
        type="file"
        accept={ACCEPT}
        className="sr-only"
        disabled={disabled}
        onChange={(event) => {
          const file = event.target.files?.[0]
          if (file) onFile(file)
          // reset so re-picking the same file fires change again
          event.target.value = ''
        }}
      />
      <label
        htmlFor="image-upload"
        className={`block cursor-pointer text-sm font-semibold ${
          disabled ? 'cursor-not-allowed text-fg-muted' : 'text-accent'
        }`}
      >
        <span className="btn-primary pointer-events-none">Choose an image</span>
      </label>
      <p className="mt-3 text-sm text-fg-muted">
        or drag and drop it here. {hint ?? `JPEG or PNG, up to 10 MB (${ALLOWED_EXTENSIONS.join(', ')}).`}
      </p>
    </div>
  )
}