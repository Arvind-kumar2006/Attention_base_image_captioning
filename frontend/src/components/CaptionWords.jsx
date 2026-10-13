/**
 * The generated caption, one button per word.
 *
 * Keyboard: Left/Right move between words, Home/End jump to the ends, Enter or
 * Space toggles the overlay. Roving tabindex keeps the group a single tab stop,
 * which is the standard pattern for a toolbar-like widget.
 */
import { useCallback, useEffect, useRef } from 'react'

export default function CaptionWords({ words = [], selectedIndex, onSelect }) {
  const containerRef = useRef(null)
  const buttonRefs = useRef([])

  const count = words.length
  const hasSelection = selectedIndex >= 0 && selectedIndex < count

  // When selection moves by keyboard, keep the focused word in view.
  useEffect(() => {
    if (!hasSelection) return
    const button = buttonRefs.current[selectedIndex]
    button?.scrollIntoView({ block: 'nearest', inline: 'nearest' })
  }, [selectedIndex, hasSelection])

  const focusWord = useCallback((index) => {
    const clamped = Math.max(0, Math.min(count - 1, index))
    buttonRefs.current[clamped]?.focus()
    onSelect(clamped)
  }, [count, onSelect])

  const handleKeyDown = useCallback(
    (event, index) => {
      switch (event.key) {
        case 'ArrowRight':
          event.preventDefault()
          focusWord(index + 1)
          break
        case 'ArrowLeft':
          event.preventDefault()
          focusWord(index - 1)
          break
        case 'Home':
          event.preventDefault()
          focusWord(0)
          break
        case 'End':
          event.preventDefault()
          focusWord(count - 1)
          break
        case 'Enter':
        case ' ':
        case 'Spacebar':
          // Space would otherwise scroll the page.
          event.preventDefault()
          onSelect(selectedIndex === index ? -1 : index)
          break
        case 'Escape':
          event.preventDefault()
          onSelect(-1)
          break
        default:
          break
      }
    },
    [count, focusWord, onSelect, selectedIndex],
  )

  if (count === 0) return null

  return (
    <div
      ref={containerRef}
      role="group"
      aria-label="Caption words. Use the left and right arrow keys to step through them and see where the model looked."
      className="flex flex-wrap items-baseline gap-1.5"
    >
      {words.map((word, index) => {
        const isSelected = index === selectedIndex
        return (
          <button
            key={`${word}-${index}`}
            ref={(element) => {
              buttonRefs.current[index] = element
            }}
            type="button"
            // roving tabindex: exactly one word is in the tab order
            tabIndex={isSelected || (selectedIndex === -1 && index === 0) ? 0 : -1}
            aria-pressed={isSelected}
            // Selection is driven by click and by the arrow keys, never by focus: a click
            // also fires focus, so selecting in onFocus would run first and make
            // every click toggle the word straight back off.
            onClick={() => onSelect(selectedIndex === index ? -1 : index)}
            onKeyDown={(event) => handleKeyDown(event, index)}
            className={`rounded px-1.5 py-0.5 text-lg leading-8 transition-colors ${
              isSelected
                ? 'bg-accent text-white underline decoration-2 underline-offset-4'
                : 'text-fg hover:bg-accent-soft'
            }`}
          >
            {word}
          </button>
        )
      })}
    </div>
  )
}