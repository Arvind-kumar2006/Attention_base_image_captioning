/** Vitest setup: jest-dom matchers and a canvas stub for jsdom. */
import '@testing-library/jest-dom/vitest'
import { afterEach, vi } from 'vitest'
import { cleanup } from '@testing-library/react'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

// jsdom has no canvas backend, so ImageData and the 2D context are stubs. The
// heatmap maths is covered in lib/attention.test.js against real buffers.
if (!globalThis.ImageData) {
  globalThis.ImageData = class ImageData {
    constructor(data, width, height) {
      if (typeof data === 'number') {
        this.width = data
        this.height = width
        this.data = new Uint8ClampedArray(data * width * 4)
      } else {
        this.data = data
        this.width = width
        this.height = height
      }
    }
  }
}

if (typeof HTMLCanvasElement !== 'undefined') {
  HTMLCanvasElement.prototype.getContext = () => ({
    clearRect: () => {},
    drawImage: () => {},
    putImageData: () => {},
    imageSmoothingEnabled: true,
    imageSmoothingQuality: 'high',
  })
}

// jsdom does not implement scrollIntoView, which CaptionWords calls to keep the
// keyboard-selected word in view.
if (typeof Element !== 'undefined') {
  Element.prototype.scrollIntoView = function scrollIntoView() {}
}