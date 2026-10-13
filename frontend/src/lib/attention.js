/**
 * Attention helpers.
 *
 * The API returns attention as num_words x 7 x 7. Each 7x7 grid is the model's
 * distribution over the 49 image regions, so its values sum to 1 already; the
 * per-word normalisation here is about *contrast*, making the peak land on 1.0
 * so the heatmap is visible regardless of how flat the model was.
 */

/** Clamp to the 0..1 range. */
function clamp01(value) {
  if (!Number.isFinite(value)) return 0
  if (value < 0) return 0
  if (value > 1) return 1
  return value
}

/**
 * Normalise one word's 7x7 grid to 0..1.
 *
 * A constant grid (every value equal) has nothing to highlight, so it becomes
 * all zeros rather than dividing by zero.
 *
 * @param {number[][]} grid 7x7 weights for a single word
 * @returns {number[][]} 7x7 values in 0..1
 */
export function normaliseGrid(grid) {
  if (!Array.isArray(grid) || grid.length === 0) return []
  const flat = grid.flat().filter((value) => Number.isFinite(value))
  if (flat.length === 0) return grid.map((row) => row.map(() => 0))

  let min = Infinity
  let max = -Infinity
  for (const value of flat) {
    if (value < min) min = value
    if (value > max) max = value
  }
  const spread = max - min
  if (spread <= 0) return grid.map((row) => row.map(() => 0))

  return grid.map((row) => row.map((value) => clamp01((value - min) / spread)))
}

/** Min/max of a 7x7 grid, for labelling the scale. */
export function gridRange(grid) {
  const flat = (grid || []).flat().filter((value) => Number.isFinite(value))
  if (flat.length === 0) return { min: 0, max: 0 }
  return { min: Math.min(...flat), max: Math.max(...flat) }
}

/**
 * Blend a 7x7 grid into an RGBA pixel buffer at full canvas size.
 *
 * The grid is drawn into a 7x7 ImageData and then upscaled with the canvas
 * `imageSmoothingEnabled` flag left on, so the browser interpolates it smoothly
 * instead of showing hard 49-pixel blocks.
 *
 * @param {number[][]} grid 7x7 values, ideally already normalised
 * @param {number} width target width in pixels
 * @param {number} height target height in pixels
 * @returns {ImageData} RGBA buffer the size of the target
 */
export function renderHeatmap(grid, width, height) {
  const blank = new ImageData(width, height)

  if (!Array.isArray(grid) || grid.length === 0 || width < 1 || height < 1) {
    return blank
  }

  // Turbo-like ramp: blue (cold) through green to red (hot). Perceptually
  // ordered, and distinguishable without relying on hue alone.
  const stops = [
    [0.0, [49, 46, 129]],
    [0.25, [33, 113, 181]],
    [0.5, [34, 168, 132]],
    [0.75, [253, 187, 43]],
    [1.0, [214, 39, 40]],
  ]

  const colourAt = (t) => {
    const clamped = clamp01(t)
    for (let i = 0; i < stops.length - 1; i += 1) {
      const [p0, c0] = stops[i]
      const [p1, c1] = stops[i + 1]
      if (clamped >= p0 && clamped <= p1) {
        const ratio = p1 === p0 ? 0 : (clamped - p0) / (p1 - p0)
        return [
          Math.round(c0[0] + (c1[0] - c0[0]) * ratio),
          Math.round(c0[1] + (c1[1] - c0[1]) * ratio),
          Math.round(c0[2] + (c1[2] - c0[2]) * ratio),
        ]
      }
    }
    return stops[stops.length - 1][1]
  }

  const rows = grid.length
  const cols = grid[0]?.length ?? 0
  if (rows === 0 || cols === 0) return blank

  // ImageData.data hands back a copy, so fill a buffer and wrap it once at the
  // end rather than writing into a detached array.
  const pixels = new Uint8ClampedArray(width * height * 4)
  for (let y = 0; y < height; y += 1) {
    // Bilinear sample between the two nearest grid rows.
    const gy = ((y + 0.5) / height) * rows - 0.5
    const y0 = Math.max(0, Math.min(rows - 1, Math.floor(gy)))
    const y1 = Math.min(rows - 1, y0 + 1)
    const ty = clamp01(gy - Math.floor(gy))

    for (let x = 0; x < width; x += 1) {
      const gx = ((x + 0.5) / width) * cols - 0.5
      const x0 = Math.max(0, Math.min(cols - 1, Math.floor(gx)))
      const x1 = Math.min(cols - 1, x0 + 1)
      const tx = clamp01(gx - Math.floor(gx))

      const top = grid[y0][x0] * (1 - tx) + grid[y0][x1] * tx
      const bottom = grid[y1][x0] * (1 - tx) + grid[y1][x1] * tx
      const value = clamp01(top * (1 - ty) + bottom * ty)

      const [r, g, b] = colourAt(value)
      const offset = (y * width + x) * 4
      pixels[offset] = r
      pixels[offset + 1] = g
      pixels[offset + 2] = b
      // Alpha rides on intensity, so cold regions stay nearly transparent and
      // the photo is only obscured where the model actually looked.
      pixels[offset + 3] = Math.round(255 * value * 0.75)
    }
  }

  return new ImageData(pixels, width, height)
}