import { describe, expect, it } from 'vitest'

import { gridRange, normaliseGrid, renderHeatmap } from './attention'

/** A 7x7 grid with a single hot spot in the given corner. */
function peaked(hotIndex, peak = 1) {
  const grid = Array.from({ length: 7 }, () => Array(7).fill(0))
  grid[Math.floor(hotIndex / 7)][hotIndex % 7] = peak
  return grid
}

describe('normaliseGrid', () => {
  it('maps the range onto 0..1', () => {
    const out = normaliseGrid(peaked(0))
    expect(out.flat().every((v) => v >= 0 && v <= 1)).toBe(true)
    expect(Math.max(...out.flat())).toBe(1)
    expect(Math.min(...out.flat())).toBe(0)
  })

  it('handles a constant grid without dividing by zero', () => {
    const flat = Array.from({ length: 7 }, () => Array(7).fill(0.25))
    const out = normaliseGrid(flat)
    expect(out).toHaveLength(7)
    expect(out.flat().every((v) => v === 0)).toBe(true)
  })

  it('preserves ordering', () => {
    const grid = Array.from({ length: 7 }, (_, r) =>
      Array.from({ length: 7 }, (_, c) => (r * 7 + c) / 48),
    )
    const out = normaliseGrid(grid)
    // the last cell had the largest value and must stay the largest
    let bestRow = 0
    let bestCol = 0
    for (let r = 0; r < 7; r += 1) {
      for (let c = 0; c < 7; c += 1) {
        if (out[r][c] > out[bestRow][bestCol]) {
          bestRow = r
          bestCol = c
        }
      }
    }
    expect([bestRow, bestCol]).toEqual([6, 6])
  })

  it('returns an empty array for junk input', () => {
    expect(normaliseGrid(null)).toEqual([])
    expect(normaliseGrid([])).toEqual([])
  })
})

describe('gridRange', () => {
  it('reports min and max', () => {
    const { min, max } = gridRange(peaked(3, 0.8))
    expect(min).toBe(0)
    expect(max).toBeCloseTo(0.8)
  })

  it('returns zeros for an empty grid', () => {
    expect(gridRange([])).toEqual({ min: 0, max: 0 })
  })
})

describe('renderHeatmap', () => {
  it('produces a full-size RGBA buffer', () => {
    const image = renderHeatmap(peaked(24), 224, 224)
    expect(image.width).toBe(224)
    expect(image.height).toBe(224)
    expect(image.data.length).toBe(224 * 224 * 4)
  })

  it('is transparent where the model did not look', () => {
    // one hot cell at the top-left, everything else 0
    const image = renderHeatmap(peaked(0), 140, 140)
    const alphaAt = (x, y) => image.data[(y * 140 + x) * 4 + 3]
    // bottom-right is far from the hot cell
    expect(alphaAt(135, 135)).toBe(0)
    expect(alphaAt(2, 2)).toBeGreaterThan(0)
  })

  it('interpolates between grid cells rather than showing 49 flat blocks', () => {
    const image = renderHeatmap(peaked(24), 140, 140)
    const distinct = new Set()
    for (let i = 0; i < image.data.length; i += 4) {
      distinct.add(
        `${image.data[i]},${image.data[i + 1]},${image.data[i + 2]},${image.data[i + 3]}`,
      )
    }
    // Nearest-neighbour upsampling of a 7x7 grid would give at most 49 distinct
    // pixels; bilinear gives far more, which is what "smooth" means here.
    expect(distinct.size).toBeGreaterThan(49 * 2)
  })

  it('returns a blank buffer for junk input', () => {
    const image = renderHeatmap(null, 10, 10)
    expect(image.data.every((byte) => byte === 0)).toBe(true)
  })

  it('handles a 1x1 grid without throwing', () => {
    const image = renderHeatmap([[1]], 8, 8)
    expect(image.data.length).toBe(8 * 8 * 4)
  })
})