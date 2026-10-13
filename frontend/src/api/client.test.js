import { describe, expect, it, vi } from 'vitest'

import {
  ACCEPT,
  API_BASE,
  ApiError,
  MAX_UPLOAD_BYTES,
  apiUrl,
  imageUrl,
  uploadImage,
  validateImageFile,
} from './client'

function file(name, type, size) {
  const blob = new Blob([new Uint8Array(size)], { type })
  return new File([blob], name, { type })
}

describe('apiUrl', () => {
  // API_BASE is resolved once at module load from VITE_API_URL, so these assert
  // against the real value rather than a per-test stub.
  it('has no trailing slash', () => {
    expect(API_BASE.endsWith('/')).toBe(false)
  })

  it('builds an absolute url from the base', () => {
    expect(apiUrl('/api/v1/stats')).toBe(`${API_BASE}/api/v1/stats`)
    expect(apiUrl('/api/v1/stats').startsWith('http')).toBe(true)
  })

  it('tolerates a path with no leading slash', () => {
    expect(apiUrl('api/v1/stats')).toBe(`${API_BASE}/api/v1/stats`)
  })
})

describe('imageUrl', () => {
  it('passes absolute urls through untouched', () => {
    expect(imageUrl('https://cdn.test/a.jpg')).toBe('https://cdn.test/a.jpg')
  })

  it('returns null for nothing', () => {
    expect(imageUrl(null)).toBeNull()
    expect(imageUrl('')).toBeNull()
  })

  it('accepts a bare id', () => {
    expect(imageUrl('abc123')).toContain('/api/v1/images/abc123')
  })
})

describe('validateImageFile', () => {
  it('accepts a png', () => {
    expect(validateImageFile(file('a.png', 'image/png', 1000))).toBeNull()
  })

  it('accepts a jpeg', () => {
    expect(validateImageFile(file('a.jpg', 'image/jpeg', 1000))).toBeNull()
    expect(validateImageFile(file('a.JPEG', 'image/jpeg', 1000))).toBeNull()
  })

  it('rejects a pdf', () => {
    expect(validateImageFile(file('a.pdf', 'application/pdf', 100))).toMatch(/JPEG and PNG/)
  })

  it('rejects a file whose extension lies about a jpeg type', () => {
    expect(validateImageFile(file('virus.jpg.exe', 'image/jpeg', 100))).toMatch(/JPEG and PNG/)
  })

  it('rejects an oversize file', () => {
    const big = file('a.png', 'image/png', MAX_UPLOAD_BYTES + 1)
    expect(validateImageFile(big)).toMatch(/10 MB/)
  })

  it('accepts a file exactly on the limit', () => {
    expect(validateImageFile(file('a.png', 'image/png', MAX_UPLOAD_BYTES))).toBeNull()
  })

  it('rejects an empty file', () => {
    expect(validateImageFile(file('a.png', 'image/png', 0))).toMatch(/empty/)
  })

  it('rejects nothing at all', () => {
    expect(validateImageFile(null)).toBeTruthy()
  })
})

describe('ACCEPT', () => {
  it('lists both mime types and extensions', () => {
    expect(ACCEPT).toContain('image/jpeg')
    expect(ACCEPT).toContain('image/png')
    expect(ACCEPT).toContain('.jpeg')
  })
})

describe('uploadImage', () => {
  it('surfaces the API detail message on 400', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({ detail: 'could not decode the file as an image' }),
    })
    await expect(uploadImage(file('a.jpg', 'image/jpeg', 10))).rejects.toMatchObject({
      status: 400,
      message: 'could not decode the file as an image',
    })
  })

  it('surfaces 413 separately so the UI can say "too large"', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 413,
      json: async () => ({ detail: 'file is larger than the 10 MB limit' }),
    })
    const error = await uploadImage(file('a.png', 'image/png', 10)).catch((e) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect(error.status).toBe(413)
  })

  it('falls back to the status text when the body is not json', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 502,
      json: async () => {
        throw new Error('not json')
      },
    })
    await expect(uploadImage(file('a.png', 'image/png', 10))).rejects.toThrow(/502/)
  })

  it('reports an unreachable API without leaking a TypeError', async () => {
    global.fetch = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'))
    await expect(uploadImage(file('a.png', 'image/png', 10))).rejects.toThrow(
      /Could not reach the API/,
    )
  })

  it('does not send a Content-Type, so the browser sets the multipart boundary', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 201,
      json: async () => ({ id: 'x', caption: 'a dog runs', words: ['a'], attention: [], time_ms: 1 }),
    })
    await uploadImage(file('a.png', 'image/png', 10))
    const [, options] = global.fetch.mock.calls[0]
    expect(options.headers).toBeUndefined()
    expect(options.body).toBeInstanceOf(FormData)
  })

  it('rejects a bad file before making a request', async () => {
    global.fetch = vi.fn()
    await expect(uploadImage(file('a.pdf', 'application/pdf', 10))).rejects.toThrow()
    expect(global.fetch).not.toHaveBeenCalled()
  })
})