/**
 * The only module that talks to the backend.
 *
 * Every request is built from VITE_API_URL so the dev server, a staging deploy
 * and production all point somewhere different without a code change.
 */

const RAW_BASE = import.meta.env.VITE_API_URL

/** Trailing slashes stripped so `${API_BASE}/api/v1` never doubles up. */
export const API_BASE = (RAW_BASE ?? 'http://localhost:8000').replace(/\/+$/, '')

/** Mirrors the backend's MAX_UPLOAD_BYTES and ALLOWED_CONTENT_TYPES. */
export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024
export const ALLOWED_TYPES = ['image/jpeg', 'image/png']
export const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png']

/** Value for the file input's accept attribute, types and extensions. */
export const ACCEPT = [...ALLOWED_TYPES, ...ALLOWED_EXTENSIONS].join(',')

/** An error carrying the HTTP status, so callers can branch on 413 vs 400. */
export class ApiError extends Error {
  constructor(message, { status = 0, detail = null } = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

/** Absolute URL for a backend path such as `/api/v1/stats`. */
export function apiUrl(path) {
  return `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`
}

/** Absolute URL for a stored image. */
export function imageUrl(pathOrId) {
  if (!pathOrId) return null
  if (/^https?:\/\//i.test(pathOrId)) return pathOrId
  if (pathOrId.startsWith('/')) return apiUrl(pathOrId)
  return apiUrl(`/api/v1/images/${pathOrId}`)
}

async function parseError(response) {
  // The API always answers with {"detail": ...}; fall back to the status text
  // if something in front of it returned an HTML error page instead.
  let detail = null
  try {
    const body = await response.json()
    if (body && typeof body.detail === 'string') detail = body.detail
  } catch {
    detail = null
  }
  return new ApiError(detail || `Request failed (${response.status})`, {
    status: response.status,
    detail,
  })
}

/** JSON GET. */
async function getJson(path, { signal } = {}) {
  let response
  try {
    response = await fetch(apiUrl(path), { signal, headers: { Accept: 'application/json' } })
  } catch (error) {
    if (error?.name === 'AbortError') throw error
    throw new ApiError('Could not reach the API. Is the backend running?')
  }
  if (!response.ok) throw await parseError(response)
  return response.json()
}

/**
 * Browser-side validation, mirroring what the backend enforces.
 * Returns null when the file is acceptable, otherwise a message.
 */
export function validateImageFile(file) {
  if (!file) return 'No file selected.'

  const type = (file.type || '').toLowerCase()
  const name = (file.name || '').toLowerCase()

  if (!ALLOWED_TYPES.includes(type)) {
    return 'Only JPEG and PNG images are supported.'
  }
  // The type is a hint the client controls, so check the extension too.
  if (!ALLOWED_EXTENSIONS.some((extension) => name.endsWith(extension))) {
    return 'Only JPEG and PNG images are supported.'
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return 'File is larger than the 10 MB limit.'
  }
  if (file.size === 0) return 'That file is empty.'
  return null
}

/**
 * Upload an image and caption it.
 * @returns {Promise<object>} caption, words, attention (n x 7 x 7), time_ms, image_url
 */
export async function uploadImage(file, { signal } = {}) {
  const problem = validateImageFile(file)
  if (problem) throw new ApiError(problem)

  const body = new FormData()
  body.append('file', file)

  let response
  try {
    response = await fetch(apiUrl('/api/v1/captions'), {
      method: 'POST',
      body,
      signal,
      // no Content-Type header: the browser must set the multipart boundary
    })
  } catch (error) {
    if (error?.name === 'AbortError') throw error
    throw new ApiError('Could not reach the API. Is the backend running?')
  }
  if (!response.ok) throw await parseError(response)
  return response.json()
}

/** Caption history, newest first. `q` is an optional case-insensitive search. */
export function fetchCaptions({ limit = 50, offset = 0, q = '', signal } = {}) {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) })
  if (q) params.set('q', q)
  return getJson(`/api/v1/captions?${params.toString()}`, { signal })
}

/** A single caption by id. */
export function fetchCaption(id, { signal } = {}) {
  return getJson(`/api/v1/captions/${encodeURIComponent(id)}`, { signal })
}

/** Totals, mean inference time, and the latest evaluation if one is recorded. */
export function fetchStats({ signal } = {}) {
  return getJson('/api/v1/stats', { signal })
}

/** Liveness plus whether the model loaded. */
export function fetchHealth({ signal } = {}) {
  return getJson('/health', { signal })
}