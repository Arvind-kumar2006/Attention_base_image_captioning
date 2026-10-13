/** Page-level behaviour with a stubbed backend: no network, no weights. */
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import CaptionPage from './CaptionPage'
import DashboardPage from './DashboardPage'
import HistoryPage from './HistoryPage'
import EvaluationPage from './EvaluationPage'

/** A 3-word caption with a plausible attention map. */
const CAPTION = {
  id: 'cap-1',
  image_id: 'img-1',
  caption: 'a dog runs',
  words: ['a', 'dog', 'runs'],
  attention: [
    Array.from({ length: 7 }, (_, r) => Array.from({ length: 7 }, (_, c) => ((r + c) % 7) / 7)),
    Array.from({ length: 7 }, () => Array.from({ length: 7 }, () => 1 / 7)),
    Array.from({ length: 7 }, () => Array.from({ length: 7 }, () => 1 / 7)),
  ],
  time_ms: 222.55,
  image_url: '/api/v1/images/img-1',
  created_at: '2026-10-09T10:54:13Z',
}

function json(body, status = 200) {
  return { ok: true, status, json: async () => body }
}

function pngFile(name = 'photo.png', size = 200) {
  const blob = new Blob([new Uint8Array(size)], { type: 'image/png' })
  return new File([blob], name, { type: 'image/png' })
}

function route(element) {
  return render(<MemoryRouter>{element}</MemoryRouter>)
}

beforeEach(() => {
  URL.createObjectURL = vi.fn(() => 'blob:preview')
  URL.revokeObjectURL = vi.fn()
})

describe('CaptionPage', () => {
  it('shows the required empty state text', () => {
    route(<CaptionPage />)
    expect(screen.getByText('Upload an image to generate a caption')).toBeInTheDocument()
  })

  it('offers a labelled file input accepting jpeg and png', () => {
    route(<CaptionPage />)
    const input = screen.getByLabelText(/choose an image/i)
    expect(input).toHaveAttribute('type', 'file')
    expect(input.getAttribute('accept')).toContain('image/png')
    expect(input.getAttribute('accept')).toContain('image/jpeg')
  })

  it('names the input so screen readers announce it as a file control', () => {
    route(<CaptionPage />)
    expect(screen.getByLabelText(/choose an image/i).tagName).toBe('INPUT')
  })

  it('rejects a non-image in the browser without calling the API', async () => {
    global.fetch = vi.fn()
    // applyAccept off: the input's accept attribute would stop the browser
    // offering the file at all, and we want to prove our own guard catches it.
    const user = userEvent.setup({ applyAccept: false })
    route(<CaptionPage />)

    const pdf = new File([new Uint8Array(100)], 'a.pdf', { type: 'application/pdf' })
    await user.upload(screen.getByLabelText(/choose an image/i), pdf)

    expect(await screen.findByRole('alert')).toHaveTextContent(/JPEG and PNG/)
    expect(global.fetch).not.toHaveBeenCalled()
  })

  it('rejects a drag-and-drop of a non-image', async () => {
    global.fetch = vi.fn()
    route(<CaptionPage />)

    const pdf = new File([new Uint8Array(100)], 'a.pdf', { type: 'application/pdf' })
    const zone = document.querySelector('input[type="file"]').closest('div')
    fireEvent.drop(zone, { dataTransfer: { files: [pdf] } })

    expect(await screen.findByRole('alert')).toHaveTextContent(/JPEG and PNG/)
    expect(global.fetch).not.toHaveBeenCalled()
  })

  it('rejects an oversize file in the browser with the 10 MB message', async () => {
    global.fetch = vi.fn()
    const user = userEvent.setup()
    route(<CaptionPage />)

    const big = new File([new Uint8Array(10 * 1024 * 1024 + 1)], 'big.png', { type: 'image/png' })
    await user.upload(screen.getByLabelText(/choose an image/i), big)

    expect(await screen.findByRole('alert')).toHaveTextContent(/10 MB/)
    expect(global.fetch).not.toHaveBeenCalled()
  })

  it('shows a skeleton while the caption is being generated', async () => {
    let release
    global.fetch = vi.fn(() => new Promise((resolve) => { release = () => resolve(json(CAPTION, 201)) }))
    const user = userEvent.setup()
    route(<CaptionPage />)

    await user.upload(screen.getByLabelText(/choose an image/i), pngFile())
    expect(await screen.findByTestId('caption-skeleton')).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent(/generating a caption/i)

    release()
    await waitFor(() => expect(screen.queryByTestId('caption-skeleton')).not.toBeInTheDocument())
  })

  it('renders the caption words and the generation time after upload', async () => {
    global.fetch = vi.fn().mockResolvedValue(json(CAPTION, 201))
    const user = userEvent.setup()
    route(<CaptionPage />)

    await user.upload(screen.getByLabelText(/choose an image/i), pngFile())

    expect(await screen.findByRole('button', { name: 'dog' })).toBeInTheDocument()
    expect(screen.getByText('223 ms')).toBeInTheDocument()
    expect(screen.queryByText('Upload an image to generate a caption')).not.toBeInTheDocument()
  })

  it('surfaces the API error message when the upload fails', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({ detail: 'could not decode the file as an image' }),
    })
    const user = userEvent.setup()
    route(<CaptionPage />)

    await user.upload(screen.getByLabelText(/choose an image/i), pngFile())
    expect(await screen.findByRole('alert')).toHaveTextContent(/could not decode/i)
  })

  it('announces the selected word in a polite live region', async () => {
    global.fetch = vi.fn().mockResolvedValue(json(CAPTION, 201))
    const user = userEvent.setup()
    route(<CaptionPage />)
    await user.upload(screen.getByLabelText(/choose an image/i), pngFile())
    await screen.findByRole('button', { name: 'dog' })

    const live = document.querySelector('p[aria-live="polite"]')
    expect(live).toBeTruthy()
    expect(live).toHaveTextContent(/no word selected/i)

    await user.click(screen.getByRole('button', { name: 'dog' }))
    await waitFor(() => expect(live).toHaveTextContent(/selected word: dog/i))
  })

  it('moves between words with the arrow keys and updates the live region', async () => {
    global.fetch = vi.fn().mockResolvedValue(json(CAPTION, 201))
    const user = userEvent.setup()
    route(<CaptionPage />)
    await user.upload(screen.getByLabelText(/choose an image/i), pngFile())
    const dog = await screen.findByRole('button', { name: 'dog' })
    dog.focus()

    await user.keyboard('{ArrowRight}')
    const live = document.querySelector('p[aria-live="polite"]')
    await waitFor(() => expect(live).toHaveTextContent(/selected word: runs/i))
    expect(screen.getByRole('button', { name: 'runs' })).toHaveAttribute('aria-pressed', 'true')
  })

  it('draws the overlay canvas as aria-hidden decoration', async () => {
    global.fetch = vi.fn().mockResolvedValue(json(CAPTION, 201))
    const user = userEvent.setup()
    const { container } = route(<CaptionPage />)
    await user.upload(screen.getByLabelText(/choose an image/i), pngFile())
    await screen.findByRole('button', { name: 'dog' })

    const canvas = container.querySelector('canvas')
    expect(canvas).toBeTruthy()
    expect(canvas.getAttribute('aria-hidden')).toBe('true')

    await user.click(screen.getByRole('button', { name: 'dog' }))
    await waitFor(() => expect(canvas.className).toMatch(/opacity-100/))
  })
})

describe('HistoryPage', () => {
  beforeEach(() => {
    global.fetch = vi.fn((url) => {
      if (String(url).includes('/api/v1/captions?')) {
        return Promise.resolve(json({ total: 2, limit: 100, offset: 0, items: [CAPTION, { ...CAPTION, id: 'cap-2', caption: 'a cat sits' }] }))
      }
      return Promise.resolve(json({}))
    })
  })

  it('renders a proper table with a caption and header cells', async () => {
    route(<HistoryPage />)
    const table = await screen.findByRole('table')
    expect(table).toBeInTheDocument()

    // header cells must be <th scope="col">
    const headers = within(table).getAllByRole('columnheader')
    expect(headers.map((h) => h.textContent)).toEqual([
      'Image',
      'Caption',
      'Inference',
      'Date',
    ])

    // and the table must be labelled for screen readers
    expect(within(table).getByText(/newest first/i)).toBeInTheDocument()
  })

  it('shows caption, inference time and date per row', async () => {
    route(<HistoryPage />)
    expect(await screen.findByText('a dog runs')).toBeInTheDocument()
    // both rows share a generation time, so there are two cells
    expect(screen.getAllByText('223 ms')).toHaveLength(2)
    // a machine-readable datetime attribute on the <time> element
    expect(document.querySelector('time')).toHaveAttribute('datetime')
  })

  it('reopens a row in the caption view', async () => {
    const user = userEvent.setup()
    render(
      <MemoryRouter initialEntries={['/history']}>
        <Routes>
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/caption" element={<CaptionPage />} />
        </Routes>
      </MemoryRouter>,
    )

    const row = (await screen.findByText('a dog runs')).closest('tr')
    await user.click(row)

    // the caption view now shows the handed-over caption
    expect(await screen.findByRole('button', { name: 'dog' })).toBeInTheDocument()
    expect(screen.queryByText('Upload an image to generate a caption')).not.toBeInTheDocument()
  })

  it('is reachable by keyboard from a row', async () => {
    const user = userEvent.setup()
    render(
      <MemoryRouter initialEntries={['/history']}>
        <Routes>
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/caption" element={<CaptionPage />} />
        </Routes>
      </MemoryRouter>,
    )
    const row = (await screen.findByText('a dog runs')).closest('tr')
    row.focus()
    await user.keyboard('{Enter}')
    expect(await screen.findByRole('button', { name: 'dog' })).toBeInTheDocument()
  })

  it('passes the search term through as q, debounced', async () => {
    const user = userEvent.setup()
    route(<HistoryPage />)
    await screen.findByText('a dog runs')

    await user.type(screen.getByLabelText(/search captions/i), 'dog')

    await waitFor(() => {
      const urls = global.fetch.mock.calls.map(([url]) => String(url))
      expect(urls.some((u) => u.includes('q=dog'))).toBe(true)
    })
  })
})

describe('DashboardPage', () => {
  it('shows a dash, not zero, when no evaluation is recorded', async () => {
    global.fetch = vi.fn().mockResolvedValue(
      json({ total_captions: 7, avg_generation_time_ms: 180.4, evaluation: null }),
    )
    route(<DashboardPage />)

    expect(await screen.findByText('7')).toBeInTheDocument()
    expect(screen.getByText('180 ms')).toBeInTheDocument()
    const bleu = screen.getByText('Latest BLEU-4').closest('a')
    expect(within(bleu).getByText('—')).toBeInTheDocument()
  })

  it('shows the latest BLEU-4 and the score table when one exists', async () => {
    global.fetch = vi.fn().mockResolvedValue(
      json({
        total_captions: 7,
        avg_generation_time_ms: 180.4,
        evaluation: {
          id: 'e1',
          checkpoint: 'ml/weights/best.pt',
          split: 'test',
          tokenizer: 'PTBTokenizer',
          num_images: 5000,
          scores: { Bleu_4: 0.3123, CIDEr: 1.04 },
          created_at: '2026-10-09T10:00:00Z',
        },
      }),
    )
    route(<DashboardPage />)

    // 0.3123 appears twice: on the stat card and in the table
    expect((await screen.findAllByText('0.3123')).length).toBeGreaterThan(0)
    expect(screen.getByText('1.0400')).toBeInTheDocument()
    const table = screen.getByRole('table')
    expect(within(table).getByRole('rowheader', { name: 'Bleu_4' })).toBeInTheDocument()
    expect(screen.getByText(/PTBTokenizer/)).toBeInTheDocument()
  })

  it('reports an unreachable API instead of rendering empty cards', async () => {
    global.fetch = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'))
    route(<DashboardPage />)
    expect(await screen.findByRole('alert')).toHaveTextContent(/could not reach the api/i)
  })
})

describe('EvaluationPage', () => {
  it('is a placeholder that says so', () => {
    route(<EvaluationPage />)
    expect(screen.getByRole('heading', { name: 'Evaluation' })).toBeInTheDocument()
    expect(screen.getByText('Not built yet')).toBeInTheDocument()
    expect(screen.getByText(/ml\/evaluate\.py/)).toBeInTheDocument()
  })

  it('links to the dashboard', () => {
    route(<EvaluationPage />)
    expect(screen.getByRole('link', { name: /go to the dashboard/i })).toHaveAttribute(
      'href',
      '/dashboard',
    )
  })
})