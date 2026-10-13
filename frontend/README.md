# Lumina frontend

React + Vite + Tailwind CSS 4 + react-router. Talks to the FastAPI backend for
captioning, history and stats.

## Setup

```bash
npm install
cp .env.example .env      # set VITE_API_URL to your backend
npm run dev               # http://localhost:5173
```

The backend must be running and its `FRONTEND_ORIGIN` must allow this origin:

```bash
# from the repository root
uvicorn backend.app.main:app --reload
```

## Scripts

| Command | Purpose |
|---|---|
| `npm run dev` | Dev server with HMR |
| `npm run build` | Production bundle into `dist/` |
| `npm run preview` | Serve the built bundle |
| `npm run lint` | Oxlint |
| `npm test` | Vitest (jsdom + Testing Library) |

## Pages

| Route | What it does |
|---|---|
| `/caption` | Upload by drag-and-drop or button, then read the caption |
| `/history` | Past captions with search; a row reopens it on `/caption` |
| `/dashboard` | Totals, mean inference time, latest BLEU-4 |
| `/evaluation` | Placeholder |

## How the caption view works

The API returns `attention` as `num_words x 7 x 7` — one 7×7 grid per word,
each row being the model's distribution over the encoder's 49 image regions.

Each word is a button. Selecting one (click, or the Left/Right arrow keys)
normalises that word's grid to 0..1 and bilinearly upsamples it to the image
size on a `<canvas>`, so the overlay is a smooth field rather than 49 visible
blocks. Alpha rides on intensity, so cold regions stay transparent and the photo
is only obscured where the model actually looked.

The canvas is `aria-hidden`; the words themselves are the accessible
representation, and a polite live region announces the selected word.

## Accessibility

- Every text pair meets WCAG AA (≥ 4.5:1); the token values are commented in
  `src/index.css` with their ratios.
- Visible focus rings via `:focus-visible`, never removed without a replacement.
- The word group is a single tab stop with arrow-key navigation (roving
  `tabindex`), plus Home/End/Escape.
- Real table markup: `<caption>`, `<th scope="col">`, `<th scope="row">`.
- A skip link precedes the navigation; `prefers-reduced-motion` is respected.

## API client

`src/api/client.js` is the only module that fetches. Every URL is built from
`VITE_API_URL`. Client-side validation (type, extension, 10 MB) mirrors the
backend so a bad file never becomes a round trip — but the backend still
re-validates, since a client check can be bypassed.