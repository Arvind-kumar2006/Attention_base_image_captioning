/** Placeholder until an evaluation results viewer is built. */
import { Link } from 'react-router-dom'

export default function EvaluationPage() {
  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold tracking-tight">Evaluation</h1>
        <p className="mt-1 text-sm text-fg-muted">
          Corpus metrics for a trained checkpoint.
        </p>
      </header>

      <div className="card px-6 py-12 text-center">
        <p className="text-lg font-semibold">Not built yet</p>
        <p className="mx-auto mt-2 max-w-prose text-sm text-fg-muted">
          Scores come from <code className="font-mono">ml/evaluate.py</code>, which writes
          <code className="font-mono"> results.json</code> with BLEU-1..4, METEOR and CIDEr
          for a checkpoint. This page will list those runs, plot them per metric and show
          the best and worst captions by CIDEr.
        </p>
        <p className="mx-auto mt-4 max-w-prose text-sm text-fg-muted">
          To produce a run today:
        </p>
        <pre className="mx-auto mt-2 max-w-xl overflow-x-auto rounded-lg bg-canvas px-4 py-3 text-left text-xs">
          <code>{`python -m ml.evaluate \\
  --checkpoint ml/weights/best.pt \\
  --data-dir data/processed \\
  --coco-root data/coco \\
  --split test \\
  --out results.json`}</code>
        </pre>
        <p className="mt-6 text-sm text-fg-muted">
          The dashboard already shows the latest run once one is recorded.{' '}
          <Link to="/dashboard" className="font-semibold text-accent underline">
            Go to the dashboard
          </Link>
        </p>
      </div>
    </div>
  )
}