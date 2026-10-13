/** Red banner for a failed request. The message comes from the API's `detail`. */
export default function ErrorMessage({ message, onDismiss }) {
  if (!message) return null
  return (
    <div
      role="alert"
      className="flex items-start justify-between gap-4 rounded-lg border border-danger/40 bg-danger-soft px-4 py-3"
    >
      <p className="text-sm font-medium text-danger">{message}</p>
      {onDismiss ? (
        <button
          type="button"
          onClick={onDismiss}
          className="shrink-0 rounded px-2 py-1 text-sm font-semibold text-danger underline"
        >
          Dismiss
        </button>
      ) : null}
    </div>
  )
}