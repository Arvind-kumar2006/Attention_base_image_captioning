/** Shown while a caption is being generated. */
export default function CaptionSkeleton() {
  return (
    <div className="space-y-4" data-testid="caption-skeleton">
      <div className="card overflow-hidden">
        <div className="h-56 w-full animate-pulse bg-border" />
      </div>
      <div className="space-y-2">
        <div className="h-4 w-3/4 animate-pulse rounded bg-border" />
        <div className="h-4 w-1/2 animate-pulse rounded bg-border" />
      </div>
      <p className="muted text-sm" role="status">
        Generating a caption…
      </p>
    </div>
  )
}