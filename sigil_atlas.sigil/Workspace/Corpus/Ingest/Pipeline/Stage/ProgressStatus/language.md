---
status: idea
---

# Progress Status

Real-time state of a @stage during a @pipeline run. Reported to the @statusBar so I can watch ingest progress.

- **total**: how many @images this stage needs to process.
- **completed**: how many are done.
- **failed**: how many errored (with error details retained for inspection).
- **rate**: throughput (images per second), computed over a rolling window.
- **eta**: estimated time remaining, derived from rate and remaining count.

The @pipeline aggregates all stage statuses into an overall progress. The @statusBar shows both per-stage and overall.
