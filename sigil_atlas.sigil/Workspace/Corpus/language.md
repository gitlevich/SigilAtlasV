---
status: idea
---

# Corpus

All my @image files.

On @ingest, each @image is brought into the @corpus: thumbnails generated at multiple resolutions, projected into @contrastspace as @embeddings using each @VisionModel, and metadata extracted (capture date, camera, GPS, dimensions).

Responds to a @RelevanceFilter with a @slice — the subset of @images that match.

Physically, the @corpus lives in a @workspace.

## Data per image

- id: content-hash based, !unique, !stable, !derived from bytes
- source_path: where the original file lives
- thumbnails: micro (16px), thumb (256px), cached in @imageCache
- embeddings: one per @VisionModel (CLIP ViT-B/32 at 512 dim, DINOv2 ViT-B/14 at 768 dim)
- metadata: capture_date, pixel dimensions, GPS, camera, lens, aperture, ISO, shutter speed
