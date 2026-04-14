---
status: idea
---

# Strip

A horizontal row of @images on the @torus. A film strip that wraps.

Each @image scales uniformly to height h — display width = h * original_width / original_height. Images tile edge-to-edge with no gaps.

How images get assigned to strips is defined by @stripAssignment. The position of each image within and across strips is determined by the active @mode.

!fixed-height, !no-crop, !no-gaps, !no-overlaps, !respect-image-aspect-ratio.

A @strip #places-images-on-itself to ensure its @invariants.