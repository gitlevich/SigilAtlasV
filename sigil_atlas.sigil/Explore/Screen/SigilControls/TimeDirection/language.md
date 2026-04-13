---
status: idea
---

# Time Direction

What "forward" means when scrolling vertically in @timelike mode. A selector that picks which axis drives the vertical ordering of @strips.

## Widget

A dropdown or dial with one entry per option:
- **Capture date** (default): @strips ordered by when images were taken. Always available.
- One entry per active @contrastControl: use that contrast axis as the vertical ordering. Label matches the contrast's short name (e.g. "warm/cold", "crowded/empty").

Switching is instant — pick a different entry, the @strips reflow along the new axis.

## Behavior

When set to capture date: vertical position = time. Scrolling down = forward in time. Within each @strip, images are sorted by similarity (horizontal).

When set to a @contrastControl: vertical position = that contrast's projection value. @strips at the top are at one pole, strips at the bottom are at the other. Within each @strip, images are sorted by similarity along the remaining dimensions.

The selected contrast still functions as a band-pass filter — the time direction and the filter are independent. I can scroll along warm/cold while also constraining crowded/empty to a narrow band.

## Implementation

- Default: sort images by capture_date, group into @strips.
- Contrast axis: sort images by their projection onto the contrast direction, group into @strips. Strip boundaries at uniform intervals along the projection range, or at natural density gaps.
