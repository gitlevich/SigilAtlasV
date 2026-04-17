---
status: idea
---

# Multi Resolution

Render @images at the resolution appropriate for the current @pointOfView distance. Avoids loading full-resolution files when they'd display at a few pixels.

- Micro (16px height): maximum zoom-out. Just color and shape.
- Thumb (256px height): medium distance. Content recognizable.
- Full (original resolution): close-up. Loaded on demand from @source.

All tiers cached in @imageCache.
