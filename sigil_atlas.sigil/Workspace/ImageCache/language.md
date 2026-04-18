---
status: idea
---

# Image Cache

Cached copies of @images at multiple resolutions for rendering at different @POV distances.

Three tiers:
- Micro (16px height): for maximum zoom-out. Just color and shape. Pre-generated at ingest.
- Thumb (256px height): for medium distance. Content recognizable. Pre-generated at ingest.
- Full (original resolution): loaded on demand from @source when zoomed close.

All tiers preserve original aspect ratio. Stored in the @workspace so the app works when @source is disconnected (at micro and thumb resolution).
