---
status: idea
---

# Frame

The viewport into the @torus. A rectangle of !fixed-aspect-ratio and !fixed-resolution, like a TV screen: I look through it at the @torus surface.

At any moment, the frame shows the portion of the @torus visible from my @PointOfView. What's visible depends on:
- Position: where on the surface I am (which @images are in view).
- Distance: how far from the surface (how many @strips are visible).

The frame renders @images at the resolution appropriate for the current distance:
- Far: many images visible, each rendered as a small thumbnail.
- Medium: a few strips visible, images recognizable.
- Close: one image fills the frame at full resolution.

## Multi-resolution rendering

At far distances, individual images are too small to resolve. The renderer should use cached thumbnails at multiple scales to avoid loading full-resolution images when they'd display at a few pixels. Three levels:
- Micro (16px height): for maximum zoom-out. Just color and shape.
- Thumb (256px height): for medium distance. Content recognizable.
- Full (original resolution): for close-up. Loaded on demand.
