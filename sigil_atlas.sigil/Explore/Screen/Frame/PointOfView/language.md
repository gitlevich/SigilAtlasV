---
status: idea
---

# PointOfView

The camera. Defined by three values:
- x: horizontal position on the @torus surface (wraps).
- y: vertical position on the @torus surface (wraps).
- z: distance from the surface. Determines how much of the @torus is visible in the @frame.

Equivalent to a 45mm lens on 35mm film: a natural perspective, not wide-angle, not telephoto. The field of view is fixed. Distance z controls magnification.

At z=0 (touching the surface): one @image fills the @frame.
At z=max: the entire @torus is visible — all @images, all @strips, at micro-thumbnail resolution.

Pan changes x and y. Zoom changes z. All three are continuous.
