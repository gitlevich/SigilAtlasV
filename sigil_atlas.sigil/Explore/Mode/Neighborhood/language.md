---
status: idea
---

# Neighborhood

A compact cluster of @images on the @torus, centered on an @attractor. Images within a neighborhood are similar by the active @distanceMetric.

Internally gapless — images tile edge-to-edge via strip packing at the rectangle's width. The @attractor is the most representative image, placed at the center. Remaining images arrange by decreasing similarity outward.

Neighborhoods are placed on the surface by projecting @attractor positions from embedding space to 2D. Gaps between neighborhoods are acceptable.

The @pointOfView determines which neighborhood I am in. Zooming out reveals the neighborhood structure. Zooming in enters one neighborhood and shows individual images.
