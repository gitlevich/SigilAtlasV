---
status: idea
---

# Torus

The layout surface. A flat rectangle where both axes wrap — traversal is !endless in every direction.

The surface is tiled with @images arranged in @strips. No empty space. Dimensions derived from the images: given N images at @stripHeight h, width and height follow from aspect ratios and number of strips.

Purely topological — the @frame always renders flat. No 3D curvature. Wrapping means: moving past the edge returns to the opposite side.

My @pointOfView is a position on the surface. Coming close fills my @frame with one image. Stepping back reveals many @strips. The @mode determines how @images are assigned to @strips via @stripAssignment. Switching @mode recomputes the layout.

!endless, !flat, !gapless, !reorderable, !undistorted.
