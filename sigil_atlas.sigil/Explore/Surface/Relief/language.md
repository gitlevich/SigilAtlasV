---
status: idea
---

# Relief

An elevation function over the @surface. Each cell on the @torus has a height derived from the arrangement's structure — high where attractor pull is strong or where @proximity density peaks, low where the field is diffuse. Relief turns the flat field into a terrain. The terrain is not decorative: it compresses information — three channels the flat view does not have: parallax, shading under @Lighting, and cast shadows.

Relief is the physics the rendering engine puts on the @surface. It is objective — arrangement-derived, @POV-independent, sitting below the semantic layer. The @POV sees it as terrain only via @Perspective; Relief alone is just elevation data on the ground.

Relief is an independent mode of the @surface, toggled at will. When off, the field is flat. When on, the @POV's @Perspective gains orbit and tilt — without the elevation, there is nothing for a point of view to read.

The elevation function must be derived, not authored. In @SpaceLike, elevation is the superposition of projected @sigil spheres: each thrown @sigil contributes a well whose diameter is its taxonomy's articulated richness, whose depth is its gravity (force of the throw), and whose boundary feathers by `Discriminate` narrowness. Overlapping projections add as interference; where peer-scale throws reinforce, the well deepens; where they cancel via `Not`, the surface lifts back toward flat. In @TimeLike, elevation can reflect local similarity density within a strip.
