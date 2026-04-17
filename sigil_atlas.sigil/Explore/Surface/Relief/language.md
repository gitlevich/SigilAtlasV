---
status: idea
---

# Relief

An elevation function over the @surface. Each cell on the @torus has a height derived from the arrangement's structure — high where attractor pull is strong or where @proximity density peaks, low where the field is diffuse. Relief turns the flat field into a terrain. The terrain is not decorative: it compresses information. Parallax reveals depth as the @pointOfView moves; shading reveals local curvature under @Lighting; cast shadows reveal long-range structure. Three information channels the flat view does not have.

Relief is an independent mode of the @surface, toggled at will. When off, the field is flat. When on, the @pointOfView gains orbit and tilt — the user can look at the terrain from any angle.

The elevation function must be derived, not authored — it reflects the current arrangement's structure, not an external design decision. In @SpaceLike, elevation comes from softmax confidence in attractor pull (peaks at strongly-attracted regions). In @TimeLike, elevation can reflect local similarity density within a strip.
