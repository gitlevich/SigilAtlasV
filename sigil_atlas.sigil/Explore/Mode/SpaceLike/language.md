---
status: idea
---

# Space-like

The @torus ordered by @similarity into @neighborhoods centered on their @attractors. Moving along the surface = moving through a continuous visual space. Nearby images look alike. Distant images look different.

@Neighborhoods are local (!local-neighborhoods) — a cluster of nearby images on the surface that share qualities. They don't align with @strips. A neighborhood may span parts of several strips or sit within one. The transition across the surface is smooth — no abrupt jumps.

## Implementation

Given N images with embeddings:
1. Compute a 2D ordering that preserves embedding similarity — nearby images in embedding space should be nearby on the torus.
2. Unroll the 2D ordering into @strips: the first axis maps to position within a strip, the second axis maps to which strip.

The 2D ordering is the hard part. UMAP to 2D is one approach. A space-filling curve (Hilbert) through the embedding space is another. The requirement is that the mapping from embedding space to torus position is smooth — small changes in visual content = small changes in position.
