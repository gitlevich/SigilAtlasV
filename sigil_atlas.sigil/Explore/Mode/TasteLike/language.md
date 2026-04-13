---
status: idea
---

# Taste-like

The @torus ordered by a @RelevanceFilter I define. Forward = more relevant. Backward = less relevant.

I specify what I want to see more of and what I want to see less of:
- Text descriptions of @contrasts: "warm golden light", "solitary figure", "urban decay"
- Specific @images I point at: "more like this one"
- Exclusions: "less of this", "no cityscapes"

The @RelevanceFilter translates to a direction in @contrastspace (via CLIP text embedding or by pointing at an image's embedding). Images are scored by cosine similarity to that direction. The @strip layout orders images by score — most relevant first.

As I adjust the filter, the @torus reflows. Images that match slide forward. Images that don't recede. The experience is: I describe what attracts me and the surface rearranges to put it in front of me.

## Implementation

Given N images with CLIP embeddings and a relevance direction r (a unit vector in embedding space):
1. Score each image: score_i = dot(embedding_i, r).
2. Sort images by score descending.
3. Fill @strips in score order — the most relevant images appear in the first strips, visible from the default @PointOfView.

The relevance direction r is constructed from:
- Text input: embed with CLIP text encoder, normalize.
- Image input: use the image's CLIP embedding as r.
- Combined: weighted sum of text and image directions.
- Exclusions: subtract the exclusion direction from r.
