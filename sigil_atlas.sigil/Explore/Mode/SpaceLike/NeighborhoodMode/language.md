---
status: idea
---

# Neighborhood Mode

Operates on the current @slice. Define what a @neighborhood means.

Two clustering strategies depending on context:

**Default (no contrasts selected):** precomputed KMeans clusters from ingest, cached at multiple granularities. The #adjust-tightness slider selects which level to use. Works for all models.

**With contrasts selected:** each contrast pole becomes an @attractor. Images assign to their nearest pole by projection. Recomputes live on every contrast change. Works for CLIP.

#select-axes: checkboxes listing available dimensions — the @similarity controls I have defined, plus physical location and time. Check the ones that matter for this layout.

#adjust-tightness: one slider controlling how tight the clustering is. With precomputed clusters, selects the granularity level. With live projection, controls the number of poles.

#switch-model: choose which @visionModel provides the @embedding. Different models see different structure — CLIP groups by semantics, DINOv2 by visual texture.
