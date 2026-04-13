---
status: idea
---

# Neighborhood Mode

Operates on the current @slice. Define what a @neighborhood means.

#select-axes: checkboxes listing available dimensions — the @contrasts I have defined, plus physical location and time. Check the ones that matter for this layout.

#adjust-tightness: one slider controlling how tight the clustering is. Small = compact neighborhoods with sharp boundaries. Large = loose neighborhoods that bleed into each other.

#switch-model: choose which @visionModel provides the @embedding. Different models see different structure — CLIP groups by semantics, DINOv2 by visual texture.

With no axes selected, the layout uses raw @embedding distance from the active model.
