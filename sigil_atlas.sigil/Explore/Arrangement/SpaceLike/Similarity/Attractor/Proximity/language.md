---
status: idea
---

# Proximity

The metric of @similarity in @contrastspace. Two @images are close when they are near along whichever dimensions I have named relevant; far when they are distant along them. Proximity composes from per-@contrast proximities — one scalar per active @contrast — and from distance to @attractors, which are specific points in @contrastspace anchored by a @thing or a @TargetImage. Unnamed @contrasts are in superposition and do not contribute. If I name nothing, proximity defaults to the @visionModel's native @embedding distance — the model decides what is close.