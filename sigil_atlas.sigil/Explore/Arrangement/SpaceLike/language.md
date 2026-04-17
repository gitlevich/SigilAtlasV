---
status: idea
---

# Space-like

In this mode I make sense of the @images spatially, as a whole. The arrangement is the @spacelike shape of attention: the field is apprehended all at once, not traversed. @images are center-cropped to squares and tile the @torus in a contiguous field with no gaps.

Each @image has a target position computed as the equilibrium of gravitational pulls from the active @attractors, with @contrasts shaping the @proximity metric. Targets are then assigned to square cells on the @torus so the field stays contiguous. A @neighborhood forms around each @attractor; neighborhoods overlap smoothly where pulls are comparable.

Every change to a @control — bandpass drag, added @contrast, new @attractor, different @visionModel — recomputes targets, re-assigns cells, and animates each @image from its old cell to its new one. No snap; the surface flows like a gravitational field settling.

To shape the arrangement, I #define-controls to manipulate @attractors and @contrasts. The @arrangement is itself a @SigilML expression: each @attractor is an `Attract` node over an atom (a @thing or a @TargetImage); the @RelevanceFilter is the enclosing gate. Ordering is the business of @TimeLike.
