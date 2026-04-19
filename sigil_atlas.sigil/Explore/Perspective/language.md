---
status: idea
---

# Perspective

How this application projects the world to me — my first-person rendering of the @surface. What @Navigation moves; what @Relief becomes when read from a point of view. @Aperspectival is objective; a view from somewhere has perspective, and perspective is what makes the @surface feel like a place rather than a diagram.

Perspective reads @Relief as terrain by interpreting parallax — the differential motion of cells at different elevations as my @POV shifts. Without @Relief there is nothing for perspective to read; the surface is flat and any point of view renders the same projection. Without movement, no parallax to interpret; my perspective collapses to @Aperspectival. The coupling is: @Relief is the physics the engine puts on the ground; Perspective is the @POV - side projection that turns elevation data into depth.

Perspective is also what @Lighting lands on. Shading, shadows, and parallax-from-light only register because perspective is projecting from somewhere; under @Aperspectival, every light direction is equivalent.

!requires-relief-for-depth. !coupled-to-POV.
