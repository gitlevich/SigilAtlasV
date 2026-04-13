---
status: idea
---

# Sigil Controls

The @rightPanel. A sigil builder — define the region of @contrastspace you want to inhabit.

## Structure

A sigil is a set of @contrastControls. Each contrast is an axis with a range. The sigil boundary is the intersection of all ranges.

Each @contrastControl has a **role**:
- **Filter**: constrains the @slice. Images outside the band are excluded.
- **Order**: drives the vertical @strip ordering. Images are arranged along this axis. Only one contrast can be the active order axis at a time.
- **Attract**: pulls the @slice toward one pole. Images are scored by proximity — higher score = appears earlier in @strips.

The default order axis is capture date (always available, not a contrast — just metadata).

## Controls

- Add @contrastControl: define a new contrast axis (text poles or @compositeContrast). Starts as a filter with full-width band (no constraint).
- Set role: toggle each contrast between filter / order / attract.
- @mode toggle: switch between @timelike, @spacelike, @tastelike.
- @timescale slider: story gap threshold (in @timelike mode).
- @strip height slider.

## The query

The active set of contrast controls with their roles and ranges IS the query. The @slice is the result. Adding, removing, and adjusting controls recomputes the @slice and reflows the @torus in real time.

The sigil I'm building lives in the @rightPanel. It persists across mode switches. Switching @mode changes how the @slice is laid out on the @torus, not what's in it.
