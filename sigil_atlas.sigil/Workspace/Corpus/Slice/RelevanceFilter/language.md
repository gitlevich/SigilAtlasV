---
status: idea
---

# Relevance Filter

The structural query. Built from the active @sigilControls — a set of @contrastControls, each with a role and range.

## How it works

Each @contrastControl contributes based on its role:
- **Filter** (role=filter): hard constraint. Images outside the band are excluded.
- **Attract** (role=attract): soft ranking. Images are scored by proximity to the attract direction. Higher score = appears earlier.
- **Order** (role=order): determines @strip ordering. One axis at a time. Does not filter or score — just sequences.

## Composition

1. Start with all @corpus images.
2. Apply metadata filters if any (date range, camera, etc.).
3. Apply all filter-role @contrastControls as band-pass (AND). Images outside any band are excluded.
4. Score remaining images against all attract-role @contrastControls. Composite score = sum of per-control scores.
5. The @slice = filtered images, ranked by composite score.
6. The order-role @contrastControl (or capture date by default) determines how the @slice is laid out into @strips.

## No controls active

The @slice is the entire @corpus. Ordering is by capture date (default order axis).
