---
status: idea
---

# Relevance Filter

Defines the @invariants of the @slice interms of allowed ranges along the relevant @contrasts and named @things we want to see.

todo the stuff below does not belong here and needs to go to @Explore section.

## How it works

Each @contrastControl contributes based on its role:
- **Filter** (role=filter): hard constraint. Images outside the band are excluded.
- **Attract** (role=attract): selection. Each named thing selects its top-matching images. Multiple things take the union. Images not matching any are excluded.
- **Order** (role=order): determines @strip ordering. One axis at a time. Does not filter or score — just sequences.

## Composition

1. Start with all @corpus images.
2. Apply metadata filters if any (date range, camera, etc.).
3. Apply all filter-role @contrastControls as band-pass (AND). Images outside any band are excluded.
4. Each attract term selects its matching images. The union of all selections becomes the working set. Images not selected by any term are excluded.
5. The @slice = selected images, ranked by composite score.
6. The order-role @contrastControl (or capture date by default) determines how the @slice is laid out into @strips.

## No controls active

The @slice is the entire @corpus. Ordering depends on the active @mode.
