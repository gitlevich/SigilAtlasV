---
status: idea
---

# Contrast Control

A band-pass filter along a tension between two @poles.

## Roles

Each contrast control has a role (see @sigilControls):
- **Filter**: images outside the band are excluded from the @slice.
- **Attract**: images are scored by proximity to one pole. Higher score = appears earlier.
- **Order**: drives the vertical @strip ordering. One at a time.

## Widget layout

The range slider takes the full width — it's the primary interaction.

Below the slider, two @pole builders sit at each end. A short label (user-editable) appears on the slider rail for quick identification when many controls are stacked.

## Band-pass behavior

The range slider has two draggable handles defining an accepted band:
- Wide band (handles at extremes): permissive. Almost all images pass. This contrast is not constraining.
- Narrow band: strict. Only images within this range along the contrast axis are included.
- Band at one end: "I want only the warm pole" or "I want only the cold pole."
- Band in the middle: "I want neutral — neither extreme."

## Scoring

Each image is projected onto the contrast direction: projection_i = dot(image_embedding, contrast_direction).

The projection values across the corpus define the range. The slider maps to this range. An image passes the filter if its projection falls within the selected band.

## Multiple controls

I can create several independent contrast controls. Each defines a separate axis. They compose as intersection (AND) for filtering, sum for attraction.

## Implementation

- Contrast direction: normalize(pole_A.embedding - pole_B.embedding).
- Project all corpus images onto this direction. Store the projections.
- Compute corpus min/max along this direction to set slider range.
- On handle drag: update band bounds, recompute which images pass, reflow @torus.
- Store: direction vector (512-dim) + band_min (float) + band_max (float) + role.
