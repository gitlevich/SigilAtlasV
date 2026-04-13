---
status: idea
---

# Pole

One end of a @contrastControl. A point in @contrastspace defined by a bag of ingredients mixed together.

## Ingredients

A pole contains one or more entries. Each entry is one of:
- **Concept**: a word or phrase I type. Embedded by CLIP text encoder. "golden light", "fish and chips", "brutalist concrete."
- **Contrast position**: a reference to an existing @contrastControl plus a position along it (slider from pole A to pole B). "0.8 on my light/dark axis."

I can mix both kinds freely. A pole like "my warm" might be:
- concept: "golden"
- position 0.8 on light/dark axis
- position 0.3 on urban/natural axis

## Embedding

Each entry produces a direction vector:
- Concept: CLIP text embed, normalized.
- Contrast position: the referenced contrast's direction vector, scaled by the position value (-1 to +1).

The pole's embedding = normalize(sum of all entry vectors).

## Widget

A compact vertical list below one end of the range slider. Each entry is a row:
- Concept entry: a text pill I can type into.
- Contrast position entry: a dropdown (pick a contrast) + a mini slider (position along it).

I can add and remove entries. The pole updates live as I edit.

## Recursion

A contrast position entry references another @contrastControl. That control's direction may itself be built from poles that reference other contrasts. The system resolves the full chain to compute the final embedding.

!no-cycles: a pole cannot reference a @contrastControl that references it, directly or indirectly.
