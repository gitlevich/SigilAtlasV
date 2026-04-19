---
status: idea
---

# DINOv2

Self-supervised vision model producing 768-dimensional L2-normalized @embeddings (ViT-B/14). Groups @images by visual texture — composition, color, surface — not by semantic content.

Not text-native: there is no direct text-to-embedding projection. Text-driven @contrasts and @attractors are unavailable when @DINOv2 is the active @visionModel. @Similarity based on @TargetImage and visual @neighborhoods still work.
