---
status: idea
---

# VisionModel

The model we use to embed our @images into @contrastspace. Two kinds: @CLIP groups by semantic content and accepts text prompts, so it drives @ContrastControl and @AttractorControl. @DINOv2 groups by visual texture; text prompts are not native to it.

One @visionModel is active at a time. Switching model reshapes @proximity.
