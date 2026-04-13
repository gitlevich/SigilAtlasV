---
status: idea
---

# Slice

A subset of @corpus @images that matched a @RelevanceFilter. !ordered — the order depends on the active @mode.

A slice is the input to the @torus layout engine. Given a slice and a @mode, the layout engine computes @strips and image positions.

A slice with no filter = the entire @corpus. A slice with a text filter = images whose embeddings are closest to the text's CLIP embedding. A slice with an image filter = images most similar to the reference image.
