---
status: idea
---

# Embedding

A vector produced by a @visionModel from an @image. The position encodes what the model sees — semantic content (CLIP) or visual texture (DINOv2).

!stable: the same @visionModel on the same @image always produces the same vector. Embeddings are cached and reused until the model version changes.

Dimensionality depends on the model: CLIP ViT-B/32 at 512 dimensions, DINOv2 ViT-B/14 at 768 dimensions. All vectors are L2-normalized so cosine similarity reduces to dot product.

A @contrastControl projects images onto a direction in this space. The dot product of an image's embedding with the contrast direction gives its position along that contrast.
