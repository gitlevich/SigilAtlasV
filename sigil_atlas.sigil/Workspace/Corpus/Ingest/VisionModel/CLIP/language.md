---
status: idea
---

# CLIP

Semantic embedding model trained on image-text pairs. Produces 512-dimensional L2-normalized vectors (ViT-B/32) or 768-dimensional (ViT-L/14). Dot product = cosine similarity.

Accepts text prompts natively — "a photograph that is dark" becomes a direction in the same space as image embeddings. This is what makes @ContrastControl and @AttractorControl work: the @pole text maps to a vector; the @contrast direction is the difference of two pole vectors; images project onto it.
