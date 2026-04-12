---
status: idea
---

# Corpus

All my @image files.

On @ingest, I wrap each @image into its @ImageSigil that, aside from the @image@identity, contains all @embeddings of it for each of the @VisionModels we support. 

Responds to a @RelevanceFilter with the @Slice that contains all matched @ImageSigils.

@ImageSigils are clustered recursively into @ImageNeighborhoodSigils by taking each image and creating an @ImageNeighborhoodSigil, unless one already exists, with one less @contrast constrained than the @ImageSigil. 

Then we repeat this process for every @ImageNeighborhoodSigil, reducing relevant @contrast by one and repeating. 

In the end, we will have several sigil structures in which each @sigil participates at its scale. 