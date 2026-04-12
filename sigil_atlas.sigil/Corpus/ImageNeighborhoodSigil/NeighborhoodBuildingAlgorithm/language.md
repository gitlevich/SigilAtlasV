---
status: idea
---

# Neighborhood Building Algorithm

This is how we aggregate @ImageSigils and @ImageNeighborhoodSigils into a lattice of intersecting @ImageNeighborhoodSigils:

Take one image. It has a 768-dim DINOv2 embedding. Filter out dimensions with low amplitude. Sort remaining by absolute value descending. Keep top 40%. These are its major contrasts. Say it has 50.

Now take this image's 50 major contrasts. Pick one. Drop it. You now have 49 constraints. Find all other images in the corpus whose embeddings match this image on those 49 dimensions within some tolerance. Those images are the neighborhood formed by dropping that one contrast.

Do this for each of the 50 contrasts. That gives this image 50 neighborhoods it belongs to.

Do this for every image. 5,000 images times ~50 contrasts = ~250,000 neighborhood candidates. Many will be identical — two images in the same neighborhood will generate the same neighborhood when they drop the same contrast. Deduplicate.

Now take each neighborhood. It has a set of shared contrasts. Pick one of those shared contrasts. Drop it. Find all images matching the remaining constraints. That's a coarser neighborhood.

Repeat for each shared contrast in each neighborhood. Deduplicate again. Keep going until you reach a single root that contains everything.

The result is a lattice where each image participates in many neighborhoods, and each neighborhood has many parents and many children.