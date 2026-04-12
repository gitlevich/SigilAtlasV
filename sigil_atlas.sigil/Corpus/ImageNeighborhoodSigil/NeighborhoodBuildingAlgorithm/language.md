---
status: idea
---

# Neighborhood Building Algorithm

This is how we aggregate @ImageSigils and @ImageNeighborhoodSigils into a lattice of intersecting @ImageNeighborhoodSigils, where each image participates in many neighborhoods, and each neighborhood has many parents and many children.

This creates a rich enough structure we can slice to reveal multi-scale @tiling correlated with neighbouhoods visually, so I could keep zooming in to the neighborhoods that looks more interesting and find there images that I love.

I think of @ImageNeighborhoodSigils as predicates that admit other @ImageNeighborhoodSigils and @ImageSigils.


For each @ImageSigil S defined in terms of @MajorContrast and @SigilProximity @invariants:

1. the create an @ImageNeighborhoodSigil N with one less @invariant than S, wrapping it and all other @ImageSigils that only differ from S in that dropped @invariant.
2. make a new @ImageNeighborhoodSigil from N by dropping another @invariant, wrapping N and other @ImageNeighborhoodSigils that only differ from it in the dropped @invariant.
3. Recurse until N only have one @invariant left. 


Note that in the process, we just create a very large number of potentially duplicated predicates. We should deduplicate them using a set. 

## Algorithm Complexity

Let k = average invariants per image (about 50). Let n = number of images.

**Level 1:** n images, each drops k invariants. n*k hash operations. O(nk).

**Level 2:** The number of unique neighborhoods from level 1 is at most n*k, but with dedup probably much less — call it m1. Each has k-1 invariants to drop. m1*(k-1) hash operations.

**Level 3:** m2 unique neighborhoods, each with k-2 invariants. m2*(k-2) operations.

The question is how fast m shrinks. 

**Best case:** Heavy dedup. Most images share most contrasts, so neighborhoods collide early. m stays small at every level. Total work: O(nk) — dominated by level 1.

**Worst case:** Every image is unique, no dedup at all. Level 1: nk neighborhoods. Level 2: nk(k-1). Level 3: nk(k-1)(k-2). That's n * k! which is catastrophically exponential. But this can't actually happen because you run out of images — a neighborhood can't have more members than n, and you need 2+ members. So the number of valid neighborhoods at any level is bounded by n choose 2 at most. Realistic worst case: O(nk^2) total hash ops.

**Expected case:** Heavy dedup at each level. The set grows fast initially then plateaus as most new predicates collide with existing ones. Total: O(nk * L) where L is the number of levels (at most k, but typically much less because groups hit size 2 and stop splitting). For n=5000, k=50: about 250K * L. If L is 20, that's 5 million hash operations. A few seconds.

For 10 million images: 10M * 50 * 20 = 10 billion hash ops. At 100ns each: ~1000 seconds, 17 minutes. Shardable across cores.