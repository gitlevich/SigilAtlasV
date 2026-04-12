# Decision Frame

Viewing: Corpus

## User

Rather, re-add. So when we packed the images into image sigils, we captured every contrast for every image. Then we aggregated into ImageNeighborhoodSigils, gradually reducing contrasts by one, ending up with multiple essentially top-level sigils in this differing containment structure. That's what we're slicing, and that's through what we're traversing.

## Partner

Yes, that changes the core rule in a good way: zoom is not revealing hidden detail from one fixed embedding, it is re-adding one contrast to the sliced sigil structure at each generation. So the object being explored is the contrast-stratified containment system itself, and the frame is a slice through whichever generation of that system is currently active, with orientation held stable while the partition refines.

That means the main specification task is to define how a slice intersects an ImageNeighborhoodSigil hierarchy and how parent regions split into child regions when one contrast is reintroduced. If we get that transition rule right, the user will feel that each zoom step is the same world becoming more discriminating, rather than a jump to a different representation.

I think Explore should now be defined around one recursive promise: every visible cell at generation n maps to a bounded set of child cells at generation n+1, and their union preserves the parent cell’s place in the frame. If you want, I can turn that into exact domain language for Frame, Generation, Slice, Cell, Boundary, and Zoom so the prototype has a precise build target.