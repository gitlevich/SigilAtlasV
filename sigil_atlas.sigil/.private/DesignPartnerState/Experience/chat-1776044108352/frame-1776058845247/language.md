# Decision Frame

Viewing: Explore > Screen > SigilControls > ContrastControl

## User

All right. So how would we—I mean, what do you feel any decoherence in the spec? I need it to be projectable into code unambiguously.

So functional errors we should prevent, like we should define all the functionality, but there are going to be little UI annoyances and things on the surface that does not require changing anything deeper in the code. Because my experience is it's easier to reproject from scratch than to iterate over the same code base because it decoheres.

## Partner

Yes. The remaining decoherence I feel is less about the philosophy now and more about missing operational contracts: the spec evokes Explore very well, but it still does not always say exactly what state exists, what transforms are allowed, what remains invariant under each transform, and what data structure realizes each visible phenomenon.

The biggest risk is ambiguity between representation and interaction. Right now terms like neighborhood, slice, timelike direction, relevance, zoom generation, and preference sometimes read as experiential concepts and sometimes as implementation objects, so a builder could make several incompatible systems that all sound faithful; if you want code to stay reprojectable, each of those needs one canonical meaning, one owned state location, and one precise update rule.

The good news is this is very fixable. The spec wants a small set of hard invariants and transition semantics beneath a flexible surface, so the next step is to define Explore as a state machine over a geometric model: what the corpus embedding is, what a neighborhood is, what spacelike view returns, what timelike traversal orders, how zoom refines without breaking orientation, how preference updates the field, and which UI variations are explicitly declared non-semantic.