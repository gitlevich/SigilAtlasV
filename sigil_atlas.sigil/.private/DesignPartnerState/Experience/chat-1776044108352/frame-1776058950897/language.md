# Decision Frame

Viewing: Explore > Screen > SigilControls > ContrastControl

## User

good. Where would these improvements go? Under which sigils?

What would be the structure?

## Partner

I’d put them as close as possible to the product surface they govern, not in a separate theory layer. The main home should be the Explore sigil, with child sigils for the geometric model, traversal semantics, neighborhood semantics, preference/relevance dynamics, and view-state projection, because those seem like the core semantic units that code must implement consistently.

Then each of those sigils should carry three kinds of content in plain product language: what state exists, what operations transform it, and what invariants must survive those transformations. UI-specific sigils should only describe projection and local interaction rules, while anything that changes meaning across the whole app, like what “timelike” means or how a neighborhood is computed, should live higher in Explore so the surface can vary without changing the underlying code contract.