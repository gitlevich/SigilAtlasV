Only @sigils at the same @scope can compose within a peer-group. A peer-group declares its scope; every member must match. Bringing a term from a different scope into the current one requires explicit wrapping in a peer-group at the current scope — never implicit promotion or demotion.

This invariant is enforced structurally by the grammar: `And` and `Or` take a list of peers, and the affordance system refuses members whose scope does not match the group's.
